import multiprocessing as mp
import os
from typing import Generator, Iterable, Iterator

import tokenizers
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets as imagedatasets
from torchvision import transforms

from megatron import get_args, print_rank_0
from megatron.model.multimodal_model import MultimodalTransformer
from megatron.model.utils import init_method_normal, scaled_init_method_normal


def build_megatron_model(pre_process: bool, post_process: bool) -> MultimodalTransformer:

    print_rank_0("Building Multimodal Transformer")

    args = get_args()

    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)

    multimodal_transformer = MultimodalTransformer(
        init_method=init_method,
        scaled_init_method=scaled_init_method,
        attn_mask_type=args.attn_mask_type,
        num_tokentypes=0,
        add_pooler=args.add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        num_image_classes=args.num_image_classes,
        language_model_key='language_model',
        image_model_key='image_model'
    )

    return multimodal_transformer


def alternating_generator(frequency: int, images: Iterable, text: Iterable, first_item: str) -> Generator:
    if first_item == 'images':
        iterable1, iterable1type = images, 'images'
        iterable2, iterable2type = text, 'text'
    elif first_item == 'text':
        iterable1, iterable1type = text, 'text'
        iterable2, iterable2type = images, 'images'
    else:
        raise ValueError(
            f'First item must be either "images" or "text", got {first_item}')

    while True:
        for i in range(frequency):
            yield next(iterable1), iterable1type
        for i in range(frequency):
            yield next(iterable2), iterable2type


class MultimodalDataset(IterableDataset):
    def __init__(self, args, frequency: int = 2, first_item: str = 'images') -> None:
        super().__init__()

        image_transforms = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        self.imagenet_dataset = imagedatasets.ImageNet(
            args.image_data_path, transform=image_transforms)

        if not os.path.exists(os.path.join(args.text_data_path, 'wikitext.pth')):
            tokenizer = Tokenizer(BPE.from_file(
                vocab=os.path.join(args.text_data_path, 'vocab.json'), merges=os.path.join(args.text_data_path, 'merges.txt')))
            text_dataset = WikiTextDataset(
                os.path.join(args.text_data_path, 'WikiText'), split='train', tokenizer=tokenizer, seq_len=args.seq_len, num_preprocessing_workers=args.num_preprocessing_workers)
            text_dataset.save(os.path.join(
                args.text_data_path, 'wikitext.pth'))
        else:
            text_dataset = WikiTextDataset.from_preprocessed(
                os.path.join(args.text_data_path, 'wikitext.pth'), seq_len=args.seq_len)

        self.text_dataset = text_dataset
        self.frequency = frequency
        self.first_item = first_item

    def __len__(self) -> int:
        return len(self.text_dataset) + len(self.imagenet_dataset)

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return iter(alternating_generator(
                self.frequency, self.imagenet_dataset, self.text_dataset, self.first_item))
        else:
            per_worker = int(torch.ceil(
                (self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self))

            return iter(alternating_generator(
                self.frequency, self.imagenet_dataset[start//2:end//2], self.text_dataset[start//2:end//2], self.first_item))


class WikiTextDataset(Dataset):
    def __init__(self, path: str = None, split: str = 'train', tokenizer: tokenizers.Tokenizer = None, seq_len: int = None, num_preprocessing_workers: int = -1) -> None:
        self.data_path = path
        self.split = split
        self.tokenizer = tokenizer
        if tokenizer:
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            self.vocab_size = None
        self.seq_len = seq_len
        self.preprocessed = False
        self.num_workers = mp.cpu_count() if num_preprocessing_workers == -1 \
            else num_preprocessing_workers

        if path is not None:
            self.raw_data = list(
                open(os.path.join(path, f'wiki.{split}.tokens'), encoding="utf8").readlines())

    @classmethod
    def from_preprocessed(cls, filename: str, seq_len: int):
        instance = cls(seq_len=seq_len)
        data = torch.load(filename)

        dataset = data['dataset']
        vocab_len = data['vocab_len']

        num_sequences = dataset.size(0) // instance.seq_len
        dataset = dataset[:num_sequences * instance.seq_len]
        dataset = dataset.view(instance.seq_len, num_sequences).t()

        instance.dataset = dataset
        instance.vocab_size = vocab_len
        instance.preprocessed = True
        return instance

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx], self.dataset[idx + 1]

    def _tokenize(self, sample: str) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(sample).ids, dtype=torch.long)

    def preprocess(self) -> None:
        print(f"Preprocessing set with {self.num_workers} workers")

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                dataset = pool.map(self._tokenize, self.raw_data)

        else:
            dataset = []
            for sample in self.raw_data:
                dataset.append(torch.tensor(
                    self.tokenizer.encode(sample).ids, dtype=torch.long))

        dataset = torch.cat(
            tuple(filter(lambda x: x.numel() > 0, dataset)))

        self.saveable_dataset = dataset.clone()

        num_sequences = dataset.size(0) // self.seq_len
        dataset = dataset[:num_sequences * self.seq_len]
        self.dataset = dataset.view(self.seq_len, num_sequences).t()

        self.preprocessed = True

    def save(self, filename: str) -> None:
        if not self.preprocessed:
            self.preprocess()

        save_dict = {
            'dataset': self.saveable_dataset,
            'vocab_len': self.vocab_size,
        }

        torch.save(save_dict, filename)
        del self.saveable_dataset
