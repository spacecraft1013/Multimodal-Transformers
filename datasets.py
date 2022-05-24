import json
import multiprocessing as mp
import os
from typing import Iterable

import tokenizers
import torch
from torch.utils.data import Dataset
from torchvision import datasets as imagedatasets
from torchvision import transforms

from megatron.tokenizer.tokenizer import build_tokenizer


class DatasetConfig:
    def __init__(self, input) -> None:
        if isinstance(input, str):
            input = json.loads(input)

        self.add_dict(input)

    def add_dict(self, d: dict):
        for key, val in d.items():
            key = key.replace('-', '_')
            if isinstance(val, dict):
                setattr(self, key, DatasetConfig())
                getattr(self, key).add_dict(val)
            else:
                setattr(self, key, val)


class MultimodalDataset(Dataset):
    def __init__(self, args, name: str = None, image_indices=None, text_indices=None) -> None:
        super().__init__()

        self.name = name

        if isinstance(args.multimodal_datasets, (str, dict)):
            args.multimodal_datasets = DatasetConfig(args.multimodal_datasets)

        if args.multimodal_datasets.image_dataset.lower() == "imagenet":
            image_dataset = ImagenetDataset(args)
        else:
            raise ValueError(f"Unknown image dataset: {args.multimodal_datasets.image_dataset}")

        if args.multimodal_datasets.text_dataset.lower() == "wikitext":
            assert args.multimodal_datasets.wikitext_dir is not None, "Wikitext directory not specified"
            assert args.multimodal_datasets.wikitext_dataset is not None, "Wikitext file not specified"
            save_location = os.path.join(
                args.multimodal_datasets.wikitext_dir, args.multimodal_datasets.wikitext_dataset)
            if not os.path.exists(save_location):
                tokenizer = build_tokenizer(args)
                text_dataset = WikiTextDataset(
                    args.multimodal_datasets.wikitext_dir, split='train', tokenizer=tokenizer, seq_len=args.seq_length, num_preprocessing_workers=args.multimodal_datasets.num_preprocessing_workers)
                text_dataset.save(save_location)
            else:
                text_dataset = WikiTextDataset.from_preprocessed(
                    save_location, seq_len=args.seq_length)
        elif args.multimodal_datasets.text_dataset.lower == "pile":
            raise NotImplementedError("Pile dataset not implemented yet")
        else:
            raise ValueError(f"Unknown text dataset: {args.multimodal_datasets.text_dataset}")

        self.text_dataset = text_dataset[text_indices] if text_indices is not None else text_dataset
        self.image_dataset = image_dataset[image_indices] if image_indices is not None else image_dataset

        self.state = args.multimodal_datasets.start_sample
        self.switch = args.micro_batch_size
        self.count = 0

    def __len__(self) -> int:
        return max(len(self.image_dataset), len(self.text_dataset))

    def __getitem__(self, idx: int) -> Iterable:
        if self.count >= self.switch:
            self.count = 0
            self.state = ({"images", "text"} - {self.state})[0]
        if self.state == "images":
            self.count += 1
            if idx >= len(self.image_dataset):
                idx = idx % len(self.image_dataset)
            return self.image_dataset[idx]
        elif self.state == "text":
            self.count += 1
            if idx >= len(self.text_dataset):
                idx = idx % len(self.text_dataset)
            return self.text_dataset[idx]


class ImagenetDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        image_transforms = transforms.Compose([
            transforms.Resize(
                (args.img_dim, args.img_dim)),
            transforms.ToTensor()
        ])
        self.dataset = imagedatasets.ImageNet(
            args.multimodal_datasets.imagenet_dir, transform=image_transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Iterable:
        return self.dataset[idx]


class WikiTextDataset(Dataset):
    def __init__(self, path: str = None, split: str = 'train', tokenizer: tokenizers.Tokenizer = None, seq_len: int = None, num_preprocessing_workers: int = -1) -> None:
        self.data_path = path
        self.split = split
        self.tokenizer = tokenizer
        if tokenizer:
            self.vocab_size = tokenizer.vocab_size
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
            self.tokenizer.tokenize(sample), dtype=torch.long)

    def preprocess(self) -> None:
        print(f"Preprocessing set with {self.num_workers} workers")

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                dataset = pool.map(self._tokenize, self.raw_data)

        else:
            dataset = []
            for sample in self.raw_data:
                dataset.append(torch.tensor(
                    self.tokenizer.tokenize(sample), dtype=torch.long))

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
