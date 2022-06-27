import json
import multiprocessing as mp
import os
from typing import Iterable

import tokenizers
import torch
from torch.utils.data import Dataset
from torchvision import datasets as imagedatasets
from torchvision import transforms

from megatron.data.gpt_dataset import \
    build_train_valid_test_datasets as build_gpt_datasets
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
                setattr(self, key, DatasetConfig(val))
            else:
                setattr(self, key, val)


class MultimodalDataset(Dataset):
    def __init__(self, text_dataset, image_dataset, name: str = None) -> None:
        super().__init__()

        self.name = name

        self.text_dataset = text_dataset if text_dataset is not None else []
        self.image_dataset = image_dataset if image_dataset is not None else []

        self.dataset_list = [self.text_dataset, self.image_dataset]

    def __len__(self) -> int:
        return sum(map(len, self.dataset_list))

    def __getitem__(self, idx: int):
        if not self.text_dataset:
            dataset = self.image_dataset
        elif not self.image_dataset:
            dataset = self.text_dataset
        else:
            dataset = self.dataset_list[idx % 2]

        if idx >= len(dataset):
            idx = idx % len(dataset)
        return dataset[idx]


class ImagenetDataset(Dataset):
    def __init__(self, args, split: str = 'train') -> None:
        super().__init__()

        image_transforms = transforms.Compose([
            transforms.Resize(
                (args.img_dim, args.img_dim)),
            transforms.ToTensor()
        ])
        self.dataset = imagedatasets.ImageNet(
            args.multimodal_datasets.image_dataset.dir, split=split, transform=image_transforms)

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


def build_wikitext_datasets(args):
    datasets = []
    for split in ("train", "valid", "test"):
        save_location = os.path.join(
            args.multimodal_datasets.text_dataset.dir, f'wikitext_{split}.pth')
        if not os.path.exists(save_location):
            tokenizer = build_tokenizer(args)
            text_dataset = WikiTextDataset(
                args.multimodal_datasets.text_dataset.dir, split=split, tokenizer=tokenizer, seq_len=args.seq_length, num_preprocessing_workers=args.multimodal_datasets.text_dataset.num_preprocessing_workers)
            text_dataset.save(save_location)
        else:
            text_dataset = WikiTextDataset.from_preprocessed(
                save_location, seq_len=args.seq_length)
        datasets.append(text_dataset)
    return datasets


def build_imagenet_datasets(args):
    datasets = []
    for split in ("train", "val"):
        datasets.append(ImagenetDataset(args, split=split))
    datasets.append(None)
    return datasets


def build_multimodal_datasets(args, train_val_test_num_samples):
    if isinstance(args.multimodal_datasets, (str, dict)):
        args.multimodal_datasets = DatasetConfig(args.multimodal_datasets)

    if args.multimodal_datasets.text_dataset.type.lower() == "pile":
        text_datasets = build_gpt_datasets(
            data_prefix=args.multimodal_datasets.text_dataset.dir,
            data_impl=args.multimodal_datasets.text_dataset.data_impl,
            splits_string=args.multimodal_datasets.text_dataset.splits,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup)
        )

    elif args.multimodal_datasets.text_dataset.type.lower() == "wikitext":
        text_datasets = build_wikitext_datasets(args)
    else:
        raise NotImplementedError(
            f"Text dataset {args.multimodal_datasets.text_dataset.type.lower()} is not available")

    if args.multimodal_datasets.image_dataset.type.lower() == "imagenet":
        image_datasets = build_imagenet_datasets(args)
    else:
        raise NotImplementedError(
            f"Image dataset {args.multimodal_datasets.image_dataset.type.lower()} is not available")

    multimodal_datasets = []
    for split, text_dataset, image_dataset in zip(("train", "valid", "test"), text_datasets, image_datasets):
        dataset = MultimodalDataset(
            name=split, text_dataset=text_dataset, image_dataset=image_dataset)
        multimodal_datasets.append(dataset)

    return multimodal_datasets
