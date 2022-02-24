from __future__ import annotations

import multiprocessing as mp
import os
from typing import Generator, Iterable

import tokenizers
import torch
from torch.utils.data import Dataset


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
    def from_preprocessed(cls, filename: str, seq_len: int) -> WikiTextDataset:
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
