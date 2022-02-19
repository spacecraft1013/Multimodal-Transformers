from __future__ import annotations

import multiprocessing as mp
import os
from typing import Generator, Iterable

import tokenizers
import torch
import yaml
from torch import optim
from torch.distributed.optim.zero_redundancy_optimizer import \
    ZeroRedundancyOptimizer
from torch.utils.data import Dataset

from model import ImageTransformer, TextTransformer, Transformer


def build_model(d_model: int = None, n_layers: int = None, n_heads: int = None, head_dim: int = None, feedforward_dim: int = None, seq_len: int = None, image_size: int = None, patch_size: int = None, num_classes: int = None, vocab_len: int = None, dropout: float = None, load_path: str = None, **kwargs) -> tuple[ImageTransformer, TextTransformer, int]:
    if load_path is not None:
        data = torch.load(load_path)
        config = data['config']
        transformer_state_dict = data['transformer']
        image_transformer_state_dict = data['image_transformer_params']
        text_transformer_state_dict = data['text_transformer_params']

        for k, v in config['model'].items():
            locals()[k] = v

    transformer = Transformer(
        n_layers, d_model, n_heads, head_dim, feedforward_dim, dropout)

    if load_path is not None:
        transformer.load_state_dict(transformer_state_dict)

    image_transformer = ImageTransformer(
        transformer, d_model, num_classes, image_size, patch_size)
    text_transformer = TextTransformer(transformer, d_model, seq_len, vocab_len)

    if load_path is not None:
        image_transformer.load_state_dict(
            image_transformer_state_dict, strict=False)
        text_transformer.load_state_dict(
            text_transformer_state_dict, strict=False)

    num_parameters = sum(param.numel() for param in transformer.parameters())
    return image_transformer, text_transformer, num_parameters


def build_optimizer(image_transformer: ImageTransformer, text_transformer: TextTransformer, optimizer_config: dict) -> torch.optim.Optimizer:
    params = list(image_transformer.parameters()) + \
        list(text_transformer.parameters())
    optimizer_args = optimizer_config['params']

    if optimizer_config['type'].lower() == 'adam':
        optimizer_class = optim.Adam
    elif optimizer_config['type'].lower() == 'sgd':
        optimizer_class = optim.SGD
    else:
        raise NotImplementedError(
            "Only Adam and SGD optimizers are currently supported")

    if optimizer_config['ZeRO']:
        optimizer = ZeroRedundancyOptimizer(
            params, optimizer_class, **optimizer_args)
    else:
        optimizer = optimizer_class(params, **optimizer_args)

    return optimizer


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


class Config:
    def __init__(self, config_path: str = None) -> None:
        if config_path is not None:
            self.config_dict = yaml.safe_load(open(config_path, 'r'))
            self.add_dict(self.config_dict)

    def add_dict(self, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config())
                getattr(self, k).add_dict(v)
            else:
                setattr(self, k, v)


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
