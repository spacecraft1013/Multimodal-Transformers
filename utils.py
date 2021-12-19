from __future__ import annotations

import multiprocessing as mp
import os

import tokenizers
import torch
from torch import optim
from torch.distributed.optim.zero_redundancy_optimizer import \
    ZeroRedundancyOptimizer
from torch.utils.data import Dataset

from model import ImageTransformer, TextTransformer, TransformerEncoder


def build_model(d_model: int, n_layers: int, n_heads: int, head_dim: int, feedforward_dim: int, seq_len: int, image_size: int, patch_size: int, num_classes: int, vocab_len: int, dropout: float = 0.1, **kwargs) -> tuple[ImageTransformer, TextTransformer, int]:
    encoder = TransformerEncoder(
        n_layers, d_model, n_heads, head_dim, feedforward_dim, dropout)
    # decoder = TransformerDecoder(
    #     n_layers, d_model, n_heads, head_dim, feedforward_dim, dropout)

    image_transformer = ImageTransformer(
        encoder, d_model, num_classes, image_size, patch_size)
    text_transformer = TextTransformer(encoder, d_model, seq_len, vocab_len)

    num_parameters = sum(param.numel() for param in encoder.parameters())
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
