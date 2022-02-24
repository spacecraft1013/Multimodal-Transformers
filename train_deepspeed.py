import argparse
import itertools
import os

import deepspeed
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets as imagesdatasets
from torchvision import transforms

from model import MultimodalTransformer
from utils import WikiTextDataset, alternating_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train(args)


def train(args):
    config = yaml.safe_load(open(args.config, 'r'))

    data_dir = config['data']['data_dir']

    print("Loading Imagenet")
    image_transforms = transforms.Compose([
        transforms.Resize((config['model']['image_size'],
                           config['model']['image_size'])),
        transforms.ToTensor()
    ])

    imagenet_dataset = imagesdatasets.ImageNet(
        config['data']['imagenet_dir'], transform=image_transforms)
    imagenet_loader = DataLoader(imagenet_dataset, shuffle=True,
                                 batch_size=config['training']['image_batch_size'], pin_memory=True, drop_last=True)

    print("Loading WikiText103")
    if not os.path.exists(os.path.join(data_dir, config['data']['wikitext_dataset'])):
        tokenizer = Tokenizer(BPE.from_file(
            vocab=os.path.join(data_dir, 'vocab.json'), merges=os.path.join(data_dir, 'merges.txt')))
        text_dataset = WikiTextDataset(
            config['data']['wikitext_dir'], split='train', tokenizer=tokenizer, seq_len=config['model']['seq_len'], num_preprocessing_workers=config['data']['num_preprocessing_workers'])
        text_dataset.save(os.path.join(
            data_dir, config['data']['wikitext_dataset']))
    else:
        text_dataset = WikiTextDataset.from_preprocessed(
            os.path.join(data_dir, config['data']['wikitext_dataset']), seq_len=config['model']['seq_len'])

    text_loader = DataLoader(
        text_dataset, batch_size=config['training']['text_batch_size'], pin_memory=True, drop_last=True)

    imagenet_cycler = itertools.cycle(imagenet_loader)
    text_cycler = itertools.cycle(text_loader)

    loader = alternating_generator(config['training']['alternate_iters'],
                                   imagenet_cycler, text_cycler, first_item=config['training']['start_with'])
    loader_with_mask = zip(loader, itertools.cycle(torch.triu(torch.ones(config['model']['seq_len'], config['model']
                                                                         ['seq_len']) * float('-inf'), diagonal=1)))

    print("Building Model & Optimizer")
    multimodal_transformer = MultimodalTransformer(
        num_classes=1000, vocab_len=text_dataset.vocab_size, **config['model'])
    pipeline_model = multimodal_transformer.parallelize(
        pipeline_stages=config['parallel']['pp'])

    multimodal_transformer_parallelism_engine, _, _, _ = deepspeed.initialize(
        args=args, model=pipeline_model, model_parameters=pipeline_model.parameters(), training_data=loader_with_mask)

    num_parameters = sum(param.numel()
                         for param in pipeline_model.parameters())
    print(f"Number of transformer parameters: {num_parameters:,}")

    loss_fn = nn.CrossEntropyLoss()

    running_loss_text = 0.0
    running_loss_images = 0.0

    for step in range(config['training']['train_iters']):
        loss = multimodal_transformer_parallelism_engine.train_batch()

    multimodal_transformer_parallelism_engine.save_state_dict(
        config['data']['checkpoint_dir'])
