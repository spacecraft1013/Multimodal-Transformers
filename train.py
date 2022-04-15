import itertools
import os

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets as imagesdatasets
from torchvision import transforms
from tqdm import trange

from utils import (Config, WikiTextDataset, alternating_generator, build_model,
                   build_optimizer)

config = Config('config.yml')

device = torch.device(config.device)
data_dir = config.data.data_dir

print("Loading Imagenet")
image_transforms = transforms.Compose([
    transforms.Resize((config.model.image_size,
                       config.model.image_size)),
    transforms.ToTensor()
])

imagenet_dataset = imagesdatasets.ImageNet(
    config.data.imagenet_dir, transform=image_transforms)
imagenet_loader = DataLoader(imagenet_dataset, shuffle=True,
                             batch_size=config.training.image_batch_size, pin_memory=True, drop_last=True)

print("Loading WikiText103")
if not os.path.exists(os.path.join(data_dir, config.data.wikitext_dataset)):
    tokenizer = Tokenizer(BPE.from_file(
        vocab=os.path.join(data_dir, 'vocab.json'), merges=os.path.join(data_dir, 'merges.txt')))
    text_dataset = WikiTextDataset(
        config.data.wikitext_dir, split='train', tokenizer=tokenizer, seq_len=config.model.seq_len, num_preprocessing_workers=config.data.num_preprocessing_workers)
    text_dataset.save(os.path.join(
        data_dir, config.data.wikitext_dataset))
else:
    text_dataset = WikiTextDataset.from_preprocessed(
        os.path.join(data_dir, config.data.wikitext_dataset), seq_len=config.model.seq_len)

text_loader = DataLoader(
    text_dataset, batch_size=config.training.text_batch_size, pin_memory=True, drop_last=True)

imagenet_cycler = itertools.cycle(imagenet_loader)
text_cycler = itertools.cycle(text_loader)

loader = alternating_generator(config.training.alternate_iters,
                               imagenet_cycler, text_cycler, first_item=config.training.start_with)

print("Building Model & Optimizer")
image_transformer, text_transformer, num_parameters = build_model(
    num_classes=1000, vocab_len=text_dataset.vocab_size, **config.model.__dict__)

image_transformer.to(device)
text_transformer.to(device)

print(f"Number of transformer parameters: {num_parameters:,}")

optimizer = build_optimizer(
    image_transformer, text_transformer, config.optimizer)
scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss()

image_transformer.train()
text_transformer.train()

running_loss_text = 0.0
running_loss_images = 0.0

image_steps = 0
text_steps = 0

if config.training.start_with == 'images':
    using_images = True
    using_text = False
elif config.training.start_with == 'text':
    using_images = False
    using_text = True
else:
    raise ValueError(
        f'Start value must be either "images" or "text", got {config.training.start_with}')

mask = torch.triu(torch.ones(config.model.seq_len, config.model
                             .seq_len, device=device) * float('-inf'), diagonal=1)

progressbar = trange(config.training.train_iters, desc='Training')
for step in progressbar:

    data, datatype = next(loader)

    src, tgt = data
    src, tgt = src.to(device), tgt.to(device)

    if datatype == 'images':
        using_images = True
        using_text = False
        model = image_transformer
        image_steps += 1
        model_args = tuple(src)

    elif datatype == 'text':
        using_images = False
        using_text = True
        model = text_transformer
        text_steps += 1
        model_args = tuple(src, mask)

    with autocast():
        output = model(*model_args)

        if using_text:
            output = output.view(-1, output.size(-1))
            tgt = tgt.view(-1)

        loss = loss_fn(output, tgt)

    if using_images:
        running_loss_images += loss.item()
    elif using_text:
        running_loss_text += loss.item()

    progressbar.set_description(
        f'Training, Text Loss: {running_loss_text / (text_steps + 1):.3f}, Image Loss: {running_loss_images / (image_steps + 1):.3f}')

    scaler.scale(loss).backward()

    if step % config.training.gradient_accumulation_steps == 0:
        nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clipping)

        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

save_dict = {
    'config': config,
    'transformer': image_transformer.transformer.state_dict(),
    'image_transformer_params': {k: v for k, v in image_transformer.state_dict() if k not in image_transformer.transformer.state_dict()},
    'text_transformer_params': {k: v for k, v in text_transformer.state_dict() if k not in text_transformer.transformer.state_dict()}
}

torch.save(save_dict, 'model.pt')
