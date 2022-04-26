import os
from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args, get_timers, print_rank_0
from megatron.model import ModelType
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from utils import MultimodalDataset, args_provider, build_megatron_model


def get_batch(data_iterator):
    """Build the batch."""
    inputs = next(data_iterator)

    # only data parallelism; no need for broadcast
    data = inputs[0].cuda()
    labels = inputs[1].cuda()

    return data, labels


def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    data, labels = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    if len(data.size()) == 4:
        key = 'images'
    elif len(data.size()) == 2:
        key = 'text'
    else:
        raise ValueError('Invalid Data Size')
    output_tensor = model(data, key)

    return output_tensor, partial(loss_func, labels)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "Building Multimodal Dataset"
    )

    train_ds = MultimodalDataset(args)

    return train_ds, None, None


if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15000"

    pretrain(
        train_valid_test_datasets_provider,
        build_megatron_model,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic'},
        extra_args_provider=partial(args_provider, 'megatron_config.yml')
    )
