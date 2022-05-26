import os
from functools import partial

import torch
import torch.nn.functional as F

from datasets import MultimodalDataset
from megatron import get_args, get_timers, print_rank_0
from megatron.model import ModelType
from megatron.model.multimodal_model import build_megatron_model
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from utils import args_provider


def get_batch(data_iterator):
    """Build the batch."""
    inputs = next(data_iterator)

    # only data parallelism; no need for broadcast
    data = inputs[0].cuda()
    labels = inputs[1].cuda()

    return data, labels


def loss_func(labels, output_tensor, loss_mask=None):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    if loss_mask:
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(loss.view(-1)*loss_mask) / torch.sum(loss_mask)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    batch_data = get_batch(data_iterator)
    timers("batch-generator").stop()

    if batch_data[0].dim() == 4 and len(batch_data) == 2:
        data, labels = batch_data
        output_tensor = model(data)
        return output_tensor, partial(loss_func, labels)

    elif batch_data[0].dim() == 2 and len(batch_data) == 4:
        data, labels, loss_mask, attention_mask, position_ids = batch_data
        output_tensor = model(data, position_ids, attention_mask)
        return output_tensor, partial(loss_func, labels=labels, loss_mask=loss_mask)


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
        extra_args_provider=partial(args_provider, 'megatron_config.yml')
    )
