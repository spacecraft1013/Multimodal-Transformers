import os
from functools import partial

import torch
import torch.nn.functional as F

from datasets import build_multimodal_datasets
from megatron import get_args, get_timers, get_tokenizer, mpu, print_rank_0
from megatron.model import ModelType
from megatron.model.multimodal_model import build_megatron_model
from megatron.training import pretrain
from megatron.utils import (average_losses_across_data_parallel_group,
                            get_ltor_masks_and_position_ids)
from utils import args_provider


def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    if data[0].dim() == 4:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        return 0, inputs, labels
    else:
        args = get_args()
        tokenizer = get_tokenizer()

        keys = ['text']
        dtype = torch.int64
        data_broadcasted = mpu.broadcast_data(keys, data, dtype)
        tokens_ = data_broadcasted['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return 1, tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(output_tensor, labels=None, loss_mask=None):
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

    if batch_data[0] == 0:
        _, data, labels = batch_data
        output_tensor = model((data,))
        return output_tensor, partial(loss_func, labels=labels)

    elif batch_data[0] == 1:
        _, tokens, labels, loss_mask, attention_mask, position_ids = batch_data
        output_tensor = model((tokens, position_ids, attention_mask))
        return output_tensor, partial(loss_func, labels=labels, loss_mask=loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "Building Multimodal Dataset"
    )

    return build_multimodal_datasets(args, train_val_test_num_samples)


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
