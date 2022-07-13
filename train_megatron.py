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

    if isinstance(data, list): # image
        data = {
            'inputs': data[0],
            'labels': data[1]
        }
        keys = ['inputs', 'labels']
        dtype = torch.float32
        data_broadcasted = mpu.broadcast_data(keys, data, dtype)
        inputs = data_broadcasted['inputs']
        labels = data_broadcasted['labels']
        return 0, inputs, labels
    elif isinstance(data, dict): # text
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
    else:
        raise ValueError(f"Data does not match proper type, got type {type(data)}")


def loss_func(output_tensor, labels=None, loss_mask=None, sample_type=None):
    if sample_type == 0: # image
        logits = output_tensor.contiguous().float()
        loss = F.cross_entropy(logits, labels)

        outputs = torch.argmax(logits, -1)
        correct = (outputs == labels).float()
        accuracy = torch.mean(correct)

        averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

        return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}

    elif sample_type == 1: # text
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {"loss": averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    batch_data = get_batch(data_iterator)
    timers("batch-generator").stop()

    if batch_data[0] == 0:
        sample_type, data, labels = batch_data
        output_tensor = model((data,))
        return output_tensor, partial(loss_func, labels=labels, sample_type=sample_type)

    elif batch_data[0] == 1:
        sample_type, tokens, labels, loss_mask, attention_mask, position_ids = batch_data
        output_tensor = model((tokens, position_ids, attention_mask))
        return output_tensor, partial(loss_func, labels=labels, loss_mask=loss_mask, sample_type=sample_type)


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
        extra_args_provider=partial(args_provider, 'megatron_config.yml')
    )
