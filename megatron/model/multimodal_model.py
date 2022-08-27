"""Vision Transformer(VIT) model."""


import einops
import torch
from megatron import get_args, print_rank_0
from megatron.model.enums import AttnMaskType
from megatron.model.gpt_model import post_language_model_processing
from megatron.model.language_model import Embedding, Pooler
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import (get_linear_layer, init_method_normal,
                                  scaled_init_method_normal)
from megatron.model.vit_model import (
    VitMlpHead, twod_interpolate_position_embeddings_hook)
from torch import nn

from .module import MegatronModule


class MultimodalTransformer(nn.Module):
    def __init__(self,
                 init_method,
                 scaled_init_method,
                 attn_mask_type,
                 num_tokentypes,
                 add_pooler,
                 pre_process,
                 post_process,
                 num_image_classes) -> None:
        super().__init__()
        self.transformer = ParallelTransformer(
            init_method,
            scaled_init_method,
            self_attn_mask_type=attn_mask_type,
            pre_process=pre_process,
            post_process=post_process
        )

        self.language_model = TransformerLanguageModel(
            self.transformer,
            init_method,
            scaled_init_method,
            attn_mask_type=attn_mask_type,
            num_tokentypes=num_tokentypes,
            add_pooler=add_pooler,
            pre_process=pre_process,
            post_process=post_process
        )

        self.image_model = VitModel(
            self.transformer,
            num_classes=num_image_classes,
            pre_process=pre_process,
            post_process=post_process
        )

    def set_input_tensor(self, input_tensor):
        if input_tensor[0] is None:
            self.transformer.set_input_tensor(input_tensor)
        elif input_tensor[0].dim() == 4:
            self.transformer.set_input_tensor(input_tensor)
        elif input_tensor[0].dim() == 2:
            self.transformer.set_input_tensor(input_tensor[0])

    def forward(self, input):
        if len(input) == 1:
            return self.image_model(input[0])
        else:
            return self.language_model(*input)


class VitModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 transformer_backbone,
                 num_classes,
                 finetune=False,
                 pre_process=True,
                 post_process=True):
        super(VitModel, self).__init__(share_word_embeddings=False)
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, args.num_layers
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.patch_dim = args.patch_dim
        self.img_dim = args.img_dim
        self.finetune = finetune

        assert self.img_dim % self.patch_dim == 0
        self.num_patches_per_dim = self.img_dim // self.patch_dim
        self.num_patches = self.num_patches_per_dim ** 2
        self.seq_length = self.num_patches + 1
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels

        if self.pre_process:
            # cls_token
            self.cls_token = torch.nn.Parameter(
                torch.randn(1, 1, self.hidden_size)
            )
            torch.nn.init.zeros_(self.cls_token)

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )
            self.position_ids = torch.arange(
                self.seq_length).expand(1, -1).cuda()

            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

        # Transformer
        # self.transformer = ParallelTransformer(
        #     self.init_method,
        #     self.scaled_init_method,
        #     pre_process=self.pre_process,
        #     post_process=self.post_process
        # )

        self.transformer = transformer_backbone

        if self.post_process:
            # MLP head
            if not self.finetune:
                self.mlp_head = VitMlpHead(self.hidden_size, self.num_classes)
            else:
                self.class_head = get_linear_layer(
                    self.hidden_size, num_classes, torch.nn.init.zeros_
                )

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half
            encoder_output = self.linear_encoder(rearranged_input)
            cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
            concatenated_tokens = torch.cat(
                (cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                self.position_embeddings(self.position_ids)
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.transformer(hidden_states, None)

        if self.post_process:
            if not self.finetune:
                hidden_states = self.mlp_head(hidden_states)
            else:
                hidden_states = self.class_head(hidden_states[:, 0, :])

        return hidden_states


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 transformer_backbone,
                 init_method,
                 output_layer_init_method,
                 attn_mask_type,
                 num_tokentypes=0,
                 parallel_output=True,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.attn_mask_type = attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       self.init_method,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Transformer.
        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        # self.encoder = ParallelTransformer(
        #     self.init_method,
        #     output_layer_init_method,
        #     self_attn_mask_type=self.encoder_attn_mask_type,
        #     pre_process=self.pre_process,
        #     post_process=self.post_process
        # )
        self.encoder = transformer_backbone
        self._encoder_key = 'encoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask, labels,
                tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None):

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            return post_language_model_processing(encoder_output, labels, self.word_embeddings_weight(), self.parallel_output, self.fp16_lm_cross_entropy)
        else:
            return encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        state_dict_[self._encoder_key] \
            = self.encoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(
                        destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]
        # For backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # For backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[
                        1]] = state_dict[key]

        # For backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.",
                                                      ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)


def build_megatron_model(pre_process: bool, post_process: bool) -> MultimodalTransformer:

    print_rank_0("Building Multimodal Transformer")

    args = get_args()

    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                   args.num_layers)

    multimodal_transformer = MultimodalTransformer(
        init_method=init_method,
        scaled_init_method=scaled_init_method,
        attn_mask_type=AttnMaskType.padding,
        num_tokentypes=0,
        add_pooler=False,
        pre_process=pre_process,
        post_process=post_process,
        num_image_classes=args.num_classes
    )

    return multimodal_transformer
