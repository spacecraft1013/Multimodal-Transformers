import torch
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class ImageTransformer(nn.Module):
    def __init__(self, transformer: nn.Module, d_model: int, num_classes: int, image_size: int, patch_size: int, num_channels: int = 3) -> None:
        super().__init__()

        self.transformer = transformer
        self.patch_size = patch_size

        self.linear_projection = nn.Linear(
            patch_size**2 * num_channels, d_model)

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=self.patch_size, p2=self.patch_size)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Parameter(torch.randn(1,
                                                  ((image_size // patch_size) ** 2) + 1, d_model))

        self.classification_head = ImageTransformerClassificationHead(
            d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = self.rearrange(x)
        input_vector = self.linear_projection(patches)

        b, n = input_vector.size(0), input_vector.size(1)

        class_tokens = repeat(
            self.class_token, '() n d -> b n d', b=b)
        input_vector = torch.cat([class_tokens, input_vector], dim=1)

        input_vector += self.positions[:, :(n + 1)]
        x = self.transformer(input_vector)

        x = self.classification_head(x)

        return x


class ImageTransformerClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()

        self.reduce = Reduce('b n d -> b d', reduction='mean')
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.norm(x)
        x = self.linear(x)
        return x


class TextTransformer(nn.Module):
    def __init__(self, transformer: nn.Module, d_model: int, seq_length: int, vocab_len: int) -> None:
        super().__init__()

        self.transformer = transformer

        self.input_embedding = nn.Embedding(vocab_len, d_model)
        self.decoder = nn.Linear(d_model, vocab_len)

        self.register_buffer('positional_encoding',
                             positional_encoding(seq_length, d_model))

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        src = self.input_embedding(src)

        src += self.positional_encoding

        x = self.transformer(src, mask)
        y = self.decoder(x)
        return y


class ResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.norm(features[-1] + self.dropout(self.sublayer(*features, **kwargs)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_head: int = 64) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        inner_dim = dim_head * num_heads

        self.attention_heads = nn.ModuleList([
            AttentionHead(d_model, dim_head) for _ in range(num_heads)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        head_outputs = torch.cat([head(q, k, v, mask=mask)
                                 for head in self.attention_heads], dim=-1)
        return self.to_out(head_outputs)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, dim_head: int) -> None:
        super().__init__()

        self.q = nn.Linear(d_model, dim_head)
        self.k = nn.Linear(d_model, dim_head)
        self.v = nn.Linear(d_model, dim_head)

        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask
        attn = self.softmax(attn)
        out = torch.bmm(attn, v)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = ResidualConnection(
            MultiHeadAttention(d_model, num_heads, head_dim),
            d_model, dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )

        self.feedforward_block = ResidualConnection(
            self.feedforward, d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, x, x, mask=mask)
        x = self.feedforward_block(x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, head_dim: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerLayer(
            d_model, num_heads, head_dim, feedforward_dim, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask)

        return x


def positional_encoding(sequence_length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(sequence_length, dtype=torch.float).reshape(1, -1, 1)
    dim = torch.arange(d_model, dtype=torch.float).reshape(1, 1, -1)

    phase = pos / (10000 ** torch.div(2 * dim, d_model, rounding_mode='trunc'))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
