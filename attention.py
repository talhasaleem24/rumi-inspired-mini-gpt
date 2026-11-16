import json
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 16
BLOCK_SIZE = 128

EMBED_DIM = 64
NUM_HEADS = 4

DROPOUT = 0.1
DEVICE = "cpu"

if EMBED_DIM % NUM_HEADS != 0:
    raise ValueError("EMBED_DIM must be divisible by NUM_HEADS.")

with (PROJECT_DIR / "vocab.json").open("r", encoding="utf-8") as file:
    _stoi = json.load(file)

if sorted(_stoi.values()) != list(range(len(_stoi))):
    raise ValueError("Vocabulary IDs must be contiguous and start at zero.")

vocab_size = len(_stoi)


def apply_rope(x):
    if x.ndim != 4:
        raise ValueError(
            "RoPE expects a tensor shaped (batch, heads, sequence, head_dim)."
        )

    _, _, seq_len, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device,
                dtype=torch.float32
            ) / head_dim
        )
    )

    positions = torch.arange(
        seq_len,
        device=x.device,
        dtype=torch.float32
    )

    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :].to(x.dtype)
    sin = angles.sin()[None, None, :, :].to(x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack(
        [rotated_even, rotated_odd],
        dim=-1
    ).flatten(-2)


class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            )
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, channels)

        return self.proj(output), attention_weights


class FeedForward(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output, weights = self.attention(self.ln1(x))
        x = x + self.residual_dropout(attention_output)
        x = x + self.feedforward(self.ln2(x))
        return x, weights
