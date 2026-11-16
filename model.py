import math

import torch
import torch.nn as nn

from attention import (
    BLOCK_SIZE,
    DROPOUT,
    EMBED_DIM,
    NUM_HEADS,
    TransformerBlock,
    vocab_size
)


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError("Input tokens must have shape (batch, sequence).")

        _, seq_len = idx.shape

        if seq_len == 0:
            raise ValueError("Input sequence must not be empty.")
        if seq_len > BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {seq_len} exceeds {BLOCK_SIZE}."
            )
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("Target shape must match input shape.")

        x = self.token_embedding(idx)

        for block in self.blocks:
            x, _ = block(x)

        logits = self.lm_head(self.final_ln(x))
        loss = None

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        forbidden_token_ids=None
    ):
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(
                "Input tokens must have shape (batch, non-empty sequence)."
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Temperature must be a finite number greater than zero."
            )
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if forbidden_token_ids is not None:
            invalid_ids = [
                token_id for token_id in forbidden_token_ids
                if token_id < 0 or token_id >= self.vocab_size
            ]
            if invalid_ids:
                raise ValueError(
                    f"Forbidden token IDs are outside the vocabulary: "
                    f"{invalid_ids}"
                )
            if len(set(forbidden_token_ids)) >= self.vocab_size:
                raise ValueError("At least one token must remain sampleable.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if forbidden_token_ids:
                logits[:, forbidden_token_ids] = float("-inf")

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                threshold = values[:, [-1]]
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=2,
    dropout=DROPOUT
).to("cpu")


if __name__ == "__main__":
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    expected_parameters = 100096 + (64 * vocab_size)

    print("Vocabulary size:", vocab_size)
    print("Embedding dimension:", EMBED_DIM)
    print("Attention heads:", NUM_HEADS)
    print("Transformer blocks:", len(model.blocks))
    print("Total trainable parameters:", total_parameters)

    assert total_parameters == expected_parameters
    print("Parameter count verification passed!")
