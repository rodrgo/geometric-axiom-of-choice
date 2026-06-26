"""Full-source denoising encoder.

Transformer encoder + LM head. We don't use a separate decoder: the
training objective is whole-word masked-LM (cross-entropy on masked
positions only), which only needs the encoder output projected to
vocabulary logits.

Sizes (from research/plan_full_data.md, Step 3.1):
  d_model = 256, n_heads = 8, d_ff = 1024
  6 encoder layers
  context length 512, learned positional embeddings
  GELU, dropout 0.1
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Special-token IDs. Trained tokenizer puts specials in this order.
PAD_ID = 0
UNK_ID = 1
MASK_ID = 2
BOS_ID = 3
EOS_ID = 4


class FullSourceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights — common practice, saves ~8M params at 32K vocab.
        self.lm_head.weight = self.tok_emb.weight

        self.proj = nn.Linear(d_model, proj_dim)

        # BERT-style init: tied embeddings need a small std so initial
        # logits aren't wild (default nn.Embedding init has std=1, which
        # for tied embeddings + 30K vocab gives huge softmax peaks).
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[PAD_ID].zero_()

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits[B, L, V], pooled[B, d_model])."""
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.emb_drop(x)

        pad_mask = ids == PAD_ID  # [B, L] True where padding
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        logits = self.lm_head(h)

        # Mean pool over non-pad positions.
        mask = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return logits, pooled

    @torch.no_grad()
    def embed(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_256d, projected_128d) on the input batch."""
        _, pooled = self.forward(ids)
        return pooled, self.proj(pooled)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = FullSourceEncoder(vocab_size=30_136)
    n = count_parameters(m)
    print(f"params: {n:,} ({n/1e6:.1f}M)")
    x = torch.randint(5, 30_136, (4, 64))
    logits, pooled = m(x)
    print(f"logits: {tuple(logits.shape)}")
    print(f"pooled: {tuple(pooled.shape)}")
