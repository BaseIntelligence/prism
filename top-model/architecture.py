"""Staging e2e miner A architecture: MLP-Mixer LM (token/channel mixing, no attention)."""

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, d: int, block: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln_t = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)
        self.t_mix = nn.Sequential(
            nn.Linear(block, block * mlp_ratio), nn.GELU(), nn.Linear(block * mlp_ratio, block)
        )
        self.c_mix = nn.Sequential(
            nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d)
        )

    def forward(self, x):
        h = self.ln_t(x)
        h = h.transpose(1, 2)
        h = self.t_mix(h)
        x = x + h.transpose(1, 2)
        return x + self.c_mix(self.ln_c(x))


class MixerLM(nn.Module):
    def __init__(self, vocab=50257, d=192, n_layer=3, block=512):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([MixerBlock(d, block) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.logits = None

    def forward(self, ids):
        b, t = ids.shape
        t = min(t, self.block)
        ids = ids[:, -t:]
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.head(x)
        self.logits = logits
        return logits


def build_model(ctx):
    """Recipe contract entrypoint. ctx carries device/seed/caps (unused here)."""
    torch.manual_seed(int(ctx.get("seed", 0)))
    return MixerLM()
