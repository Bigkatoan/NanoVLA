"""From-scratch encoders for the tiny VLA.

TextEncoder : small pre-LN transformer over BPE tokens (~2M params)
VisionViT   : ViT-Tiny style encoder over 112x112 crops, patch 16 (~5.5M params)
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, ff, drop=0.0, drop_path=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d), nn.Dropout(drop))
        self.dp = drop_path

    def _dp(self, x):
        if not self.training or self.dp == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.dp
        return x * keep / (1 - self.dp)

    def forward(self, x, pad_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self._dp(a)
        x = x + self._dp(self.mlp(self.n2(x)))
        return x


# ---------------------------------------------------------------- text
class TextEncoder(nn.Module):
    def __init__(self, vocab, d=192, depth=4, heads=4, ff=768, max_len=64, drop=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, ids):
        """ids [B,L] -> tokens [B,L,d], pooled [B,d], pad_mask [B,L] (True=pad)"""
        pad = ids == self.pad_id
        x = self.drop(self.tok(ids) + self.pos[:, : ids.shape[1]])
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        m = (~pad).unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return x, pooled, pad


class MLMHead(nn.Module):
    def __init__(self, d, vocab, tok_emb):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.emb = tok_emb  # weight tying

    def forward(self, h):
        return self.t(h) @ self.emb.weight.T + self.bias


# ---------------------------------------------------------------- vision
class VisionViT(nn.Module):
    """ViT over variable-size inputs (multiples of patch); interpolated pos-emb."""

    def __init__(self, img=112, patch=16, d=192, depth=12, heads=3, ff=768, drop_path=0.1, n_views=8):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        g = img // patch
        self.grid = g
        self.pos = nn.Parameter(torch.zeros(1, g * g, d))
        self.view_emb = nn.Embedding(n_views, d)  # 0 = external cam, 1 = wrist (others spare)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(d, heads, ff, 0.0, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.d = d

    def _pos(self, gh, gw):
        if gh == self.grid and gw == self.grid:
            return self.pos
        p = self.pos.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(gh, gw), mode="bicubic", align_corners=False)
        return p.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)

    def forward(self, x, view=None):
        """x [B,3,H,W] normalized -> cls [B,d], patch tokens [B,N,d]"""
        x = self.proj(x)
        B, D, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2) + self._pos(gh, gw)
        if view is not None:
            x = x + self.view_emb(view).unsqueeze(1)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class DINOHead(nn.Module):
    def __init__(self, d, out=4096, hidden=1024, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        return self.last(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
