"""Núcleo transformer decoder-only de FUGA (RMSNorm, RoPE, GQA, QK-norm, SwiGLU)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ConfigModelo:
    vocab: int = 512
    d: int = 256
    capas: int = 8
    cabezas: int = 4
    cabezas_kv: int = 2
    ffn: int = 704
    ctx: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0
    kv_compas: int = 0          # cabezas kv cuyo RoPE gira por índice de compás (eje musical)
    compas_max: int = 4096
    ckpt: bool = False          # checkpointing de activaciones por bloque (menos memoria al entrenar)
    extra: dict = field(default_factory=dict)

    @property
    def d_cabeza(self) -> int:
        return self.d // self.cabezas


PRESETS: dict[str, dict] = {
    # ensayo en CPU
    "nano": dict(d=192, capas=6, cabezas=6, cabezas_kv=2, ffn=512, ctx=1024, kv_compas=1),
    # objetivo: profundo y estrecho (≈150M y ≈385M con vocab pequeño)
    "fuga-s": dict(d=768, capas=24, cabezas=12, cabezas_kv=4, ffn=2048, ctx=4096, kv_compas=1),
    "fuga-m": dict(d=1024, capas=32, cabezas=16, cabezas_kv=4, ffn=3072, ctx=4096, kv_compas=1),
}


def config(preset: str, vocab: int, **kw) -> ConfigModelo:
    return ConfigModelo(vocab=vocab, **{**PRESETS[preset], **kw})


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)


def tabla_rope(ctx: int, dh: int, base: float, device=None):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    t = torch.arange(ctx, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def aplicar_rope(x, cos, sin):
    # x: (B, H, T, dh); cos/sin: (B|1, H|1, T, dh/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class Atencion(nn.Module):
    def __init__(self, c: ConfigModelo):
        super().__init__()
        self.h, self.hkv, self.dh = c.cabezas, c.cabezas_kv, c.d_cabeza
        self.q = nn.Linear(c.d, self.h * self.dh, bias=False)
        self.kv = nn.Linear(c.d, 2 * self.hkv * self.dh, bias=False)
        self.o = nn.Linear(self.h * self.dh, c.d, bias=False)
        self.nq, self.nk = RMSNorm(self.dh), RMSNorm(self.dh)
        self.drop = c.dropout
        eje_kv = torch.tensor([1 if i >= self.hkv - c.kv_compas else 0 for i in range(self.hkv)])
        self.register_buffer("eje_kv", eje_kv, persistent=False)
        self.register_buffer("eje_q", eje_kv.repeat_interleave(self.h // self.hkv), persistent=False)

    def forward(self, x, rot, cache=None, mascara=None):
        """rot = (cos, sin) con forma (B, 2, T, dh/2): eje 0 token, eje 1 compás."""
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.hkv, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.nq(q), self.nk(k)
        cos, sin = rot
        q = aplicar_rope(q, cos[:, self.eje_q], sin[:, self.eje_q])
        k = aplicar_rope(k, cos[:, self.eje_kv], sin[:, self.eje_kv])
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        rep = self.h // self.hkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        causal = mascara is None and (cache is None or T > 1)
        if cache is not None and T > 1 and k.shape[2] != T:
            raise ValueError("prefill con caché previa no soportado")
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mascara, is_causal=causal,
            dropout_p=self.drop if self.training else 0.0,
        )
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: ConfigModelo):
        super().__init__()
        self.wi = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.wo = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, dim=-1)
        return self.wo(F.silu(a) * b)


class Bloque(nn.Module):
    def __init__(self, c: ConfigModelo):
        super().__init__()
        self.n1, self.att = RMSNorm(c.d), Atencion(c)
        self.n2, self.ffn = RMSNorm(c.d), SwiGLU(c)

    def forward(self, x, rot, cache=None, mascara=None):
        x = x + self.att(self.n1(x), rot, cache, mascara)
        return x + self.ffn(self.n2(x))


class Transformer(nn.Module):
    """Decoder causal. forward(ids, pos0=0, caches=None) -> (B, T, d) estados finales normalizados."""

    def __init__(self, c: ConfigModelo):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.bloques = nn.ModuleList(Bloque(c) for _ in range(c.capas))
        self.nf = RMSNorm(c.d)
        cos, sin = tabla_rope(max(c.ctx, c.compas_max), c.d_cabeza, c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.bloques:  # escalado de residuales por profundidad
            nn.init.normal_(b.att.o.weight, std=0.02 / math.sqrt(2 * c.capas))
            nn.init.normal_(b.ffn.wo.weight, std=0.02 / math.sqrt(2 * c.capas))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def estados(self, ids, pos=None, caches=None):
        """ids (B,T); pos (B,2,T) = [índice de token, índice de compás]; por defecto arange y 0."""
        B, T = ids.shape
        if pos is None:
            t = torch.arange(T, device=ids.device)
            pos = torch.stack([t, torch.zeros_like(t)]).expand(B, 2, T)
        rot = (self.cos[pos], self.sin[pos])  # (B, 2, T, dh/2)
        x = self.emb(ids)
        for i, b in enumerate(self.bloques):
            if self.c.ckpt and self.training and caches is None:
                x = checkpoint(b, x, rot, None, None, use_reentrant=False)
            else:
                x = b(x, rot, None if caches is None else caches[i])
        return self.nf(x)

    def logits(self, h):
        return F.linear(h, self.emb.weight)  # embeddings atados

    def forward(self, ids, pos=None, caches=None):
        return self.logits(self.estados(ids, pos, caches))


def config_guardable(c: ConfigModelo) -> dict:
    """Configuración para el checkpoint: sin banderas de entrenamiento (ckpt), legible por versiones anteriores."""
    return {k: v for k, v in c.__dict__.items() if k != "ckpt"}


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
