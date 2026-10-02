"""A Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU) that runs over a paged KV cache.

The forward pass takes a *flattened, variable-length* batch: every token of every
sequence scheduled this step is packed into one [T, d] tensor, exactly like
production engines (vLLM, TensorRT-LLM) do. Prefill chunks and decode tokens can
be mixed in the same step, which is what makes chunked-prefill scheduling possible.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    rope_theta: float = 10000.0
    max_seq_len: int = 4096

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        d, f, L = self.d_model, self.ffn_dim, self.n_layers
        kv = self.n_kv_heads * self.head_dim
        per_layer = d * d * 2 + d * kv * 2 + 3 * d * f + 2 * d
        return L * per_layer + 2 * self.vocab_size * d + d


PRESETS = {
    # ~60M params: fast enough to serve real traffic on a laptop CPU.
    "tiny": ModelConfig(),
    # ~0.5B-class shape for profiling closer to deployment regimes.
    "small": ModelConfig(d_model=1536, n_layers=16, n_heads=12, n_kv_heads=4, ffn_dim=4096),
    # Llama-3-8B shape, used for analytic roofline projections (not instantiated on CPU).
    "llama3-8b": ModelConfig(vocab_size=128256, d_model=4096, n_layers=32, n_heads=32,
                             n_kv_heads=8, ffn_dim=14336, rope_theta=500000.0, max_seq_len=8192),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(cfg: ModelConfig, device, dtype):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, H, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


@dataclass
class StepInput:
    """Everything the model needs to run one engine step over a packed batch.

    token_ids:    [T] all new tokens of all scheduled sequences, concatenated.
    positions:    [T] absolute position of each new token in its sequence.
    slot_mapping: [T] flat KV-cache slot each new token's K/V is written to.
    seq_spans:    per sequence, (start, end) offsets into the packed tokens.
    block_tables: per sequence, the KV blocks holding its context, in order.
    ctx_lens:     per sequence, context length after this step's tokens are added.
    logits_idx:   [S] packed index of each sequence's last token (to sample from).
    """
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    seq_spans: list[tuple[int, int]]
    block_tables: list[list[int]]
    ctx_lens: list[int]
    logits_idx: torch.Tensor


_WORKSPACE: dict[tuple, torch.Tensor] = {}


def gather_kv(cache: torch.Tensor, tables: list[list[int]], max_len: int, slot: str) -> torch.Tensor:
    """PagedAttention-style gather: [num_blocks, bs, Hkv, D] -> [B, Hkv, max_len, D].

    Copies whole blocks (bs contiguous tokens each) rather than single slots, which is
    what makes paging cheap: one index per block instead of one per token. The output
    lands in a persistent workspace: on CPU, faulting in a fresh buffer every layer
    costs ~10x more than the copy itself.
    """
    bs = cache.shape[1]
    nb = -(-max_len // bs)
    idx = torch.tensor([t[:nb] + [0] * (nb - len(t[:nb])) for t in tables], device=cache.device)
    n = idx.numel()
    key = (slot, cache.device, cache.dtype, cache.shape[1:])
    ws = _WORKSPACE.get(key)
    if ws is None or ws.shape[0] < n:
        ws = _WORKSPACE[key] = torch.empty((max(n, 1024), *cache.shape[1:]), dtype=cache.dtype, device=cache.device)
    kv = torch.index_select(cache, 0, idx.view(-1), out=ws[:n]).view(len(tables), nb * bs, *cache.shape[2:])
    return kv[:, :max_len].transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)

    def forward(self, x, inp: StepInput, k_cache, v_cache, cos, sin):
        cfg = self.cfg
        T = x.shape[0]
        q = self.wq(x).view(T, cfg.n_heads, cfg.head_dim)
        k = self.wk(x).view(T, cfg.n_kv_heads, cfg.head_dim)
        v = self.wv(x).view(T, cfg.n_kv_heads, cfg.head_dim)
        c, s = cos[inp.positions], sin[inp.positions]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)

        # Write this step's K/V into their paged slots. Cache: [num_blocks, bs, Hkv, D].
        k_cache.view(-1, cfg.n_kv_heads, cfg.head_dim)[inp.slot_mapping] = k
        v_cache.view(-1, cfg.n_kv_heads, cfg.head_dim)[inp.slot_mapping] = v

        out = torch.empty_like(q)
        decode, prefill = [], []
        for i, (a, b) in enumerate(inp.seq_spans):
            (decode if b - a == 1 else prefill).append(i)

        # Decode tokens: one query each; batch them with a padded block gather + mask.
        if decode:
            lens = torch.tensor([inp.ctx_lens[i] for i in decode], device=x.device)
            Lmax = int(lens.max())
            tables = [inp.block_tables[i] for i in decode]
            kd, vd = gather_kv(k_cache, tables, Lmax, "k"), gather_kv(v_cache, tables, Lmax, "v")  # [B, Hkv, L, D]
            valid = torch.arange(Lmax, device=x.device)[None, :] < lens[:, None]
            mask = torch.zeros(valid.shape, dtype=x.dtype, device=x.device).masked_fill_(~valid, float("-inf"))
            tok = torch.tensor([inp.seq_spans[i][0] for i in decode], device=x.device)
            qd = q[tok].unsqueeze(2)                                                       # [B, H, 1, D]
            od = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask[:, None, None, :], enable_gqa=True)
            out[tok] = od.squeeze(2)

        # Prefill chunks: causal attention of the new tokens over the full context.
        for i in prefill:
            a, b = inp.seq_spans[i]
            n, L = b - a, inp.ctx_lens[i]
            kp = gather_kv(k_cache, [inp.block_tables[i]], L, "k")[0]                           # [Hkv, L, D]
            vp = gather_kv(v_cache, [inp.block_tables[i]], L, "v")[0]
            qp = q[a:b].transpose(0, 1)                                                     # [H, n, D]
            # New token j sits at context position L-n+j and may see positions <= that.
            causal = torch.ones(n, L, dtype=torch.bool, device=x.device).tril(diagonal=L - n)
            op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=causal, enable_gqa=True)
            out[a:b] = op.transpose(0, 1)

        return self.wo(out.reshape(T, -1))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x, inp, k_cache, v_cache, cos, sin):
        x = x + self.attn(self.attn_norm(x), inp, k_cache, v_cache, cos, sin)
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope = None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def rope(self, device, dtype):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:
            self._rope = rope_tables(self.cfg, device, dtype)
        return self._rope

    @torch.inference_mode()
    def forward_step(self, inp: StepInput, k_caches, v_caches) -> torch.Tensor:
        """Run one packed step; returns logits [S, vocab] for each sequence's last token."""
        x = self.embed(inp.token_ids)
        cos, sin = self.rope(x.device, x.dtype)
        for layer, kc, vc in zip(self.layers, k_caches, v_caches):
            x = layer(x, inp, kc, vc, cos, sin)
        return self.lm_head(self.norm(x[inp.logits_idx]))

    @torch.inference_mode()
    def forward_dense(self, ids: torch.Tensor) -> torch.Tensor:
        """Reference forward with no cache (full recompute). ids: [L] -> logits [L, vocab]."""
        cfg = self.cfg
        L = ids.numel()
        x = self.embed(ids)
        cos, sin = self.rope(x.device, x.dtype)
        pos = torch.arange(L, device=ids.device)
        rep = cfg.n_heads // cfg.n_kv_heads
        for layer in self.layers:
            h = layer.attn_norm(x)
            a = layer.attn
            q = apply_rope(a.wq(h).view(L, cfg.n_heads, -1), cos[pos], sin[pos]).transpose(0, 1)
            k = apply_rope(a.wk(h).view(L, cfg.n_kv_heads, -1), cos[pos], sin[pos])
            k = k.repeat_interleave(rep, dim=1).transpose(0, 1)
            v = a.wv(h).view(L, cfg.n_kv_heads, -1).repeat_interleave(rep, dim=1).transpose(0, 1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(L, -1)
            x = x + a.wo(o)
            h = layer.ffn_norm(x)
            x = x + layer.w2(F.silu(layer.w1(h)) * layer.w3(h))
        return self.lm_head(self.norm(x))


def build_model(preset: str = "tiny", device: str = "cpu", dtype=torch.float32, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    model = Transformer(PRESETS[preset]).to(device=device, dtype=dtype).eval()
    return model


def model_flops_per_token(cfg: ModelConfig) -> float:
    """Dense matmul FLOPs per token (2 * params touched), excluding attention scores."""
    return 2.0 * (cfg.n_params() - cfg.vocab_size * cfg.d_model)  # embedding lookup is not a matmul


def attn_flops(cfg: ModelConfig, q_tokens: int, ctx_len: int) -> float:
    """QK^T and PV FLOPs for q_tokens queries attending over ctx_len keys, all layers."""
    return 4.0 * cfg.n_layers * cfg.n_heads * cfg.head_dim * q_tokens * ctx_len


def _check():
    cfg = PRESETS["tiny"]
    print(f"tiny params: {cfg.n_params()/1e6:.1f}M, llama3-8b params: {PRESETS['llama3-8b'].n_params()/1e9:.2f}B")


if __name__ == "__main__":
    _check()
