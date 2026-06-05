import torch
from torch import nn

from prism_challenge.evaluator.interface import PrismContext, TrainingRecipe

DEFAULT_DIM = 256
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 4
DEFAULT_MLP_RATIO = 4
DEFAULT_NUM_EXPERTS = 4
DEFAULT_TOP_K = 2
DEFAULT_EPS = 0.000001
DEFAULT_LR = 0.0003
DEFAULT_BATCH_SIZE = 4
DEFAULT_WEIGHT_DECAY = 0.01


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.mul(x).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if n_heads <= 0:
            raise Exception("n_heads must be positive")
        if dim % n_heads != 0:
            raise Exception("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool)
        causal = torch.triu(causal, diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(context)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class LightMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise Exception("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise Exception("top_k must be in range [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.eps = eps
        self.router = nn.Linear(dim, num_experts, bias=False)
        experts: list[GatedMLP] = []
        for _ in range(num_experts):
            experts.append(GatedMLP(dim, hidden_dim))
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        gate = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(gate, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + self.eps)
        dispatch = torch.zeros_like(gate)
        dispatch = dispatch.scatter(-1, top_i, top_w)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            weight_e = dispatch.select(-1, e).unsqueeze(-1)
            out = out + weight_e * self.experts[e](x)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int,
        use_moe: bool,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm_mlp = RMSNorm(dim)
        if use_moe:
            self.mlp = LightMoE(dim, hidden_dim, num_experts, top_k)
        else:
            self.mlp = GatedMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: int,
        use_moe: bool,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[TransformerBlock] = []
        for _ in range(n_layers):
            blocks.append(
                TransformerBlock(dim, n_heads, mlp_ratio, use_moe, num_experts, top_k)
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)


def build_model(ctx: PrismContext) -> DecoderLM:
    return DecoderLM(
        vocab_size=ctx.vocab_size,
        dim=DEFAULT_DIM,
        n_heads=DEFAULT_HEADS,
        n_layers=DEFAULT_LAYERS,
        mlp_ratio=DEFAULT_MLP_RATIO,
        use_moe=False,
        num_experts=DEFAULT_NUM_EXPERTS,
        top_k=DEFAULT_TOP_K,
    )


def get_recipe(ctx: PrismContext) -> TrainingRecipe:
    return TrainingRecipe(
        learning_rate=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE,
        optimizer="adamw",
        scheduler="cosine",
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
