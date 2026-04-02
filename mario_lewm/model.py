from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-(t.square()) / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if inner_dim != dim else nn.Identity()

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for layer in self.layers:
            x = layer(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    def __init__(self, input_dim: int, smoothed_dim: int, emb_dim: int, mlp_scale: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        norm_fn: type[nn.Module] | None = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    def __init__(
        self,
        num_frames: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = self.dropout(x + self.pos_embedding[:, :t])
        return self.transformer(x, c)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        return self.patch_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.encoder(x)
        return self.norm(x[:, 0])


@dataclass
class LeWorldModelConfig:
    image_size: int = 224
    patch_size: int = 14
    encoder_hidden_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_dim: int = 768
    encoder_dropout: float = 0.0
    action_dim: int = 8
    action_embed_dim: int = 192
    action_smoothed_dim: int = 32
    history_size: int = 3
    num_preds: int = 1
    predictor_hidden_dim: int = 192
    predictor_output_dim: int = 192
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    projector_hidden_dim: int = 2048
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeWorldModel(nn.Module):
    def __init__(self, config: LeWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_dim=config.encoder_hidden_dim,
            depth=config.encoder_depth,
            heads=config.encoder_heads,
            mlp_dim=config.encoder_mlp_dim,
            dropout=config.encoder_dropout,
        )
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.action_smoothed_dim,
            emb_dim=config.action_embed_dim,
        )
        self.projector = MLP(
            input_dim=config.encoder_hidden_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.action_embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.predictor = ARPredictor(
            num_frames=config.history_size,
            input_dim=config.action_embed_dim,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=config.predictor_output_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            mlp_dim=config.predictor_mlp_dim,
            dim_head=config.predictor_dim_head,
            dropout=config.predictor_dropout,
            emb_dropout=config.predictor_emb_dropout,
        )
        self.pred_proj = MLP(
            input_dim=config.predictor_output_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.action_embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = batch["pixels"].float()
        b = pixels.size(0)
        flat_pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
        pixels_emb = self.encoder(flat_pixels)
        emb = self.projector(pixels_emb)
        output = {"emb": rearrange(emb, "(b t) d -> b t d", b=b)}
        if "action" in batch:
            output["act_emb"] = self.action_encoder(batch["action"])
        return output

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
        ctx_len = self.config.history_size
        n_preds = self.config.num_preds
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        target_emb = emb[:, n_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = (pred_emb - target_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.config.sigreg_weight * sigreg_loss
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "emb": emb,
            "pred_emb": pred_emb,
            "target_emb": target_emb,
        }

    @torch.no_grad()
    def encode_goal(self, goal_pixels: torch.Tensor) -> torch.Tensor:
        return self.encode({"pixels": goal_pixels})["emb"]

    @torch.no_grad()
    def rollout(self, init_pixels: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        if init_pixels.dim() != 5:
            raise ValueError("init_pixels must be shaped [B, H, C, H, W].")
        if action_sequences.dim() != 4:
            raise ValueError("action_sequences must be shaped [B, S, T, action_dim].")
        b, s, horizon, _ = action_sequences.shape
        history = self.config.history_size
        init_encoded = self.encode({"pixels": init_pixels})["emb"]
        emb = init_encoded.unsqueeze(1).expand(b, s, -1, -1)
        emb = rearrange(emb, "b s t d -> (b s) t d").clone()
        actions = rearrange(action_sequences, "b s t a -> (b s) t a")
        act = actions[:, :history].clone()
        future = actions[:, history:]
        for step in range(max(0, horizon - history)):
            act_emb = self.action_encoder(act)
            pred = self.predict(emb[:, -history:], act_emb[:, -history:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, future[:, step : step + 1]], dim=1)
        act_emb = self.action_encoder(act)
        pred = self.predict(emb[:, -history:], act_emb[:, -history:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        return rearrange(emb, "(b s) t d -> b s t d", b=b, s=s)
