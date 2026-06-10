"""
phase1_model.py
Dual-Horizon Hybrid Attention Block for quadruped adaptive estimation.
Phase 1 of the multi-rate control framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Feature map for linear attention (ELU+1, hardware-portable) ───────────────
def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


# ── Short-term head: causal full-attention over a local window ─────────────────
class CausalWindowAttention(nn.Module):
    """
    Scaled-dot-product attention over the last W timesteps only.
    Complexity: O(W^2 * d). With W<=15 this is negligible.
    At inference: caller slices the buffer to W steps before passing in.
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int = 12):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.W        = window_size
        self.scale    = math.sqrt(self.d_head)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, _ = x.shape
        T = min(T, self.W)
        x = x[:, -T:, :]                            # enforce window

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        mask = torch.ones(T, T, device=x.device).tril().bool()
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out[:, -1, :])          # [B, d_model]


# ── Long-term head: streaming linear attention ─────────────────────────────────
class StreamingLinearAttention(nn.Module):
    """
    Linear attention via the kernel trick.
    O(T * d^2) training, O(d^2) per step at inference.
    The recurrent state S replaces the KV-cache — no memory growth.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model  = d_model
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                S_init=None, z_init=None):
        """
        Full-sequence pass (training or offline).
        Returns: (last_step_output [B,d], S_final [B,d,d], z_final [B,d])
        """
        B, T, d = x.shape
        Q = elu_feature_map(self.q_proj(x))
        K = elu_feature_map(self.k_proj(x))
        V = self.v_proj(x)

        S = S_init if S_init is not None else torch.zeros(B, d, d, device=x.device)
        z = z_init if z_init is not None else torch.zeros(B, d,    device=x.device)

        outputs = []
        for t in range(T):
            S = S + torch.bmm(K[:, t].unsqueeze(2), V[:, t].unsqueeze(1))
            z = z + K[:, t]
            num   = torch.bmm(Q[:, t].unsqueeze(1), S).squeeze(1)
            denom = (Q[:, t] * z).sum(-1, keepdim=True) + 1e-6
            outputs.append(num / denom)

        out_seq = torch.stack(outputs, dim=1)        # [B, T, d]
        return self.out_proj(out_seq[:, -1]), S, z

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, S: torch.Tensor, z: torch.Tensor):
        """
        Single-step inference call. O(d^2) per call.
        Returns: (output [B,d], S_new [B,d,d], z_new [B,d])
        """
        q = elu_feature_map(self.q_proj(x_t))
        k = elu_feature_map(self.k_proj(x_t))
        v = self.v_proj(x_t)
        S = S + torch.bmm(k.unsqueeze(2), v.unsqueeze(1))
        z = z + k
        num   = torch.bmm(q.unsqueeze(1), S).squeeze(1)
        denom = (q * z).sum(-1, keepdim=True) + 1e-6
        return self.out_proj(num / denom), S, z


# ── Hybrid block ───────────────────────────────────────────────────────────────
class DualHorizonHybridBlock(nn.Module):
    """
    Main Phase 1 model. Combines both attention heads.
    Input:  raw sensor buffer [B, T_long, d_in]
    Output: x_latent [B, d_latent] + carry states for linear head
    """
    def __init__(self,
                 d_in:     int   = 24,   # 12 joint pos + 12 joint vel
                 d_model:  int   = 64,
                 d_latent: int   = 8,
                 n_heads:  int   = 4,
                 window:   int   = 12):
        super().__init__()
        self.d_model    = d_model
        self.d_latent   = d_latent
        self.input_proj = nn.Linear(d_in, d_model)
        self.short_head = CausalWindowAttention(d_model, n_heads, window)
        self.long_head  = StreamingLinearAttention(d_model)
        self.fusion     = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(self, x_raw: torch.Tensor, S=None, z=None):
        """
        x_raw: [B, T, d_in]
        Returns: x_latent [B, d_latent], S_new, z_new
        """
        x               = self.input_proj(x_raw)
        h_short         = self.short_head(x)
        h_long, S, z    = self.long_head(x, S, z)
        fused           = torch.cat([h_short, h_long], dim=-1)
        return self.fusion(fused), S, z

    def init_states(self, batch_size: int = 1, device=None):
        """Convenience: create zero carry states for the linear head."""
        d = self.d_model
        return (
            torch.zeros(batch_size, d, d, device=device),
            torch.zeros(batch_size, d,    device=device),
        )


# ── State augmentation ─────────────────────────────────────────────────────────
def build_augmented_state(x_phys: torch.Tensor,
                          x_latent: torch.Tensor) -> torch.Tensor:
    """
    Concatenate physical kinematics state with learned latent features.
    x_phys:   [B, n_phys]   e.g. [Vx, Vy, omega_z, roll, pitch, yaw_rate]
    x_latent: [B, d_latent]
    Returns:  [B, n_phys + d_latent]
    """
    return torch.cat([x_phys, x_latent], dim=-1)
