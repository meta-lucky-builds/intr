"""Dynamics model f(s, a, z) = W * phi(s, a, z) predicting a normalized observation delta.

phi is an MLP with a separate final layer ``last`` (what AdaJEPA-style adaptation touches), W is a
linear readout of shape (obs_dim, feature_dim) (what RLS writes), z is an environment latent.

Shapes: leading dimensions are free. o (..., O), a (..., A), z (..., Z) with the same leading
dimensions; features (..., F); output (..., O).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    obs_dim: int
    act_dim: int
    z_dim: int = 8  # 0 disables the latent
    hidden: int = 128
    n_hidden: int = 2
    feature_dim: int = 64
    input_extra: str = ""  # a key of INPUT_EXTRAS; empty means the input is exactly (norm_obs, a, z)


def _puck_relative(o: Tensor) -> Tensor:
    """Agent-to-puck geometry from a raw puck observation. (..., 8) -> (..., 8).

    The contact boundary is a circle in relative position. An MLP on absolute positions has to build
    that from scratch for every pair of positions; here it is a coordinate. PUCK_CONTACT_DISTANCE is 2 * PuckEnv.radius.
    """
    d = o[..., 2:4] - o[..., :2]
    relative_velocity = o[..., 6:8] - o[..., 4:6]
    dist = torch.sqrt((d * d).sum(-1, keepdim=True) + 1e-8)
    normal = d / dist
    gap = dist - PUCK_CONTACT_DISTANCE
    approach = (relative_velocity * normal).sum(-1, keepdim=True)
    return torch.cat([d, 0.3 * relative_velocity, 5.0 * gap, normal, 0.3 * approach], dim=-1)


PUCK_CONTACT_DISTANCE = 0.2
INPUT_EXTRAS = {"puck_relative": (8, _puck_relative)}  # name: (output size, function of the raw observation)


@dataclass
class Counts:
    """Compute counters. A call is one model evaluation over a batch; samples counts the batch."""

    forward_calls: int = 0
    forward_samples: int = 0
    backward_calls: int = 0
    writes: int = 0  # closed-form or analytic-gradient updates that need no autograd pass

    def as_dict(self) -> dict:
        return asdict(self)


class DynamicsModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        d = cfg.obs_dim + cfg.act_dim + cfg.z_dim
        if cfg.input_extra:
            d += INPUT_EXTRAS[cfg.input_extra][0]
        for _ in range(cfg.n_hidden):
            layers.append(nn.Linear(d, cfg.hidden))
            d = cfg.hidden
        self.body = nn.ModuleList(layers)
        self.last = nn.Linear(d, cfg.feature_dim)  # weight (F, h), bias (F,)
        self.W = nn.Parameter(torch.empty(cfg.obs_dim, cfg.feature_dim))  # (D, F), with D = O
        nn.init.xavier_uniform_(self.W)
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))
        self.register_buffer("delta_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("delta_std", torch.ones(cfg.obs_dim))

    def set_normalization(self, obs: Tensor, next_obs: Tensor) -> None:
        """obs, next_obs: (..., O). Call once on training data before training."""
        O = self.cfg.obs_dim
        o = obs.reshape(-1, O)
        d = (next_obs - obs).reshape(-1, O)
        self.obs_mean.copy_(o.mean(0))
        self.obs_std.copy_(o.std(0).clamp_min(1e-6))
        self.delta_mean.copy_(d.mean(0))
        self.delta_std.copy_(d.std(0).clamp_min(1e-6))

    def norm_obs(self, o: Tensor) -> Tensor:
        return (o - self.obs_mean) / self.obs_std

    def target(self, o: Tensor, next_o: Tensor) -> Tensor:
        """Normalized delta, the regression target of the readout."""
        return (next_o - o - self.delta_mean) / self.delta_std

    def apply_delta(self, o: Tensor, normalized_delta: Tensor) -> Tensor:
        return o + normalized_delta * self.delta_std + self.delta_mean

    def hidden(self, o: Tensor, a: Tensor, z: Tensor | None = None) -> Tensor:
        """Output of the layers before ``last``. (..., h)."""
        parts = [self.norm_obs(o), a]
        if self.cfg.input_extra:
            parts.append(INPUT_EXTRAS[self.cfg.input_extra][1](o))
        if self.cfg.z_dim > 0:
            if z is None:
                raise ValueError("model has z_dim > 0 but z is None")
            parts.append(z)
        x = torch.cat(parts, dim=-1)
        for layer in self.body:
            x = F.gelu(layer(x))
        return x

    def phi_from_hidden(self, h: Tensor) -> Tensor:
        return F.gelu(self.last(h))

    def features(self, o: Tensor, a: Tensor, z: Tensor | None = None) -> Tensor:
        """phi(s, a, z). (..., F)."""
        return self.phi_from_hidden(self.hidden(o, a, z))

    def forward(self, o: Tensor, a: Tensor, z: Tensor | None = None) -> Tensor:
        """Normalized delta. (..., F) @ (F, D) = (..., D)."""
        return self.features(o, a, z) @ self.W.t()

    def step(self, o: Tensor, a: Tensor, z: Tensor | None = None) -> Tensor:
        return self.apply_delta(o, self.forward(o, a, z))


class ContextEncoder(nn.Module):
    """Permutation-invariant encoder: mean-pooled per-transition embedding, mapped into [0, 1]^Z.

    Mean pooling keeps the output scale independent of how many transitions are given.
    """

    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden: int = 64):
        super().__init__()
        self.per_transition = nn.Sequential(
            nn.Linear(obs_dim + act_dim + obs_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, z_dim))

    def forward(self, norm_obs: Tensor, act: Tensor, norm_delta: Tensor) -> Tensor:
        """Inputs (E, N, ·), output (E, Z)."""
        x = torch.cat([norm_obs, act, norm_delta], dim=-1)  # (E, N, 2 O + A)
        return torch.sigmoid(self.head(self.per_transition(x).mean(dim=1)))


def clip_obs(o: Tensor, low: Tensor | None, high: Tensor | None) -> Tensor:
    """Keep a diverging rollout finite. A non-finite value becomes the bound, so it scores badly."""
    if low is None:
        return o
    big = 1e6
    o = torch.nan_to_num(o, nan=big, posinf=big, neginf=-big)
    return torch.max(torch.min(o, high), low)


def rollout(step_fn, o0: Tensor, actions: Tensor, obs_low: Tensor | None = None, obs_high: Tensor | None = None) -> Tensor:
    """Feed predictions back in. o0 (..., O), actions (..., H, A); returns (..., H + 1, O).

    step_fn(o, a) takes and returns (..., O). Works for any model, adapted or not.
    """
    out = [o0]
    o = o0
    for t in range(actions.shape[-2]):
        o = clip_obs(step_fn(o, actions[..., t, :]), obs_low, obs_high)
        out.append(o)
    return torch.stack(out, dim=-2)
