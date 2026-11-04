"""Training across many environments.

Four backbones:
    latent    T2-Z scheme: one learnable latent per training environment, alternating model updates
              (latents fixed) and latent updates (model frozen), latents clipped to [0, 1].
    nolatent  env-agnostic baseline, z_dim = 0.
    context   jointly trained permutation-invariant encoder producing z.
    control   nolatent continued for as many extra backward passes as the latent updates cost
              (protocol: "matched continued-training control").

T2-Z's block schedule (warmup then 200 model / 200 latent) is kept in shape and scaled by config.
Its learning rates are not: 1e-5 is specific to a video diffusion model, 1e-3 is used here.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .envs import BatchEnv, Transitions, collect_mixed, derive_seed, make_generator
from .models import ContextEncoder, DynamicsModel, ModelConfig


@dataclass
class TrainConfig:
    model_steps: int = 8000
    warmup_steps: int = 300
    block_steps: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    lr_floor: float = 0.05  # cosine decays to lr * lr_floor
    latent_lr: float = 0.03
    grad_clip: float = 5.0
    log_every: int = 500
    context_envs_per_step: int = 32
    context_queries: int = 8
    context_max_k: int = 3
    control_lr_scale: float = 0.3  # warm restart: the original schedule has already decayed
    # Opt-in options for Thread G. The defaults below reproduce the first round's training exactly.
    loss: str = "mse"  # "mse" or "huber", for model and latent updates alike
    huber_delta: float = 1.0  # in units of the normalized target
    latent_samples_per_env: int = 0  # 0: a latent step uses batch_size draws, about 2 per environment at round-one sizes
    latent_block_steps: int = 0  # latent updates per block; 0: block_steps
    latent_lr_decay: bool = False  # follow the model's cosine; False keeps latent_lr constant


def regression_loss(pred: Tensor, target: Tensor, tcfg: TrainConfig) -> Tensor:
    if tcfg.loss == "mse":
        return ((pred - target) ** 2).mean()
    if tcfg.loss == "huber":
        return F.huber_loss(pred, target, delta=tcfg.huber_delta)
    raise ValueError(f"unknown loss {tcfg.loss!r}")


@dataclass
class TransitionData:
    """Training transitions, grouped so a context batch can take whole trajectories."""

    obs: Tensor  # (E, N, O) with N = n_traj * n_steps
    act: Tensor  # (E, N, A)
    next_obs: Tensor  # (E, N, O)
    n_traj: int
    n_steps: int

    @property
    def n_envs(self) -> int:
        return self.obs.shape[0]

    @classmethod
    def collect(cls, env: BatchEnv, n_traj: int, n_steps: int, gen) -> "TransitionData":
        trans: Transitions = collect_mixed(env, n_traj, n_steps, env.data_policies, gen).flat()
        return cls(trans.obs, trans.act, trans.next_obs, n_traj, n_steps)

    def sample(self, batch: int, gen) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """env_idx (B,), o (B, O), a (B, A), next_o (B, O)."""
        env_idx = torch.randint(self.n_envs, (batch,), generator=gen)
        t_idx = torch.randint(self.obs.shape[1], (batch,), generator=gen)
        return env_idx, self.obs[env_idx, t_idx], self.act[env_idx, t_idx], self.next_obs[env_idx, t_idx]

    def sample_per_env(self, per_env: int, gen) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """per_env draws from every environment. env_idx (B,), o (B, O), a (B, A), next_o (B, O), B = E * per_env."""
        E = self.n_envs
        env_idx = torch.arange(E)[:, None].expand(E, per_env).reshape(-1)
        t_idx = torch.randint(self.obs.shape[1], (E * per_env,), generator=gen)
        return env_idx, self.obs[env_idx, t_idx], self.act[env_idx, t_idx], self.next_obs[env_idx, t_idx]

    def extended(self, extra: Transitions) -> "TransitionData":
        """More transitions from the same environments, after the existing ones.

        n_traj and n_steps still describe the leading whole trajectories, so a context batch stays valid.
        """
        if extra.obs.shape[0] != self.n_envs:
            raise ValueError(f"extra data has {extra.obs.shape[0]} environments, expected {self.n_envs}")
        return TransitionData(
            torch.cat([self.obs, extra.obs], dim=1),
            torch.cat([self.act, extra.act], dim=1),
            torch.cat([self.next_obs, extra.next_obs], dim=1),
            self.n_traj,
            self.n_steps,
        )


@dataclass
class Backbone:
    """A trained model plus what test-time methods need from training. Parameters are frozen."""

    model: DynamicsModel
    kind: str  # "latent" | "nolatent" | "context"
    z_default: Tensor | None = None  # (Z,) mean training latent, the start and the fallback z
    z_table: Tensor | None = None  # (E_train, Z)
    encoder: ContextEncoder | None = None
    log: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def freeze(self) -> "Backbone":
        self.model.requires_grad_(False)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
        return self


def _cosine_lr(base: float, step: int, total: int, floor: float) -> float:
    progress = min(step / max(total, 1), 1.0)
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class _Logger:
    def __init__(self, log_every: int):
        self.log_every = log_every
        self.records: list[dict] = []
        self._sum = {}
        self._n = {}
        self.backward = 0

    def add(self, phase: str, loss: float) -> None:
        self.backward += 1
        self._sum[phase] = self._sum.get(phase, 0.0) + loss
        self._n[phase] = self._n.get(phase, 0) + 1
        if self.backward % self.log_every == 0:
            self.flush()

    def flush(self) -> None:
        for phase, n in self._n.items():
            if n > 0:
                self.records.append({"backward": self.backward, "phase": phase, "loss": self._sum[phase] / n})
        self._sum, self._n = {}, {}


def _build_model(data: TransitionData, cfg: ModelConfig, seed, label: str) -> DynamicsModel:
    torch.manual_seed(derive_seed("init", seed, label))
    model = DynamicsModel(cfg)
    model.set_normalization(data.obs, data.next_obs)
    return model


def _model_updates(model, optimizer, data, tcfg, gen, n, start, total, logger, z_table=None, lr_base=None, phase="model"):
    """n model updates with z_table (E, Z) held fixed. Returns the new update count."""
    base = tcfg.lr if lr_base is None else lr_base
    for i in range(n):
        _set_lr(optimizer, _cosine_lr(base, start + i, total, tcfg.lr_floor))
        env_idx, o, a, next_o = data.sample(tcfg.batch_size, gen)
        z = None if z_table is None else z_table[env_idx].detach()  # (B, Z)
        loss = regression_loss(model(o, a, z), model.target(o, next_o), tcfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        logger.add(phase, loss.item())
    return start + n


def train_nolatent(data: TransitionData, cfg: ModelConfig, tcfg: TrainConfig, seed) -> Backbone:
    if cfg.z_dim != 0:
        raise ValueError("train_nolatent needs z_dim = 0")
    started = time.perf_counter()
    gen = make_generator("train", seed, "nolatent")
    model = _build_model(data, cfg, seed, "nolatent")
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    logger = _Logger(tcfg.log_every)
    _model_updates(model, optimizer, data, tcfg, gen, tcfg.model_steps, 0, tcfg.model_steps, logger)
    logger.flush()
    stats = {"model_updates": tcfg.model_steps, "latent_updates": 0, "wall_s": time.perf_counter() - started}
    return Backbone(model, "nolatent", log=logger.records, stats=stats).freeze()


def train_latent(data: TransitionData, cfg: ModelConfig, tcfg: TrainConfig, seed) -> Backbone:
    """The T2-Z alternating scheme."""
    if cfg.z_dim <= 0:
        raise ValueError("train_latent needs z_dim > 0")
    started = time.perf_counter()
    gen = make_generator("train", seed, "latent")
    model = _build_model(data, cfg, seed, "latent")
    z_table = torch.rand(data.n_envs, cfg.z_dim, generator=gen).requires_grad_(True)  # (E, Z)
    opt_model = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    opt_latent = torch.optim.Adam([z_table], lr=tcfg.latent_lr)
    logger = _Logger(tcfg.log_every)

    total = tcfg.model_steps
    warmup = min(tcfg.warmup_steps, total)
    done = _model_updates(model, opt_model, data, tcfg, gen, warmup, 0, total, logger, z_table)
    latent_updates = 0

    def latent_block(n: int) -> None:
        model.requires_grad_(False)
        if tcfg.latent_lr_decay:
            _set_lr(opt_latent, _cosine_lr(tcfg.latent_lr, done, total, tcfg.lr_floor))
        for _ in range(n):
            if tcfg.latent_samples_per_env > 0:
                env_idx, o, a, next_o = data.sample_per_env(tcfg.latent_samples_per_env, gen)
            else:
                env_idx, o, a, next_o = data.sample(tcfg.batch_size, gen)
            loss = regression_loss(model(o, a, z_table[env_idx]), model.target(o, next_o), tcfg)
            opt_latent.zero_grad(set_to_none=True)
            loss.backward()
            opt_latent.step()
            with torch.no_grad():
                z_table.clamp_(0.0, 1.0)
            logger.add("latent", loss.item())
        model.requires_grad_(True)

    # Latents get updated right after warmup, then model and latent blocks alternate. A latent block
    # comes last, so the final latents are consistent with the final model.
    per_block = tcfg.latent_block_steps or tcfg.block_steps
    latent_block(per_block)
    latent_updates += per_block
    while done < total:
        n = min(tcfg.block_steps, total - done)
        done = _model_updates(model, opt_model, data, tcfg, gen, n, done, total, logger, z_table)
        latent_block(per_block)
        latent_updates += per_block
    logger.flush()

    z_final = z_table.detach().clone()
    stats = {"model_updates": done, "latent_updates": latent_updates, "wall_s": time.perf_counter() - started}
    return Backbone(model, "latent", z_final.mean(0), z_final, log=logger.records, stats=stats).freeze()


def train_context(data: TransitionData, cfg: ModelConfig, tcfg: TrainConfig, seed) -> Backbone:
    """Encoder and model trained jointly; the encoder sees 1 to context_max_k whole trajectories."""
    if cfg.z_dim <= 0:
        raise ValueError("train_context needs z_dim > 0")
    started = time.perf_counter()
    gen = make_generator("train", seed, "context")
    model = _build_model(data, cfg, seed, "context")
    encoder = ContextEncoder(cfg.obs_dim, cfg.act_dim, cfg.z_dim)
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=tcfg.lr)
    logger = _Logger(tcfg.log_every)
    Eb, Q, T = tcfg.context_envs_per_step, tcfg.context_queries, data.n_steps
    N = data.obs.shape[1]
    max_k = min(tcfg.context_max_k, data.n_traj)
    steps_t = torch.arange(T)
    for step in range(tcfg.model_steps):
        _set_lr(optimizer, _cosine_lr(tcfg.lr, step, tcfg.model_steps, tcfg.lr_floor))
        env_idx = torch.randint(data.n_envs, (Eb, 1), generator=gen)  # (Eb, 1), broadcasts in indexing
        k = int(torch.randint(1, max_k + 1, (1,), generator=gen))
        traj = torch.randint(data.n_traj, (Eb, k), generator=gen)  # (Eb, k)
        ctx_idx = (traj[:, :, None] * T + steps_t[None, None, :]).reshape(Eb, k * T)  # (Eb, k T)
        co, ca, cn = data.obs[env_idx, ctx_idx], data.act[env_idx, ctx_idx], data.next_obs[env_idx, ctx_idx]
        z = encoder(model.norm_obs(co), ca, model.target(co, cn))  # (Eb, Z)
        q_idx = torch.randint(N, (Eb, Q), generator=gen)  # (Eb, Q)
        qo, qa, qn = data.obs[env_idx, q_idx], data.act[env_idx, q_idx], data.next_obs[env_idx, q_idx]
        pred = model(qo, qa, z[:, None, :].expand(-1, Q, -1))  # (Eb, Q, O)
        loss = ((pred - model.target(qo, qn)) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
        optimizer.step()
        logger.add("joint", loss.item())
    logger.flush()
    stats = {"model_updates": tcfg.model_steps, "latent_updates": 0, "wall_s": time.perf_counter() - started}
    return Backbone(model, "context", encoder=encoder, log=logger.records, stats=stats).freeze()


def continued_training(backbone: Backbone, data: TransitionData, extra_steps: int, tcfg: TrainConfig, seed) -> Backbone:
    """Copy a nolatent backbone and train it further. The original is untouched.

    For the T2-Z control, pass extra_steps = latent_updates: same number of backward passes in total,
    same data. Latent backward passes cost less than model updates, so this slightly favours the
    control over a wall-clock match. A warm restart is used because the cosine has already decayed.
    """
    if backbone.kind != "nolatent":
        raise ValueError("the continued-training control is defined for the nolatent backbone")
    started = time.perf_counter()
    gen = make_generator("train", seed, "control")
    model = copy.deepcopy(backbone.model).requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr * tcfg.control_lr_scale)
    logger = _Logger(tcfg.log_every)
    _model_updates(
        model, optimizer, data, tcfg, gen, extra_steps, 0, extra_steps, logger,
        lr_base=tcfg.lr * tcfg.control_lr_scale, phase="control",
    )
    logger.flush()
    stats = {
        "model_updates": backbone.stats.get("model_updates", 0) + extra_steps,
        "latent_updates": 0,
        "extra_steps": extra_steps,
        "wall_s": backbone.stats.get("wall_s", 0.0) + time.perf_counter() - started,
    }
    return Backbone(model, "nolatent", log=logger.records, stats=stats).freeze()


def latent_r2(z_table: Tensor, hidden: Tensor) -> list[float]:
    """R^2 of each hidden parameter regressed linearly on the latents.

    Diagnostic only: if latents do not encode the parameters, a null H1 means little. Association,
    not disentanglement. z_table (E, Z), hidden (E, H). Returns H values.
    """
    X = torch.cat([z_table.double(), torch.ones(z_table.shape[0], 1, dtype=torch.float64)], dim=1)
    Y = hidden.double()
    beta = torch.linalg.lstsq(X, Y).solution
    residual = Y - X @ beta
    total = ((Y - Y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return (1.0 - (residual**2).sum(0) / total).tolist()
