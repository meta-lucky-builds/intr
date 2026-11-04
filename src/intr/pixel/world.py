"""Encoder, JEPA training, state probe, and the dynamics core on embeddings.

The recipe, with reasons:

1. Train a small conv encoder with a JEPA loss: online encoder plus a residual predictor rolled 3 steps toward an EMA target
   encoder, with variance and covariance terms on the online embedding. EMA and stop-gradient on the target stop the easy solution of
   moving the target toward the prediction. Variance and covariance stop collapse and force several independent dimensions. A
   3-step rollout rewards encoding velocity, since position errors from a missing velocity compound. No state label is used here.
   GroupNorm, not BatchNorm, so an embedding never depends on what else is in the batch.

2. Freeze the EMA encoder. From here an embedding is a fixed function of images, so the state-based code applies unchanged with
   obs := embedding. intr.train trains W * phi(s, a, z), intr.adapt and intr.ebayes adapt it. The target of an RLS write is
   DynamicsModel.target: (s_next - s - delta_mean) / delta_std per dimension, with those statistics from training embeddings. Both
   fixed, so a write has a stationary target, and the Bayesian reading of P still holds as well as it did on state.

3. latent and nolatent backbones share one encoder. Their prediction losses are then in the same units, which is what makes Gate
   GP's "latent lowers prediction loss by 10%" a measurement rather than a comparison of scales.

4. Encoding for any metric or write runs in float32, outside autocast, always. bfloat16 has 8 mantissa bits: on embedding values
   near 3 with delta_std near 0.1, that is about 10% noise on a normalized target. Autocast is for JEPA training only.

5. torch.compile is not used. Training takes minutes, compile costs a good part of a minute per process, it could not be tested
   here, and EMA parameter handling across a compiled module is one more way to be wrong.

6. Two optional encoder terms, both off by default, built only when their weight is non-zero, and dropped when training ends. Only
   the EMA encoder leaves train_jepa. Inverse dynamics (as in PLDM, Sobal et al.): windows 0 and 1 of the online encoder predict action
   0, which is the action taking the last frame of window 0 to the last frame of window 1. State control: window 0 of the online
   encoder predicts the standardized observation of its last frame, obs[t] for frames t .. t + stack - 1, since stored states start
   stack - 1 steps before obs[0]. env.obs reads the state and no hidden parameter, and train_jepa never sees an environment.

The probe reads embeddings computed under no_grad on training environments only. fit_probe refuses any other tier.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..adapt import AdaptedPredictor
from ..envs import Trajectories, derive_seed, make_generator
from ..models import DynamicsModel, ModelConfig
from ..train import Backbone, TrainConfig, TransitionData, _cosine_lr, _set_lr, continued_training, train_latent, train_nolatent
from .config import CoreSettings, JepaSettings, Rung
from .render import render, stack_windows
from .sim import StateTrajectories


class ProbeLeakError(ValueError):
    """Something other than training environments reached the probe."""


class ConvEncoder(nn.Module):
    """(N, 3 stack, H, W) float in [0, 1] -> (N, D). Four stride-2 convs and a linear head."""

    def __init__(self, in_channels: int, size: int, embed_dim: int, width: int = 32):
        super().__init__()
        channels = [in_channels, width, 2 * width, 4 * width, 4 * width]
        layers: list[nn.Module] = []
        side = size
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.GELU()]
            side = (side + 1) // 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] * side * side, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x).flatten(1))


class JepaPredictor(nn.Module):
    """Residual: s_next = s + net(s, a)."""

    def __init__(self, embed_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim + act_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        return s + self.net(torch.cat([s, a.to(s.dtype)], dim=-1))


def aux_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class InverseDynamicsHead(nn.Module):
    """(s_t, s_t+1) -> a_t. Training only."""

    def __init__(self, embed_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = aux_mlp(2 * embed_dim, hidden, act_dim)

    def forward(self, s0: Tensor, s1: Tensor) -> Tensor:
        return self.net(torch.cat([s0, s1], dim=-1))


class StateAuxHead(nn.Module):
    """s_t -> standardized observable obs_t. Training only. The scale comes from the training observations given here."""

    def __init__(self, embed_dim: int, train_obs: Tensor, hidden: int):
        super().__init__()
        flat = train_obs.reshape(-1, train_obs.shape[-1])
        self.register_buffer("obs_mean", flat.mean(0))
        self.register_buffer("obs_std", flat.std(0).clamp_min(1e-6))
        self.net = aux_mlp(embed_dim, hidden, flat.shape[-1])

    def standardize(self, obs: Tensor) -> Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def forward(self, s0: Tensor) -> Tensor:
        return self.net(s0)


def var_cov_terms(z: Tensor) -> tuple[Tensor, Tensor]:
    """VICReg's two anti-collapse terms on (N, D), float32."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.t() @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return variance, (off**2).sum() / d


def jepa_loss(
    online: nn.Module, target: nn.Module, predictor: nn.Module, frames: Tensor, actions: Tensor, stack: int, s: JepaSettings, device_type: str, amp: bool,
    idm_head: nn.Module | None = None, state_head: nn.Module | None = None, obs0: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """frames (b, H + stack, 3, h, w) uint8, actions (b, H, A), obs0 (b, O) the observation of window 0's last frame.

    A term with weight zero is not computed, whatever heads are given, so the online encoder runs no extra forward for it.
    """
    b, H = actions.shape[0], actions.shape[1]
    extra: dict[str, tuple[float, Tensor]] = {}
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        windows = torch.stack([frames[:, i : i + stack] for i in range(1, H + 1)], dim=1)  # (b, H, stack, 3, h, w)
        with torch.no_grad():
            goal = target(stack_windows(windows.flatten(0, 1))).reshape(b, H, -1).float()
        s0 = online(stack_windows(frames[:, :stack]))
        current = s0
        sim = 0.0
        for i in range(H):
            current = predictor(current, actions[:, i])
            sim = sim + ((current.float() - goal[:, i]) ** 2).mean()
        sim = sim / H
        if s.idm_weight != 0.0:
            s1 = online(stack_windows(frames[:, 1 : 1 + stack]))
            extra["idm"] = (s.idm_weight, ((idm_head(s0, s1).float() - actions[:, 0].float()) ** 2).mean())
        if s.state_weight != 0.0:
            extra["state"] = (s.state_weight, ((state_head(s0).float() - state_head.standardize(obs0)) ** 2).mean())
    with torch.autocast(device_type=device_type, enabled=False):
        variance, covariance = var_cov_terms(s0.float())
    loss = s.sim_weight * sim + s.var_weight * variance + s.cov_weight * covariance
    parts = {"sim": sim.detach(), "var": variance.detach(), "cov": covariance.detach()}
    for name, (weight, term) in extra.items():
        loss = loss + weight * term
        parts[name] = term.detach()
    return loss, parts


def train_jepa(frames: Tensor, act: Tensor, stack: int, embed_dim: int, rung: Rung, s: JepaSettings, seed, device, amp: bool | None = None, obs: Tensor | None = None) -> tuple[ConvEncoder, dict]:
    """frames (E, B, stack + T, 3, h, w) uint8, act (E, B, T, A), both on device. Returns the frozen EMA encoder.

    obs (E, B, T + 1, O) on device, training environments only, read only when state_weight is non-zero.
    """
    started = time.perf_counter()
    device = torch.device(device)
    E, B, F_total = frames.shape[:3]
    T = act.shape[2]
    if F_total != stack + T:
        raise ValueError(f"{F_total} frames do not fit {T} transitions with a stack of {stack}")
    if s.state_weight != 0.0 and (obs is None or obs.shape[:3] != (E, B, T + 1)):
        raise ValueError("state_weight needs observations of shape (E, B, T + 1, O)")
    H = min(s.rollout, T)
    use_amp = amp if amp is not None else (device.type == "cuda" and torch.cuda.is_bf16_supported())
    torch.manual_seed(derive_seed("init", seed, "jepa"))
    online = ConvEncoder(3 * stack, rung.size, embed_dim, rung.width).to(device)
    predictor = JepaPredictor(embed_dim, act.shape[-1], s.predictor_hidden).to(device)
    target = copy.deepcopy(online).requires_grad_(False)
    params = list(online.parameters()) + list(predictor.parameters())
    # Built after the encoder and predictor, so every variant starts from the same weights and draws the same batches.
    idm_head = InverseDynamicsHead(embed_dim, act.shape[-1], s.predictor_hidden).to(device) if s.idm_weight != 0.0 else None
    state_head = StateAuxHead(embed_dim, obs, s.predictor_hidden).to(device) if s.state_weight != 0.0 else None
    for head in (idm_head, state_head):
        if head is not None:
            params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=s.lr)
    gen = make_generator("train", seed, "jepa")
    total = s.steps * rung.jepa_scale
    frame_offsets = torch.arange(H + stack, device=device)[None]
    action_offsets = torch.arange(H, device=device)[None]
    log, every = [], max(total // 10, 1)
    for step in range(total):
        _set_lr(optimizer, _cosine_lr(s.lr, step, total, s.lr_floor))
        e = torch.randint(E, (s.batch, 1), generator=gen).to(device)
        b = torch.randint(B, (s.batch, 1), generator=gen).to(device)
        t = torch.randint(T - H + 1, (s.batch, 1), generator=gen).to(device)
        obs0 = obs[e[:, 0], b[:, 0], t[:, 0]] if state_head is not None else None
        loss, parts = jepa_loss(online, target, predictor, frames[e, b, t + frame_offsets], act[e, b, t + action_offsets], stack, s, device.type, use_amp, idm_head, state_head, obs0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, s.grad_clip)
        optimizer.step()
        with torch.no_grad():
            for p_target, p_online in zip(target.parameters(), online.parameters()):
                p_target.lerp_(p_online, 1.0 - s.ema)
        if step % every == 0 or step == total - 1:
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"JEPA loss became {value} at step {step}")
            log.append({"step": step, "loss": value, **{k: float(v) for k, v in parts.items()}})
    target.eval()
    params_total = sum(p.numel() for p in target.parameters()) + sum(p.numel() for p in predictor.parameters())
    return target, {"log": log, "steps": total, "params": params_total, "amp": use_amp, "wall_s": time.perf_counter() - started}


@torch.no_grad()
def embed_frames(encoder: nn.Module, frames: Tensor, stack: int, chunk: int = 4096) -> Tensor:
    """frames (E, B, Tf, 3, h, w) -> (E, B, Tf - stack + 1, D) on the CPU, float32, outside autocast. Batch layout has no effect."""
    E, B, Tf = frames.shape[:3]
    windows = Tf - stack + 1
    flat = frames.reshape(E * B, Tf, *frames.shape[3:])
    per_chunk = max(chunk // windows, 1)
    out = []
    for i in range(0, E * B, per_chunk):
        x = flat[i : i + per_chunk].unfold(1, stack, 1).permute(0, 1, 5, 2, 3, 4)  # (m, windows, stack, 3, h, w)
        out.append(encoder(stack_windows(x.reshape(-1, *x.shape[2:]))).float().cpu())
    return torch.cat(out, dim=0).reshape(E, B, windows, -1)


@dataclass
class Observed:
    """Embedded trajectories, CPU. Index t of emb comes from frames t .. t + stack - 1 of the stored states."""

    emb: Tensor  # (E, B, T + 1, D)
    obs: Tensor  # (E, B, T + 1, O)
    act: Tensor  # (E, B, T, A)

    def take(self, n: int) -> "Observed":
        return Observed(self.emb[:, :n], self.obs[:, :n], self.act[:, :n])

    def trajectories(self) -> Trajectories:
        return Trajectories(self.emb, self.act)

    def as_dict(self) -> dict:
        return {"emb": self.emb, "obs": self.obs, "act": self.act}


def observe(traj: StateTrajectories, family: str, size: int, encoder: nn.Module, stack: int, palette: Tensor | None = None) -> Observed:
    """Render stored states and encode them."""
    if traj.burn != stack - 1:
        raise ValueError(f"burn-in {traj.burn} does not fit a stack of {stack}")
    frames = render(family, traj.states, size, palette)
    emb = embed_frames(encoder, frames, stack)
    return Observed(emb, traj.obs.cpu(), traj.act.cpu())


def embedding_stats(emb: Tensor) -> dict:
    """Collapse diagnostics on (..., D). Participation ratio is the effective number of dimensions."""
    x = emb.reshape(-1, emb.shape[-1]).double()
    x = x - x.mean(dim=0)
    eig = torch.linalg.eigvalsh((x.t() @ x) / max(x.shape[0] - 1, 1)).clamp_min(0.0)
    return {
        "std_min": x.std(dim=0).min().item(),
        "std_mean": x.std(dim=0).mean().item(),
        "participation_ratio": (eig.sum() ** 2 / (eig**2).sum().clamp_min(1e-30)).item(),
    }


class StateProbe(nn.Module):
    """Embedding to physical observation. Fit on training environments, stop-gradient by construction, used for every verdict metric."""

    def __init__(self, embed_dim: int, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, obs_dim))
        self.register_buffer("emb_mean", torch.zeros(embed_dim))
        self.register_buffer("emb_std", torch.ones(embed_dim))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, s: Tensor) -> Tensor:
        return self.obs_mean + self.obs_std * self.net((s - self.emb_mean) / self.emb_std)


def fit_probe(emb: Tensor, obs: Tensor, tier: str, scale: Tensor, hidden: int, steps: int, batch: int, seed, device) -> tuple[StateProbe, dict]:
    """emb (N, D), obs (N, O), CPU, no grad history. tier must be train."""
    if tier != "train":
        raise ProbeLeakError(f"the probe trains on training environments only, got tier {tier!r}")
    started = time.perf_counter()
    torch.manual_seed(derive_seed("init", seed, "probe"))
    probe = StateProbe(emb.shape[-1], obs.shape[-1], hidden)
    probe.emb_mean.copy_(emb.mean(0))
    probe.emb_std.copy_(emb.std(0).clamp_min(1e-6))
    probe.obs_mean.copy_(obs.mean(0))
    probe.obs_std.copy_(obs.std(0).clamp_min(1e-6))
    probe = probe.to(device)
    x, y = emb.detach().to(device), obs.to(device)
    scale = scale.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    gen = make_generator("train", seed, "probe")
    value = float("nan")
    for step in range(steps):
        _set_lr(optimizer, _cosine_lr(1e-3, step, steps, 0.05))
        idx = torch.randint(x.shape[0], (batch,), generator=gen).to(device)
        loss = (((probe(x[idx]) - y[idx]) / scale) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = loss.item() if step == steps - 1 else value
    probe.requires_grad_(False).eval()
    return probe, {"final_loss": value, "source_tiers": [tier], "n_samples": x.shape[0], "wall_s": time.perf_counter() - started}


@torch.no_grad()
def probe_r2(probe: StateProbe, emb: Tensor, obs: Tensor) -> dict:
    """R2 per observation dimension, on data the probe did not train on. emb (..., D), obs (..., O), any device."""
    device = probe.obs_mean.device
    x = emb.reshape(-1, emb.shape[-1]).to(device)
    y = obs.reshape(-1, obs.shape[-1]).to(device)
    pred = probe(x)
    total = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    r2 = 1.0 - ((pred - y) ** 2).sum(0) / total
    return {"per_dim": r2.tolist(), "mean": r2.mean().item()}


def transition_data(observed: Observed) -> TransitionData:
    trans = observed.trajectories().flat()
    B, T = observed.act.shape[1], observed.act.shape[2]
    return TransitionData(trans.obs, trans.act, trans.next_obs, B, T)


def train_config(core: CoreSettings) -> TrainConfig:
    return TrainConfig(
        model_steps=core.model_steps,
        warmup_steps=core.warmup_steps,
        block_steps=core.block_steps,
        batch_size=core.batch_size,
        latent_block_steps=core.latent_block_steps,
    )


def model_config(embed_dim: int, act_dim: int, z_dim: int, core: CoreSettings) -> ModelConfig:
    return ModelConfig(embed_dim, act_dim, z_dim, core.hidden, core.n_hidden, core.feature_dim)


def train_core(data: TransitionData, embed_dim: int, act_dim: int, z_dim: int, core: CoreSettings, seed) -> dict[str, Backbone]:
    """The existing training code on embeddings, CPU. latent, nolatent, and the matched continued-training control."""
    tcfg = train_config(core)
    latent = train_latent(data, model_config(embed_dim, act_dim, z_dim, core), tcfg, seed)
    nolatent = train_nolatent(data, model_config(embed_dim, act_dim, 0, core), tcfg, seed)
    control = continued_training(nolatent, data, latent.stats["latent_updates"], tcfg, seed)
    return {"latent": latent, "nolatent": nolatent, "control": control}


@torch.no_grad()
def exact_train_losses(backbones: dict[str, Backbone], data: TransitionData) -> dict:
    """Prediction loss over all training transitions, in the shared embedding's target units. Gate GP reads this."""
    latent = backbones["latent"]
    N = data.obs.shape[1]
    target = latent.model.target(data.obs, data.next_obs)
    own_z = latent.z_table[:, None, :].expand(-1, N, -1)
    mean_z = latent.z_default[None, None, :].expand(data.obs.shape[0], N, -1)
    out = {
        "latent": ((latent.model(data.obs, data.act, own_z) - target) ** 2).mean().item(),
        "latent_at_mean_z": ((latent.model(data.obs, data.act, mean_z) - target) ** 2).mean().item(),
    }
    for name in ("nolatent", "control"):
        model = backbones[name].model
        out[name] = ((model(data.obs, data.act) - model.target(data.obs, data.next_obs)) ** 2).mean().item()
    return out


def embedding_bounds(emb: Tensor) -> tuple[Tensor, Tensor]:
    """Rollout clip, the analogue of a family's observation bounds: the training range plus half of it on each side."""
    x = emb.reshape(-1, emb.shape[-1])
    low, high = x.min(0).values, x.max(0).values
    margin = 0.5 * (high - low).clamp_min(1e-6)
    return low - margin, high + margin


def on_device(predictor: AdaptedPredictor, model_dev: DynamicsModel, device) -> AdaptedPredictor:
    """Copy an adaptation result to a model on the device. The per-environment z and W are what adaptation produced."""
    W = predictor.rls.W32 if predictor.rls is not None else predictor.W
    return AdaptedPredictor(
        model_dev,
        z=None if predictor.z is None else predictor.z.to(device),
        W=None if W is None else W.to(device),
    )


def backbone_to_dict(b: Backbone) -> dict:
    return {
        "kind": b.kind,
        "model_cfg": asdict(b.model.cfg),
        "state": {k: v.cpu() for k, v in b.model.state_dict().items()},
        "z_default": b.z_default,
        "z_table": b.z_table,
        "stats": b.stats,
    }


def backbone_from_dict(d: dict) -> Backbone:
    model = DynamicsModel(ModelConfig(**d["model_cfg"]))
    model.load_state_dict(d["state"])
    return Backbone(model, d["kind"], d["z_default"], d["z_table"], stats=d["stats"]).freeze()
