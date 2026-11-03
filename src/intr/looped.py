"""Looped dynamics core and the measurements for Thread C (H7, H8, H9 in docs/HYPOTHESES.md).

One rollout step, with o (..., O), a (..., A), z (..., Z):

    e_s, e_a = enc_state(norm(o)), enc_action(a)
    z^0      = rescale(z_prev + prelude(z_prev, e_s, e_a))     z_prev = 0 on the first step
    z^(k+1)  = rescale(z^k + update(z^k, inj))                 same weights for every k, no k input
    next_o   = o + decoded delta of z^K

Switches (LoopConfig). Parameter count is the same under all of them, so an ablation changes behaviour,
not capacity. Disabled inputs are zeroed rather than removed.

    reinject_input     e_s enters inj at every loop. Off: e_s reaches the loop only through z^0.
    action_every_loop  e_a enters inj at every loop. Off: e_a reaches the loop only through z^0 (H8).
                       H7's "no input re-injection" removes both, since the input is (s, a).
    normalize          rescale() fixes each sample's state RMS to rms_target, one scalar per state vector,
                       as in Solv3R. Off: rescale() is the identity.
    mixed_depth        training depth is random; see below.
    carry              z_prev is the previous step's final z. Off: z_prev is always 0, which re-initializes
                       from the encoded (s, a).

The update, decoder and prelude have default initialization. Solv3R zero-initializes its injection map,
which would make "re-injection on" and "off" identical at initialization. That is not done here.

The decoder reads z only. If it also saw e_s or e_a, input could bypass the loop and the ablations would
measure little.

Bounding gradient depth. Training is teacher-forced over short windows. At each rollout step:
    r no-grad calls, then g = grad_loops differentiable calls, with z_prev detached at every step boundary.
So backpropagation crosses at most g update calls per step, whatever K and the window length are.
    mixed_depth on:  r ~ Uniform{0, ..., 2 (K - g)}, total depth in [g, 2 K - g], mean K.
    mixed_depth off: r = K - g, total depth exactly K.
Both arms have the same mean forward calls and exactly the same backward depth, so this ablation isolates
depth randomness, and mixed depth costs no extra compute on average. No-grad calls build no graph.

Without a path around the no-grad calls, anything entering only through z^0 would get no gradient when
r > 0, and the prelude-only and no-reinjection arms would barely train their encoders. The differentiable
calls therefore start from z_pre + z^0 - detach(z^0): same value as z_pre, with d z_pre / d z^0 approximated
by the identity (Jacobian-free backprop, as used for deep equilibrium models). The approximation is biased,
and it is the same in every arm.

Fragile by construction: training windows carry z for only a few steps, so long carried rollouts extrapolate
in carry age as well as in loop count. Trained values are recorded with every result.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor, nn

from .envs import BatchEnv, PuckEnv, Trajectories, _policy_action, derive_seed, make_generator
from .models import Counts, DynamicsModel, ModelConfig, clip_obs
from .train import _cosine_lr, _set_lr

BLOWUP_LIMIT = 0.05  # H7: "stable" requires fewer than this fraction of clipped or non-finite rollouts
AUROC_FLOOR = 0.7  # H9 kill threshold from HYPOTHESES.md
TWO_X_THRESHOLD = 0.5  # paired_comparison(full, ablated, threshold=0.5) wins exactly when ablated >= 2 x full


@dataclass
class LoopConfig:
    obs_dim: int
    act_dim: int
    z_dim: int = 48
    emb_dim: int = 24
    hidden: int = 96
    loops: int = 8  # K: the trained loop count, the mean total calls per rollout step
    grad_loops: int = 4  # g: differentiable calls per rollout step, which bounds gradient depth
    reinject_input: bool = True
    action_every_loop: bool = True
    normalize: bool = True
    mixed_depth: bool = True
    carry: bool = True
    rms_target: float = 0.85  # Solv3R's rho

    def __post_init__(self):
        if not 1 <= self.grad_loops <= self.loops:
            raise ValueError(f"need 1 <= grad_loops <= loops, got {self.grad_loops} and {self.loops}")

    def pre_call_range(self) -> tuple[int, int]:
        """Inclusive range of no-grad calls per rollout step in training."""
        extra = self.loops - self.grad_loops
        return (0, 2 * extra) if self.mixed_depth else (extra, extra)

    def trained_depth_range(self) -> tuple[int, int]:
        """Inclusive range of total calls per rollout step that training saw."""
        low, high = self.pre_call_range()
        return low + self.grad_loops, high + self.grad_loops


def draw_pre_calls(cfg: LoopConfig, gen: torch.Generator) -> int:
    low, high = cfg.pre_call_range()
    if low == high:
        return low
    return int(torch.randint(low, high + 1, (1,), generator=gen))


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def linear_macs(module: nn.Module) -> int:
    """Multiply-accumulates of every nn.Linear inside the module, per sample."""
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


class LoopedDynamics(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg
        O, A, Z, E, H = cfg.obs_dim, cfg.act_dim, cfg.z_dim, cfg.emb_dim, cfg.hidden
        self.enc_state = _mlp(O, H, E)
        self.enc_action = _mlp(A, E, E)
        self.prelude_net = _mlp(Z + 2 * E, H, Z)
        self.update = _mlp(Z + 2 * E, H, Z)
        self.decoder = _mlp(Z, H, O)
        self.register_buffer("obs_mean", torch.zeros(O))
        self.register_buffer("obs_std", torch.ones(O))
        self.register_buffer("delta_mean", torch.zeros(O))
        self.register_buffer("delta_std", torch.ones(O))

    def set_normalization(self, obs: Tensor, next_obs: Tensor) -> None:
        """Same statistics as DynamicsModel.set_normalization."""
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
        return (next_o - o - self.delta_mean) / self.delta_std

    def apply_delta(self, o: Tensor, normalized_delta: Tensor) -> Tensor:
        return o + normalized_delta * self.delta_std + self.delta_mean

    def rescale(self, u: Tensor) -> Tensor:
        """Fix the RMS of each state vector. The small constant keeps a zero state at zero."""
        if not self.cfg.normalize:
            return u
        rms = (u.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        return self.cfg.rms_target * u / rms

    def injection(self, e_s: Tensor, e_a: Tensor) -> Tensor:
        """What each loop sees of the input. (..., 2 E)."""
        s = e_s if self.cfg.reinject_input else torch.zeros_like(e_s)
        a = e_a if self.cfg.action_every_loop else torch.zeros_like(e_a)
        return torch.cat([s, a], dim=-1)

    def begin_step(self, o: Tensor, a: Tensor, z_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Returns z^0 (..., Z) and the injection (..., 2 E). z_prev is ignored when carry is off."""
        e_s = self.enc_state(self.norm_obs(o))
        e_a = self.enc_action(a)
        if z_prev is None or not self.cfg.carry:
            z_prev = o.new_zeros(*o.shape[:-1], self.cfg.z_dim)
        z0 = self.rescale(z_prev + self.prelude_net(torch.cat([z_prev, e_s, e_a], dim=-1)))
        return z0, self.injection(e_s, e_a)

    def cell(self, z: Tensor, inj: Tensor) -> Tensor:
        """One update. Stationary: it takes no loop index and holds no per-call state."""
        return self.rescale(z + self.update(torch.cat([z, inj], dim=-1)))

    def solve(self, z: Tensor, inj: Tensor, n: int) -> Tensor:
        for _ in range(n):
            z = self.cell(z, inj)
        return z

    def solve_traced(self, z: Tensor, inj: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """Also returns ||z_{k+1} - z_k|| / ||z_k|| for each call. Second output (..., n)."""
        residuals = []
        for _ in range(n):
            next_z = self.cell(z, inj)
            residuals.append((next_z - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-12))
            z = next_z
        return z, torch.stack(residuals, dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def step(self, o: Tensor, a: Tensor, z_prev: Tensor | None = None, loops: int | None = None) -> tuple[Tensor, Tensor]:
        """Returns next_o (..., O) and the final state (..., Z)."""
        n = self.cfg.loops if loops is None else loops
        z0, inj = self.begin_step(o, a, z_prev)
        z = self.solve(z0, inj, n)
        return self.apply_delta(o, self.decode(z)), z

    def training_step(self, o: Tensor, a: Tensor, z_prev: Tensor | None, pre_calls: int) -> tuple[Tensor, Tensor]:
        """Returns the normalized delta and final z. See the module docstring for the gradient path."""
        z0, inj = self.begin_step(o, a, z_prev)
        z = z0
        if pre_calls > 0:
            with torch.no_grad():
                z_pre = self.solve(z0, inj, pre_calls)
            z = z_pre + z0 - z0.detach()
        z = self.solve(z, inj, self.cfg.grad_loops)
        return self.decode(z), z

    def cell_macs(self) -> int:
        return linear_macs(self.update)

    def base_macs(self) -> int:
        """Forward cost per rollout step that does not scale with the loop count."""
        return (
            linear_macs(self.enc_state)
            + linear_macs(self.enc_action)
            + linear_macs(self.prelude_net)
            + linear_macs(self.decoder)
        )

    def macs_per_step(self, loops: int | None = None) -> int:
        n = self.cfg.loops if loops is None else loops
        return self.base_macs() + n * self.cell_macs()


def check_ablations(base_cfg: LoopConfig, arms: dict[str, dict], seed=0) -> dict:
    """Guard against ablations that are silently identical. Raises RuntimeError if one is.

    Every arm gets the same weights and inputs. An arm that changes the forward computation must change
    the output. An arm that changes only the training schedule must change the depth schedule.
    """
    gen = make_generator("ablation_check", seed)
    o = torch.randn(16, base_cfg.obs_dim, generator=gen)
    a = 2.0 * torch.rand(16, base_cfg.act_dim, generator=gen) - 1.0
    z_prev = torch.randn(16, base_cfg.z_dim, generator=gen)
    torch.manual_seed(derive_seed("ablation_check_init", seed))
    base = LoopedDynamics(base_cfg)
    with torch.no_grad():
        reference, _ = base.step(o, a, z_prev)
    record = {}
    for name, overrides in arms.items():
        if not overrides:
            continue
        cfg = replace(base_cfg, **overrides)
        model = LoopedDynamics(cfg)
        model.load_state_dict(base.state_dict())
        with torch.no_grad():
            out, _ = model.step(o, a, z_prev)
        diff = (out - reference).abs().max().item()
        schedule_differs = cfg.pre_call_range() != base_cfg.pre_call_range()
        forward_keys = {"reinject_input", "action_every_loop", "normalize", "carry"}
        changes_forward = any(k in forward_keys and v != getattr(base_cfg, k) for k, v in overrides.items())
        if changes_forward and not diff > 1e-6:
            raise RuntimeError(f"ablation {name} leaves the forward pass unchanged (max diff {diff})")
        if not changes_forward and not schedule_differs:
            raise RuntimeError(f"ablation {name} changes neither the forward pass nor the depth schedule")
        record[name] = {"max_output_diff": diff, "schedule_differs": schedule_differs, "params": count_params(model)}
    return record


# ---------------------------------------------------------------- training


@dataclass
class LoopTrainConfig:
    steps: int = 3000
    batch_windows: int = 64
    window: int = 4  # 64 windows of 4 is 256 transitions, the batch size of Thread A
    lr: float = 1e-3
    lr_floor: float = 0.05
    grad_clip: float = 5.0
    log_every: int = 500


@dataclass
class SequenceData:
    obs: Tensor  # (E, B, T + 1, O)
    act: Tensor  # (E, B, T, A)

    @classmethod
    def from_trajectories(cls, trajs: Trajectories) -> "SequenceData":
        return cls(trajs.obs, trajs.act)

    def sample(self, batch: int, window: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """o (batch, window + 1, O), a (batch, window, A), from random windows."""
        E, B, T, _ = self.act.shape
        if window > T:
            raise ValueError(f"window {window} is longer than the {T} steps of data")
        env_idx = torch.randint(E, (batch, 1), generator=gen)
        traj_idx = torch.randint(B, (batch, 1), generator=gen)
        start = torch.randint(T - window + 1, (batch, 1), generator=gen)
        t_idx = start + torch.arange(window + 1)[None, :]  # (batch, window + 1)
        return self.obs[env_idx, traj_idx, t_idx], self.act[env_idx, traj_idx, t_idx[:, :window]]


@dataclass
class TrainedLooped:
    model: LoopedDynamics
    stats: dict = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)


def train_looped(data: SequenceData, cfg: LoopConfig, tcfg: LoopTrainConfig, seed) -> TrainedLooped:
    """Teacher-forced windows with carried, detached state.

    Initial weights and batches depend only on the seed, so arms of one seed start identically and see the
    same data. Depth draws come from a separate generator, so mixed and fixed depth see the same batches.
    A non-finite loss or gradient skips the update and is counted, so a diverging arm cannot silently
    poison the optimizer or hide what happened.
    """
    started = time.perf_counter()
    torch.manual_seed(derive_seed("init", seed, "looped"))
    model = LoopedDynamics(cfg)
    model.set_normalization(data.obs[:, :, :-1], data.obs[:, :, 1:])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    batch_gen = make_generator("train", seed, "looped", "batches")
    depth_gen = make_generator("train", seed, "looped", "depth")
    base_macs, cell_macs = model.base_macs(), model.cell_macs()
    counts = Counts()
    no_grad_calls, grad_calls, macs_total, skipped = 0, 0, 0, 0
    log: list[dict] = []
    running, running_n = 0.0, 0
    first_loss = None
    recent: list[float] = []
    recent_size = max(1, tcfg.steps // 10)
    for step in range(tcfg.steps):
        _set_lr(optimizer, _cosine_lr(tcfg.lr, step, tcfg.steps, tcfg.lr_floor))
        o, a = data.sample(tcfg.batch_windows, tcfg.window, batch_gen)
        z = None
        loss = o.new_zeros(())
        for t in range(tcfg.window):
            pre = draw_pre_calls(cfg, depth_gen)
            pred, z = model.training_step(o[:, t], a[:, t], z, pre)
            loss = loss + ((pred - model.target(o[:, t], o[:, t + 1])) ** 2).mean()
            z = z.detach()
            no_grad_calls += pre
            grad_calls += cfg.grad_loops
            macs_total += o.shape[0] * (base_macs + (pre + cfg.grad_loops) * cell_macs)
            counts.forward_calls += 1
            counts.forward_samples += o.shape[0]
        loss = loss / tcfg.window
        value = loss.item()
        if not math.isfinite(value):
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        counts.backward_calls += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        if not math.isfinite(float(norm)):
            skipped += 1
            continue
        optimizer.step()
        if first_loss is None:
            first_loss = value
        recent.append(value)
        if len(recent) > recent_size:
            recent.pop(0)
        running += value
        running_n += 1
        if running_n >= tcfg.log_every:
            log.append({"update": step + 1, "loss": running / running_n})
            running, running_n = 0.0, 0
    if running_n > 0:
        log.append({"update": tcfg.steps, "loss": running / running_n})
    model.requires_grad_(False)
    stats = {
        "updates": tcfg.steps,
        "skipped_nonfinite_updates": skipped,
        "t_calls_no_grad": no_grad_calls,
        "t_calls_grad": grad_calls,
        "forward_macs_total": macs_total,
        "first_loss": first_loss,
        "final_loss": float(np.mean(recent)) if recent else None,
        "counts": counts.as_dict(),
        "wall_s": time.perf_counter() - started,
    }
    return TrainedLooped(model, stats, log)


# ---------------------------------------------------------------- baselines


def mlp_param_count(cfg: ModelConfig) -> int:
    """Trainable parameters of DynamicsModel(cfg)."""
    d = cfg.obs_dim + cfg.act_dim + cfg.z_dim
    n = 0
    for _ in range(cfg.n_hidden):
        n += d * cfg.hidden + cfg.hidden
        d = cfg.hidden
    return n + d * cfg.feature_dim + cfg.feature_dim + cfg.obs_dim * cfg.feature_dim


def mlp_macs(cfg: ModelConfig) -> int:
    """Forward multiply-accumulates of DynamicsModel(cfg) per sample."""
    d = cfg.obs_dim + cfg.act_dim + cfg.z_dim
    n = 0
    for _ in range(cfg.n_hidden):
        n += d * cfg.hidden
        d = cfg.hidden
    return n + d * cfg.feature_dim + cfg.obs_dim * cfg.feature_dim


def _matching_config(count_fn, target: int, obs_dim: int, act_dim: int, feature_dim: int, n_hidden: int) -> ModelConfig:
    make = lambda width: ModelConfig(obs_dim, act_dim, 0, width, n_hidden, feature_dim)
    low, high = 1, 16384
    while low < high:
        mid = (low + high) // 2
        if count_fn(make(mid)) < target:
            low = mid + 1
        else:
            high = mid
    best = low
    if low > 1 and abs(count_fn(make(low - 1)) - target) < abs(count_fn(make(low)) - target):
        best = low - 1
    return make(best)


def mlp_matching_params(target: int, obs_dim: int, act_dim: int, feature_dim: int, n_hidden: int = 2) -> ModelConfig:
    return _matching_config(mlp_param_count, target, obs_dim, act_dim, feature_dim, n_hidden)


def mlp_matching_macs(target: int, obs_dim: int, act_dim: int, feature_dim: int, n_hidden: int = 2) -> ModelConfig:
    return _matching_config(mlp_macs, target, obs_dim, act_dim, feature_dim, n_hidden)


# ---------------------------------------------------------------- predictors


class LoopedPredictor:
    """Carried rollouts at a chosen loop count."""

    def __init__(self, model: LoopedDynamics, loops: int | None = None):
        self.model = model
        self.loops = model.cfg.loops if loops is None else loops
        self.counts = Counts()
        self.t_calls = 0

    def advance(self, o: Tensor, a: Tensor, carry: Tensor | None) -> tuple[Tensor, Tensor]:
        self.counts.forward_calls += 1
        self.counts.forward_samples += int(np.prod(o.shape[:-1]))
        self.t_calls += self.loops
        return self.model.step(o, a, carry, self.loops)


class MemorylessPredictor:
    """Wraps any step(o, a), such as DynamicsModel.step or a persistence model."""

    def __init__(self, step_fn):
        self.step_fn = step_fn
        self.counts = Counts()

    def advance(self, o: Tensor, a: Tensor, carry) -> tuple[Tensor, None]:
        self.counts.forward_calls += 1
        self.counts.forward_samples += int(np.prod(o.shape[:-1]))
        return self.step_fn(o, a), None

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        return self.advance(o, a, None)[0]


class PlanningPredictor:
    """Stateful step(o, a) for eval.cem_plan, which has no reset hook.

    cem_plan makes exactly `horizon` calls per imagined rollout, so state resets every `horizon` calls.
    Each imagination therefore starts from z = 0 at the observed state. State is carried inside an
    imagination but not across real steps. This is an interface limit of eval.closed_loop.
    """

    def __init__(self, model: LoopedDynamics, loops: int | None, horizon: int):
        self.model = model
        self.loops = model.cfg.loops if loops is None else loops
        self.horizon = horizon
        self.counts = Counts()
        self.t_calls = 0
        self._calls = 0
        self._carry: Tensor | None = None

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        if self._calls % self.horizon == 0:
            self._carry = None
        self._calls += 1
        self.counts.forward_calls += 1
        self.counts.forward_samples += int(np.prod(o.shape[:-1]))
        self.t_calls += self.loops
        next_o, self._carry = self.model.step(o, a, self._carry, self.loops)
        return next_o


# ---------------------------------------------------------------- open-loop rollouts


def make_windows(holdout: Trajectories, starts, horizon: int) -> tuple[Tensor, Tensor]:
    """Same windows as eval.open_loop_errors. true (E, B S, H + 1, O), act (E, B S, H, A)."""
    E, B, T, A = holdout.act.shape
    O = holdout.obs.shape[-1]
    if max(starts) + horizon > T:
        raise ValueError(f"holdout of {T} steps is too short for start {max(starts)} and horizon {horizon}")
    S = len(starts)
    true = torch.stack([holdout.obs[:, :, s : s + horizon + 1] for s in starts], dim=2)
    act = torch.stack([holdout.act[:, :, s : s + horizon] for s in starts], dim=2)
    return true.reshape(E, B * S, horizon + 1, O), act.reshape(E, B * S, horizon, A)


def _state_stats(z: Tensor) -> dict:
    """z (E, W, Z). Catches norm collapse, explosion, and every window sharing one state."""
    rms = z.pow(2).mean(dim=-1).sqrt()  # (E, W)
    finite = torch.isfinite(rms)
    stats = {
        "z_rms_median": rms[finite].median().item() if finite.any() else float("nan"),
        "z_collapse_frac": (rms < 1e-3).float().mean().item(),
        "z_explode_frac": ((~finite) | (rms > 1e3)).float().mean().item(),
        "z_spread": float("nan"),
    }
    rows = z[finite]
    if rows.shape[0] >= 2:
        total = rows.pow(2).sum(dim=-1).mean().clamp_min(1e-30)
        stats["z_spread"] = (1.0 - rows.mean(dim=0).pow(2).sum() / total).item()
    return stats


@torch.no_grad()
def rollout_metrics(
    predictor,
    true: Tensor,
    act: Tensor,
    horizons,
    obs_scale: Tensor,
    low: Tensor,
    high: Tensor,
    keep_per_env: bool = False,
    per_window: bool = False,
) -> dict[str, dict]:
    """Open-loop rollouts that count blow-ups instead of hiding them.

    err matches eval.open_loop_errors: predictions are clipped to the observation bounds, a non-finite value
    becomes a bound, and the error stays finite and large. Flags are sticky over a rollout:
        nonfinite_frac  a prediction or carried state was non-finite at some step up to this horizon
        clipped_frac    a finite prediction left the observation bounds at some step up to this horizon
        blowup_frac     either of the above
        err_intact      RMSE over rollouts with neither flag, None if there are none
    Returns {"h<H>": cell}. With per_window, each cell holds "_sq" (E, W), for the caller to pop.
    """
    H = max(horizons)
    if true.shape[2] <= H:
        raise ValueError(f"windows of {true.shape[2] - 1} steps are too short for horizon {H}")
    wanted = set(horizons)
    o = true[:, :, 0]
    carry = None
    nonfinite = torch.zeros(true.shape[:2], dtype=torch.bool)
    clipped = torch.zeros(true.shape[:2], dtype=torch.bool)
    out = {}
    for t in range(H):
        raw, carry = predictor.advance(o, act[:, :, t], carry)
        bad = ~torch.isfinite(raw).all(dim=-1)
        if carry is not None:
            bad = bad | ~torch.isfinite(carry).all(dim=-1)
        nonfinite = nonfinite | bad
        clipped = clipped | (raw < low).any(dim=-1) | (raw > high).any(dim=-1)
        o = clip_obs(raw, low, high)
        h = t + 1
        if h not in wanted:
            continue
        sq = (((o - true[:, :, h]) / obs_scale) ** 2).mean(dim=-1)  # (E, W)
        per_env = sq.mean(dim=1).sqrt()  # (E,)
        blowup = nonfinite | clipped
        intact = ~blowup
        cell = {
            "err": per_env.mean().item(),
            "nonfinite_frac": nonfinite.float().mean().item(),
            "clipped_frac": clipped.float().mean().item(),
            "blowup_frac": blowup.float().mean().item(),
            "err_intact": sq[intact].mean().sqrt().item() if intact.any() else None,
            "n_rollouts": int(sq.numel()),
        }
        if carry is not None:
            cell.update(_state_stats(carry))
        if keep_per_env:
            cell["per_env"] = per_env.tolist()
        if per_window:
            cell["_sq"] = sq
        out[f"h{h}"] = cell
    return out


def loop_relation(loops: int, trained_low: int, trained_high: int) -> str:
    if loops < trained_low:
        return "below_trained"
    if loops > trained_high:
        return "above_trained"
    return "within_trained"


def rollout_grid(model: LoopedDynamics, true: Tensor, act: Tensor, loop_grid, horizons, obs_scale, low, high) -> dict:
    """Loops by horizon, with every cell tagged by its relation to the trained depth.

    The model's own trained loop count is always added to the grid, so at-trained behaviour is present
    whether or not it is one of the requested values.
    """
    trained_low, trained_high = model.cfg.trained_depth_range()
    loops_out = {}
    for n in sorted(set(loop_grid) | {model.cfg.loops}):
        started = time.perf_counter()
        predictor = LoopedPredictor(model, n)
        cells = rollout_metrics(predictor, true, act, horizons, obs_scale, low, high)
        loops_out[str(n)] = {
            "relation": loop_relation(n, trained_low, trained_high),
            "is_trained_loops": n == model.cfg.loops,
            "ratio_to_trained_max": n / trained_high,
            "t_calls": predictor.t_calls,
            "wall_s": time.perf_counter() - started,
            "horizons": cells,
        }
    return {
        "trained_loops": model.cfg.loops,
        "trained_depth_range": [trained_low, trained_high],
        "grad_loops": model.cfg.grad_loops,
        "loops": loops_out,
    }


def saturation_reference(true: Tensor, horizons, obs_scale: Tensor, stride: int) -> dict[str, float]:
    """Error of predicting another trajectory's state at the same step.

    A prediction with no information about the rollout scores near this. Merely inaccurate models saturate
    here, so a 2x error ratio is reachable only through blow-up once a model is above half of this value.
    """
    other = torch.roll(true, shifts=stride, dims=1)
    out = {}
    for h in horizons:
        sq = (((other[:, :, h] - true[:, :, h]) / obs_scale) ** 2).mean(dim=-1)
        out[f"h{h}"] = sq.mean(dim=1).sqrt().mean().item()
    return out


@torch.no_grad()
def sensitivity(model: LoopedDynamics, o: Tensor, a: Tensor, gen: torch.Generator) -> dict:
    """How much carried state and action change a prediction, in delta-std units. o (N, O), a (N, A).

    carry_sensitivity near 0 means a carried arm is effectively memoryless, so a "product" test would not be
    testing carried state. action_sensitivity near 0 means the action was lost, a risk for prelude-only.
    """
    _, z = model.step(o, a)
    base, _ = model.step(o, a, z)
    fresh, _ = model.step(o, a, None)
    perm = torch.randperm(a.shape[0], generator=gen)
    other, _ = model.step(o, a[perm], z)
    return {
        "carry_sensitivity": ((base - fresh).abs() / model.delta_std).mean().item(),
        "action_sensitivity": ((base - other).abs() / model.delta_std).mean().item(),
    }


# ---------------------------------------------------------------- contact and regime switches


def single_substep_env(env: BatchEnv) -> BatchEnv:
    """A shallow copy whose step() runs one physics substep of env, using env's own physics code."""
    sub = copy.copy(env)
    sub.dt = env.dt / env.n_substeps
    sub.n_substeps = 1
    return sub


@torch.no_grad()
def step_with_contact(env: PuckEnv, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """env.step, plus whether contact occurred during this control step.

    PuckEnv does not expose contact, and a collision can begin and end inside one 0.1 s control step, so
    checking states at control boundaries would miss it. This runs the env's own substeps and evaluates the
    env's contact conditions at each one:
        agent_puck  center distance below 2 radius
        wall        agent or puck beyond half_size - radius on some axis
    The resulting state equals env.step(state, action) exactly. All shapes (E, B, ...).
    """
    sub = single_substep_env(env)
    limit = env.half_size - env.radius
    agent_puck = torch.zeros(state.shape[:-1], dtype=torch.bool)
    wall = torch.zeros(state.shape[:-1], dtype=torch.bool)
    for _ in range(env.n_substeps):
        pa, pp = state[..., :2], state[..., 2:4]
        d = pp - pa
        agent_puck = agent_puck | (torch.sqrt((d * d).sum(-1) + 1e-12) < 2 * env.radius)
        wall = wall | (pa.abs() > limit).any(dim=-1) | (pp.abs() > limit).any(dim=-1)
        state = sub.step(state, action)
    return state, agent_puck, wall


@torch.no_grad()
def collect_contact_trajectories(env: PuckEnv, n_traj: int, n_steps: int, policy: str, gen) -> tuple[Trajectories, Tensor, Tensor]:
    """Trajectories plus per-transition flags agent_puck and wall, each (E, B, T)."""
    state = env.reset(n_traj, gen)
    o = env.obs(state)
    obs, act, ap, wall = [o], [], [], []
    noise = None
    for _ in range(n_steps):
        a, noise = _policy_action(env, policy, o, noise, gen)
        state, c, w = step_with_contact(env, state, a)
        o = env.obs(state)
        obs.append(o)
        act.append(a)
        ap.append(c)
        wall.append(w)
    return Trajectories(torch.stack(obs, dim=2), torch.stack(act, dim=2)), torch.stack(ap, dim=2), torch.stack(wall, dim=2)


@torch.no_grad()
def replay_contact_flags(env: PuckEnv, trajs: Trajectories) -> tuple[Tensor, Tensor]:
    """Flags for recorded trajectories, replaying each transition from its stored float32 observation."""
    ap, wall = [], []
    for t in range(trajs.act.shape[2]):
        state = env.state_from_obs(trajs.obs[:, :, t])
        _, c, w = step_with_contact(env, state, trajs.act[:, :, t])
        ap.append(c)
        wall.append(w)
    return torch.stack(ap, dim=2), torch.stack(wall, dim=2)


def window_contact_mask(agent_puck: Tensor, starts, horizon: int) -> Tensor:
    """Which rollout windows contain contact. (E, B, T) to (E, B S), in make_windows order."""
    E = agent_puck.shape[0]
    return torch.stack([agent_puck[:, :, s : s + horizon].any(dim=-1) for s in starts], dim=2).reshape(E, -1)


@torch.no_grad()
def collect_switching(env_pre: BatchEnv, env_post: BatchEnv, n_traj: int, n_steps: int, t_switch: int, policy: str, gen) -> Trajectories:
    """Transitions t < t_switch use env_pre, later ones env_post. obs[t_switch + 1] first shows the change.

    With env_post = env_pre and the same generator seed, this is the matched no-switch control: identical
    initial states and action noise.
    """
    if env_pre.n_envs != env_post.n_envs:
        raise ValueError("both regimes need the same number of environments")
    state = env_pre.reset(n_traj, gen)
    o = env_pre.obs(state)
    obs, act = [o], []
    noise = None
    for t in range(n_steps):
        a, noise = _policy_action(env_pre, policy, o, noise, gen)
        state = (env_pre if t < t_switch else env_post).step(state, a)
        o = env_pre.obs(state)
        obs.append(o)
        act.append(a)
    return Trajectories(torch.stack(obs, dim=2), torch.stack(act, dim=2))


# ---------------------------------------------------------------- H9 signals


@torch.no_grad()
def teacher_forced_trace(model: LoopedDynamics, obs: Tensor, act: Tensor, loops: int | None = None) -> dict[str, Tensor]:
    """Online use: each step sees the true observation, with z carried. obs (N, T + 1, O), act (N, T, A).

    residual_last (N, T)   ||z_{k+1} - z_k|| / ||z_k|| at the final loop of each step, the H9 signal
    residual_mean (N, T)   the same, averaged over the step's loops
    pred (N, T, O)         pred[:, t] predicts obs[:, t + 1]
    """
    n = model.cfg.loops if loops is None else loops
    z = None
    last, mean, pred = [], [], []
    for t in range(act.shape[1]):
        z0, inj = model.begin_step(obs[:, t], act[:, t], z)
        z, residuals = model.solve_traced(z0, inj, n)
        last.append(residuals[..., -1])
        mean.append(residuals.mean(dim=-1))
        pred.append(model.apply_delta(obs[:, t], model.decode(z)))
    return {
        "residual_last": torch.stack(last, dim=1),
        "residual_mean": torch.stack(mean, dim=1),
        "pred": torch.stack(pred, dim=1),
    }


@dataclass
class GaussianResidualModel:
    """Fixed Gaussian on scaled one-step prediction errors, fit on clean data.

    A simplified stand-in for ARCADE's predictive likelihood. ARCADE uses a Bayesian linear decoder with a
    state-dependent predictive variance and a beam of changepoint hypotheses. This uses a fixed covariance
    and a CUSUM of the log likelihood ratio between "unchanged" and "variance inflated by kappa^2".
    """

    mean: Tensor  # (D,)
    precision: Tensor  # (D, D)
    logdet: float
    dim: int

    @classmethod
    def fit(cls, residuals: Tensor, ridge: float = 1e-6) -> "GaussianResidualModel":
        """residuals (N, D). Non-finite rows are dropped. Raises ValueError if too few remain."""
        r = residuals.double()
        r = r[torch.isfinite(r).all(dim=-1)]
        D = r.shape[-1]
        if r.shape[0] < D + 1:
            raise ValueError(f"need at least {D + 1} finite residuals, got {r.shape[0]}")
        mean = r.mean(dim=0)
        c = r - mean
        cov = c.t() @ c / (r.shape[0] - 1)
        cov = cov + ridge * (cov.diagonal().mean() + 1e-12) * torch.eye(D, dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        return cls(mean, torch.cholesky_inverse(chol), 2.0 * chol.diagonal().log().sum().item(), D)

    def mahalanobis(self, r: Tensor) -> Tensor:
        d = r.double() - self.mean
        return ((d @ self.precision) * d).sum(dim=-1)

    def nll(self, r: Tensor) -> Tensor:
        return 0.5 * (self.mahalanobis(r) + self.logdet + self.dim * math.log(2.0 * math.pi))

    def cusum(self, r_seq: Tensor, kappa: float = 2.0) -> Tensor:
        """Page's CUSUM along time. r_seq (N, T, D) to (N, T). A non-finite step counts as strong evidence."""
        llr = 0.5 * self.mahalanobis(r_seq) * (1.0 - 1.0 / kappa**2) - self.dim * math.log(kappa)
        llr = torch.nan_to_num(llr, nan=1e6, posinf=1e6, neginf=0.0)
        out = []
        running = torch.zeros(r_seq.shape[0], dtype=torch.float64)
        for t in range(r_seq.shape[1]):
            running = (running + llr[:, t]).clamp_min(0.0)
            out.append(running)
        return torch.stack(out, dim=1)


def to_signal_time(x: Tensor) -> Tensor:
    """Align a signal about obs[t + 1], stored at index t, to the time it becomes known. (N, T) to (N, T).

    Index t of the result uses observations up to obs[t], matching the loop residual at step t. Index 0 is NaN.
    """
    pad = torch.full((x.shape[0], 1), float("nan"), dtype=x.dtype)
    return torch.cat([pad, x[:, :-1]], dim=1)


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN if either class is empty."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse.ravel()]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_groups(positive: Tensor, negative: Tensor) -> dict:
    """AUROC with sample counts. A non-finite score counts as maximally alarming, and is reported."""
    pos = positive.double().reshape(-1)
    neg = negative.double().reshape(-1)
    scores = torch.cat([pos, neg])
    finite = torch.isfinite(scores)
    n_nonfinite = int((~finite).sum())
    ceiling = scores[finite].max().item() + 1.0 if finite.any() else 1.0
    scores = torch.where(finite, scores, torch.full_like(scores, ceiling))
    labels = torch.cat([torch.ones(pos.numel(), dtype=torch.bool), torch.zeros(neg.numel(), dtype=torch.bool)])
    return {
        "auroc": auroc(scores.numpy(), labels.numpy()),
        "n_pos": int(pos.numel()),
        "n_neg": int(neg.numel()),
        "n_nonfinite": n_nonfinite,
    }


def _any_at_lags(x: Tensor, lags) -> Tensor:
    """out[:, t] is True if x[:, t - l] is True for some l in lags. Out-of-range positions are False."""
    out = torch.zeros_like(x)
    T = x.shape[1]
    for lag in lags:
        if abs(lag) >= T:
            continue
        if lag >= 0:
            out[:, lag:] |= x[:, : T - lag]
        else:
            out[:, : T + lag] |= x[:, -lag:]
    return out


def contact_masks(agent_puck: Tensor, wall: Tensor, tol: int = 1, burn: int = 5, alignment: str = "lag") -> dict[str, Tensor]:
    """Labels on signal time t, where signal t has seen obs up to obs[t]. Flags (N, T), per transition.

    Onset at transition s: agent-puck contact in s but not in s - 1. Contact already there at reset is no onset.
    lag   (primary): positive if an onset happened in the last 1 + tol transitions, so obs[t] can show it.
          This is the fair comparison: the loop residual and predictive likelihood both have seen obs[t].
    lead  (secondary): positive if an onset happens in the upcoming transition, so only geometry can reveal
          it. This is what allocating loops in advance would need.
    Negatives have no contact of any kind, walls included, near t. Sustained contact belongs to neither class.
    sustained: positive if the last transition had agent-puck contact, against the same negatives.
    """
    if burn < 1:
        raise ValueError("burn must be at least 1: signal index 0 has no predictive likelihood")
    onset = torch.zeros_like(agent_puck)
    onset[:, 1:] = agent_puck[:, 1:] & ~agent_puck[:, :-1]
    touching = agent_puck | wall
    if alignment == "lag":
        positive = _any_at_lags(onset, range(1, tol + 2))
        negative = ~_any_at_lags(touching, range(0, tol + 2))
    elif alignment == "lead":
        positive = onset.clone()
        negative = ~_any_at_lags(touching, range(-1, tol + 2))
    else:
        raise ValueError(f"unknown alignment {alignment!r}")
    sustained = _any_at_lags(agent_puck, [1])
    for mask in (positive, negative, sustained):
        mask[:, :burn] = False
    return {"positive": positive, "negative": negative, "sustained": sustained}


def shift_auroc(
    switched: Tensor, control: Tensor, t_switch: int, window: int, burn: int = 5, keep: Tensor | None = None
) -> dict:
    """AUROC for shift onset. Signals (N, T) on signal time.

    Positive: the first `window` signals that can see post-switch dynamics, t_switch + 1 to t_switch + window.
    Negative: the same run before the switch, plus the same time steps of the matched no-switch control.
    keep (N,) bool restricts to some rollouts, for per-mechanism breakdowns.
    """
    if burn < 1:
        raise ValueError("burn must be at least 1")
    T = switched.shape[1]
    stop = min(t_switch + window, T - 1) + 1
    if keep is not None:
        switched, control = switched[keep], control[keep]
    positive = switched[:, t_switch + 1 : stop]
    negative = torch.cat([switched[:, burn : t_switch + 1].reshape(-1), control[:, t_switch + 1 : stop].reshape(-1)])
    return auroc_groups(positive, negative)


# ---------------------------------------------------------------- pre-registered outcome rules


def h7_outcome(verdict_by_arm: dict[str, str], plain_blowup_frac: float, full_beats_persistence: bool) -> str:
    """Rule for H7. Verdicts come from paired_comparison(full, ablated, threshold=TWO_X_THRESHOLD).

    HYPOTHESES.md does not define "stable". Operational meaning used for the kill: plain is less than 2x worse
    than the full recipe, and under BLOWUP_LIMIT of its rollouts blow up.
    """
    needed = ("no_reinject", "no_norm", "no_mixed", "plain")
    if any(verdict_by_arm[a] == "insufficient_seeds" for a in needed):
        return "insufficient_seeds"
    if not full_beats_persistence:
        return "uninformative_full_recipe_not_better_than_persistence"
    degrades = {a: verdict_by_arm[a] == "a_wins" for a in needed}
    if not degrades["plain"] and plain_blowup_frac < BLOWUP_LIMIT:
        return "killed_plain_core_stable"
    if degrades["no_reinject"] and degrades["no_norm"] and degrades["no_mixed"]:
        return "supported"
    return "not_supported_some_ablations_under_2x"


def h8_outcome(verdict: str) -> str:
    return {
        "a_wins": "supported",
        "b_wins": "reversed_prelude_only_better",
        "no_detectable_difference": "killed_no_detectable_difference",
    }.get(verdict, verdict)


def h9_target_passes(mean_auroc: float, verdict: str) -> bool:
    """Passing needs both: AUROC at least 0.7, and a detectable gain over the baseline."""
    return bool(math.isfinite(mean_auroc) and mean_auroc >= AUROC_FLOOR and verdict == "a_wins")
