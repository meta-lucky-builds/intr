"""A looped dynamics core that sees its own prediction error (H14p in docs/HYPOTHESES.md). Moved from src/intr, unreviewed until tests/test_pixel_error_fed.py.

Round one's carried state was inert (Problem 6). The state is fully observed, the loop never saw its own error, and
training detached z at every step, so nothing could reward remembering. Hidden physics is what history reveals. This core
gets the error, and gradients cross real steps, so the carried state can act as an online filter: an amortized form of
T2-Z's latent optimization, with no test-time gradient.

One step, with o (..., O), a (..., A), z (..., Z), and an innovation block i (..., O + A + 1):

    e_s, e_a, e_i = enc_state(norm(o)), enc_action(a), enc_innovation(i)
    z^0      = rescale(z_prev + prelude(z_prev, e_s, e_a, e_i))
    z^(k+1)  = rescale(z^k + update(z^k, e_s, e_a, e_i))        same weights for every k
    next_o   = o + decoded delta of z^K                         the decoder reads z only, as in looped.py

The innovation block says what the last real transition taught:

    i = (asinh(gain * err), a_prev, 1)     err = target(o_prev, o) - pred_prev, in delta-std units, pred_prev detached
    i = 0                                  no observation: the first step of a filter, and every imagined step after the first

Filtering and imagination, the design point where bugs hide:

    Real step t has seen obs[t] and is predicting obs[t + 1]. err is known only once obs[t + 1] arrives, so it enters at
    step t + 1. So after T transitions the T-th error has not entered any step yet: it is held as the pending innovation in
    Belief. The first imagined step consumes it, which is why all T transitions count, as they do for latent_opt. Every later
    imagined step gets the explicit all-zero block, whose flag says "no observation". Imagination never takes a true state
    other than its start, and it cannot compute an error. The flag lets the core tell "error is zero" from "there is no
    error", and the block is the same in training and at test time.

    Open-loop tests start from a state in another trajectory than the adaptation data, as every comparator's do. So in
    training the imagined phase hops to another trajectory of the same environment with probability hop_prob. A state that
    only remembered its own pose would fail this. The closed loop continues a trajectory, so continuation is trained too.

Why these inputs:
    - a_prev is included because an error only means something next to the action that caused it, and z cannot be assumed to
      remember it. obs[t - 1] is deliberately not included. With it, the raw delta could be computed from inputs alone, and the
      shuffled control would still have a clean channel to its own physics.
    - asinh(gain * err): the error shrinks as the model improves (RMS about 0.02 to 0.15 on pendulum) and spikes on puck contact.
      This is linear near zero and logarithmic beyond, needing no running statistics.
    - use_error = False zeroes only err. The flag and a_prev remain, parameters are identical, and init is identical. The
      control therefore still sees the history of (o, a) through its state: a tie means the error adds nothing to history,
      not that history is useless.
    - Shuffled errors replace only err, with a donor environment's error at the same time index.

Training differs from train_looped on purpose. z is not detached across real steps, since a filter cannot be learned without
gradient through time. Within a step, gradient depth is bounded as in looped.py: no-grad calls, then grad_loops calls, through
the same identity shortcut. Predicted observations are detached when fed back and the error input is always detached, so the only
long gradient path is through z. update and prelude have their last layers shrunk at init (init_scale): at default init they
overwrite z within a few calls, and backpropagation through dozens of steps would then see nothing. The first-round core keeps its
default init.

No hidden parameter appears anywhere in this file. Functions see observations and actions. Hidden values are only passed to
heldout_r2, as targets of a diagnostic, by the experiment script.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..envs import derive_seed, make_generator
from ..looped import SequenceData, _mlp, draw_pre_calls, linear_macs, to_signal_time
from ..models import Counts, clip_obs
from ..train import _cosine_lr, _set_lr



@dataclass
class ErrorFedConfig:
    obs_dim: int
    act_dim: int
    z_dim: int = 48
    emb_dim: int = 24
    hidden: int = 96
    loops: int = 4  # round one selected 4 on both families
    grad_loops: int = 2
    use_error: bool = True  # False is the matched control: same inputs with err zeroed
    error_gain: float = 20.0
    rms_target: float = 0.85
    init_scale: float = 0.1

    def __post_init__(self):
        if not 1 <= self.grad_loops <= self.loops:
            raise ValueError(f"need 1 <= grad_loops <= loops, got {self.grad_loops} and {self.loops}")

    @property
    def innovation_dim(self) -> int:
        return self.obs_dim + self.act_dim + 1

    def pre_call_range(self) -> tuple[int, int]:
        """Mixed depth, as in the full recipe: total depth in [g, 2 K - g], mean K."""
        return 0, 2 * (self.loops - self.grad_loops)

    def trained_depth_range(self) -> tuple[int, int]:
        low, high = self.pre_call_range()
        return low + self.grad_loops, high + self.grad_loops


class ErrorFedDynamics(nn.Module):
    def __init__(self, cfg: ErrorFedConfig):
        super().__init__()
        self.cfg = cfg
        O, A, Z, E, H = cfg.obs_dim, cfg.act_dim, cfg.z_dim, cfg.emb_dim, cfg.hidden
        self.enc_state = _mlp(O, H, E)
        self.enc_action = _mlp(A, E, E)
        self.enc_innovation = _mlp(cfg.innovation_dim, H, E)
        self.prelude_net = _mlp(Z + 3 * E, H, Z)
        self.update = _mlp(Z + 3 * E, H, Z)
        self.decoder = _mlp(Z, H, O)
        with torch.no_grad():
            for net in (self.prelude_net, self.update):
                net[-1].weight.mul_(cfg.init_scale)
                net[-1].bias.mul_(cfg.init_scale)
        self.register_buffer("obs_mean", torch.zeros(O))
        self.register_buffer("obs_std", torch.ones(O))
        self.register_buffer("delta_mean", torch.zeros(O))
        self.register_buffer("delta_std", torch.ones(O))

    def set_normalization(self, obs: Tensor, next_obs: Tensor) -> None:
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
        """Fix the RMS of each state vector, as in the full recipe."""
        rms = (u.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        return self.cfg.rms_target * u / rms

    # ---------------------------------------- the innovation block

    def no_observation(self, o: Tensor) -> Tensor:
        """The explicit block for a step with nothing new to report. (..., O + A + 1), all zero, flag included."""
        return o.new_zeros(*o.shape[:-1], self.cfg.innovation_dim)

    def prediction_error(self, o_prev: Tensor, o_now: Tensor, pred: Tensor) -> Tensor:
        """Observed minus predicted normalized delta, detached, so training sees what test time sees."""
        return (self.target(o_prev, o_now) - pred).detach()

    def innovation(self, err: Tensor, a_prev: Tensor) -> Tensor:
        """err (..., O), a_prev (..., A). A non-finite error from a diverged prediction becomes a large finite one."""
        clean = torch.nan_to_num(err, nan=1e3, posinf=1e3, neginf=-1e3)
        features = torch.asinh(self.cfg.error_gain * clean)
        flag = err.new_ones(*err.shape[:-1], 1)
        return torch.cat([features, a_prev, flag], dim=-1)

    def visible_innovation(self, innovation: Tensor) -> Tensor:
        """What the encoder sees. The control has err zeroed and everything else unchanged."""
        if self.cfg.use_error:
            return innovation
        O = self.cfg.obs_dim
        return torch.cat([torch.zeros_like(innovation[..., :O]), innovation[..., O:]], dim=-1)

    # ---------------------------------------- one step

    def begin_step(self, o: Tensor, a: Tensor, z_prev: Tensor | None = None, innovation: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Returns z^0 (..., Z) and the injection (..., 3 E)."""
        if z_prev is None:
            z_prev = o.new_zeros(*o.shape[:-1], self.cfg.z_dim)
        if innovation is None:
            innovation = self.no_observation(o)
        e_s = self.enc_state(self.norm_obs(o))
        e_a = self.enc_action(a)
        e_i = self.enc_innovation(self.visible_innovation(innovation))
        inj = torch.cat([e_s, e_a, e_i], dim=-1)
        return self.rescale(z_prev + self.prelude_net(torch.cat([z_prev, inj], dim=-1))), inj

    def cell(self, z: Tensor, inj: Tensor) -> Tensor:
        return self.rescale(z + self.update(torch.cat([z, inj], dim=-1)))

    def solve(self, z: Tensor, inj: Tensor, n: int) -> Tensor:
        for _ in range(n):
            z = self.cell(z, inj)
        return z

    def solve_traced(self, z: Tensor, inj: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """Also returns ||z_{k+1} - z_k|| / ||z_k|| for each call, (..., n), the first round's residual."""
        residuals = []
        for _ in range(n):
            next_z = self.cell(z, inj)
            residuals.append((next_z - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-12))
            z = next_z
        return z, torch.stack(residuals, dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def predict(self, o: Tensor, a: Tensor, z_prev: Tensor | None = None, innovation: Tensor | None = None, loops: int | None = None) -> tuple[Tensor, Tensor]:
        """Returns the normalized delta and the final state."""
        z0, inj = self.begin_step(o, a, z_prev, innovation)
        z = self.solve(z0, inj, self.cfg.loops if loops is None else loops)
        return self.decode(z), z

    def step(self, o: Tensor, a: Tensor, z_prev: Tensor | None = None, innovation: Tensor | None = None, loops: int | None = None) -> tuple[Tensor, Tensor]:
        """Returns next_o and the final state. looped.sensitivity calls this exactly as it calls LoopedDynamics.step."""
        pred, z = self.predict(o, a, z_prev, innovation, loops)
        return self.apply_delta(o, pred), z

    def training_step(self, o: Tensor, a: Tensor, z_prev: Tensor | None, innovation: Tensor | None, pre_calls: int) -> tuple[Tensor, Tensor]:
        """Differentiable in z_prev. See looped.py for the shortcut around the no-grad calls."""
        z0, inj = self.begin_step(o, a, z_prev, innovation)
        z = z0
        if pre_calls > 0:
            with torch.no_grad():
                z_pre = self.solve(z0, inj, pre_calls)
            z = z_pre + z0 - z0.detach()
        z = self.solve(z, inj, self.cfg.grad_loops)
        return self.decode(z), z

    # ---------------------------------------- cost

    def cell_macs(self) -> int:
        return linear_macs(self.update)

    def base_macs(self) -> int:
        return (
            linear_macs(self.enc_state)
            + linear_macs(self.enc_action)
            + linear_macs(self.enc_innovation)
            + linear_macs(self.prelude_net)
            + linear_macs(self.decoder)
        )

    def macs_per_step(self, loops: int | None = None) -> int:
        return self.base_macs() + (self.cfg.loops if loops is None else loops) * self.cell_macs()


# ---------------------------------------------------------------- filtering


@dataclass
class Belief:
    """What a filter hands to imagination. z (N, Z), None if nothing was observed; pending (N, O + A + 1)."""

    z: Tensor | None
    pending: Tensor


def match_batch(x: Tensor, o: Tensor) -> Tensor:
    """(E, D) to the leading shape of o, for (E, O), (E, W, O) or (E, N, O)."""
    while x.dim() < o.dim():
        x = x.unsqueeze(1)
    return x.expand(*o.shape[:-1], x.shape[-1])


def other_env_donor(n: int) -> Tensor:
    """donor[i] is another row than i. For a batch with one row per environment, another environment."""
    if n < 2:
        raise ValueError("a donor needs at least two environments")
    return torch.roll(torch.arange(n), 1)


def make_donor(n: int, gen: torch.Generator) -> Tensor:
    """A random cyclic shift, so it has no fixed point. Training batches hold distinct environments."""
    if n < 2:
        raise ValueError("a donor needs at least two environments")
    return torch.roll(torch.arange(n), int(torch.randint(1, n, (1,), generator=gen)))


@torch.no_grad()
def run_filter(
    model: ErrorFedDynamics, obs: Tensor, act: Tensor, donor: Tensor | None = None, belief: Belief | None = None, loops: int | None = None
) -> tuple[Belief, dict[str, Tensor]]:
    """Consume real transitions. obs (N, T + 1, O), act (N, T, A). Returns the belief after T transitions, and a trace.

    With donor, each row is fed donor row's error at the same time step, errors coming from the donor row's own predictions.

    Trace, all (N, T, ...):
        pred_next      pred_next[:, t] predicts obs[:, t + 1] and has seen observations up to obs[:, t] only
        own_error      the row's own error about obs[:, t + 1]
        fed_error      what the core received, next step, in its place
        residual_last, residual_mean   the first round's fixed-point residual at step t
        state_change   ||z_t - z_{t-1}|| / ||z_{t-1}||, NaN at the first step
        error_signal   ||own_error|| aligned as in looped.to_signal_time: index t knows observations up to obs[:, t]
    """
    n = model.cfg.loops if loops is None else loops
    N, T, _ = act.shape
    z = None if belief is None else belief.z
    pending = model.no_observation(obs[:, 0]) if belief is None else belief.pending
    if T == 0:
        return Belief(z, pending), {}
    preds, own, fed, last, mean, change = [], [], [], [], [], []
    for t in range(T):
        z0, inj = model.begin_step(obs[:, t], act[:, t], z, pending)
        next_z, residuals = model.solve_traced(z0, inj, n)
        pred = model.decode(next_z)
        err = model.prediction_error(obs[:, t], obs[:, t + 1], pred)
        given = err if donor is None else err[donor]
        pending = model.innovation(given, act[:, t])
        if z is None:
            change.append(torch.full((N,), float("nan"), dtype=obs.dtype))
        else:
            change.append((next_z - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-12))
        z = next_z
        preds.append(model.apply_delta(obs[:, t], pred))
        own.append(err)
        fed.append(given)
        last.append(residuals[..., -1])
        mean.append(residuals.mean(dim=-1))
    own_error = torch.stack(own, dim=1)
    trace = {
        "pred_next": torch.stack(preds, dim=1),
        "own_error": own_error,
        "fed_error": torch.stack(fed, dim=1),
        "residual_last": torch.stack(last, dim=1),
        "residual_mean": torch.stack(mean, dim=1),
        "state_change": torch.stack(change, dim=1),
        "error_signal": to_signal_time(own_error.norm(dim=-1)),
    }
    return Belief(z, pending), trace


class ErrorFedPredictor:
    """Imagination from a belief, for looped.rollout_metrics (advance) and for eval.cem_plan (step).

    Both start from the belief, consume its pending innovation on the first step, and then use the no-observation block.
    step is stateful because cem_plan has no reset hook: state resets to the belief every `horizon` calls.
    observe writes a real transition into the belief. With online False it does nothing, so the belief stays where
    adaptation left it, as a latent would.
    """

    filters_online = True

    def __init__(self, model: ErrorFedDynamics, belief: Belief | None = None, loops: int | None = None, horizon: int | None = None, online: bool = True):
        self.model = model
        self.belief = belief
        self.loops = model.cfg.loops if loops is None else loops
        self.horizon = horizon
        self.online = online
        self.counts = Counts()
        self.t_calls = 0
        self._calls = 0
        self._carry: Tensor | None = None

    def _start(self, o: Tensor) -> tuple[Tensor | None, Tensor]:
        if self.belief is None:
            return None, self.model.no_observation(o)
        z = None if self.belief.z is None else match_batch(self.belief.z, o)
        return z, match_batch(self.belief.pending, o)

    def advance(self, o: Tensor, a: Tensor, carry: Tensor | None) -> tuple[Tensor, Tensor]:
        if carry is None:
            z, innovation = self._start(o)
        else:
            z, innovation = carry, self.model.no_observation(o)
        self.counts.forward_calls += 1
        self.counts.forward_samples += int(np.prod(o.shape[:-1]))
        self.t_calls += self.loops
        return self.model.step(o, a, z, innovation, self.loops)

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        if self.horizon is None:
            raise ValueError("step needs a planning horizon")
        if self._calls % self.horizon == 0:
            self._carry = None
        self._calls += 1
        next_o, self._carry = self.advance(o, a, self._carry)
        return next_o

    @torch.no_grad()
    def observe(self, o: Tensor, a: Tensor, next_o: Tensor) -> None:
        """One real transition per environment, (E, O), (E, A), (E, O)."""
        if not self.online:
            return
        z, innovation = self._start(o)
        pred, z = self.model.predict(o, a, z, innovation, self.loops)
        err = self.model.prediction_error(o, next_o, pred)
        self.belief = Belief(z, self.model.innovation(err, a))
        self.counts.forward_calls += 1
        self.counts.forward_samples += o.shape[0]
        self.t_calls += self.loops


# ---------------------------------------------------------------- training on sequences


@dataclass
class ErrorFedTrainConfig:
    steps: int = 3000
    batch_seqs: int = 32  # one sequence per distinct environment; must not exceed the number of training environments
    filter_max: int = 50  # the filtering phase is uniform on 0 to this. The test budget of 40 sits inside the range
    horizon_max: int = 30  # the imagined phase is uniform on 1 to this
    hop_prob: float = 0.5
    lr: float = 1e-3
    lr_floor: float = 0.05
    grad_clip: float = 5.0
    imagination_huber: float = 1.0
    log_every: int = 500

    def expected_steps_per_sequence(self) -> float:
        return self.filter_max / 2.0 + (self.horizon_max + 1) / 2.0

    def expected_transitions(self) -> float:
        return self.steps * self.batch_seqs * self.expected_steps_per_sequence()


@dataclass
class SequenceBatch:
    env_idx: Tensor  # (b,), distinct
    filter_obs: Tensor  # (b, L + 1, O)
    filter_act: Tensor  # (b, L, A)
    imag_obs: Tensor  # (b, H + 1, O), true states: only index 0 may ever be an input
    imag_act: Tensor  # (b, H, A)
    hopped: Tensor  # (b,) bool, the imagined phase left the filtering trajectory


def sample_sequences(data: SequenceData, batch: int, filter_len: int, horizon: int, hop_prob: float, gen: torch.Generator) -> SequenceBatch:
    """Filter on one trajectory, then imagine on the same environment.

    It draws the same number of random values whatever the phases decide, up to L and H themselves.
    """
    E, B, T, _ = data.act.shape
    if batch > E:
        raise ValueError(f"{batch} sequences need {batch} distinct environments, there are {E}")
    if filter_len > T or horizon > T:
        raise ValueError(f"phases {filter_len} and {horizon} do not fit in trajectories of {T} steps")
    env = torch.randperm(E, generator=gen)[:batch]
    traj_f = torch.randint(B, (batch,), generator=gen)
    start_f = torch.randint(T - filter_len + 1, (batch,), generator=gen)
    wants_hop = torch.rand(batch, generator=gen) < hop_prob
    traj_shift = torch.randint(1, B, (batch,), generator=gen) if B > 1 else torch.zeros(batch, dtype=torch.long)
    start_hop = torch.randint(T - horizon + 1, (batch,), generator=gen)
    fits = start_f + filter_len + horizon <= T
    hopped = wants_hop | ~fits
    traj_i = torch.where(hopped, (traj_f + traj_shift) % B, traj_f)
    start_i = torch.where(hopped, start_hop, start_f + filter_len)
    env_col = env[:, None]
    f_t = start_f[:, None] + torch.arange(filter_len + 1)[None, :]
    i_t = start_i[:, None] + torch.arange(horizon + 1)[None, :]
    return SequenceBatch(
        env,
        data.obs[env_col, traj_f[:, None], f_t],
        data.act[env_col, traj_f[:, None], f_t[:, :filter_len]],
        data.obs[env_col, traj_i[:, None], i_t],
        data.act[env_col, traj_i[:, None], i_t[:, :horizon]],
        hopped,
    )


def _imagined_loss(pred: Tensor, target: Tensor, delta: float) -> Tensor:
    """Twice the Huber loss, so it equals MSE for small errors and its gradient is bounded once a rollout has drifted."""
    return 2.0 * F.huber_loss(pred, target, delta=delta)


def sequence_loss(
    model: ErrorFedDynamics, batch: SequenceBatch, tcfg: ErrorFedTrainConfig, depth_gen: torch.Generator,
    low: Tensor | None = None, high: Tensor | None = None, donor: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """One filtering phase, then one imagined phase, with gradient through z the whole way. Mean loss per step.

    Filtering: teacher-forced MSE, feeding the error from the model's own detached prediction.
    Imagination: the predicted state is fed back, detached and clipped to the bounds, and the target is the delta that would
    land on the true next state. The input after the first step is always the no-observation block, and no true state enters.
    """
    cfg = model.cfg
    L, H = batch.filter_act.shape[1], batch.imag_act.shape[1]
    total = batch.filter_obs.new_zeros(())
    z = None
    pending = model.no_observation(batch.filter_obs[:, 0])
    calls = 0
    for t in range(L):
        o, a, next_o = batch.filter_obs[:, t], batch.filter_act[:, t], batch.filter_obs[:, t + 1]
        pre = draw_pre_calls(cfg, depth_gen)
        calls += pre + cfg.grad_loops
        pred, z = model.training_step(o, a, z, pending, pre)
        total = total + ((pred - model.target(o, next_o)) ** 2).mean()
        err = model.prediction_error(o, next_o, pred)
        pending = model.innovation(err if donor is None else err[donor], a)
    o = batch.imag_obs[:, 0]
    for h in range(H):
        pre = draw_pre_calls(cfg, depth_gen)
        calls += pre + cfg.grad_loops
        pred, z = model.training_step(o, batch.imag_act[:, h], z, pending, pre)
        pending = model.no_observation(o)
        total = total + _imagined_loss(pred, model.target(o, batch.imag_obs[:, h + 1]), tcfg.imagination_huber)
        o = clip_obs(model.apply_delta(o, pred.detach()), low, high)
    return total / (L + H), {"steps": L + H, "cell_calls": calls}


@dataclass
class TrainedRecurrent:
    model: nn.Module
    stats: dict = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)


def _train_loop(model: nn.Module, data: SequenceData, tcfg: ErrorFedTrainConfig, seed, batch_loss, macs_of) -> tuple[dict, list[dict]]:
    """Same batches and phase lengths for every arm of a seed, whatever else it does. A non-finite update is skipped and counted."""
    started = time.perf_counter()
    model.set_normalization(data.obs[:, :, :-1], data.obs[:, :, 1:])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    batch_gen = make_generator("train", seed, "error_fed", "batches")
    phase_gen = make_generator("train", seed, "error_fed", "phases")
    skipped, transitions, macs, backward = 0, 0, 0, 0
    log: list[dict] = []
    running, running_n = 0.0, 0
    first_loss = None
    recent: list[float] = []
    recent_size = max(1, tcfg.steps // 10)
    for update in range(tcfg.steps):
        _set_lr(optimizer, _cosine_lr(tcfg.lr, update, tcfg.steps, tcfg.lr_floor))
        L = int(torch.randint(tcfg.filter_max + 1, (1,), generator=phase_gen))
        H = int(torch.randint(1, tcfg.horizon_max + 1, (1,), generator=phase_gen))
        batch = sample_sequences(data, tcfg.batch_seqs, L, H, tcfg.hop_prob, batch_gen)
        loss, info = batch_loss(batch)
        transitions += info["steps"] * tcfg.batch_seqs
        macs += tcfg.batch_seqs * macs_of(info)
        value = loss.item()
        if not math.isfinite(value):
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        backward += 1
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
            log.append({"update": update + 1, "loss": running / running_n})
            running, running_n = 0.0, 0
    if running_n > 0:
        log.append({"update": tcfg.steps, "loss": running / running_n})
    model.requires_grad_(False)
    stats = {
        "updates": tcfg.steps,
        "skipped_nonfinite_updates": skipped,
        "transitions": transitions,
        "backward_calls": backward,
        "forward_macs_total": macs,
        "first_loss": first_loss,
        "final_loss": float(np.mean(recent)) if recent else None,
        "wall_s": time.perf_counter() - started,
    }
    return stats, log


def train_error_fed(
    data: SequenceData, cfg: ErrorFedConfig, tcfg: ErrorFedTrainConfig, seed, low: Tensor | None = None, high: Tensor | None = None, shuffle_errors: bool = False
) -> TrainedRecurrent:
    """Arms of a seed share initial weights and batches. Takes observations and actions only.

    shuffle_errors: every sequence is fed the error of another environment in the batch, in lockstep. The donor changes per update.
    """
    if shuffle_errors and tcfg.batch_seqs < 2:
        raise ValueError("shuffled errors need at least two sequences per batch")
    torch.manual_seed(derive_seed("init", seed, "error_fed"))
    model = ErrorFedDynamics(cfg)
    depth_gen = make_generator("train", seed, "error_fed", "depth")
    donor_gen = make_generator("train", seed, "error_fed", "donor")
    base, cell = model.base_macs(), model.cell_macs()

    def batch_loss(batch: SequenceBatch):
        donor = make_donor(tcfg.batch_seqs, donor_gen) if shuffle_errors else None
        return sequence_loss(model, batch, tcfg, depth_gen, low, high, donor)

    stats, log = _train_loop(model, data, tcfg, seed, batch_loss, lambda info: info["steps"] * base + info["cell_calls"] * cell)
    return TrainedRecurrent(model, stats, log)

