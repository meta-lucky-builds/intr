"""Decoded open-loop error and device-aware CEM.

eval.open_loop_errors compares rollouts with truth in the space they roll in. Here rollouts live in embedding space and truth in physical
state, so the same windows, clipping and scaling are applied after decoding through the probe. With an identity probe this equals
eval.open_loop_errors, and a test checks that.

eval.cem_plan creates CPU tensors. plan below is its logic with those moved to the device, and with step and cost passed in, so the oracle
and every model share one planner. Noise still comes from a CPU generator, so it is the same across methods and machines. On CPU it
equals eval.cem_plan, and a test checks that.

Cost. Models imagine in embedding space and pay env.cost on the probe's decoding, the same function and goal the oracle pays on true
observations. So a planner's target needs no goal image, puck needs no made-up agent pose in a goal, and success stays the simulator's.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from ..envs import OBS_DT, BatchEnv, Task
from ..eval import START_STEPS, CEMConfig, OraclePredictor
from ..models import clip_obs, rollout
from .render import render, stack_windows
from .sim import burn_in

VERDICT_HORIZONS = (0, 1, 10, 30)  # 0 is the probe's own floor: decoding a true embedding


@dataclass
class EvalContext:
    """On the device."""

    probe: torch.nn.Module
    obs_scale: Tensor
    obs_low: Tensor
    obs_high: Tensor
    emb_low: Tensor
    emb_high: Tensor


@dataclass
class Windows:
    """Fixed open-loop test windows on the device. s0 (E, B S, D), act (E, B S, H, A), true (E, B S, H + 1, O)."""

    s0: Tensor
    act: Tensor
    true: Tensor


def make_windows(emb: Tensor, obs: Tensor, act: Tensor, device, horizon: int = 30, starts=START_STEPS) -> Windows:
    """emb (E, B, T + 1, D), obs (E, B, T + 1, O), act (E, B, T, A)."""
    E, B, T, A = act.shape
    if max(starts) + horizon > T:
        raise ValueError(f"holdout of {T} steps is too short for start {max(starts)} and horizon {horizon}")
    n = len(starts)
    s0 = torch.stack([emb[:, :, s] for s in starts], dim=2).reshape(E, B * n, emb.shape[-1])
    windowed_act = torch.stack([act[:, :, s : s + horizon] for s in starts], dim=2).reshape(E, B * n, horizon, A)
    true = torch.stack([obs[:, :, s : s + horizon + 1] for s in starts], dim=2).reshape(E, B * n, horizon + 1, obs.shape[-1])
    return Windows(s0.to(device), windowed_act.to(device), true.to(device))


@torch.no_grad()
def decoded_errors(predictor, windows: Windows, ctx: EvalContext, horizons=VERDICT_HORIZONS) -> dict[int, Tensor]:
    """{h: (E,)} on the CPU. predictor.step takes only the current embedding and an action: imagination cannot read a later image."""
    pred = rollout(predictor.step, windows.s0, windows.act, ctx.emb_low, ctx.emb_high)  # (E, B S, H + 1, D)
    out = {}
    for h in horizons:
        decoded = clip_obs(ctx.probe(pred[:, :, h]), ctx.obs_low, ctx.obs_high)
        scaled = (decoded - windows.true[:, :, h]) / ctx.obs_scale
        out[h] = (scaled**2).mean(dim=(1, 2)).sqrt().cpu()
    return out


def errors_record(errors: dict[int, Tensor], mechanisms: list[str]) -> dict:
    record = {f"h{h}": e.mean().item() for h, e in errors.items()}
    record["per_env"] = {f"h{h}": e.tolist() for h, e in errors.items()}
    names = sorted(set(mechanisms))
    if len(names) > 1:
        record["by_mechanism"] = {
            name: {f"h{h}": e[torch.tensor([m == name for m in mechanisms])].mean().item() for h, e in errors.items()} for name in names
        }
    return record


@torch.no_grad()
def plan(x0: Tensor, step_fn, cost_fn, mean: Tensor, cfg: CEMConfig, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """x0 (E, X), mean (E, H, A). step_fn(x (E, N, X), a (E, N, A)) -> x; cost_fn(x, a) -> (E, N). Returns (E, A) and the shifted mean."""
    E, H, A = mean.shape
    N = cfg.n_candidates
    device = mean.device
    std = torch.full_like(mean, cfg.init_std)
    best_cost = torch.full((E,), float("inf"), device=device)
    best_action = mean[:, 0].clone()
    keep = math.sqrt(1.0 - cfg.noise_beta**2)
    for _ in range(cfg.n_iters):
        noise = torch.randn(E, N, H, A, generator=gen)
        for t in range(1, H):
            noise[:, :, t] = cfg.noise_beta * noise[:, :, t - 1] + keep * noise[:, :, t]
        candidates = (mean[:, None] + std[:, None] * noise.to(device)).clamp(-1.0, 1.0)  # (E, N, H, A)
        x = x0[:, None, :].expand(E, N, -1)
        cost = torch.zeros(E, N, device=device)
        for t in range(H):
            a = candidates[:, :, t]
            x = step_fn(x, a)
            cost = cost + cost_fn(x, a)
        top_cost, idx = cost.topk(cfg.n_elites, dim=1, largest=False)
        elites = candidates.gather(1, idx[:, :, None, None].expand(-1, -1, H, A))
        mean = elites.mean(dim=1)
        std = elites.std(dim=1, unbiased=False).clamp_min(cfg.min_std)
        better = top_cost[:, 0] < best_cost
        best_cost = torch.where(better, top_cost[:, 0], best_cost)
        best_action = torch.where(better[:, None], elites[:, 0, 0], best_action)
    return best_action, torch.cat([mean[:, 1:], torch.zeros(E, 1, A, device=device)], dim=1)


@torch.no_grad()
def run_episode(env_dev: BatchEnv, start: Tensor, goal: Tensor | None, observe, step_fn, cost_fn, cfg: CEMConfig, gen: torch.Generator, steps: int) -> dict:
    """Plan, then act in the true simulator. observe(state) gives the planner's input; it sees only states reached so far."""
    E = env_dev.n_envs
    state = start
    o = env_dev.obs(state)[:, 0]
    mean = torch.zeros(E, cfg.horizon, env_dev.act_dim, dtype=OBS_DT, device=start.device)
    observed = [o]
    total_cost = torch.zeros(E, device=start.device)
    actions = []
    started = time.perf_counter()
    for _ in range(steps):
        action, mean = plan(observe(state), step_fn, cost_fn, mean, cfg, gen)
        state = env_dev.step(state, action[:, None, :])
        o = env_dev.obs(state)[:, 0]
        observed.append(o)
        actions.append(action)
        total_cost += env_dev.cost(o[:, None], action[:, None], goal)[:, 0].to(total_cost.dtype)
    traj = torch.stack(observed, dim=1)
    success = env_dev.is_success(traj, goal).cpu()
    return {
        "success": success,
        "mean_cost": (total_cost / max(steps, 1)).cpu(),
        "actions": torch.stack(actions, dim=1).cpu(),  # (E, steps, A)
        "wall_s": time.perf_counter() - started,
    }


class Camera:
    """Renders each new true state and encodes the last `stack` frames. It holds no future: it is only given states already reached."""

    def __init__(self, family: str, size: int, stack: int, encoder, prefill: list[Tensor], palette: Tensor | None = None, render_fn=render):
        self.family, self.size, self.encoder, self.palette = family, size, encoder, palette
        self.render_fn = render_fn
        self.frames: deque = deque(maxlen=stack)
        self.stack = stack
        for state in prefill:
            self.frames.append(self._draw(state))

    def _draw(self, state: Tensor) -> Tensor:
        return self.render_fn(self.family, state[:, 0], self.size, self.palette)  # (E, 3, h, w)

    @torch.no_grad()
    def __call__(self, state: Tensor) -> Tensor:
        self.frames.append(self._draw(state))
        if len(self.frames) != self.stack:
            raise RuntimeError("camera was not prefilled with the burn-in frames")
        return self.encoder(stack_windows(torch.stack(list(self.frames), dim=1))).float()


def _task_on(task: Task, device) -> tuple[Tensor, Tensor | None]:
    return task.state.to(device), None if task.goal is None else task.goal.to(device)


def record_of(out: dict, oracle_mask: Tensor | None = None) -> dict:
    success = out["success"].bool()
    record = {
        "success_rate": success.float().mean().item(),
        "failure_rate": 1.0 - success.float().mean().item(),
        "mean_cost": out["mean_cost"].mean().item(),
        "per_env_success": success.tolist(),
        "per_env_cost": out["mean_cost"].tolist(),
        "wall_s": out["wall_s"],
    }
    if oracle_mask is not None:
        solved = int(oracle_mask.sum().item())
        record["failure_rate_on_oracle_solved"] = 1.0 - success[oracle_mask].float().mean().item() if solved > 0 else float("nan")
    return record


def closed_loop_oracle(env_dev: BatchEnv, task: Task, cfg: CEMConfig, gen: torch.Generator, steps: int, burn: int) -> dict:
    """The true simulator on the same planner, from the state after the same burn-in models get."""
    state, goal = _task_on(task, env_dev.params[next(iter(env_dev.params))].device)
    start = burn_in(env_dev, state, burn)[-1]
    oracle = OraclePredictor(env_dev)
    return run_episode(
        env_dev,
        start,
        goal,
        lambda s: env_dev.obs(s)[:, 0],
        lambda o, a: clip_obs(oracle.step(o, a), env_dev.obs_low, env_dev.obs_high),
        lambda o, a: env_dev.cost(o, a, goal),
        cfg,
        gen,
        steps,
    )


def closed_loop_model(env_dev: BatchEnv, predictor, ctx: EvalContext, family: str, size: int, stack: int, encoder, task: Task, cfg: CEMConfig, gen: torch.Generator, steps: int, palette: Tensor | None = None, render_fn=render) -> dict:
    """CEM in embedding space, executed in the true simulator, rendering one image per real step."""
    state, goal = _task_on(task, ctx.obs_low.device)
    states = burn_in(env_dev, state, stack - 1)
    camera = Camera(family, size, stack, encoder, states[:-1], palette, render_fn)
    return run_episode(
        env_dev,
        states[-1],
        goal,
        camera,
        lambda s, a: clip_obs(predictor.step(s, a), ctx.emb_low, ctx.emb_high),
        lambda s, a: env_dev.cost(clip_obs(ctx.probe(s), ctx.obs_low, ctx.obs_high), a, goal),
        cfg,
        gen,
        steps,
    )
