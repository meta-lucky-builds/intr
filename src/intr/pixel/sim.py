"""The existing simulators on any device. Physics is imported, not rewritten.

BatchEnv.step, obs, cost and is_success create no tensors of their own, so they run on a GPU once the hidden parameters are there.
Only the samplers are CPU bound, through CPU generators. Those stay on the CPU here on purpose: the same seed gives the same draws
on any device, so pairing does not depend on the machine. Draws move to the device afterwards.

Burn-in. An embedding needs `stack` frames, so every trajectory starts with stack - 1 zero-action steps in the true simulator, before
index 0. The closed loop does the same, for the oracle too. All arms of a tier therefore start from the same window, whichever policy
follows. Data index t of an embedding means frames t .. t + stack - 1 of the stored states.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..envs import OBS_DT, BatchEnv
from .config import decision_steps


def env_on(env: BatchEnv, device, repeat: int = 1) -> BatchEnv:
    """Shallow copy with hidden parameters and bounds on device. Sample with the original, step with this.

    repeat holds each action for that many simulator steps. Burn-in, data, the oracle's model and the closed loop all call this step, so
    they all see it, and one stored step is then one decision. dt and physics are untouched. The success window is the one number in an
    environment that counts observations, so it is converted: 10 observations 0.1 s apart and 4 observations 0.3 s apart both span 0.9 s.
    """
    moved = copy.copy(env)
    moved.params = {k: v.to(device) for k, v in env.params.items()}
    moved.obs_low = env.obs_low.to(device)
    moved.obs_high = env.obs_high.to(device)
    if repeat != 1:
        one_step = moved.step

        def held_step(state: Tensor, action: Tensor) -> Tensor:
            for _ in range(repeat):
                state = one_step(state, action)
            return state

        moved.step = held_step
        if hasattr(moved, "success_window"):
            moved.success_window = decision_steps(moved.success_window, repeat)
    return moved


def policy_action(env: BatchEnv, policy: str, obs: Tensor, noise: Tensor | None, gen: torch.Generator, rho: float = 0.9):
    """envs._policy_action with draws on the CPU generator, then moved. Numerically the same on CPU; a test checks it."""
    E, B, _ = obs.shape
    device = obs.device
    if policy == "iid":
        return (2.0 * torch.rand(E, B, env.act_dim, generator=gen) - 1.0).to(device), None
    eps = torch.randn(E, B, env.act_dim, generator=gen).to(device)
    noise = eps if noise is None else rho * noise + math.sqrt(1.0 - rho * rho) * eps
    if policy == "smooth":
        return (0.8 * noise).clamp(-1.0, 1.0), noise
    if policy == "push":
        return (0.8 * env.push_direction(obs) + 0.4 * noise).clamp(-1.0, 1.0), noise
    if policy == "level":
        return (env.level_action(obs) + 0.2 * noise).clamp(-1.0, 1.0), noise
    raise ValueError(f"unknown policy {policy!r}")


@dataclass
class StateTrajectories:
    """Stored states include the burn-in. T transitions at data index 0 .. T - 1."""

    states: Tensor  # (E, B, burn + T + 1, S), float64
    obs: Tensor  # (E, B, T + 1, O), float32, from data index 0
    act: Tensor  # (E, B, T, A), float32
    burn: int

    def take(self, n: int) -> "StateTrajectories":
        return StateTrajectories(self.states[:, :n], self.obs[:, :n], self.act[:, :n], self.burn)


def burn_in(env_dev: BatchEnv, state: Tensor, burn: int) -> list[Tensor]:
    """States s_0 .. s_burn under zero actions. state (E, B, S)."""
    out = [state]
    zero = torch.zeros(*state.shape[:-1], env_dev.act_dim, dtype=OBS_DT, device=state.device)
    for _ in range(burn):
        state = env_dev.step(state, zero)
        out.append(state)
    return out


@torch.no_grad()
def rollout_policy(env_dev: BatchEnv, s0: Tensor, n_steps: int, policy: str, gen: torch.Generator, burn: int) -> StateTrajectories:
    """Burn-in from s0 (E, B, S) on the device, then the policy."""
    states = burn_in(env_dev, s0, burn)
    state = states[-1]
    o = env_dev.obs(state)
    obs, act = [o], []
    noise = None
    for _ in range(n_steps):
        a, noise = policy_action(env_dev, policy, o, noise, gen)
        state = env_dev.step(state, a)
        o = env_dev.obs(state)
        states.append(state)
        obs.append(o)
        act.append(a)
    return StateTrajectories(torch.stack(states, dim=2), torch.stack(obs, dim=2), torch.stack(act, dim=2), burn)


@torch.no_grad()
def collect_states(env: BatchEnv, env_dev: BatchEnv, n_traj: int, n_steps: int, policy: str, gen: torch.Generator, burn: int, device) -> StateTrajectories:
    """envs.collect_trajectories, keeping states. With burn = 0 on CPU it gives the same numbers; a test checks it."""
    return rollout_policy(env_dev, env.reset(n_traj, gen).to(device), n_steps, policy, gen, burn)


def collect_mixed_states(env: BatchEnv, env_dev: BatchEnv, n_traj: int, n_steps: int, policies, gen: torch.Generator, burn: int, device) -> StateTrajectories:
    """envs.collect_mixed: split n_traj as evenly as possible, in order."""
    parts = []
    for i, policy in enumerate(policies):
        count = n_traj // len(policies) + (1 if i < n_traj % len(policies) else 0)
        if count > 0:
            parts.append(collect_states(env, env_dev, count, n_steps, policy, gen, burn, device))
    return StateTrajectories(
        torch.cat([p.states for p in parts], dim=1),
        torch.cat([p.obs for p in parts], dim=1),
        torch.cat([p.act for p in parts], dim=1),
        burn,
    )


@torch.no_grad()
def collect_iid_states_with_burn(env: BatchEnv, env_dev: BatchEnv, n: int, gen: torch.Generator, burn: int, device) -> StateTrajectories:
    """The iid-state ceiling for H5p: n one-step trajectories from independent diverse states. Not available on a real robot."""
    return rollout_policy(env_dev, env.reset_diverse(n, gen).to(device), 1, "iid", gen, burn)


def state_sha(traj: StateTrajectories) -> str:
    import hashlib

    digest = hashlib.sha256()
    for x in (traj.obs, traj.act):
        digest.update(x.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
