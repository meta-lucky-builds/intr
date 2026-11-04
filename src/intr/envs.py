"""Batched torch simulators with per-environment hidden parameters.

Shape convention used everywhere in this package:
    E = number of environments (each with its own hidden parameters)
    B = number of parallel trajectories/candidates per environment
    T = time steps; S/O/A = state/observation/action dimension

Simulators are stateless: ``env.step(state, action)`` maps (E, B, S) and (E, B, A) to (E, B, S).
That makes vectorized planning, an oracle model, and iid-state sampling all the same code.
State is float64 for integration accuracy; observations and actions at the API are float32.

Time steps. All simulators use a control step dt = 0.1 s:
    pendulum: RK4 with 4 substeps (h = 0.025 s). Smooth dynamics, so RK4 conserves energy closely.
    puck: semi-implicit Euler with 20 substeps (h = 0.005 s). Penalty contact with k = 400 and the
        lightest puck (m = 0.15, reduced mass 0.13) has omega = sqrt(k / m) ~ 55 rad/s, so
        omega * h ~ 0.28, well inside the symplectic Euler limit of 2.
    drone: RK4 with 4 substeps (h = 0.025 s). The fastest mode is rotational damping, c / I ~ 8 / s at
        the lowest extrapolation inertia, so lambda * h ~ 0.2.

Shift tiers (docs/HYPOTHESES.md):
    train      even "stripes" of each parameter's range.
    val_in     validation: new draws from the train stripes.
    val_shift  validation: train stripes plus a weakened actuator, a mechanism used by no test tier.
               It lets hyperparameters see *some* shift without seeing a test shift.
    interp     odd stripes: strictly inside the training range, never a value training can draw.
    extrap     every hidden parameter outside the training range.
    oof        train stripes plus a mechanism never varied in training. Mechanisms cycle by index, so
               each is equally represented; ``env.mechanisms`` names the one per environment.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

DT = torch.float64
OBS_DT = torch.float32

FAMILIES = ("pendulum", "puck", "drone")
TIERS = ("train", "val_in", "val_shift", "interp", "extrap", "oof")
N_STRIPES = 9  # even stripes (5) for training, odd stripes (4) for interpolation

G = 9.81

# Training ranges and extrapolation bands. Extrapolation stays physically feasible by construction:
# the puck friction force stays below F_MAX, and pendulum torque can still pump energy.
PENDULUM_RANGES = {"mass": (0.7, 1.3), "length": (0.7, 1.3), "damping": (0.0, 0.3)}
PENDULUM_EXTRAP = {
    "mass": [(0.5, 0.6), (1.5, 1.9)],
    "length": [(0.55, 0.62), (1.4, 1.6)],
    "damping": [(0.4, 0.7)],
}
PUCK_RANGES = {"mass": (0.3, 1.0), "friction": (0.05, 0.3)}
PUCK_EXTRAP = {
    "mass": [(0.15, 0.25), (1.2, 1.5)],
    "friction": [(0.01, 0.03), (0.35, 0.45)],
}
# Maximum thrust is 1.8 weights of a unit mass, so even the heaviest extrapolation band can hover and tilt.
DRONE_RANGES = {"mass": (0.7, 1.3), "inertia": (0.10, 0.20), "drag": (0.2, 1.0)}
DRONE_EXTRAP = {
    "mass": [(0.5, 0.6), (1.4, 1.55)],
    "inertia": [(0.06, 0.085), (0.24, 0.32)],
    "drag": [(0.02, 0.1), (1.2, 1.6)],
}


def derive_seed(*parts) -> int:
    """Stable seed from any labels. Unlike hash(), this does not change between Python runs."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def make_generator(*parts) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(derive_seed(*parts))
    return gen


def _rand(gen: torch.Generator, shape, low=0.0, high=1.0) -> Tensor:
    return low + (high - low) * torch.rand(shape, generator=gen, dtype=DT)


def _sample_stripes(gen: torch.Generator, n: int, low: float, high: float, odd: bool) -> Tensor:
    allowed = torch.arange(1 if odd else 0, N_STRIPES, 2, dtype=DT)
    idx = allowed[torch.randint(len(allowed), (n,), generator=gen)]
    return low + (high - low) * (idx + _rand(gen, (n,))) / N_STRIPES


def _sample_bands(gen: torch.Generator, n: int, bands: list[tuple[float, float]]) -> Tensor:
    table = torch.tensor(bands, dtype=DT)  # (n_bands, 2)
    idx = torch.randint(len(bands), (n,), generator=gen)
    return table[idx, 0] + (table[idx, 1] - table[idx, 0]) * _rand(gen, (n,))


def _sample_hidden(gen, n, tier, ranges, extrap) -> dict[str, Tensor]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}")
    out = {}
    for name, (low, high) in ranges.items():
        if tier == "extrap":
            out[name] = _sample_bands(gen, n, extrap[name])
        else:
            out[name] = _sample_stripes(gen, n, low, high, odd=tier == "interp")
    return out


@dataclass
class Trajectories:
    obs: Tensor  # (E, B, T + 1, O)
    act: Tensor  # (E, B, T, A)

    def take(self, n_traj: int) -> "Trajectories":
        return Trajectories(self.obs[:, :n_traj], self.act[:, :n_traj])

    def flat(self) -> "Transitions":
        E, B, T, A = self.act.shape
        O = self.obs.shape[-1]
        return Transitions(
            obs=self.obs[:, :, :-1].reshape(E, B * T, O),
            act=self.act.reshape(E, B * T, A),
            next_obs=self.obs[:, :, 1:].reshape(E, B * T, O),
        )


@dataclass
class Transitions:
    """Time order within each trajectory is preserved by ``Trajectories.flat``."""

    obs: Tensor  # (E, N, O)
    act: Tensor  # (E, N, A)
    next_obs: Tensor  # (E, N, O)

    def permuted(self, gen: torch.Generator) -> "Transitions":
        """Independent random order per environment. Does not change the set of transitions."""
        E, N, _ = self.obs.shape
        perm = torch.rand(E, N, generator=gen).argsort(dim=1)  # (E, N)
        pick = lambda x: x.gather(1, perm[:, :, None].expand(-1, -1, x.shape[-1]))
        return Transitions(pick(self.obs), pick(self.act), pick(self.next_obs))


@dataclass
class Task:
    state: Tensor  # (E, 1, S) initial state
    goal: Tensor | None  # (E, G) or None


class BatchEnv:
    """Base class. Subclasses hold hidden parameters of shape (E, ...) in ``self.params``."""

    name = ""
    obs_dim = 0
    act_dim = 0
    state_dim = 0
    dt = 0.1
    hidden_names: tuple[str, ...] = ()
    data_policies: tuple[str, ...] = ("smooth", "iid")
    default_probe = "smooth"
    episode_steps = 50
    plan_horizon = 10

    def __init__(self, params: dict[str, Tensor], mechanisms: list[str], tier: str = "custom"):
        self.params = params
        self.mechanisms = mechanisms
        self.tier = tier
        self.n_envs = len(mechanisms)

    def hidden_matrix(self) -> Tensor:
        """(E, n_hidden). Never shown to a model; used only for diagnostics."""
        return torch.stack([self.params[k] for k in self.hidden_names], dim=1)

    def reset(self, n_traj: int, gen: torch.Generator) -> Tensor:
        raise NotImplementedError

    def reset_diverse(self, n_traj: int, gen: torch.Generator) -> Tensor:
        """States for iid-state sampling. Must excite the same mechanisms as trajectories do."""
        return self.reset(n_traj, gen)

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError

    def obs(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    def state_from_obs(self, obs: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_task(self, gen: torch.Generator) -> Task:
        raise NotImplementedError

    def cost(self, obs: Tensor, action: Tensor, goal: Tensor | None) -> Tensor:
        """Per-step cost computed from observations only. (E, B, O), (E, B, A) -> (E, B)."""
        raise NotImplementedError

    def is_success(self, obs_traj: Tensor, goal: Tensor | None) -> Tensor:
        """(E, T + 1, O) -> (E,) bool."""
        raise NotImplementedError

    def push_direction(self, obs: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.name} has no push policy")

    def level_action(self, obs: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.name} has no level policy")


class PendulumEnv(BatchEnv):
    """Point-mass pendulum. theta = 0 is upright, theta = pi hangs down.

    m l^2 theta'' = m g l sin(theta) - b theta' + tau
    E = 1/2 m l^2 theta'^2 + m g l cos(theta), so dE/dt = theta' (tau - b theta').

    Torque limit 5 N m against m g l in 4.8 to 16.6 over training: a heavy long pendulum cannot be
    lifted directly and must be swung up. thetadot is clipped at 15 rad/s, which an unforced pendulum
    in these ranges does not reach.
    """

    name = "pendulum"
    obs_dim = 3
    act_dim = 1
    state_dim = 2
    hidden_names = ("mass", "length", "damping")
    data_policies = ("smooth", "iid")
    default_probe = "smooth"
    episode_steps = 60
    plan_horizon = 12
    n_substeps = 4
    tau_max = 5.0
    max_speed = 15.0
    success_window = 10
    success_angle = 0.35
    success_speed = 2.0

    obs_low = torch.tensor([-1.5, -1.5, -20.0], dtype=OBS_DT)
    obs_high = torch.tensor([1.5, 1.5, 20.0], dtype=OBS_DT)

    @classmethod
    def from_values(cls, mass, length, damping, tau_bias=None, act_gain=None) -> "PendulumEnv":
        mass = torch.as_tensor(mass, dtype=DT)
        n = mass.shape[0]
        params = {
            "mass": mass,
            "length": torch.as_tensor(length, dtype=DT),
            "damping": torch.as_tensor(damping, dtype=DT),
            "tau_bias": torch.zeros(n, dtype=DT) if tau_bias is None else torch.as_tensor(tau_bias, dtype=DT),
            "act_gain": torch.ones(n, dtype=DT) if act_gain is None else torch.as_tensor(act_gain, dtype=DT),
        }
        return cls(params, ["none"] * n)

    def energy(self, state: Tensor) -> Tensor:
        m = self.params["mass"][:, None]
        l = self.params["length"][:, None]
        return 0.5 * m * l * l * state[..., 1] ** 2 + m * G * l * torch.cos(state[..., 0])

    def reset(self, n_traj, gen):
        E = self.n_envs
        theta = _rand(gen, (E, n_traj), -math.pi, math.pi)
        speed = _rand(gen, (E, n_traj), -3.0, 3.0)
        return torch.stack([theta, speed], dim=-1)

    def step(self, state, action):
        assert state.ndim == 3 and state.shape[0] == self.n_envs, state.shape
        u = action.to(DT).clamp(-1.0, 1.0)[..., 0]  # (E, B)
        m = self.params["mass"][:, None]  # (E, 1)
        l = self.params["length"][:, None]
        b = self.params["damping"][:, None]
        inertia = m * l * l
        tau = self.tau_max * self.params["act_gain"][:, None] * u + self.params["tau_bias"][:, None]

        def accel(th, om):
            return (G / l) * torch.sin(th) - (b / inertia) * om + tau / inertia

        th, om = state[..., 0], state[..., 1]
        h = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            k1t, k1o = om, accel(th, om)
            k2t = om + 0.5 * h * k1o
            k2o = accel(th + 0.5 * h * k1t, k2t)
            k3t = om + 0.5 * h * k2o
            k3o = accel(th + 0.5 * h * k2t, k3t)
            k4t = om + h * k3o
            k4o = accel(th + h * k3t, k4t)
            th = th + (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
            om = om + (h / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        th = torch.atan2(torch.sin(th), torch.cos(th))
        om = om.clamp(-self.max_speed, self.max_speed)
        return torch.stack([th, om], dim=-1)

    def obs(self, state):
        th, om = state[..., 0], state[..., 1]
        return torch.stack([torch.cos(th), torch.sin(th), om], dim=-1).to(OBS_DT)

    def state_from_obs(self, obs):
        o = obs.to(DT)
        return torch.stack([torch.atan2(o[..., 1], o[..., 0]), o[..., 2]], dim=-1)

    def sample_task(self, gen):
        """Swing-up from near the bottom, nearly at rest."""
        E = self.n_envs
        theta = math.pi + _rand(gen, (E, 1), -0.1, 0.1)
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        speed = _rand(gen, (E, 1), -0.1, 0.1)
        return Task(state=torch.stack([theta, speed], dim=-1), goal=None)

    def cost(self, obs, action, goal):
        return (1.0 - obs[..., 0]) + 0.01 * obs[..., 2] ** 2 + 1e-3 * (action**2).sum(-1)

    def is_success(self, obs_traj, goal):
        """Upright within success_angle and slower than success_speed for the last success_window obs."""
        window = min(self.success_window, obs_traj.shape[1])
        last = obs_traj[:, -window:]
        ok = (last[..., 0] >= math.cos(self.success_angle)) & (last[..., 2].abs() < self.success_speed)
        return ok.all(dim=1)


class PuckEnv(BatchEnv):
    """Force-controlled round agent pushes a round puck in a walled box [-L, L]^2.

    state = (agent_xy, puck_xy, agent_v, puck_v), 8 numbers; observation = state.
    Agent: mass 1, viscous damping C_AGENT, so free terminal speed is F_MAX / C_AGENT = 2.5 m/s.
    Puck: hidden mass and Coulomb friction, no viscous damping.
    Friction is applied as a velocity shrink after the force update: speed drops by mu g h, stopping
    at zero. This models stiction exactly and cannot reverse velocity, at any step size.
    Contact and walls are one-sided spring-dampers, never adhesive.
    """

    name = "puck"
    obs_dim = 8
    act_dim = 2
    state_dim = 8
    hidden_names = ("mass", "friction")
    data_policies = ("push", "smooth", "iid")
    default_probe = "push"
    episode_steps = 50
    plan_horizon = 10
    n_substeps = 20
    half_size = 1.0
    radius = 0.1
    contact_k = 400.0
    contact_c = 2.0
    f_max = 10.0
    c_agent = 4.0
    agent_mass = 1.0
    success_dist = 0.15

    obs_low = torch.tensor([-1.5] * 4 + [-8.0] * 4, dtype=OBS_DT)
    obs_high = torch.tensor([1.5] * 4 + [8.0] * 4, dtype=OBS_DT)

    @classmethod
    def from_values(cls, mass, friction, wind=None, rotation=None, aniso=None, act_gain=None) -> "PuckEnv":
        mass = torch.as_tensor(mass, dtype=DT)
        n = mass.shape[0]
        params = {
            "mass": mass,
            "friction": torch.as_tensor(friction, dtype=DT),
            "wind": torch.zeros(n, 2, dtype=DT) if wind is None else torch.as_tensor(wind, dtype=DT),
            "rotation": torch.zeros(n, dtype=DT) if rotation is None else torch.as_tensor(rotation, dtype=DT),
            "aniso": torch.ones(n, 2, dtype=DT) if aniso is None else torch.as_tensor(aniso, dtype=DT),
            "act_gain": torch.ones(n, dtype=DT) if act_gain is None else torch.as_tensor(act_gain, dtype=DT),
        }
        return cls(params, ["none"] * n)

    def _separate(self, agent: Tensor, puck: Tensor) -> Tensor:
        """Move puck out of initial overlap along the line of centers."""
        d = puck - agent
        dist = d.norm(dim=-1, keepdim=True)
        min_dist = 2 * self.radius + 0.02
        moved = agent + d / (dist + 1e-9) * min_dist
        limit = self.half_size - self.radius
        return torch.where(dist < min_dist, moved, puck).clamp(-limit, limit)

    def reset(self, n_traj, gen):
        E = self.n_envs
        agent = _rand(gen, (E, n_traj, 2), -0.8, 0.8)
        puck = self._separate(agent, _rand(gen, (E, n_traj, 2), -0.7, 0.7))
        agent_v = _rand(gen, (E, n_traj, 2), -0.5, 0.5)
        moving = _rand(gen, (E, n_traj, 1)) < 0.5  # half the pucks start moving, showing friction early
        puck_v = _rand(gen, (E, n_traj, 2), -1.0, 1.0) * moving
        return torch.cat([agent, puck, agent_v, puck_v], dim=-1)

    def reset_diverse(self, n_traj, gen):
        """Half the samples start close to contact, since iid positions almost never touch."""
        state = self.reset(n_traj, gen)
        E = self.n_envs
        angle = _rand(gen, (E, n_traj), 0.0, 2 * math.pi)
        reach = 2 * self.radius + _rand(gen, (E, n_traj), 0.0, 0.15)
        offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * reach[..., None]
        limit = self.half_size - self.radius
        near_puck = (state[..., :2] + offset).clamp(-limit, limit)
        near = _rand(gen, (E, n_traj, 1)) < 0.5
        puck = torch.where(near, near_puck, state[..., 2:4])
        agent_v = torch.where(near, _rand(gen, (E, n_traj, 2), -2.0, 2.0), state[..., 4:6])
        return torch.cat([state[..., :2], puck, agent_v, state[..., 6:8]], dim=-1)

    def _wall_force(self, pos: Tensor, vel: Tensor) -> Tensor:
        """Per-axis one-sided spring-damper. (E, B, 2) -> (E, B, 2)."""
        limit = self.half_size - self.radius
        over_hi = (pos - limit).clamp(min=0.0)
        over_lo = (-limit - pos).clamp(min=0.0)
        push_lo = (self.contact_k * over_lo - self.contact_c * vel).clamp(min=0.0) * (over_lo > 0)
        push_hi = (self.contact_k * over_hi + self.contact_c * vel).clamp(min=0.0) * (over_hi > 0)
        return push_lo - push_hi

    def step(self, state, action):
        assert state.ndim == 3 and state.shape[0] == self.n_envs, state.shape
        a = action.to(DT).clamp(-1.0, 1.0)  # (E, B, 2)
        rot = self.params["rotation"][:, None]  # (E, 1)
        c, s = torch.cos(rot), torch.sin(rot)
        gain = self.params["act_gain"][:, None]
        force_cmd = self.f_max * gain[..., None] * torch.stack(
            [c * a[..., 0] - s * a[..., 1], s * a[..., 0] + c * a[..., 1]], dim=-1
        )  # (E, B, 2)

        mass = self.params["mass"][:, None, None]  # (E, 1, 1)
        wind = self.params["wind"][:, None, :]  # (E, 1, 2)
        mu = self.params["friction"][:, None, None] * self.params["aniso"][:, None, :]  # (E, 1, 2)

        pa, pp, va, vp = state[..., :2], state[..., 2:4], state[..., 4:6], state[..., 6:8]
        h = self.dt / self.n_substeps
        min_dist = 2 * self.radius
        for _ in range(self.n_substeps):
            d = pp - pa
            dist = torch.sqrt((d * d).sum(-1, keepdim=True) + 1e-12)  # (E, B, 1)
            normal = d / dist  # points from agent to puck
            overlap = (min_dist - dist).clamp(min=0.0)
            approach = ((vp - va) * normal).sum(-1, keepdim=True)  # negative when closing
            contact = (self.contact_k * overlap - self.contact_c * approach).clamp(min=0.0) * (overlap > 0)
            f_on_puck = contact * normal

            va = va + h * (force_cmd - self.c_agent * va - f_on_puck + self._wall_force(pa, va)) / self.agent_mass
            vp = vp + h * (f_on_puck + wind + self._wall_force(pp, vp)) / mass

            # Friction as a velocity shrink. With isotropic mu this keeps direction and removes
            # exactly mu g h of speed; per-axis clipping generalizes it to anisotropic mu.
            speed = torch.sqrt((vp * vp).sum(-1, keepdim=True) + 1e-24)
            shrink = mu * G * h * vp.abs() / speed
            vp = torch.sign(vp) * (vp.abs() - shrink).clamp(min=0.0)

            pa = pa + h * va
            pp = pp + h * vp
        return torch.cat([pa, pp, va, vp], dim=-1)

    def obs(self, state):
        return state.to(OBS_DT)

    def state_from_obs(self, obs):
        return obs.to(DT)

    def push_direction(self, obs):
        d = obs[..., 2:4] - obs[..., :2]
        return d / (d.norm(dim=-1, keepdim=True) + 1e-6)

    def sample_task(self, gen):
        """Puck and agent at rest, goal at least 0.4 from the puck. Best effort rejection, 20 tries."""
        E = self.n_envs
        agent = _rand(gen, (E, 1, 2), -0.8, 0.8)
        puck = self._separate(agent, _rand(gen, (E, 1, 2), -0.6, 0.6))
        goal = _rand(gen, (E, 2), -0.7, 0.7)
        for _ in range(20):
            bad = (goal - puck[:, 0]).norm(dim=-1) < 0.4
            if not bad.any():
                break
            goal = torch.where(bad[:, None], _rand(gen, (E, 2), -0.7, 0.7), goal)
        state = torch.cat([agent, puck, torch.zeros(E, 1, 4, dtype=DT)], dim=-1)
        return Task(state=state, goal=goal.to(OBS_DT))

    def cost(self, obs, action, goal):
        to_goal = (obs[..., 2:4] - goal[:, None, :]).norm(dim=-1)
        to_puck = (obs[..., :2] - obs[..., 2:4]).norm(dim=-1)
        return to_goal + 0.1 * to_puck + 1e-3 * (action**2).sum(-1)

    def is_success(self, obs_traj, goal):
        """Puck center within success_dist of the goal at the final step."""
        return (obs_traj[:, -1, 2:4] - goal).norm(dim=-1) < self.success_dist

    def reset_near_contact(self, n_traj: int, gen: torch.Generator) -> Tensor:
        """Agent close to the puck and slow, the situation the push task needs. Used only by collect_contact_rich."""
        state = self.reset(n_traj, gen)
        E = self.n_envs
        angle = _rand(gen, (E, n_traj), 0.0, 2 * math.pi)
        reach = 2 * self.radius + _rand(gen, (E, n_traj), 0.0, 0.3)
        offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * reach[..., None]
        limit = self.half_size - self.radius
        puck = (state[..., :2] + offset).clamp(-limit, limit)
        return torch.cat([state[..., :2], puck, state[..., 4:]], dim=-1)


class DroneEnv(BatchEnv):
    """Planar quadrotor: two rotors on an arm, hidden mass, moment of inertia and translational drag.

    state = (x, z, theta, vx, vz, omega); observation = (x, z, cos theta, sin theta, vx, vz, omega).

        T_i = hover_thrust * rotor_gain_i * (1 + thrust_authority * act_gain * u_i),  u_i in [-1, 1]
        m vx' = -(T_1 + T_2) sin(theta) - drag vx + wind_x
        m vz' =  (T_1 + T_2) cos(theta) - drag vz + wind_z - m g
        I omega' = arm (T_2 - T_1) - ang_damping omega

    Why this is plannable with 32 candidates, 3 CEM iterations and a 1.2 s horizon, given a 0.1 s control step:
    - Zero action is close to hover (it hovers at mass 1), so the zero-mean initial candidates stay near the
      region that matters, as they do not for a system with a large gravity bias.
    - Thrust ranges 0.2 to 1.8 weights of a unit mass. Hover needs u = (m - 1) / 0.8: 0.375 at the heaviest
      training mass, 0.69 at the heaviest extrapolation mass. There is authority left to climb and to tilt.
    - One full unit of rotor difference gives 0.785 N m, so 5.2 rad/s^2 at I = 0.15, settling at 1.6 rad/s
      with time constant I / ang_damping = 0.3 s. Tilting 0.3 rad takes about 3 control steps: slower than
      the control step, so a random-looking plan cannot flip the vehicle in one step, and much faster than the horizon.
    - At 0.3 rad the lateral acceleration is about 3 m/s^2. From rest that covers more than 1 m in 1.2 s.
      Stopping from 2 m/s takes about 0.3 s to reverse the tilt plus 0.7 s to brake, which fits the horizon,
      so the planner sees an overshoot before committing to it. Drag helps.

    The dynamics share nothing with the pendulum's: there is no restoring torque at all, and hidden mass acts
    through thrust and drag rather than through a gravity torque. The attitude is an integrator with a lag, and
    position hangs two more integrators below that.

    Outside box_size a smooth (C1) force, quadratic in the excess, pushes back, so uncontrolled data stays bounded
    without any discontinuity. Inside the box it is exactly zero; the task never needs to leave. Speeds are clipped as in the
    pendulum, which only tumbling trajectories reach.
    """

    name = "drone"
    obs_dim = 7
    act_dim = 2
    state_dim = 6
    hidden_names = ("mass", "inertia", "drag")
    data_policies = ("level", "iid")
    default_probe = "level"
    episode_steps = 60
    plan_horizon = 12
    n_substeps = 4
    hover_thrust = 1.0 * G / 2.0  # per rotor, for unit mass
    thrust_authority = 0.8
    arm_length = 0.2
    ang_damping = 0.5
    box_size = 2.5
    box_k = 5.0
    max_speed = 15.0
    max_omega = 15.0
    success_window = 10
    success_dist = 0.35
    success_speed = 1.5
    success_angle = 0.6

    obs_low = torch.tensor([-6.0, -6.0, -1.5, -1.5, -16.0, -16.0, -16.0], dtype=OBS_DT)
    obs_high = torch.tensor([6.0, 6.0, 1.5, 1.5, 16.0, 16.0, 16.0], dtype=OBS_DT)

    @classmethod
    def from_values(cls, mass, inertia, drag, wind=None, rotor_gain=None, act_gain=None) -> "DroneEnv":
        mass = torch.as_tensor(mass, dtype=DT)
        n = mass.shape[0]
        params = {
            "mass": mass,
            "inertia": torch.as_tensor(inertia, dtype=DT),
            "drag": torch.as_tensor(drag, dtype=DT),
            "wind": torch.zeros(n, 2, dtype=DT) if wind is None else torch.as_tensor(wind, dtype=DT),
            "rotor_gain": torch.ones(n, 2, dtype=DT) if rotor_gain is None else torch.as_tensor(rotor_gain, dtype=DT),
            "act_gain": torch.ones(n, dtype=DT) if act_gain is None else torch.as_tensor(act_gain, dtype=DT),
        }
        return cls(params, ["none"] * n)

    def reset(self, n_traj, gen):
        E = self.n_envs
        pos = _rand(gen, (E, n_traj, 2), -1.2, 1.2)
        theta = _rand(gen, (E, n_traj, 1), -0.3, 0.3)
        vel = _rand(gen, (E, n_traj, 2), -1.0, 1.0)
        omega = _rand(gen, (E, n_traj, 1), -1.0, 1.0)
        return torch.cat([pos, theta, vel, omega], dim=-1)

    def reset_diverse(self, n_traj, gen):
        """Wide attitudes and speeds. Hidden mass and inertia show at any state, so no special cases are needed."""
        E = self.n_envs
        pos = _rand(gen, (E, n_traj, 2), -self.box_size, self.box_size)
        theta = _rand(gen, (E, n_traj, 1), -1.0, 1.0)
        vel = _rand(gen, (E, n_traj, 2), -3.0, 3.0)
        omega = _rand(gen, (E, n_traj, 1), -3.0, 3.0)
        return torch.cat([pos, theta, vel, omega], dim=-1)

    def _box_accel(self, pos: Tensor) -> Tensor:
        """(E, B, 2) -> (E, B, 2). Zero inside the box, with zero slope at its edge."""
        excess = (pos.abs() - self.box_size).clamp(min=0.0)
        return -torch.sign(pos) * self.box_k * excess * excess

    def step(self, state, action):
        assert state.ndim == 3 and state.shape[0] == self.n_envs, state.shape
        u = action.to(DT).clamp(-1.0, 1.0)  # (E, B, 2)
        m = self.params["mass"][:, None]  # (E, 1)
        inertia = self.params["inertia"][:, None]
        c = self.params["drag"][:, None]
        wind = self.params["wind"][:, None, :]  # (E, 1, 2)
        gain = self.params["act_gain"][:, None, None]  # (E, 1, 1)
        rotor = self.params["rotor_gain"][:, None, :]  # (E, 1, 2)
        thrust = self.hover_thrust * rotor * (1.0 + self.thrust_authority * gain * u)  # (E, B, 2)
        total = thrust.sum(dim=-1)  # (E, B)
        moment = self.arm_length * (thrust[..., 1] - thrust[..., 0])  # (E, B)

        def deriv(s: Tensor) -> Tensor:
            th, vx, vz, om = s[..., 2], s[..., 3], s[..., 4], s[..., 5]
            box = self._box_accel(s[..., :2])
            ax = (-total * torch.sin(th) - c * vx + wind[..., 0]) / m + box[..., 0]
            az = (total * torch.cos(th) - c * vz + wind[..., 1]) / m - G + box[..., 1]
            alpha = (moment - self.ang_damping * om) / inertia
            return torch.stack([vx, vz, om, ax, az, alpha], dim=-1)

        s = state
        h = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            k1 = deriv(s)
            k2 = deriv(s + 0.5 * h * k1)
            k3 = deriv(s + 0.5 * h * k2)
            k4 = deriv(s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        th = torch.atan2(torch.sin(s[..., 2]), torch.cos(s[..., 2]))
        vel = s[..., 3:5].clamp(-self.max_speed, self.max_speed)
        om = s[..., 5].clamp(-self.max_omega, self.max_omega)
        return torch.cat([s[..., :2], th[..., None], vel, om[..., None]], dim=-1)

    def obs(self, state):
        x, z, th, vx, vz, om = state.unbind(dim=-1)
        return torch.stack([x, z, torch.cos(th), torch.sin(th), vx, vz, om], dim=-1).to(OBS_DT)

    def state_from_obs(self, obs):
        o = obs.to(DT)
        th = torch.atan2(o[..., 3], o[..., 2])
        return torch.stack([o[..., 0], o[..., 1], th, o[..., 4], o[..., 5], o[..., 6]], dim=-1)

    def level_action(self, obs):
        """Cascaded feedback toward the origin, using no hidden parameter. Data policy; never the task's controller.

        The gains keep attitude near level and the vehicle near the task's region, which free random actions do
        not, since mass away from 1 alone accelerates at up to 3 m/s^2. Noise is added by the caller.
        """
        theta = torch.atan2(obs[..., 3], obs[..., 2])
        theta_ref = (0.08 * obs[..., 0] + 0.12 * obs[..., 4]).clamp(-0.4, 0.4)
        difference = 1.5 * (theta_ref - theta) - 0.3 * obs[..., 6]
        collective = -0.4 * obs[..., 1] - 0.4 * obs[..., 5]
        return torch.stack([collective - 0.5 * difference, collective + 0.5 * difference], dim=-1)

    def sample_task(self, gen):
        """Fly from near rest to a goal at least 0.8 away and hold there. Best effort rejection, 20 tries."""
        E = self.n_envs
        start = _rand(gen, (E, 2), -1.0, 1.0)
        goal = _rand(gen, (E, 2), -1.0, 1.0)
        for _ in range(20):
            bad = (goal - start).norm(dim=-1) < 0.8
            if not bad.any():
                break
            goal = torch.where(bad[:, None], _rand(gen, (E, 2), -1.0, 1.0), goal)
        theta = _rand(gen, (E, 1), -0.1, 0.1)
        state = torch.cat([start, theta, torch.zeros(E, 3, dtype=DT)], dim=-1)[:, None, :]  # (E, 1, 6)
        return Task(state=state, goal=goal.to(OBS_DT))

    def cost(self, obs, action, goal):
        to_goal = (obs[..., :2] - goal[:, None, :]).norm(dim=-1)
        speed2 = (obs[..., 4:6] ** 2).sum(-1)
        return to_goal + 0.02 * speed2 + 0.5 * (1.0 - obs[..., 2]) + 0.005 * obs[..., 6] ** 2 + 1e-3 * (action**2).sum(-1)

    def is_success(self, obs_traj, goal):
        """Near the goal, slow and roughly level for the last success_window observations."""
        window = min(self.success_window, obs_traj.shape[1])
        last = obs_traj[:, -window:]
        near = (last[..., :2] - goal[:, None, :]).norm(dim=-1) < self.success_dist
        slow = last[..., 4:6].norm(dim=-1) < self.success_speed
        level = last[..., 2] >= math.cos(self.success_angle)
        return (near & slow & level).all(dim=1)


def _sample_pendulum(n: int, tier: str, seed) -> PendulumEnv:
    gen = make_generator("pendulum", tier, seed)
    hidden = _sample_hidden(gen, n, tier, PENDULUM_RANGES, PENDULUM_EXTRAP)
    tau_bias = torch.zeros(n, dtype=DT)
    act_gain = torch.ones(n, dtype=DT)
    mechanisms = ["none"] * n
    if tier == "val_shift":
        act_gain = _rand(gen, (n,), 0.5, 0.7)
        mechanisms = ["actuator_gain"] * n
    elif tier == "oof":
        sign = torch.where(_rand(gen, (n,)) < 0.5, -1.0, 1.0).to(DT)
        magnitude = _rand(gen, (n,), 0.15, 0.3) * PendulumEnv.tau_max
        mechanisms = ["torque_bias" if i % 2 == 0 else "action_flip" for i in range(n)]
        is_bias = torch.tensor([m == "torque_bias" for m in mechanisms])
        tau_bias = torch.where(is_bias, sign * magnitude, tau_bias)
        act_gain = torch.where(is_bias, act_gain, -act_gain)  # 1-D analogue of a rotated action
    params = {**hidden, "tau_bias": tau_bias, "act_gain": act_gain}
    return PendulumEnv(params, mechanisms, tier)


def _sample_puck(n: int, tier: str, seed) -> PuckEnv:
    gen = make_generator("puck", tier, seed)
    hidden = _sample_hidden(gen, n, tier, PUCK_RANGES, PUCK_EXTRAP)
    wind = torch.zeros(n, 2, dtype=DT)
    rotation = torch.zeros(n, dtype=DT)
    aniso = torch.ones(n, 2, dtype=DT)
    act_gain = torch.ones(n, dtype=DT)
    mechanisms = ["none"] * n
    if tier == "val_shift":
        act_gain = _rand(gen, (n,), 0.5, 0.7)
        mechanisms = ["actuator_gain"] * n
    elif tier == "oof":
        names = ("wind", "rotated_actions", "anisotropic_friction")
        mechanisms = [names[i % 3] for i in range(n)]
        # Draw every field for every env, then mask, so the draws do not depend on the mix.
        angle = _rand(gen, (n,), 0.0, 2 * math.pi)
        strength = _rand(gen, (n,), 1.0, 2.5)  # N; friction force is 0.15 to 2.9 N over training
        wind_all = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * strength[:, None]
        sign = torch.where(_rand(gen, (n,)) < 0.5, -1.0, 1.0).to(DT)
        rot_all = sign * _rand(gen, (n,), math.radians(30.0), math.radians(60.0))
        ratio = _rand(gen, (n,), 1.5, 2.5)
        swap = _rand(gen, (n,)) < 0.5
        aniso_all = torch.stack(
            [torch.where(swap, 1.0 / ratio, ratio), torch.where(swap, ratio, 1.0 / ratio)], dim=1
        )
        m = lambda name: torch.tensor([x == name for x in mechanisms])
        wind = torch.where(m("wind")[:, None], wind_all, wind)
        rotation = torch.where(m("rotated_actions"), rot_all, rotation)
        aniso = torch.where(m("anisotropic_friction")[:, None], aniso_all, aniso)
    params = {**hidden, "wind": wind, "rotation": rotation, "aniso": aniso, "act_gain": act_gain}
    return PuckEnv(params, mechanisms, tier)


def _sample_drone(n: int, tier: str, seed) -> DroneEnv:
    gen = make_generator("drone", tier, seed)
    hidden = _sample_hidden(gen, n, tier, DRONE_RANGES, DRONE_EXTRAP)
    wind = torch.zeros(n, 2, dtype=DT)
    rotor_gain = torch.ones(n, 2, dtype=DT)
    act_gain = torch.ones(n, dtype=DT)
    mechanisms = ["none"] * n
    if tier == "val_shift":
        act_gain = _rand(gen, (n,), 0.5, 0.7)  # both rotors respond less; no test tier does this
        mechanisms = ["actuator_gain"] * n
    elif tier == "oof":
        names = ("wind", "weak_rotor")
        mechanisms = [names[i % 2] for i in range(n)]
        # Draw every field for every env, then mask, so the draws do not depend on the mix.
        angle = _rand(gen, (n,), 0.0, 2 * math.pi)
        strength = _rand(gen, (n,), 1.0, 2.5)  # N, against a weight of 6.9 to 12.8 N over training
        wind_all = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * strength[:, None]
        # One rotor loses 10% to 25%. Level flight then needs a steady rotor difference no training
        # environment needed. At 0.75, thrust with zero torque tops out at 13.2 N, above the heaviest weight.
        weak = _rand(gen, (n,), 0.75, 0.9)
        left = _rand(gen, (n,)) < 0.5
        one = torch.ones(n, dtype=DT)
        rotor_all = torch.stack([torch.where(left, weak, one), torch.where(left, one, weak)], dim=1)
        m = lambda name: torch.tensor([x == name for x in mechanisms])
        wind = torch.where(m("wind")[:, None], wind_all, wind)
        rotor_gain = torch.where(m("weak_rotor")[:, None], rotor_all, rotor_gain)
    params = {**hidden, "wind": wind, "rotor_gain": rotor_gain, "act_gain": act_gain}
    return DroneEnv(params, mechanisms, tier)


def make_env(family: str, n_envs: int, tier: str, seed) -> BatchEnv:
    """Sample n_envs environments. Same (family, n_envs, tier, seed) gives identical parameters."""
    if family == "pendulum":
        return _sample_pendulum(n_envs, tier, seed)
    if family == "puck":
        return _sample_puck(n_envs, tier, seed)
    if family == "drone":
        return _sample_drone(n_envs, tier, seed)
    raise ValueError(f"unknown family {family!r}, expected one of {FAMILIES}")


def _policy_action(env: BatchEnv, policy: str, obs: Tensor, noise: Tensor | None, gen, rho=0.9):
    """Returns (action, noise_state). action is (E, B, A), float32, in [-1, 1]."""
    E, B, _ = obs.shape
    if policy == "iid":
        return 2.0 * torch.rand(E, B, env.act_dim, generator=gen) - 1.0, None
    eps = torch.randn(E, B, env.act_dim, generator=gen)
    noise = eps if noise is None else rho * noise + math.sqrt(1.0 - rho * rho) * eps
    if policy == "smooth":
        return (0.8 * noise).clamp(-1.0, 1.0), noise
    if policy == "push":
        return (0.8 * env.push_direction(obs) + 0.4 * noise).clamp(-1.0, 1.0), noise
    if policy == "level":
        # Half the push policy's noise: a rotor difference held for the 1 s noise correlation tilts about 0.2 rad, so
        # lateral speed reaches about 1 to 2 m/s, as in the task. At 0.4 most trajectories would leave the box sideways.
        return (env.level_action(obs) + 0.2 * noise).clamp(-1.0, 1.0), noise
    raise ValueError(f"unknown policy {policy!r}")


@torch.no_grad()
def collect_trajectories(env: BatchEnv, n_traj: int, n_steps: int, policy: str, gen) -> Trajectories:
    state = env.reset(n_traj, gen)
    o = env.obs(state)
    obs, act = [o], []
    noise = None
    for _ in range(n_steps):
        a, noise = _policy_action(env, policy, o, noise, gen)
        state = env.step(state, a)
        o = env.obs(state)
        obs.append(o)
        act.append(a)
    return Trajectories(torch.stack(obs, dim=2), torch.stack(act, dim=2))


@torch.no_grad()
def collect_contact_rich(env: PuckEnv, n_traj: int, n_steps: int, gen) -> Trajectories:
    """Pushes at a strength drawn per trajectory, from close by. Opt-in training data; no default path uses it.

    The default push policy reaches about 2 m/s, so a push is one hard collision and the puck flies off. The task
    is won by slow repeated contact. Terminal speed is 2.5 * strength m/s, so strengths from 0.15 cover the
    slow pushes where friction stops the puck between touches and the agent catches up.
    """
    E = env.n_envs
    state = env.reset_near_contact(n_traj, gen)
    strength = _rand(gen, (E, n_traj, 1), 0.15, 1.0).to(OBS_DT)
    o = env.obs(state)
    obs, act = [o], []
    noise = None
    rho = 0.9
    for _ in range(n_steps):
        eps = torch.randn(E, n_traj, env.act_dim, generator=gen)
        noise = eps if noise is None else rho * noise + math.sqrt(1.0 - rho * rho) * eps
        a = (strength * env.push_direction(o) + 0.25 * noise).clamp(-1.0, 1.0)
        state = env.step(state, a)
        o = env.obs(state)
        obs.append(o)
        act.append(a)
    return Trajectories(torch.stack(obs, dim=2), torch.stack(act, dim=2))


def collect_mixed(env: BatchEnv, n_traj: int, n_steps: int, policies, gen) -> Trajectories:
    """Split n_traj as evenly as possible across policies, in the order given."""
    parts = []
    for i, policy in enumerate(policies):
        count = n_traj // len(policies) + (1 if i < n_traj % len(policies) else 0)
        if count > 0:
            parts.append(collect_trajectories(env, count, n_steps, policy, gen))
    return Trajectories(torch.cat([p.obs for p in parts], dim=1), torch.cat([p.act for p in parts], dim=1))


@torch.no_grad()
def collect_iid_states(env: BatchEnv, n_transitions: int, gen) -> Trajectories:
    """One-step trajectories from independent states, the reduced-collinearity data condition.

    Not available on a real robot; it is a controlled manipulation for H3. Shape (E, n, 2, O).
    """
    state = env.reset_diverse(n_transitions, gen)
    o = env.obs(state)
    a, _ = _policy_action(env, "iid", o, None, gen)
    next_o = env.obs(env.step(state, a))
    return Trajectories(torch.stack([o, next_o], dim=2), a[:, :, None, :])
