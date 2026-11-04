"""The drone family: physics, tiers, the task, and that the oracle can do it with the shared planner settings."""

import math

import pytest
import torch

from intr.envs import (
    DRONE_RANGES,
    N_STRIPES,
    DroneEnv,
    collect_iid_states,
    collect_mixed,
    collect_trajectories,
    make_env,
    make_generator,
)
from intr.eval import CEMConfig, OraclePredictor, closed_loop, compute_obs_scale


def _rest(n=1, **fields):
    state = torch.zeros(n, 1, 6, dtype=torch.float64)
    for name, value in fields.items():
        state[:, 0, ("x", "z", "theta", "vx", "vz", "omega").index(name)] = value
    return state


def test_unit_mass_hovers_at_zero_action():
    env = DroneEnv.from_values([1.0], [0.15], [0.5])
    state = _rest()
    for _ in range(50):
        state = env.step(state, torch.zeros(1, 1, 2))
    assert state.abs().max() < 1e-6


def test_mass_changes_vertical_acceleration_by_the_known_amount():
    env = DroneEnv.from_values([0.8, 1.25], [0.15, 0.15], [0.0, 0.0])
    after = env.step(_rest(2), torch.zeros(2, 1, 2))
    expected = [(1.0 / m - 1.0) * 9.81 * env.dt for m in (0.8, 1.25)]
    assert abs(after[0, 0, 4].item() - expected[0]) < 1e-6
    assert abs(after[1, 0, 4].item() - expected[1]) < 1e-6


def test_rotor_difference_turns_with_known_acceleration_and_tilt_moves_sideways():
    inertia = 0.15
    env = DroneEnv.from_values([1.0], [inertia], [0.5])
    action = torch.tensor([[[-0.5, 0.5]]])  # right rotor stronger: positive moment
    after = env.step(_rest(), action)
    moment = env.arm_length * env.hover_thrust * env.thrust_authority * 1.0
    # With rotational damping c, omega(t) = (M / c) (1 - exp(-c t / I)).
    expected = moment / env.ang_damping * (1.0 - math.exp(-env.ang_damping * env.dt / inertia))
    assert abs(after[0, 0, 5].item() - expected) < 1e-4
    # A positive tilt points thrust to -x.
    tilted = env.step(_rest(theta=0.3), torch.zeros(1, 1, 2))
    assert tilted[0, 0, 3] < 0


def test_inertia_and_drag_change_dynamics():
    env = DroneEnv.from_values([1.0, 1.0, 1.0], [0.1, 0.2, 0.1], [0.5, 0.5, 1.0])
    state = _rest(3, vx=2.0)
    action = torch.tensor([[-0.5, 0.5]]).expand(3, 2)[:, None]
    after = env.step(state, action)
    assert after[0, 0, 5] > after[1, 0, 5]  # more inertia turns slower
    assert after[2, 0, 3] < after[0, 0, 3]  # more drag slows faster


def test_box_force_is_zero_inside_and_pushes_back_outside():
    env = DroneEnv.from_values([1.0], [0.15], [0.5])
    inside = torch.tensor([[[env.box_size - 0.01, -env.box_size + 0.01]]], dtype=torch.float64)
    assert (env._box_accel(inside) == 0).all()
    outside = torch.tensor([[[env.box_size + 1.0, -env.box_size - 1.0]]], dtype=torch.float64)
    accel = env._box_accel(outside)
    assert accel[0, 0, 0] == -env.box_k and accel[0, 0, 1] == env.box_k


def test_observation_round_trip():
    env = make_env("drone", 2, "train", 0)
    state = env.reset(3, make_generator("rt"))
    assert env.obs(state).dtype == torch.float32
    assert torch.allclose(env.state_from_obs(env.obs(state)), state, atol=1e-5)


def _stripe(values, low, high):
    return torch.floor((values - low) / (high - low) * N_STRIPES).long()


def test_tiers():
    n = 300
    train, interp = make_env("drone", n, "train", 0), make_env("drone", n, "interp", 0)
    extrap, oof = make_env("drone", n, "extrap", 0), make_env("drone", n, "oof", 0)
    for name, (low, high) in DRONE_RANGES.items():
        assert (_stripe(train.params[name], low, high) % 2 == 0).all()
        assert (_stripe(interp.params[name], low, high) % 2 == 1).all()
        assert ((extrap.params[name] < low) | (extrap.params[name] > high)).all()
        assert (_stripe(oof.params[name], low, high) % 2 == 0).all()
    assert all(m == "none" for m in train.mechanisms)


def test_oof_mechanisms_are_active_and_differ_from_validation_shift():
    env = make_env("drone", 40, "oof", 0)
    assert set(env.mechanisms) == {"wind", "weak_rotor"}
    for i, mechanism in enumerate(env.mechanisms):
        assert (env.params["wind"][i].norm() > 0) == (mechanism == "wind")
        asymmetric = env.params["rotor_gain"][i, 0] != env.params["rotor_gain"][i, 1]
        assert asymmetric == (mechanism == "weak_rotor")
    assert (env.params["act_gain"] == 1).all()  # val_shift weakens both rotors; no test tier does


@pytest.mark.parametrize("tier", ["train", "interp", "extrap", "oof", "val_shift"])
def test_hover_is_possible_with_authority_to_spare(tier):
    """Solves for level hover directly: rotors must give equal thrust, with wind in the vertical helping or hurting."""
    env = make_env("drone", 200, tier, 3)
    p = env.params
    top = env.hover_thrust * p["rotor_gain"] * (1.0 + env.thrust_authority * p["act_gain"][:, None])  # (E, 2)
    level_max = 2.0 * top.min(dim=1).values
    weight = p["mass"] * 9.81 - p["wind"][:, 1]
    assert (level_max > weight).all()
    if tier in ("train", "interp"):
        assert (level_max > 1.3 * weight).all()  # room to tilt and climb where the gate is scored


def test_stays_bounded_and_finite_under_every_data_policy():
    env = make_env("drone", 6, "extrap", 1)
    for policy in ("level", "iid", "smooth"):
        trajs = collect_trajectories(env, 4, 80, policy, make_generator("bounded", policy))
        assert torch.isfinite(trajs.obs).all()
        assert trajs.obs[..., 4:6].abs().max() <= env.max_speed + 1e-3, policy
        assert trajs.obs[..., :2].abs().max() < 3 * env.box_size, policy  # the box holds even a tumbling vehicle


def test_level_policy_keeps_data_near_the_task_region():
    env = make_env("drone", 32, "train", 2)
    trajs = collect_trajectories(env, 8, 40, "level", make_generator("level"))
    assert trajs.obs[..., 2].mean() > 0.8  # mostly near level
    assert (trajs.obs[..., :2].abs() < env.box_size).float().mean() > 0.8


def test_data_collection_and_normalization_work_with_the_shared_helpers():
    env = make_env("drone", 3, "train", 0)
    mixed = collect_mixed(env, 4, 5, env.data_policies, make_generator("m"))
    assert mixed.obs.shape == (3, 4, 6, 7)
    assert collect_iid_states(env, 5, make_generator("i")).act.shape == (3, 5, 1, 2)
    scale = compute_obs_scale("drone")
    assert scale.shape == (7,) and (scale > 0).all()


def test_success_criterion():
    env = make_env("drone", 2, "train", 0)
    goal = torch.tensor([[0.5, -0.5], [0.0, 0.0]])
    obs = torch.zeros(2, 12, 7)
    obs[..., :2] = goal[:, None]
    obs[..., 2] = 1.0
    assert env.is_success(obs, goal).all()
    moved = obs.clone()
    moved[:, -1, 0] += 0.5  # away at the last step only: the whole window has to hold
    assert not env.is_success(moved, goal).any()
    fast = obs.clone()
    fast[:, -1, 4] = 3.0
    assert not env.is_success(fast, goal).any()
    flipped = obs.clone()
    flipped[:, -1, 2] = -1.0
    assert not env.is_success(flipped, goal).any()
    earlier = obs.clone()
    earlier[:, 0, 0] += 2.0  # before the window: does not matter
    assert env.is_success(earlier, goal).all()


def test_task_needs_the_vehicle_to_travel():
    env = make_env("drone", 50, "interp", 0)
    task = env.sample_task(make_generator("task"))
    assert task.state.shape == (50, 1, 6) and task.goal.shape == (50, 2)
    assert ((task.state[:, 0, :2].float() - task.goal).norm(dim=-1) >= 0.8 - 1e-6).all()


def test_zero_action_does_not_solve_the_task():
    """A success criterion that a do-nothing controller meets would tell nothing about a model."""
    env = make_env("drone", 40, "interp", 0)
    task = env.sample_task(make_generator("trivial"))
    state = task.state
    observed = [env.obs(state)[:, 0]]
    for _ in range(env.episode_steps):
        state = env.step(state, torch.zeros(40, 1, 2))
        observed.append(env.obs(state)[:, 0])
    assert env.is_success(torch.stack(observed, dim=1), task.goal).float().mean() <= 0.1


def test_oracle_succeeds_with_the_shared_planner_settings():
    """The gate is scored on interp. If this fails, change the family, not the planner. Other tiers are reported per run."""
    env = make_env("drone", 12, "interp", 5)
    task = env.sample_task(make_generator("oracle_task"))
    cfg = CEMConfig(horizon=env.plan_horizon, n_candidates=32, n_elites=6, n_iters=3)
    out = closed_loop(env, OraclePredictor(env), task, cfg, make_generator("oracle_cem"))
    assert out["success"].float().mean() >= 0.5, out["success"].tolist()
