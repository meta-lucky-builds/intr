import math

import pytest
import torch

from intr.envs import (
    FAMILIES,
    N_STRIPES,
    PENDULUM_RANGES,
    PUCK_RANGES,
    PendulumEnv,
    PuckEnv,
    collect_iid_states,
    collect_mixed,
    collect_trajectories,
    derive_seed,
    make_env,
    make_generator,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_same_seed_same_rollout(family):
    def run():
        env = make_env(family, 4, "train", 7)
        return env, collect_trajectories(env, 3, 25, env.default_probe, make_generator("g", 1))

    env_a, a = run()
    env_b, b = run()
    for key in env_a.params:
        assert torch.equal(env_a.params[key], env_b.params[key])
    assert torch.equal(a.obs, b.obs)
    assert torch.equal(a.act, b.act)


@pytest.mark.parametrize("family", FAMILIES)
def test_different_seed_differs(family):
    a = make_env(family, 4, "train", 1).hidden_matrix()
    b = make_env(family, 4, "train", 2).hidden_matrix()
    assert not torch.equal(a, b)


def test_derive_seed_separates_labels():
    assert derive_seed("a", 1) == derive_seed("a", 1)
    assert derive_seed("a", 1) != derive_seed("a", 2)
    assert derive_seed("a", 1) >= 0


@pytest.mark.parametrize("family", FAMILIES)
def test_shapes(family):
    env = make_env(family, 3, "oof", 0)
    trajs = collect_trajectories(env, 2, 5, env.default_probe, make_generator("s"))
    assert trajs.obs.shape == (3, 2, 6, env.obs_dim)
    assert trajs.act.shape == (3, 2, 5, env.act_dim)
    assert trajs.obs.dtype == torch.float32
    flat = trajs.flat()
    assert flat.obs.shape == (3, 10, env.obs_dim)
    # Flattening keeps transitions intact: next_obs of step t is obs of step t + 1 in a trajectory.
    assert torch.equal(flat.next_obs[:, 0], flat.obs[:, 1])
    assert torch.equal(flat.obs[:, 5], trajs.obs[:, 1, 0])
    iid = collect_iid_states(env, 7, make_generator("i"))
    assert iid.obs.shape == (3, 7, 2, env.obs_dim)
    assert iid.flat().obs.shape == (3, 7, env.obs_dim)


def test_collect_mixed_counts():
    env = make_env("puck", 2, "train", 0)
    trajs = collect_mixed(env, 4, 3, env.data_policies, make_generator("m"))
    assert trajs.obs.shape[1] == 4


def test_permuted_keeps_transitions_together():
    env = make_env("pendulum", 2, "train", 0)
    flat = collect_trajectories(env, 2, 10, "smooth", make_generator("p")).flat()
    shuffled = flat.permuted(make_generator("q"))
    assert not torch.equal(flat.obs, shuffled.obs)
    for e in range(2):
        original = torch.cat([flat.obs[e], flat.act[e], flat.next_obs[e]], dim=1)
        moved = torch.cat([shuffled.obs[e], shuffled.act[e], shuffled.next_obs[e]], dim=1)
        assert torch.equal(original[original[:, 0].argsort()], moved[moved[:, 0].argsort()])


def test_pendulum_energy_conserved_without_damping():
    n = 5
    env = PendulumEnv.from_values([1.0] * n, [0.8, 0.9, 1.0, 1.1, 1.2], [0.0] * n)
    state = env.reset(4, make_generator("energy"))
    start = env.energy(state)
    scale = (env.params["mass"] * 9.81 * env.params["length"])[:, None]
    action = torch.zeros(n, 4, 1)
    for _ in range(200):  # 20 seconds
        state = env.step(state, action)
    assert ((env.energy(state) - start).abs() / scale).max() < 2e-3


def test_pendulum_damping_dissipates_energy():
    env = PendulumEnv.from_values([1.0], [1.0], [0.3])
    state = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    start = env.energy(state)
    for _ in range(100):
        state = env.step(state, torch.zeros(1, 1, 1))
    assert (env.energy(state) < start).all()


def test_pendulum_signs():
    env = PendulumEnv.from_values([1.0], [1.0], [0.0])
    # Slightly off upright falls away from upright; slightly off the bottom swings back.
    near_top = torch.tensor([[[0.05, 0.0]]], dtype=torch.float64)
    assert env.step(near_top, torch.zeros(1, 1, 1))[0, 0, 0] > 0.05
    near_bottom = torch.tensor([[[math.pi - 0.05, 0.0]]], dtype=torch.float64)
    after = env.step(near_bottom, torch.zeros(1, 1, 1))
    assert after[0, 0, 1] > 0  # theta rises back toward pi
    # Positive torque accelerates positively.
    rest = torch.tensor([[[math.pi, 0.0]]], dtype=torch.float64)
    assert env.step(rest, torch.ones(1, 1, 1))[0, 0, 1] > 0


def test_pendulum_obs_round_trip():
    env = make_env("pendulum", 2, "train", 0)
    state = env.reset(3, make_generator("rt"))
    back = env.state_from_obs(env.obs(state))
    assert torch.allclose(back, state, atol=1e-5)


def test_pendulum_task_success_criterion():
    env = make_env("pendulum", 2, "train", 0)
    up = torch.tensor([1.0, 0.0, 0.0]).expand(2, 20, 3)
    down = torch.tensor([-1.0, 0.0, 0.0]).expand(2, 20, 3)
    assert env.is_success(up, None).all()
    assert not env.is_success(down, None).any()
    a = torch.zeros(2, 1, 1)
    assert (env.cost(up[:, :1], a, None) < env.cost(down[:, :1], a, None)).all()
    start = env.sample_task(make_generator("t")).state
    assert start.shape == (2, 1, 2)
    assert (torch.cos(start[..., 0]) < -0.9).all()  # starts hanging


def test_pendulum_hidden_parameters_change_dynamics():
    env = PendulumEnv.from_values([0.7, 1.3, 1.0, 1.0, 1.0], [1.0, 1.0, 0.7, 1.0, 1.0], [0.1, 0.1, 0.1, 0.0, 0.3])
    state = torch.tensor([[1.0, 2.0]], dtype=torch.float64).expand(5, 1, 2)
    action = torch.full((5, 1, 1), 0.5)
    for _ in range(5):
        state = env.step(state, action)
    # All five should differ pairwise.
    for i in range(5):
        for j in range(i + 1, 5):
            assert (state[i] - state[j]).abs().max() > 1e-3


def _head_on(env_size: int):
    """Agent left of puck, touching distance plus a gap, pushed to the right. (E, 1, 8)."""
    state = torch.zeros(env_size, 1, 8, dtype=torch.float64)
    state[:, 0, 0] = -0.4  # agent x
    state[:, 0, 2] = 0.0  # puck x
    action = torch.zeros(env_size, 1, 2)
    action[:, 0, 0] = 1.0
    return state, action


def test_puck_is_pushed_and_mass_matters():
    env = PuckEnv.from_values([0.3, 1.0], [0.1, 0.1])
    state, action = _head_on(2)
    for _ in range(6):
        state = env.step(state, action)
    assert state[0, 0, 2] > 0.01 and state[1, 0, 2] > 0.01  # both moved in +x
    assert abs(state[0, 0, 3]) < 1e-6  # no sideways motion on a head-on push
    assert state[0, 0, 6] != state[1, 0, 6]
    assert (state[0, 0, 2] - state[1, 0, 2]).abs() > 1e-3


def test_puck_friction_stops_at_known_rate():
    mu = 0.2
    env = PuckEnv.from_values([0.5], [mu])
    state = torch.zeros(1, 1, 8, dtype=torch.float64)
    state[0, 0, 0] = -0.8  # agent far away
    state[0, 0, 1] = -0.8
    state[0, 0, 6] = 1.0  # puck moving in +x
    next_state = env.step(state, torch.zeros(1, 1, 2))
    expected = 1.0 - mu * 9.81 * env.dt
    assert abs(next_state[0, 0, 6].item() - expected) < 1e-6
    for _ in range(20):
        next_state = env.step(next_state, torch.zeros(1, 1, 2))
    assert next_state[0, 0, 6] == 0.0  # friction stops the puck, never reverses it
    assert abs(next_state[0, 0, 2].item() - 1.0 / (2 * mu * 9.81)) < 0.03  # v^2 / (2 mu g)


def test_puck_stays_in_box_and_finite():
    env = make_env("puck", 8, "extrap", 3)
    state = env.reset(4, make_generator("box"))
    gen = make_generator("box_actions")
    limit = env.half_size + 0.12  # stiff but soft walls allow some penetration
    for _ in range(100):
        action = 2 * torch.rand(8, 4, 2, generator=gen) - 1
        state = env.step(state, action)
        assert torch.isfinite(state).all()
        assert state[..., :4].abs().max() < limit


def test_puck_contact_speeds_stay_bounded():
    # Light puck struck at full agent speed: bounded by twice the agent's terminal speed.
    env = PuckEnv.from_values([0.15], [0.01])
    state, action = _head_on(1)
    action = action.expand(1, 1, 2)
    for _ in range(30):
        state = env.step(state, action)
        assert state[..., 6:8].abs().max() < 2.5 * env.f_max / env.c_agent


def test_puck_task_and_success():
    env = make_env("puck", 6, "train", 0)
    task = env.sample_task(make_generator("task"))
    assert task.state.shape == (6, 1, 8)
    assert task.goal.shape == (6, 2)
    # Goal is at least 0.4 from the puck, unless rejection ran out, which is very unlikely.
    assert ((task.state[:, 0, 2:4].float() - task.goal).norm(dim=-1) >= 0.4 - 1e-6).all()
    obs = torch.zeros(6, 3, 8)
    obs[:, -1, 2:4] = task.goal
    assert env.is_success(obs, task.goal).all()
    obs[:, -1, 2:4] = task.goal + 0.5
    assert not env.is_success(obs, task.goal).any()


def _stripe(values, low, high):
    return torch.floor((values - low) / (high - low) * N_STRIPES).long()


@pytest.mark.parametrize(
    "family,ranges", [("pendulum", PENDULUM_RANGES), ("puck", PUCK_RANGES)]
)
def test_tiers_differ_from_train(family, ranges):
    n = 300
    train = make_env(family, n, "train", 0)
    interp = make_env(family, n, "interp", 0)
    extrap = make_env(family, n, "extrap", 0)
    oof = make_env(family, n, "oof", 0)
    for name, (low, high) in ranges.items():
        # Train stripes are even, interpolation stripes odd: disjoint values, inside the range.
        assert (_stripe(train.params[name], low, high) % 2 == 0).all()
        assert (_stripe(interp.params[name], low, high) % 2 == 1).all()
        assert interp.params[name].min() > low and interp.params[name].max() < high
        # Extrapolation: outside the training range for every parameter.
        assert ((extrap.params[name] < low) | (extrap.params[name] > high)).all()
        # Out-of-family keeps hidden parameters in family.
        assert (_stripe(oof.params[name], low, high) % 2 == 0).all()
    assert all(m == "none" for m in train.mechanisms)
    assert all(m != "none" for m in oof.mechanisms)


def test_oof_mechanisms_are_active_and_change_dynamics():
    pendulum = make_env("pendulum", 8, "oof", 0)
    assert set(pendulum.mechanisms) == {"torque_bias", "action_flip"}
    for i, m in enumerate(pendulum.mechanisms):
        if m == "torque_bias":
            assert pendulum.params["tau_bias"][i] != 0 and pendulum.params["act_gain"][i] == 1
        else:
            assert pendulum.params["tau_bias"][i] == 0 and pendulum.params["act_gain"][i] == -1

    puck = make_env("puck", 9, "oof", 0)
    assert set(puck.mechanisms) == {"wind", "rotated_actions", "anisotropic_friction"}
    for i, m in enumerate(puck.mechanisms):
        assert (puck.params["wind"][i].norm() > 0) == (m == "wind")
        assert (puck.params["rotation"][i] != 0) == (m == "rotated_actions")
        assert (puck.params["aniso"][i] != 1).any() == (m == "anisotropic_friction")


def test_wind_moves_a_resting_puck_and_rotation_turns_the_agent():
    env = PuckEnv.from_values([0.5, 0.5], [0.05, 0.05], wind=[[2.0, 0.0], [0.0, 0.0]])
    state = torch.zeros(2, 1, 8, dtype=torch.float64)
    state[:, 0, 0] = -0.8
    after = env.step(state, torch.zeros(2, 1, 2))
    assert after[0, 0, 2] > 0 and after[1, 0, 2] == 0

    env = PuckEnv.from_values([0.5], [0.05], rotation=[math.pi / 2])
    state = torch.zeros(1, 1, 8, dtype=torch.float64)
    state[0, 0, 0] = -0.8
    action = torch.tensor([[[1.0, 0.0]]])
    after = env.step(state, action)
    assert after[0, 0, 5] > 0 and abs(after[0, 0, 4]) < 1e-6  # +x command gives +y motion


def test_anisotropic_friction_slows_axes_differently():
    env = PuckEnv.from_values([0.5], [0.1], aniso=[[2.0, 0.5]])
    state = torch.zeros(1, 1, 8, dtype=torch.float64)
    state[0, 0, 0] = -0.8
    state[0, 0, 6:8] = 1.0
    after = env.step(state, torch.zeros(1, 1, 2))
    assert after[0, 0, 6] < after[0, 0, 7] < 1.0


def test_val_shift_uses_a_mechanism_no_test_tier_uses():
    for family in FAMILIES:
        val = make_env(family, 6, "val_shift", 0)
        oof = make_env(family, 30, "oof", 0)
        assert set(val.mechanisms) == {"actuator_gain"}
        assert "actuator_gain" not in oof.mechanisms
        assert (val.params["act_gain"] < 1).all()
