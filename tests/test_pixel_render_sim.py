"""Rendering is deterministic and shows no hidden parameter; device-aware collection matches the originals on CPU."""

import inspect

import pytest
import torch

from intr.envs import FAMILIES, collect_mixed, collect_trajectories, make_env, make_generator
from intr.pixel.render import CANVAS, appearance_palette, offscreen_fraction, render, stack_windows, train_palette
from intr.pixel.sim import burn_in, collect_iid_states_with_burn, collect_mixed_states, collect_states, env_on, rollout_policy

CPU = torch.device("cpu")


def states_for(family: str, n: int = 4) -> torch.Tensor:
    return make_env(family, n, "train", 0).reset(1, make_generator("render", family))[:, 0]


@pytest.mark.parametrize("family", FAMILIES)
def test_render_is_deterministic_uint8_and_not_constant(family):
    s = states_for(family)
    a, b = render(family, s, 32), render(family, s, 32)
    assert a.dtype == torch.uint8 and a.shape == (4, 3, 32, 32)
    assert torch.equal(a, b)
    assert a.max() > a.min()


@pytest.mark.parametrize("family", FAMILIES)
def test_batch_shape_and_chunking_do_not_change_pixels(family):
    s = states_for(family, 6)
    flat = render(family, s, 24)
    assert torch.equal(flat, render(family, s, 24, chunk=2))
    assert torch.equal(render(family, s.reshape(2, 3, -1), 24).reshape(6, 3, 24, 24), flat)


def test_renderer_takes_no_environment():
    assert set(inspect.signature(render).parameters) == {"family", "state", "size", "palette", "chunk"}


@pytest.mark.parametrize("family", FAMILIES)
def test_hidden_parameters_show_only_through_motion(family):
    """Two environments with different hidden values: same state, same picture. After a step under force, different pictures."""
    env = make_env(family, 2, "extrap", 3)
    assert not torch.equal(env.hidden_matrix()[0], env.hidden_matrix()[1])
    state = states_for(family, 1).expand(2, -1).clone()
    if family == "puck":
        state[:, 6] = 1.0  # a moving puck, so friction acts without needing contact
    frames = render(family, state, 48)
    assert torch.equal(frames[0], frames[1])  # the same state in both environments
    action = torch.ones(2, 1, env.act_dim)
    moved = env.step(state[:, None], action)
    for _ in range(4):
        moved = env.step(moved, action)
    assert not torch.equal(render(family, moved[:, 0], 48)[0], render(family, moved[:, 0], 48)[1])


@pytest.mark.parametrize("family,velocity_dims", [("pendulum", [1]), ("puck", [4, 5, 6, 7]), ("drone", [3, 4, 5])])
def test_velocity_is_invisible_in_one_frame_and_visible_across_two(family, velocity_dims):
    env = make_env(family, 2, "train", 0)
    state = states_for(family, 1).expand(2, -1).clone()
    state[:, velocity_dims] = 0.0
    state[1, velocity_dims] = 1.0
    assert torch.equal(render(family, state, 48)[0], render(family, state, 48)[1])
    next_state = env_on(env, CPU).step(state[:, None], torch.zeros(2, 1, env.act_dim))[:, 0]
    # Same first frame, so any difference between the stacks is motion.
    a = stack_windows(torch.stack([render(family, state, 48), render(family, next_state, 48)], dim=1))
    assert not torch.equal(a[0], a[1])


def test_stack_windows_keeps_order():
    frames = torch.arange(2).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 4, 4).to(torch.uint8)
    out = stack_windows(frames)
    assert out.shape == (1, 6, 4, 4)
    assert (out[:, :3] == 0).all() and (out[:, 3:] > 0).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_appearance_palette_changes_pixels_and_keeps_polarity(family):
    s = states_for(family, 4)
    palette = appearance_palette(family, 4, make_generator("a"))
    assert palette.shape == (4, train_palette(family).shape[0], 3)
    assert torch.equal(palette, appearance_palette(family, 4, make_generator("a")))
    assert not torch.equal(render(family, s, 32), render(family, s, 32, palette))
    assert palette[:, 0].max() < palette[:, 1:].min()  # dark background, bright objects, as in training


def test_per_environment_palette_broadcasts_over_time():
    s = states_for("puck", 2)[:, None, None].expand(2, 3, 5, -1)
    palette = appearance_palette("puck", 2, make_generator("b"))
    frames = render("puck", s, 16, palette)
    assert frames.shape == (2, 3, 5, 3, 16, 16)
    assert torch.equal(frames[0, 1, 2], render("puck", s[0, 1, 2][None], 16, palette[:1])[0])


def test_offscreen_fraction_only_counts_clamped_drone_positions():
    s = torch.zeros(4, 6)
    s[0, 0] = CANVAS["drone"] + 1.0
    assert offscreen_fraction("drone", s) == pytest.approx(0.25)
    assert offscreen_fraction("puck", torch.zeros(2, 8)) == 0.0


@pytest.mark.parametrize("family", FAMILIES)
def test_collect_states_matches_collect_trajectories(family):
    env = make_env(family, 5, "train", 1)
    for policy in env.data_policies:
        original = collect_trajectories(env, 3, 12, policy, make_generator("c", policy))
        mine = collect_states(env, env_on(env, CPU), 3, 12, policy, make_generator("c", policy), 0, CPU)
        assert torch.equal(original.obs, mine.obs)
        assert torch.equal(original.act, mine.act)
        assert torch.equal(env.obs(mine.states[:, :, 5]), mine.obs[:, :, 5])


@pytest.mark.parametrize("family", FAMILIES)
def test_collect_mixed_matches(family):
    env = make_env(family, 4, "train", 2)
    original = collect_mixed(env, 5, 8, env.data_policies, make_generator("m"))
    mine = collect_mixed_states(env, env_on(env, CPU), 5, 8, env.data_policies, make_generator("m"), 0, CPU)
    assert torch.equal(original.obs, mine.obs) and torch.equal(original.act, mine.act)


def test_burn_in_aligns_states_observations_and_actions():
    env = make_env("drone", 3, "train", 0)
    dev = env_on(env, CPU)
    traj = collect_states(env, dev, 2, 6, env.default_probe, make_generator("b"), 1, CPU)
    assert traj.states.shape[2] == 1 + 6 + 1 and traj.obs.shape[2] == 7 and traj.act.shape[2] == 6
    assert torch.equal(env.obs(traj.states[:, :, 1]), traj.obs[:, :, 0])  # data index 0 is after the burn-in step
    assert torch.equal(env.obs(dev.step(traj.states[:, :, 1], traj.act[:, :, 0])), traj.obs[:, :, 1])
    zero = torch.zeros(3, 2, env.act_dim)
    assert torch.equal(dev.step(traj.states[:, :, 0], zero), traj.states[:, :, 1])


def test_arms_from_the_same_start_share_their_first_window():
    env = make_env("pendulum", 3, "interp", 0)
    dev = env_on(env, CPU)
    default = collect_states(env, dev, 3, 5, env.default_probe, make_generator("t", "probe"), 1, CPU)
    s0 = env.reset(3, make_generator("t", "probe"))
    other = rollout_policy(dev, s0, 5, "iid", make_generator("t", "h5"), 1)
    assert torch.equal(default.obs[:, :, 0], other.obs[:, :, 0])
    assert not torch.equal(default.obs[:, :, 3], other.obs[:, :, 3])
    assert len(burn_in(dev, s0, 2)) == 3


def test_iid_state_ceiling_has_one_transition_per_trajectory():
    env = make_env("puck", 2, "train", 0)
    traj = collect_iid_states_with_burn(env, env_on(env, CPU), 7, make_generator("i"), 1, CPU)
    assert traj.act.shape == (2, 7, 1, 2) and traj.obs.shape == (2, 7, 2, 8) and traj.states.shape[2] == 3
