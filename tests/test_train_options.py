"""Opt-in training and data options: they do what they say, and the defaults reproduce the first round exactly."""

import math

import pytest
import torch

from intr.envs import PuckEnv, collect_contact_rich, collect_mixed, collect_trajectories, make_env, make_generator
from intr.eval import CEMConfig, OraclePredictor, closed_loop, collect_planned
from intr.looped import replay_contact_flags
from intr.models import INPUT_EXTRAS, PUCK_CONTACT_DISTANCE, DynamicsModel, ModelConfig
from intr.train import TrainConfig, TransitionData, _cosine_lr, _set_lr, regression_loss, train_latent, train_nolatent

TINY = TrainConfig(model_steps=60, warmup_steps=20, block_steps=20, batch_size=32, log_every=20)


@pytest.fixture(scope="module")
def puck_data():
    env = make_env("puck", 6, "train", 0)
    return env, TransitionData.collect(env, 2, 10, make_generator("options"))


def _old_model_updates(model, optimizer, data, tcfg, gen, n, start, total, z_table=None):
    """train.py's model loop as the first round ran it, copied here so a later edit cannot move both sides."""
    for i in range(n):
        _set_lr(optimizer, _cosine_lr(tcfg.lr, start + i, total, tcfg.lr_floor))
        env_idx, o, a, next_o = data.sample(tcfg.batch_size, gen)
        z = None if z_table is None else z_table[env_idx].detach()
        loss = ((model(o, a, z) - model.target(o, next_o)) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
    return start + n


def _old_train_latent(data, cfg, tcfg, seed):
    from intr.envs import derive_seed

    gen = make_generator("train", seed, "latent")
    torch.manual_seed(derive_seed("init", seed, "latent"))
    model = DynamicsModel(cfg)
    model.set_normalization(data.obs, data.next_obs)
    z_table = torch.rand(data.n_envs, cfg.z_dim, generator=gen).requires_grad_(True)
    opt_model = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    opt_latent = torch.optim.Adam([z_table], lr=tcfg.latent_lr)
    total = tcfg.model_steps
    done = _old_model_updates(model, opt_model, data, tcfg, gen, min(tcfg.warmup_steps, total), 0, total, z_table)

    def block(n):
        model.requires_grad_(False)
        for _ in range(n):
            env_idx, o, a, next_o = data.sample(tcfg.batch_size, gen)
            loss = ((model(o, a, z_table[env_idx]) - model.target(o, next_o)) ** 2).mean()
            opt_latent.zero_grad(set_to_none=True)
            loss.backward()
            opt_latent.step()
            with torch.no_grad():
                z_table.clamp_(0.0, 1.0)
        model.requires_grad_(True)

    block(tcfg.block_steps)
    while done < total:
        n = min(tcfg.block_steps, total - done)
        done = _old_model_updates(model, opt_model, data, tcfg, gen, n, done, total, z_table)
        block(tcfg.block_steps)
    return model, z_table.detach()


def test_option_defaults_are_off():
    t = TrainConfig()
    assert (t.loss, t.latent_samples_per_env, t.latent_block_steps, t.latent_lr_decay) == ("mse", 0, 0, False)
    assert ModelConfig(8, 2).input_extra == ""


def test_default_latent_training_equals_the_first_round_code_bit_for_bit(puck_data):
    env, data = puck_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, 3, 16, 2, 8)
    model, z = _old_train_latent(data, cfg, TINY, seed=0)
    now = train_latent(data, cfg, TINY, seed=0)
    assert torch.equal(z, now.z_table)
    for (name, old), (_, new) in zip(model.named_parameters(), now.model.named_parameters()):
        assert torch.equal(old, new), name


def test_default_nolatent_training_equals_the_first_round_code_bit_for_bit(puck_data):
    from intr.envs import derive_seed

    env, data = puck_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, 0, 16, 2, 8)
    gen = make_generator("train", 0, "nolatent")
    torch.manual_seed(derive_seed("init", 0, "nolatent"))
    model = DynamicsModel(cfg)
    model.set_normalization(data.obs, data.next_obs)
    optimizer = torch.optim.Adam(model.parameters(), lr=TINY.lr)
    _old_model_updates(model, optimizer, data, TINY, gen, TINY.model_steps, 0, TINY.model_steps)
    now = train_nolatent(data, cfg, TINY, seed=0)
    for (name, old), (_, new) in zip(model.named_parameters(), now.model.named_parameters()):
        assert torch.equal(old, new), name


def test_default_input_is_unchanged():
    cfg = ModelConfig(8, 2, 3, 16, 2, 8)
    assert DynamicsModel(cfg).body[0].in_features == 13
    extended = DynamicsModel(ModelConfig(8, 2, 3, 16, 2, 8, "puck_relative"))
    assert extended.body[0].in_features == 13 + INPUT_EXTRAS["puck_relative"][0]


def test_collection_and_policies_of_old_families_are_untouched_by_the_new_policy():
    """The level branch is after the others, and collect_mixed for puck uses its old policy list."""
    assert make_env("puck", 2, "train", 0).data_policies == ("push", "smooth", "iid")
    assert make_env("pendulum", 2, "train", 0).data_policies == ("smooth", "iid")


def test_regression_loss():
    pred, target = torch.tensor([[0.0, 3.0]]), torch.tensor([[0.5, 0.0]])
    assert regression_loss(pred, target, TrainConfig()).item() == ((pred - target) ** 2).mean().item()
    huber = regression_loss(pred, target, TrainConfig(loss="huber")).item()
    assert abs(huber - (0.5 * 0.25 + (3.0 - 0.5)) / 2) < 1e-6  # quadratic below delta, linear above
    with pytest.raises(ValueError):
        regression_loss(pred, target, TrainConfig(loss="l1"))


def test_huber_reaches_a_large_error_less_than_mse():
    pred = torch.tensor([[10.0]], requires_grad=True)
    regression_loss(pred, torch.zeros(1, 1), TrainConfig(loss="huber")).backward()
    assert pred.grad.abs().item() == pytest.approx(1.0)


def test_sample_per_env_covers_every_environment_evenly(puck_data):
    _, data = puck_data
    env_idx, o, a, next_o = data.sample_per_env(5, make_generator("spe"))
    assert env_idx.shape == (30,)
    assert (torch.bincount(env_idx) == 5).all()
    assert o.shape == (30, 8) and a.shape == (30, 2) and next_o.shape == (30, 8)
    # Transitions stay whole: each drawn observation exists in its environment's data.
    for i in range(30):
        assert (data.obs[env_idx[i]] == o[i]).all(dim=1).any()


def test_extended_keeps_the_original_data_first(puck_data):
    env, data = puck_data
    extra = collect_trajectories(env, 1, 4, "smooth", make_generator("extra")).flat()
    more = data.extended(extra)
    assert more.obs.shape[1] == data.obs.shape[1] + 4
    assert torch.equal(more.obs[:, : data.obs.shape[1]], data.obs)
    assert (more.n_traj, more.n_steps) == (data.n_traj, data.n_steps)
    wrong = collect_trajectories(make_env("puck", 2, "train", 0), 1, 4, "smooth", make_generator("x")).flat()
    with pytest.raises(ValueError):
        data.extended(wrong)


def test_latent_options_take_effect(puck_data):
    env, data = puck_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, 3, 16, 2, 8)
    fewer = TrainConfig(model_steps=60, warmup_steps=20, block_steps=20, batch_size=32, log_every=20, latent_block_steps=5)
    assert train_latent(data, cfg, fewer, seed=0).stats["latent_updates"] == 3 * 5
    assert train_latent(data, cfg, TINY, seed=0).stats["latent_updates"] == 3 * 20
    options = TrainConfig(model_steps=60, warmup_steps=20, block_steps=20, batch_size=32, log_every=20, latent_samples_per_env=4, latent_lr_decay=True)
    decayed = train_latent(data, cfg, options, seed=0)
    assert decayed.z_table.min() >= 0.0 and decayed.z_table.max() <= 1.0
    assert not torch.equal(decayed.z_table, train_latent(data, cfg, TINY, seed=0).z_table)


def test_relative_inputs_describe_the_geometry():
    assert PUCK_CONTACT_DISTANCE == 2 * PuckEnv.radius
    fn = INPUT_EXTRAS["puck_relative"][1]
    o = torch.zeros(1, 8)
    o[0, 2] = 0.3  # puck 0.3 to the right of the agent
    o[0, 4] = 1.0  # agent moving toward it at 1 m/s
    out = fn(o)[0]
    assert out.shape == (8,)
    assert out[:2].tolist() == pytest.approx([0.3, 0.0], abs=1e-4)  # relative position
    assert out[4].item() == pytest.approx(5.0 * 0.1, abs=1e-3)  # gap
    assert out[5:7].tolist() == pytest.approx([1.0, 0.0], abs=1e-3)  # normal
    assert out[7].item() == pytest.approx(0.3 * -1.0, abs=1e-3)  # closing is negative


def test_relative_inputs_are_differentiable_at_zero_distance():
    fn = INPUT_EXTRAS["puck_relative"][1]
    o = torch.zeros(1, 8, requires_grad=True)
    fn(o).sum().backward()
    assert torch.isfinite(o.grad).all()


def _fraction(env, trajs):
    flags, _ = replay_contact_flags(env, trajs)
    return flags.float().mean().item()


def test_contact_rich_data_has_more_contact_and_slower_agent():
    env = make_env("puck", 8, "train", 1)
    default = collect_mixed(env, 6, 40, env.data_policies, make_generator("default"))
    rich = collect_contact_rich(env, 6, 40, make_generator("rich"))
    assert rich.obs.shape == default.obs.shape and rich.act.shape == default.act.shape
    assert _fraction(env, rich) > 1.2 * _fraction(env, default)
    assert (rich.act.abs() <= 1.0).all()


def test_contact_rich_needs_a_family_with_contact():
    with pytest.raises((NotImplementedError, AttributeError)):
        collect_contact_rich(make_env("drone", 2, "train", 0), 2, 3, make_generator("x"))


def test_collect_planned_matches_closed_loop_without_noise():
    env = make_env("puck", 3, "train", 2)
    cfg = CEMConfig(horizon=3, n_candidates=6, n_elites=2, n_iters=1)
    task = env.sample_task(make_generator("t"))
    trajs = collect_planned(env, OraclePredictor(env), task, cfg, make_generator("c"), steps=4)
    assert trajs.obs.shape == (3, 1, 5, 8) and trajs.act.shape == (3, 1, 4, 2)
    out = closed_loop(env, OraclePredictor(env), task, cfg, make_generator("c"), steps=4)
    assert torch.equal(env.is_success(trajs.obs[:, 0], task.goal), out["success"])
    # Transitions are real: stepping the simulator reproduces the next observation.
    state = env.state_from_obs(trajs.obs[:, :, 0])
    again = env.obs(env.step(state, trajs.act[:, :, 0]))
    assert torch.allclose(again, trajs.obs[:, :, 1], atol=1e-4)


def test_collect_planned_noise_changes_actions_and_stays_in_bounds():
    env = make_env("puck", 3, "train", 2)
    cfg = CEMConfig(horizon=3, n_candidates=6, n_elites=2, n_iters=1)
    task = env.sample_task(make_generator("t"))
    clean = collect_planned(env, OraclePredictor(env), task, cfg, make_generator("c"), steps=4)
    noisy = collect_planned(env, OraclePredictor(env), task, cfg, make_generator("c"), steps=4, action_noise=0.3)
    assert not torch.equal(clean.act, noisy.act)
    assert noisy.act.abs().max() <= 1.0
    assert math.isfinite(noisy.obs.sum().item())
