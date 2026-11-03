import numpy as np
import pytest
import torch

from intr.adapt import adapt
from intr.envs import Trajectories, collect_trajectories, make_env, make_generator
from intr.eval import (
    CEMConfig,
    OraclePredictor,
    PersistencePredictor,
    ValSet,
    bootstrap_ci,
    cem_plan,
    closed_loop,
    collect_holdout,
    compute_obs_scale,
    feature_collinearity,
    open_loop_errors,
    paired_comparison,
    select_hyperparameters,
)


def test_bootstrap_ci_on_known_data():
    # Constant values: no variance, so the interval collapses on the value.
    low, high = bootstrap_ci([2.0] * 5)
    assert low == pytest.approx(2.0) and high == pytest.approx(2.0)
    # Large sample: matches the normal theory interval for the mean.
    x = np.random.default_rng(0).normal(1.0, 1.0, size=400)
    low, high = bootstrap_ci(x, n_boot=4000)
    half_width = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert low == pytest.approx(x.mean() - half_width, abs=0.03)
    assert high == pytest.approx(x.mean() + half_width, abs=0.03)
    # Reproducible by seed.
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)


def test_paired_comparison_verdicts():
    # a is clearly and consistently 50% lower.
    a = [1.0, 1.1, 0.9, 1.05, 0.95]
    b = [2.0, 2.2, 1.8, 2.1, 1.9]
    assert paired_comparison(a, b)["verdict"] == "a_wins"
    assert paired_comparison(b, a)["verdict"] == "b_wins"
    result = paired_comparison(a, b)
    assert result["reduction"] == pytest.approx(0.5)
    assert result["ci_high"] < 0
    # Consistent but under 10%: CI excludes 0, effect is too small.
    small = paired_comparison([0.95, 0.96, 0.94, 0.95, 0.95], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert small["ci_high"] < 0
    assert small["verdict"] == "no_detectable_difference"
    # Seeds disagree in sign: CI includes 0.
    noisy = paired_comparison([1.0, 3.0, 1.0, 3.0, 2.0], [3.0, 1.0, 3.0, 1.0, 2.0])
    assert noisy["verdict"] == "no_detectable_difference"
    assert paired_comparison([1.0], [2.0])["verdict"] == "insufficient_seeds"
    with pytest.raises(ValueError):
        paired_comparison([1.0, 2.0], [1.0])


def _holdout(family, seed=0):
    env = make_env(family, 3, "interp", seed)
    return env, collect_holdout(env, 4, 60, make_generator("holdout", family, seed))


@pytest.mark.parametrize("family", ["pendulum", "puck"])
def test_oracle_rollout_has_near_zero_error(family):
    """If time indexing or action alignment in the evaluator were off, this would fail."""
    env, holdout = _holdout(family)
    scale = compute_obs_scale(family)
    errors = open_loop_errors(OraclePredictor(env), holdout, scale, env.obs_low, env.obs_high)
    assert set(errors) == {1, 10, 30}
    for h, e in errors.items():
        assert e.shape == (3,)
        assert e.max() < 0.02, f"horizon {h}"


def test_persistence_error_grows_with_horizon_and_matches_hand_calculation():
    env, holdout = _holdout("pendulum")
    scale = compute_obs_scale("pendulum")
    errors = open_loop_errors(PersistencePredictor(), holdout, scale, env.obs_low, env.obs_high, starts=(0,))
    assert errors[30].mean() > errors[1].mean() > 0
    by_hand = (((holdout.obs[:, :, 1] - holdout.obs[:, :, 0]) / scale) ** 2).mean(dim=(1, 2)).sqrt()
    assert torch.allclose(errors[1], by_hand, atol=1e-5)


def test_open_loop_rejects_short_holdout():
    env = make_env("pendulum", 2, "train", 0)
    short = collect_trajectories(env, 1, 20, "iid", make_generator("short"))
    with pytest.raises(ValueError):
        open_loop_errors(PersistencePredictor(), short, torch.ones(3), env.obs_low, env.obs_high)


def test_obs_scale_is_fixed_and_positive():
    a, b = compute_obs_scale("puck"), compute_obs_scale("puck")
    assert torch.equal(a, b)
    assert a.shape == (8,) and (a > 0).all()


class _LinearGoalEnv:
    """o' = o + a in 1-D, cost is distance to +1. CEM must push positive from below the goal."""

    obs_low = torch.tensor([-100.0])
    obs_high = torch.tensor([100.0])

    @staticmethod
    def cost(o, a, goal):
        return (o[..., 0] - 1.0).abs()


class _Integrator:
    counts = None

    @staticmethod
    def step(o, a):
        return o + 0.1 * a


def test_cem_plans_toward_lower_cost():
    env = _LinearGoalEnv()
    cfg = CEMConfig(horizon=5, n_candidates=64, n_elites=8, n_iters=3)
    obs = torch.tensor([[-1.0], [3.0]])  # (E, O): first must go up, second must go down
    action, next_mean = cem_plan(_Integrator, env, obs, None, torch.zeros(2, 5, 1), cfg, make_generator("cem"))
    assert action.shape == (2, 1)
    assert next_mean.shape == (2, 5, 1)
    assert action[0, 0] > 0.3
    assert action[1, 0] < -0.3
    assert (action.abs() <= 1).all()


def test_closed_loop_is_paired_and_deterministic():
    env = make_env("puck", 2, "train", 0)
    cfg = CEMConfig(horizon=3, n_candidates=4, n_elites=2, n_iters=1)

    def run():
        return closed_loop(env, OraclePredictor(env), env.sample_task(make_generator("t")), cfg, make_generator("c"), 3)

    a, b = run(), run()
    assert a["success"].shape == (2,)
    assert torch.equal(a["mean_cost"], b["mean_cost"])
    assert torch.equal(a["success"], b["success"])


def test_oracle_mpc_makes_progress_on_the_puck_task():
    """Task sanity. A short true-model plan should reduce agent-to-puck distance."""
    env = make_env("puck", 3, "train", 1)
    cfg = CEMConfig(horizon=6, n_candidates=16, n_elites=4, n_iters=2)
    task = env.sample_task(make_generator("progress"))
    start = (task.state[:, 0, :2] - task.state[:, 0, 2:4]).norm(dim=-1)
    state = task.state
    gen = make_generator("mpc")
    mean = torch.zeros(3, 6, 2)
    oracle = OraclePredictor(env)
    with torch.no_grad():
        for _ in range(8):
            o = env.obs(state)[:, 0]
            action, mean = cem_plan(oracle, env, o, task.goal, mean, cfg, gen)
            state = env.step(state, action[:, None])
    end = (state[:, 0, :2] - state[:, 0, 2:4]).norm(dim=-1)
    assert end.mean() < start.mean()


class _Features:
    def __init__(self, phi):
        self.phi = phi

    def features(self, obs, act):
        return self.phi


def _transitions_stub(n):
    class Data:
        obs = torch.zeros(1, n, 1)
        act = torch.zeros(1, n, 1)

    return Data


def test_collinearity_measure_separates_known_cases():
    gen = torch.Generator().manual_seed(0)
    N, F = 100, 8
    well_spread = torch.randn(1, N, F, generator=gen)
    line = torch.randn(1, 1, F, generator=gen) * torch.linspace(0.5, 1.5, N)[None, :, None]
    nearly_collinear = line + 1e-3 * torch.randn(1, N, F, generator=gen)
    spread = feature_collinearity(_Features(well_spread), _transitions_stub(N))
    collinear = feature_collinearity(_Features(nearly_collinear), _transitions_stub(N))
    assert collinear["log10_cond_reg"] > spread["log10_cond_reg"] + 1
    assert collinear["top_eig_fraction"] > 0.99 > spread["top_eig_fraction"]
    assert collinear["participation_ratio"] < 1.1
    assert spread["participation_ratio"] > 5


def test_collinearity_ignores_order_and_handles_too_few_transitions(latent_backbone):
    env = make_env("pendulum", 2, "interp", 0)
    data = collect_trajectories(env, 1, 10, "smooth", make_generator("few")).flat()  # N < F = 24
    predictor = adapt("frozen", latent_backbone, data).predictor
    a = feature_collinearity(predictor, data)
    b = feature_collinearity(predictor, data.permuted(make_generator("few2")))
    # log10_cond_nonzero is left out: its smallest eigenvalue sits near the float32 noise floor of phi.
    for key in ("log10_cond_reg", "participation_ratio", "top_eig_fraction"):
        assert a[key] == pytest.approx(b[key], rel=1e-4, abs=1e-4)
    assert all(np.isfinite(v) for v in a.values())


def test_selection_uses_grid_records_table_and_prefers_better(latent_backbone):
    env = make_env("pendulum", 3, "val_in", 0)
    data = collect_trajectories(env, 1, 25, env.default_probe, make_generator("val")).flat()
    holdout = collect_holdout(env, 2, 60, make_generator("val_holdout"))
    scale = compute_obs_scale("pendulum")
    out = select_hyperparameters(
        "rls", latent_backbone, [ValSet(data, holdout)], scale, env.obs_low, env.obs_high,
        grid=[{"lambda_": 0.1}, {"lambda_": 10.0}],
    )
    assert len(out["table"]) == 2
    assert out["hp"] in ({"lambda_": 0.1}, {"lambda_": 10.0})
    assert out["score"] == min(row["score"] for row in out["table"])
    assert out["relative_to_frozen"]


def test_trajectories_take():
    trajs = Trajectories(torch.zeros(2, 5, 4, 3), torch.zeros(2, 5, 3, 1))
    assert trajs.take(2).obs.shape == (2, 2, 4, 3)
    assert trajs.take(2).flat().obs.shape == (2, 6, 3)
