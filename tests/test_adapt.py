import pytest
import torch

from intr.adapt import METHODS, AdaptedPredictor, adapt, register
from intr.envs import collect_trajectories, make_env, make_generator
from intr.models import Counts


@pytest.fixture(scope="module")
def new_envs():
    env = make_env("pendulum", 5, "interp", 11)
    data = collect_trajectories(env, 1, 25, env.default_probe, make_generator("adapt")).flat()
    return env, data


def _params(model):
    return {k: v.clone() for k, v in model.named_parameters()}


def _adapted_loss(predictor, model, data):
    pred = predictor.predict_delta(data.obs, data.act)
    return ((pred - model.target(data.obs, data.next_obs)) ** 2).mean(dim=(1, 2))  # (E,)


@pytest.mark.parametrize("name", ["frozen", "context", "latent_opt", "rls", "grad_write", "last_block", "hybrid"])
def test_every_method_predicts_right_shape_and_leaves_model_alone(name, new_envs, latent_backbone, context_backbone):
    env, data = new_envs
    backbone = context_backbone if name == "context" else latent_backbone
    before = _params(backbone.model)
    result = adapt(name, backbone, data)
    o = torch.zeros(5, 7, env.obs_dim)
    a = torch.zeros(5, 7, env.act_dim)
    out = result.predictor.step(o, a)
    assert out.shape == (5, 7, env.obs_dim)
    assert torch.isfinite(out).all()
    assert result.wall_s >= 0
    assert result.predictor.counts.forward_calls == 1
    for key, value in backbone.model.named_parameters():
        assert torch.equal(value, before[key]), key


def test_methods_run_without_a_latent(new_envs, nolatent_backbone):
    _, data = new_envs
    for name in ("frozen", "rls", "grad_write", "last_block"):
        adapt(name, nolatent_backbone, data)
    for name in ("latent_opt", "hybrid"):
        with pytest.raises(ValueError):
            adapt(name, nolatent_backbone, data)


def test_latent_opt_lowers_adaptation_loss_and_clips(new_envs, latent_backbone):
    _, data = new_envs
    result = adapt("latent_opt", latent_backbone, data)
    assert result.info["loss_after"] < result.info["loss_before"]
    z = result.predictor.z
    assert z.shape == (5, 4)
    assert z.min() >= 0.0 and z.max() <= 1.0
    # Latents actually moved, and differently per environment.
    assert result.info["z_shift"] > 0
    assert (z[0] - z[1]).abs().max() > 0
    assert result.counts.backward_calls == 40


def test_latent_stays_clipped_at_a_large_learning_rate(new_envs, latent_backbone):
    _, data = new_envs
    z = adapt("latent_opt", latent_backbone, data, lr_scale=50.0).predictor.z
    assert z.min() >= 0.0 and z.max() <= 1.0
    assert ((z == 0.0) | (z == 1.0)).any()  # the bound was hit, so the clip was exercised


def test_latent_opt_zero_steps_equals_frozen(new_envs, latent_backbone):
    env, data = new_envs
    a = adapt("frozen", latent_backbone, data).predictor
    b = adapt("latent_opt", latent_backbone, data, steps=0).predictor
    assert torch.allclose(a.predict_delta(data.obs, data.act), b.predict_delta(data.obs, data.act))


def test_adaptations_batch_equals_alone(new_envs, latent_backbone):
    """Batching environments must not couple them. Adam acts per element on a summed loss."""
    from intr.envs import Transitions

    _, data = new_envs
    together = adapt("latent_opt", latent_backbone, data).predictor.z
    only_second = Transitions(data.obs[1:2], data.act[1:2], data.next_obs[1:2])
    alone = adapt("latent_opt", latent_backbone, only_second).predictor.z
    assert torch.allclose(together[1], alone[0], atol=1e-4)


def test_rls_lowers_loss_on_seen_data_and_prior_strength_matters(new_envs, latent_backbone):
    _, data = new_envs
    model = latent_backbone.model
    frozen = adapt("frozen", latent_backbone, data).predictor
    weak = adapt("rls", latent_backbone, data, lambda_=1e-3).predictor
    strong = adapt("rls", latent_backbone, data, lambda_=1e6).predictor
    base = _adapted_loss(frozen, model, data)
    assert (_adapted_loss(weak, model, data) < base).all()
    # A huge lambda keeps the prior: W stays at W0, so predictions equal frozen.
    assert torch.allclose(_adapted_loss(strong, model, data), base, rtol=1e-2, atol=1e-5)


def test_rls_exposes_epistemic_variance_and_online_writes(new_envs, latent_backbone):
    env, data = new_envs
    predictor = adapt("rls", latent_backbone, data, lambda_=0.1).predictor
    variance = predictor.variance(data.obs, data.act)
    assert variance.shape == data.obs.shape[:2]
    assert (variance >= 0).all()
    assert predictor.rls.W.dtype == torch.float64 and predictor.rls.P.dtype == torch.float64
    # An online write of a transition lowers its own variance and changes the prediction.
    o, a, n = data.obs[:, 0], data.act[:, 0], data.next_obs[:, 0]
    q_o, q_a = o[:, None], a[:, None]
    before_var = predictor.variance(q_o, q_a)
    before_pred = predictor.step(q_o, q_a)
    predictor.observe(o, a, n)
    assert (predictor.variance(q_o, q_a) < before_var).all()
    assert not torch.equal(predictor.step(q_o, q_a), before_pred)
    # Non-RLS predictors refuse.
    with pytest.raises(ValueError):
        adapt("frozen", latent_backbone, data).predictor.variance(q_o, q_a)


def test_rls_is_order_invariant_through_the_method(new_envs, latent_backbone):
    env, data = new_envs
    shuffled = data.permuted(make_generator("order"))
    a = adapt("rls", latent_backbone, data, lambda_=0.01).predictor
    b = adapt("rls", latent_backbone, shuffled, lambda_=0.01).predictor
    assert torch.allclose(a.rls.W, b.rls.W, atol=1e-5, rtol=1e-5)


def test_grad_write_learning_and_counts(new_envs, latent_backbone):
    _, data = new_envs
    model = latent_backbone.model
    base = _adapted_loss(adapt("frozen", latent_backbone, data).predictor, model, data)
    result = adapt("grad_write", latent_backbone, data, lr=0.3, passes=5)
    assert _adapted_loss(result.predictor, model, data).mean() < base.mean()
    assert result.counts.writes == 5 * data.obs.shape[1]
    assert result.counts.backward_calls == 0
    # Order matters to gradient writes, unlike RLS.
    shuffled = data.permuted(make_generator("order2"))
    other = adapt("grad_write", latent_backbone, shuffled, lr=0.3, passes=1).predictor
    same = adapt("grad_write", latent_backbone, data, lr=0.3, passes=1).predictor
    assert not torch.allclose(other.W, same.W, atol=1e-6)


def test_last_block_lowers_loss_and_batch_form_matches_model(new_envs, latent_backbone):
    _, data = new_envs
    model = latent_backbone.model
    # Zero steps: the batched per-environment form must reproduce the shared layer exactly.
    zero = adapt("last_block", latent_backbone, data, steps=0).predictor
    frozen = adapt("frozen", latent_backbone, data).predictor
    assert torch.allclose(zero.predict_delta(data.obs, data.act), frozen.predict_delta(data.obs, data.act), atol=1e-5)
    result = adapt("last_block", latent_backbone, data, lr=3e-3, steps=40)
    assert result.info["loss_after"] < result.info["loss_before"]
    assert result.predictor.last_weight.shape == (5, model.cfg.feature_dim, model.cfg.hidden)
    assert result.counts.backward_calls == 40


def test_hybrid_fits_seen_data_at_least_as_well_as_latent_alone(new_envs, latent_backbone):
    _, data = new_envs
    model = latent_backbone.model
    latent_only = adapt("latent_opt", latent_backbone, data).predictor
    hybrid = adapt("hybrid", latent_backbone, data, lambda_=0.01).predictor
    assert torch.equal(hybrid.z, latent_only.z)  # same latent stage, same deterministic path
    assert _adapted_loss(hybrid, model, data).mean() <= _adapted_loss(latent_only, model, data).mean()
    assert hybrid.rls is not None


def test_context_gives_latents_without_test_time_gradients(new_envs, context_backbone):
    _, data = new_envs
    result = adapt("context", context_backbone, data)
    assert result.counts.backward_calls == 0
    z = result.predictor.z
    assert z.shape == (5, 4)
    assert z.min() >= 0.0 and z.max() <= 1.0
    # Permutation invariance survives through the method.
    shuffled = data.permuted(make_generator("order3"))
    assert torch.allclose(adapt("context", context_backbone, shuffled).predictor.z, z, atol=1e-5)


def test_registry_rejects_wrong_backbone_and_accepts_new_methods(new_envs, latent_backbone, context_backbone):
    _, data = new_envs
    with pytest.raises(ValueError):
        adapt("context", latent_backbone, data)
    with pytest.raises(ValueError):
        adapt("rls", context_backbone, data)

    @register("test_only_method", backbones=("latent",), defaults={"gain": 2.0})
    def _method(backbone, transitions, gain):
        return AdaptedPredictor(backbone.model, z=backbone.z_default[None].expand(5, -1)), {"gain": gain}, Counts()

    try:
        assert adapt("test_only_method", latent_backbone, data).info == {"gain": 2.0}
        assert adapt("test_only_method", latent_backbone, data, gain=3.0).info == {"gain": 3.0}
    finally:
        del METHODS["test_only_method"]


def test_every_grid_entry_uses_known_hyperparameters():
    for spec in METHODS.values():
        for entry in spec.grid:
            assert set(entry) <= set(spec.defaults), spec.name
