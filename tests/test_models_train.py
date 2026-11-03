import torch

from intr.envs import collect_trajectories, make_env, make_generator
from intr.models import ContextEncoder, DynamicsModel, ModelConfig, rollout
from intr.train import latent_r2


def test_forward_shapes_and_delta_round_trip():
    model = DynamicsModel(ModelConfig(obs_dim=3, act_dim=1, z_dim=2, hidden=16, feature_dim=8))
    o = torch.randn(4, 5, 3)
    a = torch.randn(4, 5, 1)
    z = torch.rand(4, 5, 2)
    assert model.features(o, a, z).shape == (4, 5, 8)
    assert model(o, a, z).shape == (4, 5, 3)
    next_o = torch.randn(4, 5, 3)
    model.set_normalization(o, next_o)
    v = model.target(o, next_o)
    assert torch.allclose(model.apply_delta(o, v), next_o, atol=1e-5)
    # Normalized targets have unit scale on the data that set the normalization.
    assert torch.allclose(v.reshape(-1, 3).std(0), torch.ones(3), atol=1e-4)
    assert torch.allclose(v.reshape(-1, 3).mean(0), torch.zeros(3), atol=1e-5)


def test_zero_latent_dimension_disables_it():
    model = DynamicsModel(ModelConfig(obs_dim=3, act_dim=1, z_dim=0, hidden=16, feature_dim=8))
    assert model.body[0].in_features == 4
    assert model(torch.randn(2, 3), torch.randn(2, 1)).shape == (2, 3)


def test_rollout_feeds_predictions_back():
    # o_next = o + a, so the result is the cumulative sum.
    o0 = torch.zeros(2, 3, 1)
    actions = torch.ones(2, 3, 4, 1)
    out = rollout(lambda o, a: o + a, o0, actions)
    assert out.shape == (2, 3, 5, 1)
    assert torch.equal(out[0, 0, :, 0], torch.arange(5.0))


def test_rollout_clip_keeps_divergence_finite():
    o0 = torch.ones(1, 1, 1)
    actions = torch.zeros(1, 1, 4, 1)
    low, high = torch.tensor([-2.0]), torch.tensor([2.0])
    out = rollout(lambda o, a: o * float("inf"), o0, actions, low, high)
    assert torch.isfinite(out).all()
    assert out[0, 0, 1:].eq(2.0).all()
    out = rollout(lambda o, a: o * float("nan"), o0, actions, low, high)
    assert out[0, 0, 1:].eq(2.0).all()


def test_tiny_model_overfits_a_batch():
    torch.manual_seed(0)
    env = make_env("pendulum", 2, "train", 0)
    flat = collect_trajectories(env, 2, 8, "iid", make_generator("overfit")).flat()
    o = flat.obs.reshape(-1, 3)
    a = flat.act.reshape(-1, 1)
    n = flat.next_obs.reshape(-1, 3)
    model = DynamicsModel(ModelConfig(3, 1, z_dim=0, hidden=64, feature_dim=32))
    model.set_normalization(o, n)
    target = model.target(o, n)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(800):
        loss = ((model(o, a) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.05 * first


def test_context_encoder_is_permutation_invariant_and_bounded():
    torch.manual_seed(0)
    encoder = ContextEncoder(3, 1, 4)
    o, a, d = torch.randn(2, 10, 3), torch.randn(2, 10, 1), torch.randn(2, 10, 3)
    z = encoder(o, a, d)
    assert z.shape == (2, 4)
    assert (z >= 0).all() and (z <= 1).all()
    perm = torch.randperm(10)
    assert torch.allclose(z, encoder(o[:, perm], a[:, perm], d[:, perm]), atol=1e-5)


def test_training_reduces_loss(latent_backbone, nolatent_backbone, context_backbone):
    for backbone in (latent_backbone, nolatent_backbone, context_backbone):
        model_log = [r for r in backbone.log if r["phase"] in ("model", "joint")]
        assert model_log[-1]["loss"] < 0.8 * model_log[0]["loss"], backbone.kind


def test_alternation_counts_and_latents_clipped(latent_backbone):
    stats = latent_backbone.stats
    assert stats["model_updates"] == 600
    assert stats["latent_updates"] > 0
    table = latent_backbone.z_table
    assert table.shape == (16, 4)
    assert table.min() >= 0.0 and table.max() <= 1.0
    assert torch.allclose(latent_backbone.z_default, table.mean(0))
    phases = {r["phase"] for r in latent_backbone.log}
    assert phases == {"model", "latent"}


def test_latents_moved_from_initialization_during_training(latent_backbone):
    # z_table is the first draw from this generator in train_latent, so this recreates the start.
    initial = torch.rand(16, 4, generator=make_generator("train", 0, "latent"))
    assert (latent_backbone.z_table - initial).abs().max() > 0.01
    assert not latent_backbone.z_table.requires_grad


def test_backbones_are_frozen_after_training(latent_backbone, context_backbone, control_backbone):
    assert not any(p.requires_grad for p in latent_backbone.model.parameters())
    assert not any(p.requires_grad for p in context_backbone.encoder.parameters())
    assert not any(p.requires_grad for p in control_backbone.model.parameters())


def test_continued_training_changes_a_copy_only(nolatent_backbone, control_backbone):
    assert control_backbone.stats["model_updates"] == nolatent_backbone.stats["model_updates"] + 50
    assert nolatent_backbone.stats["model_updates"] == 600
    before = dict(nolatent_backbone.model.named_parameters())
    after = dict(control_backbone.model.named_parameters())
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_latent_r2_on_a_known_relation():
    gen = torch.Generator().manual_seed(0)
    z = torch.rand(100, 3, generator=gen)
    hidden = torch.stack([2 * z[:, 0] + 1, torch.rand(100, generator=gen)], dim=1)
    r2 = latent_r2(z, hidden)
    assert r2[0] > 0.999
    assert r2[1] < 0.3
