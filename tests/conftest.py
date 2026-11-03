"""Shared tiny trained backbones. Session scoped, so they train once."""

import pytest

from intr.envs import make_env, make_generator
from intr.models import ModelConfig
from intr.train import TrainConfig, TransitionData, continued_training, train_context, train_latent, train_nolatent

TINY_TRAIN = TrainConfig(
    model_steps=600, warmup_steps=100, block_steps=100, batch_size=128, log_every=100, context_envs_per_step=8
)


@pytest.fixture(scope="session")
def pendulum_data():
    env = make_env("pendulum", 16, "train", 0)
    return env, TransitionData.collect(env, 4, 20, make_generator("test_data"))


@pytest.fixture(scope="session")
def latent_backbone(pendulum_data):
    env, data = pendulum_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, z_dim=4, hidden=48, feature_dim=24)
    return train_latent(data, cfg, TINY_TRAIN, seed=0)


@pytest.fixture(scope="session")
def nolatent_backbone(pendulum_data):
    env, data = pendulum_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, z_dim=0, hidden=48, feature_dim=24)
    return train_nolatent(data, cfg, TINY_TRAIN, seed=0)


@pytest.fixture(scope="session")
def context_backbone(pendulum_data):
    env, data = pendulum_data
    cfg = ModelConfig(env.obs_dim, env.act_dim, z_dim=4, hidden=48, feature_dim=24)
    return train_context(data, cfg, TINY_TRAIN, seed=0)


@pytest.fixture(scope="session")
def control_backbone(nolatent_backbone, pendulum_data):
    _, data = pendulum_data
    return continued_training(nolatent_backbone, data, 50, TINY_TRAIN, seed=0)
