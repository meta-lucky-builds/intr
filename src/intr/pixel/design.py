"""D-optimal probes for H5p, from Thread F's design, on embeddings. A test checks the scores match thread_f.dopt_scores.

Receding horizon. Candidate action sequences are scored by the sum of log(1 + phi' P phi) on predicted features, with P updated on those
predicted features inside the sequence, which is the log det gain by the determinant lemma. The first action runs in the true simulator,
the camera encodes the new image, and that real transition is written. Trajectories run in sequence and share P.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from .. import ebayes
from ..adapt import AdaptedPredictor, RLSState, _start_z
from ..envs import OBS_DT, BatchEnv, Transitions
from ..models import clip_obs
from ..train import Backbone
from .planning import Camera
from .sim import burn_in
from .world import Observed


def sample_action_sequences(E: int, C: int, H: int, A: int, gen: torch.Generator) -> Tensor:
    """(E, C, H, A) in [-1, 1]. A quarter constant at corners, a quarter constant inside, and the rest iid per step."""
    seqs = (2.0 * torch.rand(E, C, 1, A, generator=gen) - 1.0).expand(-1, -1, H, -1).clone()
    n_iid = C // 2
    if H > 1 and n_iid > 0:
        seqs[:, C - n_iid :] = 2.0 * torch.rand(E, n_iid, H, A, generator=gen) - 1.0
    n_corner = C // 4
    if n_corner > 0:
        seqs[:, :n_corner] = torch.sign(seqs[:, :n_corner])
    return seqs


@torch.no_grad()
def dopt_scores(predictor: AdaptedPredictor, s: Tensor, seqs: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Predicted log det gain per candidate. s (E, D), seqs (E, C, H, A); returns (E, C). Does not change the predictor."""
    E, C, H, _ = seqs.shape
    P = predictor.rls.P[:, None]
    W = predictor.rls.W32.transpose(1, 2)
    x = s[:, None].expand(E, C, -1)
    score = torch.zeros(E, C, dtype=torch.float64)
    for h in range(H):
        phi = predictor.features(x, seqs[:, :, h])
        p64 = phi.to(torch.float64)
        u = (P @ p64[..., None]).squeeze(-1)
        q = (p64 * u).sum(-1).clamp_min(0.0)
        score = score + torch.log1p(q)
        if h + 1 < H:
            P = P - u[..., None] * u[..., None, :] / (1.0 + q)[..., None, None]
            x = clip_obs(predictor.model.apply_delta(x, phi @ W), low, high)
    return score


def design_predictor(backbone: Backbone, n_envs: int, design_lambda: float) -> AdaptedPredictor:
    z = _start_z(backbone, n_envs)
    W0 = backbone.model.W.detach()[None].expand(n_envs, -1, -1)
    return AdaptedPredictor(backbone.model, z=z, rls=RLSState(W0, design_lambda))


@torch.no_grad()
def collect_dopt(env_dev: BatchEnv, s0: Tensor, n_steps: int, predictor: AdaptedPredictor, family: str, size: int, stack: int, encoder, n_candidates: int, horizon: int, gen: torch.Generator, emb_low: Tensor, emb_high: Tensor) -> tuple[Observed, dict]:
    """s0 (E, B, S) on the device. The design predictor and its bounds are on the CPU. Every action comes from past images only."""
    E, B, _ = s0.shape
    rows = torch.arange(E)
    started = time.perf_counter()
    emb_all, obs_all, act_all = [], [], []
    for b in range(B):
        states = burn_in(env_dev, s0[:, [b]], stack - 1)
        camera = Camera(family, size, stack, encoder, states[:-1])
        state = states[-1]
        s = camera(state).cpu()
        o = env_dev.obs(state)[:, 0]
        embs, obs, acts = [s], [o], []
        for _ in range(n_steps):
            seqs = sample_action_sequences(E, n_candidates, horizon, env_dev.act_dim, gen).to(OBS_DT)
            best = dopt_scores(predictor, s, seqs, emb_low, emb_high).argmax(dim=1)
            a = seqs[rows, best, 0]
            state = env_dev.step(state, a[:, None, :].to(s0.device))
            next_s = camera(state).cpu()
            predictor.observe(s, a, next_s)
            s = next_s
            embs.append(s)
            obs.append(env_dev.obs(state)[:, 0])
            acts.append(a)
        emb_all.append(torch.stack(embs, dim=1))
        obs_all.append(torch.stack(obs, dim=1).cpu())
        act_all.append(torch.stack(acts, dim=1))
    info = {"wall_s": time.perf_counter() - started, "forward_samples": E * B * n_steps * n_candidates * horizon}
    return Observed(torch.stack(emb_all, dim=1), torch.stack(obs_all, dim=1), torch.stack(act_all, dim=1)), info


@torch.no_grad()
def coverage_measures(backbone: Backbone, data: Transitions, holdout: Transitions, lambda_: float) -> dict:
    """The manipulation check for H5p. All at one lambda, on frozen features, so arms are comparable."""
    E = data.obs.shape[0]
    frozen = AdaptedPredictor(backbone.model, z=_start_z(backbone, E))
    phi = frozen.features(data.obs, data.act).double()
    gain = ebayes.logdet_precision_gain(phi, lambda_)
    P = ebayes.posterior_covariance(phi, lambda_)
    query = frozen.features(holdout.obs, holdout.act).double()
    q = ((query @ P) * query).sum(-1).clamp_min(0.0)
    return {
        "lambda": lambda_,
        "logdet_gain": gain.mean().item(),
        "query_log1p_q": torch.log1p(q).mean().item(),  # lower: the data cover the held-out queries better
        "state_spread": backbone.model.norm_obs(data.obs).std(dim=1).mean().item(),
    }
