"""Test-time adaptation methods and their registry.

Every method has the signature ``fn(backbone, data, **hyperparameters) -> (predictor, info, counts)``,
where data is ``Transitions`` of shape (E, N, ·) from K observed trajectories in each of E new
environments. Methods adapt each environment independently but in one batch: summing per-environment
losses makes gradients independent, and Adam acts per element, so the result equals separate runs.

Add a method from anywhere with the ``@register`` decorator. Later threads need not edit this file.

Starting point: frozen is the common origin. latent_opt with 0 steps, rls with no data, grad_write with
no data and last_block with 0 steps all equal frozen on the same backbone, since each starts from the
mean training latent z_default and the trained W0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .envs import Transitions
from .models import Counts, DynamicsModel
from .train import Backbone

RLS_DTYPE = torch.float64


class RLSState:
    """Recursive least squares on a readout W, one independent problem per environment.

    Minimizes sum_i w_i ||v_i - W phi_i||^2 + lambda ||W - W0||_F^2 exactly at every step, so with
    forgetting = 1 the result does not depend on data order and equals batch ridge.

        u = P phi;  g = u / (gamma / w + phi' u)
        W += (v - W phi) g';  P = (P - g u') / gamma

    With gamma = 1 this is the World-T3 update (u' = phi' P, since P is symmetric). With gamma < 1 it
    solves the exponentially weighted problem A_t = gamma A_{t-1} + w phi phi', whose regularizer also
    decays as gamma^t lambda. So forgetting is not order invariant, and P grows without excitation
    (wind-up), which long H10 runs must watch.

    P is float64 and re-symmetrized after each write. With P0 = 1000 I the eigenvalue spread of P is
    around 1e6, far from float64 limits but not from float32.

    W: (E, D, F), P: (E, F, F).
    """

    def __init__(self, W0: Tensor, lambda_: float, forgetting: float = 1.0):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.W = W0.to(RLS_DTYPE).clone()
        E, _, feature_dim = self.W.shape
        self.P = torch.eye(feature_dim, dtype=RLS_DTYPE).expand(E, -1, -1).clone() / lambda_
        self.forgetting = forgetting
        self._W32: Tensor | None = None

    def write(self, phi: Tensor, v: Tensor, w: float | Tensor = 1.0) -> None:
        """One observation per environment. phi (E, F), v (E, D), w scalar or (E,)."""
        phi = phi.to(RLS_DTYPE)
        v = v.to(RLS_DTYPE)
        w = torch.as_tensor(w, dtype=RLS_DTYPE)
        u = (self.P @ phi[:, :, None]).squeeze(-1)  # (E, F)
        q = (phi * u).sum(-1)  # (E,), phi' P phi >= 0
        g = u / (self.forgetting / w + q)[:, None]  # (E, F)
        err = v - (self.W @ phi[:, :, None]).squeeze(-1)  # (E, D)
        self.W = self.W + err[:, :, None] * g[:, None, :]  # (E, D, 1) * (E, 1, F) = (E, D, F)
        P = (self.P - g[:, :, None] * u[:, None, :]) / self.forgetting  # (E, F, 1) * (E, 1, F)
        self.P = 0.5 * (P + P.transpose(1, 2))
        self._W32 = None

    def write_sequence(self, phi: Tensor, v: Tensor, w: float | Tensor = 1.0) -> None:
        """phi (E, N, F), v (E, N, D). Writes in order along N."""
        for t in range(phi.shape[1]):
            self.write(phi[:, t], v[:, t], w)

    def variance(self, phi: Tensor) -> Tensor:
        """Epistemic variance phi' P phi, up to sigma^2 (H4). phi (E, B, F) -> (E, B)."""
        phi = phi.to(RLS_DTYPE)
        return ((phi @ self.P) * phi).sum(-1).clamp_min(0.0)

    @property
    def W32(self) -> Tensor:
        if self._W32 is None:
            self._W32 = self.W.to(torch.float32)
        return self._W32


class AdaptedPredictor:
    """Per-environment adapted parameters on a shared backbone.

    step(o, a): o (E, B, O), a (E, B, A) -> (E, B, O). This is the model interface eval.py relies on,
    so anything with ``step`` and ``counts`` can be evaluated (Thread C's looped core included).
    """

    def __init__(
        self,
        model: DynamicsModel,
        z: Tensor | None = None,  # (E, Z)
        W: Tensor | None = None,  # (E, D, F)
        last_weight: Tensor | None = None,  # (E, F, h)
        last_bias: Tensor | None = None,  # (E, F)
        rls: RLSState | None = None,
    ):
        self.model = model
        self.z = z
        self.W = W
        self.last_weight = last_weight
        self.last_bias = last_bias
        self.rls = rls
        self.counts = Counts()

    def features(self, o: Tensor, a: Tensor) -> Tensor:
        """(E, B, F)."""
        z = None
        if self.z is not None:
            z = self.z[:, None, :].expand(o.shape[0], o.shape[1], -1)  # (E, B, Z)
        h = self.model.hidden(o, a, z)  # (E, B, h)
        if self.last_weight is None:
            return self.model.phi_from_hidden(h)
        # (E, B, h) @ (E, h, F) = (E, B, F)
        return F.gelu(h @ self.last_weight.transpose(1, 2) + self.last_bias[:, None, :])

    def predict_delta(self, o: Tensor, a: Tensor) -> Tensor:
        """Normalized delta, (E, B, D)."""
        phi = self.features(o, a)
        W = self.rls.W32 if self.rls is not None else self.W
        if W is None:
            return phi @ self.model.W.t()  # (E, B, F) @ (F, D)
        return phi @ W.transpose(1, 2)  # (E, B, F) @ (E, F, D)

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        self.counts.forward_calls += 1
        self.counts.forward_samples += o.shape[0] * o.shape[1]
        return self.model.apply_delta(o, self.predict_delta(o, a))

    def variance(self, o: Tensor, a: Tensor) -> Tensor:
        """Epistemic variance at (o, a), (E, B). Only for RLS-based predictors."""
        if self.rls is None:
            raise ValueError("this predictor has no RLS state")
        return self.rls.variance(self.features(o, a))

    def observe(self, o: Tensor, a: Tensor, next_o: Tensor, w: float | Tensor = 1.0) -> None:
        """Online RLS write of one transition per environment. Inputs (E, O), (E, A), (E, O)."""
        if self.rls is None:
            raise ValueError("this predictor has no RLS state")
        phi = self.features(o[:, None], a[:, None])[:, 0]  # (E, F)
        self.rls.write(phi, self.model.target(o, next_o), w)


@dataclass
class AdaptResult:
    predictor: AdaptedPredictor
    info: dict
    counts: Counts  # adaptation cost only; predictor.counts tracks later use
    wall_s: float


@dataclass
class MethodSpec:
    name: str
    fn: Callable
    backbones: tuple[str, ...]  # backbone kinds this method can run on
    defaults: dict
    grid: list[dict]  # validation grid; each entry overrides defaults


METHODS: dict[str, MethodSpec] = {}


def register(name: str, backbones: tuple[str, ...], defaults: dict | None = None, grid: list[dict] | None = None):
    def wrap(fn: Callable) -> Callable:
        METHODS[name] = MethodSpec(name, fn, backbones, defaults or {}, grid or [{}])
        return fn

    return wrap


def adapt(name: str, backbone: Backbone, data: Transitions, **hp) -> AdaptResult:
    spec = METHODS[name]
    if backbone.kind not in spec.backbones:
        raise ValueError(f"{name} cannot run on a {backbone.kind} backbone, needs one of {spec.backbones}")
    merged = {**spec.defaults, **hp}
    started = time.perf_counter()
    predictor, info, counts = spec.fn(backbone, data, **merged)
    return AdaptResult(predictor, info, counts, time.perf_counter() - started)


def _start_z(backbone: Backbone, n_envs: int) -> Tensor | None:
    """(E, Z) copy of the mean training latent, or None on a no-latent backbone."""
    if backbone.z_default is None:
        return None
    return backbone.z_default[None, :].expand(n_envs, -1).clone()


def _per_env_mse(pred: Tensor, target: Tensor) -> Tensor:
    """(E, N, D) -> (E,)."""
    return ((pred - target) ** 2).mean(dim=(1, 2))


def _features_and_targets(model: DynamicsModel, z: Tensor | None, data: Transitions, counts: Counts):
    """No autograd. phi (E, N, F) and v (E, N, D)."""
    with torch.no_grad():
        phi = AdaptedPredictor(model, z=z).features(data.obs, data.act)
    counts.forward_calls += 1
    counts.forward_samples += data.obs.shape[0] * data.obs.shape[1]
    return phi, model.target(data.obs, data.next_obs)


def _coarse_to_fine(step: int, steps: int, lr_scale: float) -> float:
    """T2-Z says coarse-to-fine without exact values. Here: thirds at 0.1, 0.03, 0.01."""
    fraction = step / max(steps, 1)
    base = 0.1 if fraction < 1 / 3 else (0.03 if fraction < 2 / 3 else 0.01)
    return base * lr_scale


def _optimize_latent(backbone: Backbone, data: Transitions, steps: int, lr_scale: float, counts: Counts):
    """Model frozen, z starts at the mean training latent, clipped to [0, 1]. Returns (z, info)."""
    if backbone.z_default is None:
        raise ValueError("latent optimization needs a backbone with a latent")
    model = backbone.model
    E, N, _ = data.obs.shape
    target = model.target(data.obs, data.next_obs)
    z = _start_z(backbone, E).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=_coarse_to_fine(0, steps, lr_scale))
    with torch.no_grad():
        before = _per_env_mse(model(data.obs, data.act, z[:, None, :].expand(-1, N, -1)), target)
    with torch.enable_grad():
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = _coarse_to_fine(step, steps, lr_scale)
            pred = model(data.obs, data.act, z[:, None, :].expand(-1, N, -1))  # (E, N, D)
            loss = _per_env_mse(pred, target).sum()  # gradient of z_e depends on environment e only
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z.clamp_(0.0, 1.0)
    z = z.detach()
    with torch.no_grad():
        after = _per_env_mse(model(data.obs, data.act, z[:, None, :].expand(-1, N, -1)), target)
    counts.forward_calls += steps + 2
    counts.forward_samples += (steps + 2) * E * N
    counts.backward_calls += steps
    info = {
        "loss_before": before.mean().item(),
        "loss_after": after.mean().item(),
        "z_saturation": ((z <= 1e-4) | (z >= 1.0 - 1e-4)).float().mean().item(),  # the H1 mechanism
        "z_shift": (z - backbone.z_default).norm(dim=1).mean().item(),
    }
    return z, info


def _run_rls(backbone: Backbone, data: Transitions, z: Tensor | None, lambda_: float, forgetting: float, w: float, counts: Counts) -> RLSState:
    phi, v = _features_and_targets(backbone.model, z, data, counts)
    E = data.obs.shape[0]
    rls = RLSState(backbone.model.W.detach()[None].expand(E, -1, -1), lambda_, forgetting)
    rls.write_sequence(phi, v, w)
    counts.writes += phi.shape[1]
    return rls


@register("frozen", backbones=("latent", "nolatent"))
def frozen(backbone: Backbone, data: Transitions):
    """No adaptation: mean training latent (if any) and trained weights."""
    return AdaptedPredictor(backbone.model, z=_start_z(backbone, data.obs.shape[0])), {}, Counts()


@register("context", backbones=("context",))
def context(backbone: Backbone, data: Transitions):
    """Encoder forward pass over the observed transitions. No test-time gradients."""
    model = backbone.model
    with torch.no_grad():
        z = backbone.encoder(model.norm_obs(data.obs), data.act, model.target(data.obs, data.next_obs))
    counts = Counts(forward_calls=1, forward_samples=data.obs.shape[0] * data.obs.shape[1])
    return AdaptedPredictor(model, z=z), {}, counts


@register(
    "latent_opt",
    backbones=("latent",),
    defaults={"steps": 40, "lr_scale": 1.0},
    grid=[{"lr_scale": s} for s in (0.5, 1.0, 2.0)],
)
def latent_opt(backbone: Backbone, data: Transitions, steps: int, lr_scale: float):
    counts = Counts()
    z, info = _optimize_latent(backbone, data, steps, lr_scale, counts)
    return AdaptedPredictor(backbone.model, z=z), info, counts


_LAMBDAS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)


@register(
    "rls",
    backbones=("latent", "nolatent"),
    defaults={"lambda_": 1.0, "forgetting": 1.0, "w": 1.0},
    grid=[{"lambda_": lam} for lam in _LAMBDAS],
)
def rls(backbone: Backbone, data: Transitions, lambda_: float, forgetting: float, w: float):
    counts = Counts()
    z = _start_z(backbone, data.obs.shape[0])
    state = _run_rls(backbone, data, z, lambda_, forgetting, w, counts)
    return AdaptedPredictor(backbone.model, z=z, rls=state), {}, counts


GRAD_WRITE_LRS = (0.01, 0.03, 0.1, 0.3, 1.0)
GRAD_WRITE_ONE_PASS_GRID = [{"lr": lr, "passes": 1} for lr in GRAD_WRITE_LRS]


@register(
    "grad_write",
    backbones=("latent", "nolatent"),
    defaults={"lr": 0.1, "passes": 1, "nlms": True},
    grid=[{"lr": lr, "passes": p} for lr in GRAD_WRITE_LRS for p in (1, 5, 20)],
)
def grad_write(backbone: Backbone, data: Transitions, lr: float, passes: int, nlms: bool):
    """Sequential gradient writes on W from W0, visiting the same transitions as RLS, in order.

    Each write descends 1/2 ||v - W phi||^2 analytically: W += lr * err phi'. With nlms = True the step
    is divided by ||phi||^2, which removes the feature scale from the learning rate. That favours the
    gradient method, so H3 is tested against its strongest simple form.
    """
    counts = Counts()
    E = data.obs.shape[0]
    z = _start_z(backbone, E)
    phi, v = _features_and_targets(backbone.model, z, data, counts)
    W = backbone.model.W.detach()[None].expand(E, -1, -1).clone()  # (E, D, F)
    for _ in range(passes):
        for t in range(phi.shape[1]):
            p = phi[:, t]  # (E, F)
            err = v[:, t] - (W @ p[:, :, None]).squeeze(-1)  # (E, D)
            step = lr / ((p * p).sum(-1) + 1e-8) if nlms else torch.full((E,), lr)  # (E,)
            W = W + step[:, None, None] * err[:, :, None] * p[:, None, :]  # (E, D, F)
    counts.writes += passes * phi.shape[1]
    return AdaptedPredictor(backbone.model, z=z, W=W), {}, counts


@register(
    "last_block",
    backbones=("latent", "nolatent"),
    defaults={"lr": 1e-3, "steps": 10},
    grid=[{"lr": lr, "steps": s} for lr in (1e-4, 3e-4, 1e-3, 3e-3) for s in (1, 10, 40)],
)
def last_block(backbone: Backbone, data: Transitions, lr: float, steps: int):
    """Adam on the last phi layer plus W, the AdaJEPA analogue.

    AdaJEPA adapts online, one step per MPC step, over a small replay buffer. Here the same parameter
    subset is fit offline on the K given trajectories, so all methods see identical data.
    """
    model = backbone.model
    E, N, _ = data.obs.shape
    z = _start_z(backbone, E)
    z_expanded = None if z is None else z[:, None, :].expand(-1, N, -1)
    with torch.no_grad():
        h = model.hidden(data.obs, data.act, z_expanded)  # (E, N, h); earlier layers stay frozen
    target = model.target(data.obs, data.next_obs)
    weight = model.last.weight.detach()[None].expand(E, -1, -1).clone().requires_grad_(True)  # (E, F, h)
    bias = model.last.bias.detach()[None].expand(E, -1).clone().requires_grad_(True)  # (E, F)
    W = model.W.detach()[None].expand(E, -1, -1).clone().requires_grad_(True)  # (E, D, F)
    optimizer = torch.optim.Adam([weight, bias, W], lr=lr)

    def per_env_loss() -> Tensor:
        phi = F.gelu(h @ weight.transpose(1, 2) + bias[:, None, :])  # (E, N, F)
        return _per_env_mse(phi @ W.transpose(1, 2), target)  # (E, N, D) vs (E, N, D)

    with torch.no_grad():
        before = per_env_loss()
    with torch.enable_grad():
        for _ in range(steps):
            loss = per_env_loss().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        after = per_env_loss()
    counts = Counts(
        forward_calls=steps + 3,
        forward_samples=(steps + 3) * E * N,
        backward_calls=steps,
    )
    info = {"loss_before": before.mean().item(), "loss_after": after.mean().item()}
    predictor = AdaptedPredictor(model, z=z, W=W.detach(), last_weight=weight.detach(), last_bias=bias.detach())
    return predictor, info, counts


@register(
    "hybrid",
    backbones=("latent",),
    defaults={"steps": 40, "lr_scale": 1.0, "lambda_": 1.0, "forgetting": 1.0, "w": 1.0},
    grid=[{"lambda_": lam} for lam in _LAMBDAS],
)
def hybrid(backbone: Backbone, data: Transitions, steps: int, lr_scale: float, lambda_: float, forgetting: float, w: float):
    """latent_opt, then RLS from W0 at phi(s, a, z*). RLS fits what the latent left unexplained,
    since its update is driven by v - W phi. Both stages use the same data; there is no held-out split.
    """
    counts = Counts()
    z, info = _optimize_latent(backbone, data, steps, lr_scale, counts)
    state = _run_rls(backbone, data, z, lambda_, forgetting, w, counts)
    return AdaptedPredictor(backbone.model, z=z, rls=state), info, counts
