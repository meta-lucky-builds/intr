"""Thread P settings. Standard library only, so the runner can read this before torch exists.

Every number here is fixed before any Thread P result. A change alters the run fingerprint, so results of different settings
never mix in one folder.

The fixed ladder: per family, the smallest rung that passes Gate GP on validation environments of the pilot seed. Rungs differ in
image size and, last, in encoder width and JEPA steps. Puck's dynamics core is larger on every rung, since Thread G found state-based
puck passes its gate only with capacity_updates (seeds 5-9), and drone and pendulum pass with round_one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

FAMILIES = ("pendulum", "drone", "puck")
REGISTERED_SEEDS = tuple(range(10, 20))
PILOT_SEED = 1010  # outside the registered seeds, so the ladder is chosen without touching any of their data
NO_WRITE_LAMBDA = 1e12  # intr.ebayes.NO_WRITE_LAMBDA; a test checks they agree
PROBE_TRAJECTORIES = 3  # always drawn, then cut to K, so K = 1 data never depends on which K a config lists (Problem 12)
EMBED_DIMS = {"pendulum": 16, "drone": 24, "puck": 32}  # about 4x the physical state, so anti-collapse need not invent features
REPAIR_VARIANTS = {"idm": {"idm_weight": 1.0}, "state": {"state_weight": 1.0}}  # registered in docs/HYPOTHESES.md, Thread P
ACTION_REPEATS = (1, 3)  # simulator steps per action. 60 divides by 3, so the training data keeps its transitions exactly


def decision_steps(sim_steps: int, repeat: int) -> int:
    """A count of simulator steps as a count of decisions, rounded up, so physical time is not cut short."""
    return math.ceil(sim_steps / repeat)


@dataclass(frozen=True)
class Rung:
    size: int  # image side in pixels
    width: int = 32  # first conv channels; the encoder is 1, 2, 4, 4 times this
    jepa_scale: int = 1  # multiplies JEPA steps


FULL_RUNGS = (Rung(32), Rung(48), Rung(64), Rung(64, 48, 2))
SMOKE_RUNGS = (Rung(16, 8),)


@dataclass(frozen=True)
class CoreSettings:
    """The W * phi(s, a, z) core. Same numbers as Thread G's round_one and capacity_updates arms."""

    hidden: int = 128
    n_hidden: int = 2
    feature_dim: int = 64
    model_steps: int = 8000
    warmup_steps: int = 300
    block_steps: int = 200
    latent_block_steps: int = 0
    batch_size: int = 256


CORE_FULL = {
    "pendulum": CoreSettings(),
    "drone": CoreSettings(),
    "puck": CoreSettings(256, 3, 128, 24000, 900, 600, 200),  # thread_g.arm_train_config of capacity_updates; a test compares
}
_CORE_SMOKE = CoreSettings(24, 2, 12, 80, 20, 20, 0, 48)
CORE_SMOKE = {f: _CORE_SMOKE for f in FAMILIES}


@dataclass(frozen=True)
class JepaSettings:
    steps: int = 8000
    batch: int = 256
    lr: float = 1e-3
    lr_floor: float = 0.05
    ema: float = 0.99  # target encoder, which is also the encoder kept after training
    rollout: int = 3  # predicted steps per sample. Errors compound, so velocity must be encoded to keep them small
    sim_weight: float = 1.0  # VICReg ratios 25 : 25 : 1
    var_weight: float = 1.0
    cov_weight: float = 0.04
    predictor_hidden: int = 256
    grad_clip: float = 5.0
    idm_weight: float = 0.0  # inverse dynamics term. Zero is today's objective exactly
    state_weight: float = 0.0  # state-auxiliary control on the observable obs. Zero is today's objective exactly


@dataclass(frozen=True)
class ErrorFedSettings:
    """Same as the moved module's defaults. Its filter range covers the K = 1 budget of 40 transitions."""

    z_dim: int = 48
    emb_dim: int = 24
    hidden: int = 96
    loops: int = 4
    grad_loops: int = 2
    steps: int = 3000
    batch_seqs: int = 32
    filter_max: int = 50
    horizon_max: int = 30


@dataclass
class ThreadPConfig:
    mode: str = "full"
    seeds: list[int] = field(default_factory=lambda: list(REGISTERED_SEEDS))
    families: list[str] = field(default_factory=lambda: list(FAMILIES))
    pilot_seed: int = PILOT_SEED
    h17_seeds: list[int] = field(default_factory=lambda: [10, 11, 12])  # registered: three seeds per family
    h17_latents: list[int] = field(default_factory=lambda: [2, 32])  # 8 comes from the main job
    stack: int = 2  # two frames show velocity. Three would show acceleration, a direct channel to hidden physics
    z_dim: int = 8
    train_envs: int = 128
    train_traj: int = 16
    train_steps: int = 60  # the error-fed core trains on windows up to 50 + 30, hopping trajectories when they do not fit
    n_eval_envs: int = 42  # divisible by the 2 and 3 mechanisms per family
    n_val_envs: int = 40
    adapt_steps: int = 40
    ks: list[int] = field(default_factory=lambda: [1, 3])
    holdout_traj: int = 8
    holdout_steps: int = 60
    rungs: list[Rung] = field(default_factory=lambda: list(FULL_RUNGS))
    core: dict = field(default_factory=lambda: dict(CORE_FULL))
    jepa: JepaSettings = field(default_factory=JepaSettings)
    probe_steps: int = 3000
    probe_batch: int = 1024
    probe_hidden: int = 256
    collapse_r2: float = 0.5  # unregistered number, fixed here: mean probe R2 over observation dimensions on val_in
    n_candidates: int = 64
    n_elites: int = 8
    n_iters: int = 3
    horizon_override: int | None = None
    episode_override: int | None = None
    n_boot: int = 10000
    min_seeds: int = 5  # protocol minimum for a verdict
    enforce_gates: bool = True  # a smoke run has no meaningful numbers, so it runs every family regardless
    val_lambdas: list[float] = field(default_factory=lambda: [10.0**e for e in range(-6, 7)] + [NO_WRITE_LAMBDA])
    lr_scales: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    grad_lrs: list[float] = field(default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.5])  # NLMS is stable below 2
    grad_passes: list[int] = field(default_factory=lambda: [1, 5, 20, 80])
    dopt_candidates: int = 16
    dopt_horizon: int = 4
    design_lambda_max: float = 100.0  # as in Thread F: D-optimal design at a no-write lambda would only maximize ||phi||
    error_fed: ErrorFedSettings = field(default_factory=ErrorFedSettings)
    threads: int = 2
    repair_variant: str | None = None  # set only by with_repair_variant: the pilot stage alone, at rung 0, on the pilot seed
    action_repeat: int = 1  # set only by with_action_repeat. Above 1 the error-fed core cannot train (see docs/HYPOTHESES.md), so only a repair run takes it

    def fingerprint_view(self) -> dict:
        """Everything that changes a job's numbers. Seed and family lists do not, so a run can be extended."""
        view = asdict(self)
        for key in ("seeds", "families", "threads", "h17_seeds"):
            view.pop(key)
        return view


def full_config() -> ThreadPConfig:
    return ThreadPConfig()


def smoke_config() -> ThreadPConfig:
    """Tiny, for tests on CPU. Numbers mean nothing."""
    return ThreadPConfig(
        mode="smoke",
        seeds=[10, 11],
        h17_seeds=[10],
        h17_latents=[2],
        z_dim=3,
        train_envs=8,
        train_traj=2,
        train_steps=16,
        n_eval_envs=6,
        n_val_envs=4,
        adapt_steps=10,
        holdout_traj=2,
        rungs=list(SMOKE_RUNGS),
        core=dict(CORE_SMOKE),
        jepa=JepaSettings(steps=20, batch=16, predictor_hidden=24),
        probe_steps=40,
        probe_batch=64,
        probe_hidden=24,
        collapse_r2=-1e9,  # a tiny probe predicts little; a smoke run checks the path, not the numbers
        n_candidates=8,
        n_elites=2,
        n_iters=2,
        horizon_override=3,
        episode_override=5,
        n_boot=200,
        min_seeds=2,
        enforce_gates=False,
        val_lambdas=[1e-2, 1.0, NO_WRITE_LAMBDA],
        lr_scales=[1.0],
        grad_lrs=[0.1],
        grad_passes=[1],
        dopt_candidates=4,
        dopt_horizon=2,
        error_fed=ErrorFedSettings(z_dim=8, emb_dim=6, hidden=12, loops=2, grad_loops=1, steps=4, batch_seqs=4, filter_max=6, horizon_max=4),
        threads=1,
    )


def with_repair_variant(cfg: ThreadPConfig, variant: str) -> ThreadPConfig:
    """The same settings with one encoder objective changed, at rung 0, with no registered seed left to run."""
    cfg.jepa = replace(cfg.jepa, **REPAIR_VARIANTS[variant])
    cfg.repair_variant = variant
    cfg.rungs = cfg.rungs[:1]
    cfg.seeds = []
    cfg.h17_seeds = []
    return cfg


def with_action_repeat(cfg: ThreadPConfig, repeat: int) -> ThreadPConfig:
    """Training data keeps its physical time per trajectory and its total transitions. Holdout and adaptation keep their transitions.

    Holdout steps stay as they are because make_windows needs start 30 + horizon 30. The closed loop is scaled by cem_for and episode_steps
    in the worker, and the success window by sim.env_on, because those numbers belong to each family's environment.
    """
    if repeat not in ACTION_REPEATS:
        raise ValueError(f"action repeat {repeat} is not one of {ACTION_REPEATS}")
    cfg.action_repeat = repeat
    cfg.train_steps = decision_steps(cfg.train_steps, repeat)
    cfg.train_traj = cfg.train_traj * repeat
    return cfg


def config_for(mode: str, repair_variant: str | None = None, action_repeat: int = 1) -> ThreadPConfig:
    if mode == "full":
        cfg = full_config()
    elif mode == "smoke":
        cfg = smoke_config()
    else:
        raise ValueError(f"unknown mode {mode!r}")
    cfg = with_action_repeat(cfg, action_repeat)
    return cfg if repair_variant is None else with_repair_variant(cfg, repair_variant)


def estimate_job_seconds(cfg: ThreadPConfig, family: str, bench: dict, jobs_per_gpu: int) -> dict:
    """Rough seconds for each job kind on one family, from measured rates. Pure, so it is tested.

    bench: jepa_steps_per_s (one job alone at rung 48), core_steps_per_s (CPU), model_calls_per_s (GPU rollouts). The GPU part
    slows with jobs sharing the GPU, read from jepa_share: total rate at this many jobs divided by the rate alone.
    """
    share = max(bench.get("jepa_share", 1.0), 1e-3)
    gpu_slowdown = max(jobs_per_gpu / share, 1.0)
    core = cfg.core[family]
    latent_updates = core.model_steps  # about one latent update per model update, whichever block sizes
    core_updates = core.model_steps + latent_updates + core.model_steps + latent_updates  # latent, nolatent, control
    width_factor = (core.hidden / 128.0) * (core.n_hidden / 2.0)
    core_s = core_updates * width_factor / max(bench["core_steps_per_s"], 1e-3)
    jepa_s = cfg.jepa.steps * gpu_slowdown / max(bench["jepa_steps_per_s"], 1e-3)
    horizon = cfg.horizon_override or 12
    episode = cfg.episode_override or 60
    n_methods, n_closed_tiers = 11, 3
    cem_calls = n_methods * n_closed_tiers * episode * cfg.n_iters * horizon
    cem_s = cem_calls * gpu_slowdown / max(bench["model_calls_per_s"], 1e-3)
    # Adaptation, selection and H4 have no benchmark of their own; 120 s comes from Thread F's per-method counts, scaled by hand.
    main = jepa_s + core_s + cem_s + 120.0
    ef = cfg.error_fed
    per_update_calls = (ef.filter_max / 2.0 + (ef.horizon_max + 1) / 2.0) * (ef.loops + 4)
    error_fed_s = 3 * ef.steps * per_update_calls / max(bench["core_steps_per_s"] * 6.0, 1e-3)  # a small MLP call is about 1/6 of a core update
    return {"main": main, "h14": error_fed_s, "h5": 120.0, "h16": cem_s / 3 + 30.0, "h17": core_s / 2 + 60.0}
