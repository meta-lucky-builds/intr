# intr: when should a world model adapt, and how?

[![ci](https://github.com/meta-lucky-builds/intr/actions/workflows/ci.yml/badge.svg)](https://github.com/meta-lucky-builds/intr/actions/workflows/ci.yml)

A robot's world model is wrong the moment the world changes: a heavier load, a stiff joint, a reversed motor. Recent methods fix this at test time, using different amounts of freedom. Some tune a small latent vector. Others rewrite the model's last-layer weights in closed form.

This repo tests those ideas with pre-registered hypotheses, kill criteria, paired statistics and raw records. It compares them on small simulated systems, first from physical state and then from pixels.

**Headline:** on pendulum from state, the two main methods trade places as the physics becomes unfamiliar. From pixels, that crossover disappeared: the latent won everywhere on prediction, yet that did not produce better control.

> **Scope:** small simulators, small models and our own renderer. This is not a published benchmark, and nothing here is a claim about large models or real robots. Pixel verdicts cover pendulum only.

## Top findings

![Latent vs RLS error on pendulum, from state and from pixels](docs/figures/crossover.svg)

### 1. From state, each method wins in its own regime

Pendulum, 5 seeds, error at horizon 10, lower is better.

| Physics | Latent | RLS write | Winner |
|---|---|---|---|
| Familiar (interpolation) | **0.044** | 0.124 | Latent, 64% lower |
| Unfamiliar (out-of-family) | 0.718 | **0.348** | RLS, 51% lower |

The latent can only reach shifts it saw during training, while a weight write can reach further. Puck could not test this, because its first model could not support adaptation, so this result rests on one system.

### 2. From pixels, the crossover did not survive

Pendulum, 10 fresh seeds, same effect rule.

| Physics | Latent | Better RLS | Winner |
|---|---|---|---|
| Familiar | **0.251** | 0.401 | Latent, 37% lower, 10/10 seeds |
| Unfamiliar | **0.718** | 0.817 | Latent, 12% lower, 10/10 seeds |

The registered kill fired: one method won both tiers.

- With 40 transitions for a 64-feature readout, RLS is close to not adapting at all (under 1.5% from frozen), so it may have lost for lack of usable data.
- With three times the data, the latent still leads on unfamiliar physics in 10/10 seeds, but by 7.2%, under the 10% bar.
- The unfamiliar tier keeps normal mass, length and damping variation, which also helps the latent. Its win comes mostly from torque bias (17.7%) and less from action flip (8.9%).

### 3. Better prediction did not mean better control

Closed-loop success on the same pixel run. This did not decide any verdict, but it is the most surprising result.

| Physics | True model | No adaptation | Better RLS | Latent |
|---|---|---|---|---|
| Familiar | 0.995 | 0.807 | 0.783 | 0.733 |
| Unfamiliar | 0.971 | 0.310 | 0.310 | 0.260 |

The method with the lowest prediction error had the lowest control success. The gaps to RLS are below the effect rule, and the cause is unknown. From state, by contrast, prediction and control mostly agreed.

### 4. Pixels may bury the signal adaptation needs

Drone and puck passed their gates from state but failed them from pixels. The latent's gain went from 0.957 and 0.42 from state to under 8% and under 2% from pixels, even with an encoder trained directly on state.

A diagnostic points to a possible cause: on drone and puck, an embedding's error given the true state is about as large as its change over one step (ratio about 1.1), while on pendulum it is about 30 times smaller (0.03). This fits, but does not prove, hidden physics being buried in encoder error. It does not fully explain puck.

Two registered repairs failed, and the registered stopping rule ended the attempts:

- An inverse-dynamics loss fixed the encoder's reading of velocity on puck (R² 0.28 to 0.90) but not the gate.
- Holding each action for three steps lifted drone past the 10% loss bar but broke control on all three families.

### 5. RLS uncertainty ranks errors from state, but not usefully from pixels

An RLS write is the posterior mean of Bayesian linear regression, so `phi' P phi` should estimate uncertainty. This identity is already known (ALPaCA, Neural-Fly). The test here is whether it is useful during adaptation.

- **From state:** it ranks errors well on pendulum in every tier (Spearman about 0.84), though its magnitudes are wrong under shift.
- **From pixels:** it adds almost nothing beyond feature size (0.234 vs 0.225), and H4p was killed.

### 6. Ideas that failed

| Idea | Result |
|---|---|
| Reset vs forgetting on regime switches (H10) | Per-episode reset beat plain exponential forgetting in all 18 pendulum cells, by 15-62% |
| Fixed-point residual as a surprise signal (H9) | Near chance: AUROC 0.52 vs 0.81 for a simple Gaussian baseline |
| Re-injecting actions into every loop (H8) | No effect at trained depth (1.7%). It only matters at 16+ loops, which was post hoc |
| Latent, then RLS, with one validated strength (H2) | 41% worse than RLS alone on unfamiliar physics |
| Smarter probing with D-optimal actions (H5p) | Coverage rose but error did not, and it was worse on shifted physics |
| Core fed its own errors (H14p) | 4.8% better than its control, and most of that gap appears without the error signal |
| Adapting to appearance changes (H16, descriptive) | New colours raised latent error from 0.251 to 0.630, and no write repaired it |

## All verdicts

**From state** (seeds 0-4; [details](docs/RESULTS.md))

| Hypothesis | Pendulum | Puck |
|---|---|---|
| H1 capacity crossover | Supported | Uninformative |
| H2 hybrid beats both | Killed | Not supported |
| H3 RLS vs gradient writes | Killed in-family; collinearity half untested | Killed |
| H4 uncertainty ranks error | Supported on ranking | Killed |
| H7 loops need normalization | Inconclusive; removing normalization is unstable | Inconclusive |
| H8 action injection | Killed | Killed |
| H9 loop residual detects surprise | Killed | Killed |
| H10 forgetting beats reset | Killed | Killed, vacuously |

**From pixels, pendulum only** (seeds 10-19; [details](docs/RESULTS.md#thread-p-pendulum-on-pixels-run-41841227508-seeds-10-19))

| Hypothesis | Verdict | Note |
|---|---|---|
| H1p crossover | **Killed** | Latent wins both tiers |
| H12p evidence-chosen hybrid is no worse | Supported, weakly | 7.8% worse on unfamiliar physics in 10/10 seeds, inside the 10% allowance |
| H4p uncertainty adds beyond feature size | **Killed** | 1.1% improvement |
| H3p RLS vs tuned gradient | Not killed | Expected RLS win on shift did not appear. The gradient grid sat at an edge in 7/10 seeds |
| H5p D-optimal probing | **Killed** | No error gain |
| H14p error-fed core | **Killed** | Far behind the latent: 0.455 vs 0.251 |

**No verdict:** drone and puck from pixels failed the pixel gate. **Not run:** state-based H5, H6, H11, H12, H13 and H14. Thread F's code and tests are here, but it has no full results.

## How the claims are protected

- **Written first:** hypotheses and kill criteria are in [HYPOTHESES.md](docs/HYPOTHESES.md), committed before results. Later ideas are marked post hoc and use new seeds.
- **A high bar:** a win needs a paired bootstrap 95% CI that excludes 0 and at least a 10% reduction. Otherwise it counts as no detectable difference.
- **Clean selection:** hyperparameters are chosen on validation environments only, and methods are compared on the same adaptation data.
- **Gates before verdicts:** if the true-model planner cannot solve a task, or a model cannot support the test, that result is not counted as evidence.
- **A rechecked pixel run:** a separate agent recomputed 105 comparisons from the per-seed records and found no disagreement with the stored summary. It did not re-run the experiments.

**Known limits:** one family carries the main verdicts. Five seeds give a coarse bootstrap in round one. Some rule wordings needed a reading, and these are recorded in [RESULTS.md](docs/RESULTS.md). The pixel run used cloud CPU, and the multi-GPU path has never run on a real GPU.

## Reproduce

```bash
git clone https://github.com/meta-lucky-builds/intr && cd intr
pip install -e ".[test]"
pytest -q
bash scripts/gpu_run.sh --smoke
```

The smoke run takes minutes on a CPU, and its numbers mean nothing. For the real pixel run, start the [`pixel-pendulum`](.github/workflows/pixel-pendulum.yml) workflow from the Actions tab (about 80 minutes on free runners), or follow [GPU_RUN.md](docs/GPU_RUN.md). Runs are seeded, and source hashes and package versions are stored with every result. Reruns on cloud CPU matched on pendulum and drone. Puck's closed-loop figure moved between reruns (0.075 vs 0.175 on one seed), and reproducibility on a GPU is untested.

The recorded pixel run is commit `83be5d1`, run `41841227508`. Experiment code under `src/` and `experiments/` is unchanged since then.

## Repo map

| Path | What it holds |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every verdict, with numbers, caveats and problems found |
| [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md) | Pre-registered predictions and kill criteria |
| [`docs/lit/`](docs/lit) | Literature notes and prior-art check |
| [`docs/GPU_RUN.md`](docs/GPU_RUN.md) | How to run the pixel experiment |
| [`src/intr/`](src/intr) | Simulators, models, adaptation methods, evaluation |
| [`src/intr/pixel/`](src/intr/pixel) | Renderer, JEPA encoder, scoring rules, multi-GPU scheduler |
| [`experiments/`](experiments) | One script per thread: A adaptation, B uncertainty, C loops, F replication (not run in full), G benchmark repair, P pixels |
| [`results/`](results) | Raw results for every scored run |
| [`tests/`](tests) | Test suite, run by CI |

## Open questions

1. Why does the latent lower prediction error but not raise control success from pixels? This is the most useful next step, because prediction scores are a common way to judge world models.
2. Can multi-step training lift the signal while control still acts every 0.1 s? This could test the noise explanation and might bring drone through the gate. It is not registered yet.
3. Does the state crossover hold on a second system where both methods work?

## Provenance

This work was run by Claude agents under human direction. Commits carry co-author lines, and the earlier working notes remain in the git history and on the `mission/world-model-techniques` branch.

MIT licensed.
