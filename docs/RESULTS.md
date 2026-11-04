# Results: H1, H2, H3, H4, H7, H8, H9, H10

Scored against `docs/HYPOTHESES.md` from `results/thread_{a,b,c}/thread_{a,b,c}_full.json`: 5 seeds, 2 families.

**Scope limit:** small state-based simulators on CPU. Nothing here supports a claim about pixels, large models, or real robots.

## Summary

| Hyp. | Pendulum | Puck | Kill fired? | Deciding numbers (protocol metric, K = 1) |
|---|---|---|---|---|
| H1 capacity crossover | **Supported** | **Not supported** (uninformative) | No | Pendulum h10: interp latent 0.044 vs RLS 0.124 (64% lower); oof RLS 0.348 vs latent 0.718 (51% lower). Puck: no win, 2.0% and 6.1% |
| H2 hybrid | **Killed** | **Not supported** | Pendulum yes (oof) | Pendulum oof: hybrid 0.594 vs RLS 0.348, 41% worse, 5/5 seeds |
| H3 RLS vs gradient | **Killed** in-family; see readings | **Killed** | Yes | Tuned gradient matches RLS on interp: pendulum 0.9%, puck 4.2%. Collinearity did not fall, so that half was not tested |
| H4 calibration | **Supported** | **Killed** by the revised rule; depends on one definition | Puck, under the definition fixed in code | Within-env Spearman: pendulum 0.84 (ref 0.74); puck 0.28 (ref 0.06), `P` beats `\|\|phi\|\|^2` by only 6% of 1 - rho |
| H7 loops x rollout | **Inconclusive** | **Inconclusive** | No | `plain` is unstable everywhere (2.5x to 13x, 100% blow-up). Normalization is necessary. Mixed depth is unresolved. Carried state is nearly inert |
| H8 action injection | Killed (non-contact reference) | **Killed** (decisive family) | Yes | Puck interp h10: 0.526 vs 0.517, 1.7%, CI spans 0 |
| H9 residual signal | **Killed** | **Killed** | Yes | AUROC: pendulum shift 0.52 vs baseline 0.81; puck shift 0.48 vs 0.61, contact 0.60 vs 0.91. Baseline is simplified, not ARCADE |
| H10 forgetting | **Killed**; reset beats forgetting | **Killed**, vacuously | Yes | Pendulum: reset wins all 18 cells, by 15% to 62%. Puck: forgetting, reset, persist within 3.1% in all cells, no method adapts |

The puck arm cannot discriminate between adaptation methods. Every method on it is at the closed-loop floor (at most 7% success against an oracle at 98-100%) and every selected hyperparameter is the weakest write available. Treat "the families disagree" below as "pendulum is informative and puck is not", not as contrary evidence.

## How this was scored

- Effect rule: the paired bootstrap 95% CI over training seeds excludes 0 and the reduction is at least 10%. Otherwise "no detectable difference" (NDD).
- Primary metric: open-loop error at h10, except H4 and H10 (one-step).
- I recomputed comparisons with the exact bootstrap over all 5^5 ordered resamples, stdlib only. It reproduces the stored numpy CIs to about 4 decimals. New readings below use it.
- Fragility checks, both post hoc: a paired t interval, and leave-one-seed-out. "Fragile" means a verdict flips under one of these, or sits within 2 points of a threshold.
- The stored `outcome` fields are not always the registered rule (Problems 2 to 4). Verdicts here come from the rule text.
- Provenance is clean. All stored source hashes match the working tree, and no seed-family pair is missing.

## Sanity checks

### Oracle gates (closed-loop success, per seed)

| Tier | Pendulum | Puck |
|---|---|---|
| interp | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| extrap | 0.43, 0.33, 0.33, 0.24, 0.33 (mean 0.33) | 1.0, 1.0, 1.0, 1.0, 1.0 |
| oof | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0, 0.95, 1.0, 0.95, 1.0 |

Pendulum extrap fails the gate, so its closed-loop results are invalid and are not used as evidence. Confirming this, models beat the true simulator there: `latent_opt` 0.52 against the oracle's 0.33, ahead in 5/5 seeds. The oracle is not a ceiling on that tier.

### Do models beat persistence? Are errors saturated?

Ratio of error to the saturation reference (error of predicting another trajectory's state at the same step, from Thread C on the same tier). Near 1 means differences are not meaningful.

| Family, tier | Persistence h1 / h10 / h30 | Frozen | latent_opt | RLS |
|---|---|---|---|---|
| Pendulum interp | 0.22 / 0.99 / 0.90 | 0.02 / 0.26 / 0.61 | 0.00 / 0.03 / 0.20 | 0.01 / 0.09 / 0.34 |
| Pendulum extrap | 0.22 / 1.07 / 1.01 | 0.08 / 0.71 / 1.06 | 0.04 / 0.43 / 0.76 | 0.02 / 0.23 / 0.48 |
| Pendulum oof | 0.24 / 0.97 / 0.87 | 0.05 / 0.62 / 0.89 | 0.04 / 0.51 / 0.79 | 0.03 / 0.25 / 0.54 |
| Puck interp | 0.31 / 0.79 / 0.86 | 0.13 / 0.33 / 0.46 | 0.14 / 0.34 / 0.47 | 0.14 / 0.35 / 0.51 |
| Puck extrap | 0.33 / 0.85 / 0.85 | 0.16 / 0.42 / 0.56 | 0.16 / 0.41 / 0.57 | 0.16 / 0.43 / 0.60 |
| Puck oof | 0.34 / 0.88 / 0.91 | 0.19 / 0.47 / 0.57 | 0.19 / 0.48 / 0.59 | 0.17 / 0.45 / 0.57 |

- h1 and h10 are not saturated for any adaptive method, so the primary metric is usable.
- h30 is close to saturated for pendulum extrap and oof without adaptation (0.89 to 1.06). Differences there are compressed.
- All models beat persistence at h1 and h10. Persistence is itself saturated at h10 on pendulum, so that bar is low.
- Long-horizon looped rollouts, full recipe, error / saturation:

| Tier | Pendulum h1 / h10 / h30 / h100 | Puck h1 / h10 / h30 / h100 |
|---|---|---|
| interp | 0.02 / 0.27 / 0.64 / 0.79 | 0.15 / 0.38 / 0.50 / 0.57 |
| extrap | 0.08 / 0.72 / 1.05 / 1.27 | 0.19 / 0.42 / 0.55 / 0.62 |
| oof | 0.05 / 0.60 / 0.86 / 0.98 | 0.21 / 0.49 / 0.58 / 0.65 |

`two_x_reachable` is false in all six cells at h100.

### Does the latent identify puck physics? No

The low R2 is on all seeds, not one.

| | Pendulum | Puck |
|---|---|---|
| R2 mass, range over seeds | 0.965-0.982 | 0.363-0.497 |
| R2 length or friction | 0.989-0.994 | 0.217-0.333 |
| R2 damping | 0.901-0.969 | n/a |
| Chance R2 (8 regressors, 128 envs) | about 0.06 | about 0.06 |
| Final train loss: latent vs no latent | 0.0004 vs 0.0227 | 0.118 vs 0.117 |
| `latent_opt` h10 vs frozen, interp | 88% lower | 3.2% **higher** (5/5 seeds) |
| `latent_opt` h10 vs frozen, train tier | 88% lower | 0.5% higher, NDD |
| z entries at the clip bound, interp | 1.8% | 41% (44% even on the train tier) |
| Selected `lr_scale` | 2.0 on 5/5 | mixed, score about 1.0 of frozen |

R2 is linear and above chance, so the latents carry some association. That is not the point. Even with its own trained latent per environment, the puck model fits no better than a model with no latent, and test-time optimization lowers loss on the probe (0.31 to 0.27) while raising held-out error. There the latent acts as a free regressor, not a physics estimate.

**Effect on H1:** puck cannot test capacity matching. The latent has no usable capacity even inside its training family, so its "saturation" does not distinguish tiers (41% interp, 51% oof). A puck null says nothing against the idea. A possible cause, not verified: each latent Adam step sees about 2 samples per environment (`train.py:180`), and contact spikes dominate the normalized MSE.

### Puck model quality

- Closed-loop success is 0 to 7% for every method on every tier, against an oracle at 95-100% with the same planner. Mean cost is about 0.88 against 0.21. So the model is the limit, not the planner.
- The best puck closed-loop number in Thread A is the continued-training control (0.067 on interp), with no adaptation at all.
- Thread C: contact windows (44.5% of h10 windows) have error 0.76, free windows 0.22.
- Train-tier error equals test-tier error (0.46 and 0.46). This is underfitting of contact, not a generalization gap.

### Open-loop against closed-loop

| Cell | Open-loop h10 | Closed-loop success | Agreement |
|---|---|---|---|
| Pendulum interp | latent 0.044, RLS 0.124, frozen 0.360 | 0.99, 0.96, 0.79 (oracle 1.0) | Order agrees. Latent vs RLS gap is NDD closed-loop (ceiling) |
| Pendulum oof | RLS 0.348, hybrid 0.594, latent 0.718, frozen 0.876 | 0.73, 0.62, 0.34, 0.28 (oracle 1.0) | Agrees |
| Pendulum extrap | RLS 0.256, latent 0.479 | 0.31, 0.52 (oracle 0.33) | **Reversed**, but invalid by the gate |
| Puck, all tiers | small spread | all about 0 | Closed-loop has no resolution |
| Looped, pendulum interp | `plain` 1.02 vs `full` 0.38 at h10, 3.1x at h100 | 0.77 and 0.77 (oracle 0.99) | **Disagree.** The plan horizon is 12 and z resets each imagination (`looped.py:465-490`), so blow-up never reaches the planner |

### Continued-training control vs method gains (h10, K = 1)

- **Pendulum:** the control is a no-op. Against `frozen@nolatent`, changes are 0.1% to 0.2%, and train loss is flat (0.0227 to 0.0226). The floor is unidentified environment parameters, not training time. Every adaptive method beats the control by 18% to 88%, except context on oof (6.4%, NDD).
- **Puck:** the control is the best thing on interp, 0.435. It beats frozen@nolatent by 4.1% (NDD). It beats `rls@latent` by 10.2% (b_wins, fragile: flips leaving one seed out) and `hybrid` by 12.1% (b_wins). `latent_opt` trails by 8.4% (NDD, CI excludes 0). On puck more training helps more than any adaptation method, though the gains are small.
- **Thread C:** `full` vs matched MLP: pendulum interp 0.7% (NDD). Puck interp MLP is better by 7.8% (NDD, 5/5 seeds, CI excludes 0). Looping buys nothing at trained depth.

### Selected hyperparameters at grid edges

| Choice | Pendulum | Puck |
|---|---|---|
| RLS `lambda` (grid 1e-3 to 100) | 1e-3, lower edge, 5/5 | 100, upper edge, 5/5 |
| `rls@nolatent` | 1e-3 or 1e-2 | 100, 5/5 |
| Hybrid `lambda` | interior | 100, 5/5 |
| `latent_opt` `lr_scale` (0.5, 1, 2) | 2.0, upper edge, 5/5 | mixed |
| `grad_write` tuned | lr 1.0, passes 20, corner, 4/5 | lr 0.01, lowest lr, 4/5 |
| `last_block` | lr 3e-3, steps 40, corner, 5/5 | near the weakest setting |
| H10 forgetting `gamma` | 0.99, interior | 0.999, upper edge, 5/5 |
| H10 surprise threshold | 2.0, lower edge | 10, upper edge |
| Thread C K (4, 8) | 4, lower edge, tied with 8 (0.338 and 0.338) | 4, lower edge |
| Thread C lr (1e-3, 3e-4) | 1e-3 for every arm | 1e-3 for every arm |

What this does to fairness:

- Pendulum RLS and `latent_opt` are both pinned at their strongest setting, so the H1 interp gap could change with a wider grid. It is 64% and the validation score moves slowly, so a flip looks unlikely, but this is not measured.
- Puck RLS has no way to choose "no write", since the grid has no infinite `lambda` (`adapt.py:296`). At the best setting it is still worse than frozen on interp (5.1%, NDD, 5/5 seeds).
- Tuned gradient is at a corner, so it might gain more with more passes. That would make H3's kill stronger, not weaker.

### Compute

K = 1 adaptation per batch of 21 environments, 40 transitions. Prefer the counts. Wall-clock comes from parallel Actions jobs.

| Method | Forward calls | Backward calls | Writes | Wall (s) |
|---|---|---|---|---|
| frozen | 0 | 0 | 0 | 0.000 |
| context | 1 | 0 | 0 | 0.001 |
| RLS | 1 | 0 | 40 | 0.016-0.018 |
| grad 1 pass | 1 | 0 | 40 | 0.004 |
| grad tuned | 1 | 0 | 800 (pendulum) | 0.043 |
| last_block | 43 (puck seed 0: 13) | 40 (puck seed 0: 10) | 0 | 0.064 (puck 0.026) |
| latent_opt | 42 | 40 | 0 | 0.092-0.095 |
| hybrid | 43 | 40 | 40 | 0.106-0.115 |

Training wall time, pendulum and puck: nolatent 15 and 17 s, control 30 and 33 s, latent 26 and 28 s, context 41 and 44 s. Looped arms take 3000 updates at about 5.9e10 and 6.0e10 total forward MACs. `mlp_params` costs 2.8e10; its matched control takes 6452 and 6378 updates to reach the same MACs. Looped forward cost per step is about 78K MACs against 36K for `mlp_params`.

## Evidence per hypothesis

### H1. Capacity crossover: pendulum supported, puck not supported

a = `latent_opt`, b = RLS. Per-seed values are a - b.

| Family, tier | a | b | Diff, 95% CI | Reduction | Verdict | Per-seed diff |
|---|---|---|---|---|---|---|
| Pendulum interp | 0.0442 | 0.1237 | -0.0795 [-0.1138, -0.0522] | 64.3% | latent wins | -0.142, -0.034, -0.080, -0.061, -0.081 |
| Pendulum oof | 0.7176 | 0.3483 | +0.3693 [+0.3254, +0.4045] | 51.5% | RLS wins | +0.356, +0.421, +0.401, +0.290, +0.379 |
| Puck interp | 0.4752 | 0.4849 | -0.0097 [-0.0241, +0.0025] | 2.0% | NDD | -0.021, -0.002, +0.008, -0.033, -0.001 |
| Puck oof | 0.6571 | 0.6172 | +0.0399 [+0.0249, +0.0514] | 6.1% | NDD | +0.041, +0.055, +0.011, +0.050, +0.042 |

Registered reading, the better RLS chosen on validation:

- Pendulum picks `rls@latent` on 5/5, so the numbers do not change.
- Puck picks `rls@nolatent` on seeds 2 and 3. Interp: 0.4870, 2.4%, NDD. Oof: 0.6172, 6.1%, NDD.
- Against `rls@nolatent` alone, pendulum is 78.6% and 26.6% with the same winners.

Other horizons (reported, not deciding):

| Cell | h1 | h30 |
|---|---|---|
| Pendulum interp | latent 64.9% | latent 40.3% |
| Pendulum oof | RLS 28.3% | RLS 31.7% |
| Puck interp | 2.2% NDD | 7.6% NDD |
| Puck oof | 9.1% NDD | 1.8% NDD |

Closed-loop:

- Pendulum oof failure rate 0.657 vs 0.267, RLS wins (59.4%, CI [+0.286, +0.476]).
- Pendulum interp failure 0.010 vs 0.038, NDD.
- Pendulum extrap is invalid. Puck is at the floor.

Mechanism (registered "because the latent saturates"): pendulum z at the bound is 1.8% on interp, 62% on oof, 81% on extrap. This fits. It is an association, not a test of cause.

Oof mechanisms, descriptive only:

| Mechanism, h10 | latent_opt | RLS | Frozen |
|---|---|---|---|
| Pendulum action_flip | 0.994 | 0.514 | 1.137 |
| Pendulum torque_bias | 0.467 | 0.198 | 0.639 |
| Puck wind | 0.726 | 0.651 | 0.716 |
| Puck rotated_actions | 0.705 | 0.661 | 0.708 |
| Puck anisotropic_friction | 0.541 | 0.539 | 0.520 |

No single mechanism drives the pendulum outcome. RLS wins both. Closed-loop, `latent_opt` solves 0% of action_flip and 65% of torque_bias; RLS solves 58% and 87%. On puck, RLS gains on wind and rotated actions but not on anisotropic friction.

Kill: no method wins both tiers in either family. **Not fired.** Pendulum verdict is stable under t intervals and leaving one seed out.

Post hoc: at K = 3 pendulum widens on oof (80.0%) and narrows on interp (43.0%). Puck oof reaches 12.5%, RLS wins. On pendulum extrap (not part of H1's rule) RLS wins by 46.5%, so "inside family" means inside the training range.

### H2. Hybrid: pendulum killed, puck not supported

a = hybrid. "Better single" is by test mean, as in the code.

| Family, tier | Hybrid | Better single | Diff, CI | Reduction | Verdict | vs other single |
|---|---|---|---|---|---|---|
| Pendulum interp | 0.0560 | latent 0.0442 | +0.0118 [-0.0008, +0.0317] | 21.1% | NDD (fragile) | beats RLS 54.7% |
| Pendulum extrap | 0.2431 | RLS 0.2563 | -0.0133 [-0.0570, +0.0392] | 5.2% | NDD (fragile) | beats latent 49.3% |
| Pendulum oof | 0.5936 | RLS 0.3483 | +0.2453 [+0.1545, +0.3439] | 41.3% | **RLS wins** | beats latent 17.3% |
| Puck interp | 0.4955 | latent 0.4752 | +0.0203 [+0.0169, +0.0247] | 4.1% | NDD | RLS 2.1% NDD |
| Puck extrap | 0.6193 | latent 0.6038 | +0.0155 [+0.0094, +0.0249] | 2.5% | NDD | RLS 0.8% NDD |
| Puck oof | 0.6221 | RLS 0.6172 | +0.0049 [-0.0006, +0.0105] | 0.8% | NDD | latent 5.3% NDD |

- Pendulum oof per-seed: +0.126, +0.428, +0.142, +0.236, +0.295. Stable on all checks. **Kill fired.**
- It beats both singles in no tier on either family, so the positive claim also fails.
- Closed-loop agrees on oof: 0.62 vs 0.73.
- **Ambiguity:** "worse". By the effect rule, puck is not worse. If any CI that excludes 0 counts as worse, puck interp and extrap also fire the kill, with 5/5 seeds and small effects. I report the protocol reading: not supported, kill not fired.

Caveat on the pendulum kill, descriptive: hybrid's oof error follows its selected `lambda`.

| Seed | Hybrid lambda | Hybrid oof | RLS oof |
|---|---|---|---|
| 0 | 0.01 | 0.427 | 0.301 |
| 1 | 10 | 0.728 | 0.301 |
| 2 | 0.01 | 0.460 | 0.318 |
| 3 | 0.1 | 0.706 | 0.470 |
| 4 | 1 | 0.646 | 0.352 |

Validation mixes in-family data, where the latent has already done the job, so it prefers a weak write. Even the strongest selected write is worse than RLS. Hybrid also has a fixed `lr_scale` of 1.0 (`adapt.py:395-396`), while `latent_opt` was free to pick 2.0. The kill stands as a rule, but this says little about the best possible hybrid.

### H3. RLS vs gradient writes: killed, and half of it was not tested

"Smooth" data, h10. a = RLS.

| Family, tier | RLS | 1-pass | Reduction, verdict | Tuned | Reduction, verdict |
|---|---|---|---|---|---|
| Pendulum interp | 0.1505 | 0.2543 | 40.8%, RLS wins | 0.1518 | **0.9%, NDD**, CI [-0.0445, +0.0419] |
| Pendulum extrap | 0.2540 | 0.6100 | 58.4%, RLS wins | 0.3549 | 28.4%, RLS wins |
| Pendulum oof | 0.3745 | 0.7862 | 52.4%, RLS wins | 0.5826 | 35.7%, RLS wins |
| Puck interp | 0.4961 | 0.4640 | 6.5%, NDD (1-pass ahead, 5/5) | 0.4751 | **4.2%, NDD** |
| Puck extrap | 0.6245 | 0.6123 | 1.9%, NDD | 0.6196 | **0.8%, NDD** |
| Puck oof | 0.6239 | 0.6358 | 1.9%, NDD | 0.6323 | **1.3%, NDD** |

Pendulum tuned, interp per-seed: +0.023, +0.050, -0.063, -0.060, +0.043 in the RLS minus tuned sense, so there is no direction.

**Ambiguity in the first kill clause ("tuned gradient writes match RLS"):** it names no tier.

- Any tier, or the in-family tier: fired in both families.
- All tiers must match: fired on puck, not on pendulum, where RLS holds on shifted data at 1/20 of the writes.

**Second clause (gap depends on collinearity): the test did not test this.**

| Measure, interp | Pendulum smooth / random_action / iid_state | Puck smooth / random_action / iid_state |
|---|---|---|
| Participation ratio (higher = less collinear) | 3.35 / 3.38 / 3.09 | 2.60 / 1.89 / 2.57 |
| Top eigenvalue fraction | 0.437 / 0.445 / 0.477 | 0.595 / 0.721 / 0.609 |
| log10 regularized condition number | 4.67 / 4.57 / 4.57 | 5.60 / 5.57 / 5.58 |
| log10 nonzero condition number | 9.26 / 8.94 / 8.07 | 6.81 / 6.40 / 4.74 |

- Participation ratio does not rise meaningfully in any condition on interp (largest change up: 3.35 to 3.38), and on puck `random_action` is clearly more collinear. Extrap and oof tiers look the same.
- The regularized condition number cannot work here. With 40 transitions and 64 features, `Phi'Phi` always has an exact zero eigenvalue, so it reduces to `(eig_max + 1e-3) / 1e-3` (`eval.py:243`), which measures feature scale.
- The nonzero condition number falls, but it sits near its 1e-10 cutoff on pendulum and disagrees with the other measures.
- What did change is coverage. RLS on pendulum interp goes from 0.150 to 0.083 to 0.047 across the three conditions.

So the conditions moved data coverage, not conditioning. The stored `gap_vs_smooth` results should not be read as evidence about collinearity. For the record, the relative 1-pass gap on pendulum goes from 0.88 to 1.46 (NDD) and 2.58 (grew, CI [+1.11, +2.25]). The absolute gap is NDD for both on interp. Signs vary by tier. It does not shrink as predicted, but it could not fairly have done so.

Puck "smooth" is actually the push probe with noise (`thread_a.py:178-179`, `envs.py:323`). On puck all RLS and gradient variants are within about 11% of frozen, so this family says nothing about the question.

Checks that passed: `rls_order_invariance` is 0.0, with all 105 per-environment values identical at the stored precision, while gradient values change under shuffling. So shuffling is real and RLS order invariance holds.

### H4. Calibration: pendulum supported, puck killed under the revised rule

Primary cell: selected backbone, `lambda` from the rollout rule, K = 1, within-environment.

| Family, tier | Spearman P, CI | Reference, half | `\|\|phi\|\|^2` | Partial given phi, CI | 1 - rho reduction | Seeds |
|---|---|---|---|---|---|---|
| Pendulum interp | 0.836 [0.822, 0.851] | 0.735, 0.367 | 0.293 | 0.797 [0.783, 0.812] | 76.9%, P wins | 0.84, 0.84, 0.86, 0.83, 0.81 |
| Pendulum extrap | 0.833 [0.811, 0.855] | 0.670, 0.335 | 0.229 | 0.804 [0.784, 0.820] | 78.4%, P wins | 0.82, 0.84, 0.87, 0.85, 0.80 |
| Pendulum oof | 0.850 [0.843, 0.856] | 0.778, 0.389 | 0.319 | 0.803 [0.791, 0.815] | 78.0%, P wins | 0.85, 0.86, 0.85, 0.84, 0.85 |
| Puck interp | 0.279 [0.225, 0.339] | 0.062, 0.031 | 0.233 | 0.171 [0.138, 0.207] | **6.0%, NDD** | 0.31, 0.39, 0.28, 0.21, 0.22 |
| Puck extrap | 0.379 [0.335, 0.419] | 0.073, 0.036 | 0.330 | 0.202 [0.162, 0.235] | 7.3%, NDD | 0.42, 0.40, 0.30, 0.43, 0.35 |
| Puck oof | 0.344 [0.297, 0.388] | 0.064, 0.032 | 0.268 | 0.250 [0.214, 0.286] | 10.4%, P wins | 0.41, 0.38, 0.33, 0.34, 0.26 |

Pendulum clears every threshold, raw and revised, on every backbone (`nolatent`: 0.75 and 0.74) and every seed.

Puck against each test:

| Test | Result on puck interp |
|---|---|
| Revised: below half of reference | Passes, 0.279 vs 0.031. No force: at `lambda` = 100, `q` median is 0.06, so the reference is near 0 |
| Revised: P adds nothing beyond `\|\|phi\|\|^2`, as fixed in `thread_b.py` header (paired win on 1 - rho **and** partial CI above 0) | **Fires.** 6.0%, 5.0% (latent), 8.9% (nolatent). All NDD, stable. Nolatent is close to threshold |
| Same words, read as partial CI above 0 only | Does not fire. Partial is 0.171, CI [0.138, 0.207] |
| Raw kill below 0.3 (still reported) | 0.279 is below, but the CI straddles it. Latent 0.314, nolatent 0.242. Sensitive: by protocol this needs more seeds |
| Raw target 0.5 | Missed on all puck tiers |

Stored `outcome` for puck is "killed" for the wrong reason: the old 0.3 threshold (`thread_b.py:620`).

Two things undercut even the charitable puck reading:

- P adds almost nothing once state size is known. Partial given phi and input norm is 0.018 on interp. Simple signals beat it: input norm 0.553, predicted norm 0.464.
- The `val_calibration` sensitivity picks `lambda` = 1e-3 and passes everything (0.517, outcome "supported"), but one-step RMSE is 3.00 against 0.38 for frozen, 8 times worse, and predicted norm ranks errors at 0.946. P flags errors that the write itself created. That "support" should not be used.

Magnitudes, as distinct from ranks (coverage at nominal 0.5 and 0.9):

| Family, tier | P model coverage | NLL, P vs constant |
|---|---|---|
| Pendulum interp | 0.77 / 0.95 | -2.33 vs -1.14, P better |
| Pendulum extrap | 0.49 / 0.72 | 3.50 vs 1.08, **P worse than constant** |
| Pendulum oof | 0.47 / 0.70 | 1.34 vs 3.88, P better |
| Puck interp | 0.79 / 0.94 | 0.072 vs 0.094 |
| Puck oof | 0.65 / 0.86 | 0.524 vs 0.558 |

Pendulum P ranks well under shift, and its scale does not hold there. "Calibrated" is true of ranking only.

Registered expectation of weakening out-of-family: **wrong for ranking in both families** (oof 0.850 vs interp 0.836; 0.344 vs 0.279). Mechanisms are descriptive: pendulum 0.85 and 0.85, puck wind 0.24, friction 0.36, rotated 0.46.

Pendulum Spearman exceeds its well-specified reference by 14%. That is allowed, since misfit that scales with `q` can exceed the reference. I do not read it as a bug: pendulum at `lambda` = 1e-3 has `q` median 5 to 32, so many queries likely fall in directions the 40 transitions never touched.

Across-environment, reported only: pendulum P 0.76, phi 0.57; puck P 0.54, phi 0.69.

Consequence: H5 and H6 end "as stated" on puck only. On pendulum they may proceed, using P for ranking and not for interval widths under shift.

### H7. Loops x rollout: inconclusive, not killed

"Stable" = under 2x the full recipe's error and under 5% of rollouts non-finite or clipped.

Trained loops (K = 4), h100, interp. Ratio to `full`, blow-up fraction:

| Arm | Pendulum (full 1.158, blow 0.00) | Puck (full 0.810, blow 0.00) | Reading |
|---|---|---|---|
| `plain` | 3.13x, 1.00 | 13.05x, 1.00 | Not stable |
| `no_norm` | 3.21x, 1.00 | 12.01x, 1.00 | At least 2x, 5/5 seeds |
| `no_reinject` | 0.99x, 0.00 | 1.00x, 0.00 | Under 2x; saturated, so inconclusive |
| `no_mixed` | 0.99x, 0.00 | 1.04x, 0.01 | Under 2x; saturated, so inconclusive |

The same pattern holds on extrap and oof (`plain` 2.47x to 12.28x). At h10, which is not saturated, `no_reinject` and `no_mixed` are within 1.7% on interp, so they are not hidden by saturation at this depth. They just do not matter at K = 4.

Product cell (64 loops, h100, interp, 84 rollouts per seed):

| Arm | Pendulum (full 1.164, blow 0.00) | Puck (full 0.875, blow 0.01) |
|---|---|---|
| `plain` | 3.28x, 1.00 | 12.14x, 1.00 |
| `no_norm` | 3.28x, 1.00 | 12.13x, 1.00 |
| `no_reinject` | 2.61x (2.16-3.18), 1.00 | 11.31x (9.36-12.75), 1.00 |
| `no_mixed` | 1.20x (1.10-1.36), 0.095 (0.01-0.27) | 1.96x (1.51-2.35), 0.26 (0.00-1.00) |

**Ambiguity.** `thread_c.py` fixes trained loops as the decision cell and calls the product cell separate. The hypothesis itself is about "many inner loops", and K = 4 was the lower edge of the grid. Both readings:

- Kill ("plain core stable under the product"): **not fired in any cell, on either reading.**
- Claim ("every ablation at least 2x"):
  - Trained loops: true only of `no_norm`. The others are saturated with `two_x_reachable` false, so inconclusive by the registered rule.
  - Product: true of `no_reinject` and `no_norm`. Not of `no_mixed`, though `no_mixed` fails "stable" through blow-up on average (9.5% and 26%).
  - Puck `no_mixed` at 48.9% against the 50% bar is **fragile**: 3 of 5 leave-one-out runs stay NDD, 2 flip, and per-seed blow-up runs from 0 to 1.0.

The stored `not_supported_some_ablations_under_2x` ignores the saturation rule (Problem 2).

The live contradiction with 2610.31943 goes against removing normalization in this setup: `no_norm` is 45.5% worse at h10 on pendulum and 14.6% on puck. Its teacher-forced loss on pendulum is lower (0.021 vs 0.024) and h1 is close (0.025 vs 0.023), so one-step training metrics hide this failure.

Validity limits, each serious:

- **Carried state is nearly inert.** Carry sensitivity of `full` is 0.007 on pendulum and 0.026 on puck, in delta-std units. `reinit` (no carry) matches `full` at h10 and h100 on all tiers (at most 2.9%). At 64 loops, `reinit` does differ (puck 1.44x, 0.01 blow-up; pendulum 1.07x, 0.21 blow-up), but `full` is close to memoryless in rollouts, so the "with carried state" part of H7 is weakly tested.
- No arm combines no normalization with no carry, so normalization's effect cannot be separated from carry-age extrapolation (training windows are 4 steps).
- `plain` and `no_norm` give identical errors at 64 x 100 on pendulum (3.699, 3.822, 3.876, 3.943, 3.768). This is not a no-op: rollouts sit at the clip bound (`models.py:131-132`, nonfinite fraction about 1.0). Beyond blow-up, the ratio measures the bound, not the arm.
- `full` and MLPs of equal parameters or compute are within 8.6% at h10 and h100 on every tier, all NDD.
- Closed-loop: pendulum 0.77 for all arms, puck 0.00 for all, NDD throughout.

### H8. Action injection: killed

a = every loop (`full`), b = prelude only.

| Family, tier | a | b | Diff, CI | Reduction | Verdict |
|---|---|---|---|---|---|
| **Puck interp (decisive)** | 0.5260 | 0.5172 | +0.0089 [-0.0045, +0.0250] | 1.7% | NDD |
| Puck extrap | 0.6137 | 0.6114 | +0.0023 [-0.0054, +0.0102] | 0.4% | NDD |
| Puck oof | 0.6642 | 0.6551 | +0.0091 [+0.0022, +0.0171] | 1.4% | NDD |
| Pendulum interp | 0.3766 | 0.3796 | -0.0030 [-0.0051, -0.0006] | 0.8% | NDD |
| Puck interp, contact windows (descriptive) | 0.7635 | 0.7490 | +0.0146 [-0.0055, +0.0394] | 1.9% | NDD |
| Puck interp, free windows (descriptive) | 0.2217 | 0.2231 | -0.0014 [-0.0069, +0.0037] | 0.6% | NDD |

Puck interp per-seed: -0.001, +0.014, +0.040, +0.002, -0.011. No sign consistency, and if anything prelude-only is ahead, including on contact. Closed-loop: NDD (0.77 and 0.00). **Kill fired. Stable.**

Not a no-op: forward outputs differ (0.056 at the same weights) and action sensitivity is 0.44 and 0.43, so the prelude keeps the action through 4 loops.

Post hoc, above trained depth, re-injection does matter. Prelude-only at 64 loops, h10, interp: puck 1.072 vs 0.522 (51.3%); pendulum 0.920 vs 0.290 (68.5%). At 16 loops: 30.6% and 42.1%. All 5/5. This was not the registered cell and should not revive H8 without its own pre-registered test.

### H9. Loop residual as a surprise signal: killed on every target

a = final-loop residual, b = simplified Gaussian baseline, chosen on validation (`gauss_nll` on all seeds). This baseline is a fixed Gaussian with CUSUM, **not ARCADE**.

| Family, target | Residual AUROC | Per-seed | Baseline | Per-seed | Below 0.7? | Gain? |
|---|---|---|---|---|---|---|
| Pendulum shift, pooled | 0.521 | 0.51, 0.53, 0.52, 0.52, 0.53 | 0.810 | 0.82, 0.81, 0.82, 0.80, 0.81 | Yes | No, baseline wins 60.2% |
| Pendulum shift, extrap | 0.527 | | 0.821 | | Yes | No |
| Pendulum shift, oof | 0.516 | | 0.799 | | Yes | No |
| Puck shift, pooled | 0.484 | 0.49, 0.48, 0.49, 0.49, 0.47 | 0.605 | 0.61, 0.59, 0.62, 0.62, 0.60 | Yes | No, baseline wins 23.5% |
| Puck shift, extrap | 0.493 | | 0.566 | | Yes | No |
| Puck shift, oof | 0.474 | | 0.645 | | Yes | No |
| Puck contact onset | 0.601 | 0.59, 0.56, 0.66, 0.62, 0.57 | 0.908 | 0.89, 0.91, 0.90, 0.93, 0.91 | Yes | No, baseline wins 76.9% |

Both kill conditions fire on all targets and seeds. No other arm reaches 0.7 (best is 0.65, `no_mixed` on contact). No fragility. The kill does not depend on ARCADE, since the residual already loses to a weaker baseline. Beating or losing to ARCADE itself is untested. Allocating loops and gating writes stay untested, as registered.

Descriptive, by mechanism: pendulum action_flip 0.51 vs 0.91, torque_bias 0.52 vs 0.69. Puck anisotropic friction 0.49 vs 0.51, rotated 0.47 vs 0.78, wind 0.47 vs 0.64. Contact lead (advance warning): 0.59 vs 0.70.

Baseline note: on puck extrap the baseline is itself weak (0.566) likely because mass and friction changes barely register against a model that underfits contact.

### H10. Forgetting: killed; reset beats forgetting

Prequential one-step RMSE. "F vs R" is forgetting against per-episode reset, with reset ahead unless shown. Interp row values are forgetting / reset / persist.

Pendulum (`gamma` 0.99, `lambda` 0.1 for forgetting, 0.01 for reset, 0.1 for persist; 0 failures, not tainted):

| Schedule | Interp F / R / P | F vs R interp | F vs R extrap | F vs R oof | F vs P, 3 tiers |
|---|---|---|---|---|---|
| fast | 0.100 / 0.075 / 0.109 | 25.2% | 15.5% | 16.0% | NDD (8.1, 8.1, 6.9%) |
| medium | 0.092 / 0.059 / 0.108 | 36.5% | 29.0% | 25.1% | F wins (14.8, 13.6, 15.2%) |
| slow | 0.069 / 0.039 / 0.094 | 42.9% | 31.9% | 28.2% | F wins (27.2, 28.8, 26.0%) |
| varied | 0.081 / 0.052 / 0.099 | 35.8% | 24.9% | 23.4% | F wins (18.0, 18.4, 18.8%) |
| return | 0.062 / 0.031 / 0.070 | 50.1% | 43.4% | 33.2% | F wins (10.8, 11.7, 12.8%) |
| aligned | 0.093 / 0.035 / 0.106 | 62.0% | 54.6% | 51.8% | F wins (12.8, 12.5, 13.6%) |

All 18 reset wins exclude 0 with 5/5 seeds. Example per-seed differences, interp fast: +0.028, +0.025, +0.026, +0.024, +0.024. No-switch control: forgetting beats reset (29.8%, 62.8%, 47.6%), so the switching result is not a general weakness in forgetting.

Puck: **vacuous.** `lambda` = 100 and `gamma` = 0.999 for all seeds. Forgetting, reset, and persist are within 3.1% on every switching cell (interp fast 0.525, 0.525, 0.525, frozen 0.521). All NDD. Frozen beats every write on interp and is within about 7% of them elsewhere. P trace stays at about 1.

**Kill wording.** "Reset matches forgetting in every switching schedule." On puck they match. On pendulum reset wins outright, which fails the hypothesis more strongly than matching would. **Fired on both.** Forgetting beats reset in 0 of 36 switching cells.

"Persist is worst just after a switch" (first 20 steps):

- Pendulum: worst of the three by mean in every cell (18/18).
- By the effect rule, requiring wins by forgetting, reset, and surprise reset: 5 of 18 (interp slow and aligned, extrap slow, oof slow and varied). The gap to forgetting is often 3% to 9%.
- Puck: never.

Validity: no numerical failures or forced resets in any forgetting cell. P trace for forgetting peaks at about 8.7e5 against 6.3e2 for persist, far below the 1e12 limit. Prediction clipping is zero.

Horizon 10, reported only: gaps close sharply. Forgetting vs reset is NDD in most extrap and oof cells. On interp fast, every write (0.44 to 0.45) is worse than frozen (0.415), while one-step RMSE is better than frozen. One-step wins do not carry over reliably to rollouts in fast switching.

Post hoc: `surprise_reset`, in the code but not in the hypothesis, beats forgetting in all 18 pendulum cells. Against episode reset it wins fast, medium, slow, and varied (10.9% to 40.4%), and loses on aligned, as it should, and on interp return. Its threshold is pinned at 2.0, the lower edge.

Scope: this tests plain exponential forgetting, whose prior decays as `gamma^t` (`adapt.py:41-44`), so forgetting also forgets W0, while reset returns to it. Forgetting that keeps the prior is untested.

## Fragility to 5 seeds

No decisive verdict flips under a t interval. These sit near a threshold:

| Cell | Value | Note |
|---|---|---|
| H7 product, puck `no_mixed` | 48.9% vs 50% bar | Leave-one-out flips 2 of 5 |
| H4 raw kill, puck interp | 0.279, CI [0.225, 0.339] vs 0.3 | Revised rule does not depend on it |
| H4 adds-beyond-norm, puck nolatent | 8.9% and 9.3% vs 10% | Leave-one-out stable; backbone choice matters |
| Puck, `rls@latent` vs control | 10.2% vs 10% | Flips leaving one seed out. Post hoc |
| H2 pendulum interp and extrap | NDD | Leave-one-out unstable, but the kill rests on oof |

Protocol response is to add seeds, and that would be worthwhile only for the first row.

## Problems found

Code that computed the metrics held up: no scoring bug changes a verdict. These are validity issues.

1. **Puck model cannot support the tests.** Contact is underfit, the latent is unused, closed-loop is at the floor, and every method picks its weakest write. Puck nulls on H1, H2, H3, H10 are uninformative, and puck closed-loop results have no resolution.
2. **H7 outcome ignores the saturation rule.** `looped.py:935-951` takes no `two_x_reachable`, and `thread_c.py:847` does not pass it. Stored "not_supported" should be inconclusive under `HYPOTHESES.md`.
3. **H4 outcome uses the superseded kill.** `thread_b.py:620` kills on raw 0.3 and never tests half-reference or adds-beyond-norm as a kill. Stored outcomes are wrong in reason, and in some cells in result: `latent` puck is "below_target", where the revised rule gives a kill.
4. **H1 and H2 outcomes use `rls@latent` only.** `thread_a.py:422,427-428,445-448` skip the registered better-of-two chosen on validation. H2 also picks the "better single" by test mean. No verdict changes, but the summary fields are not the registered rule.
5. **H3 collinearity manipulation failed, and one measure is degenerate.** `eval.py:243` with N = 40 below F = 64 reports feature scale. `feature_collinearity` was built for this and its own docstring notes singularity, but the conditions were never checked to change collinearity before scoring.
6. **Carried state is inert in the full recipe** (`sensitivity` in `looped.py:640-654`). H7's product test and H9's residual both rely on memory that barely exists. There is also a structural reason for H9: the loop sees `o` and `a` but never its own prediction error, so a residual cannot see surprise except through unusual inputs.
7. **Grid edges.** See the table above: `adapt.py:296` and `:288`, `:312`, `:320`, `:348`; `thread_c.py:133-134`; `thread_b.py:157-158`. RLS has no "no write" option.
8. **Thread C selection is short and coarse.** 600 steps (`thread_c.py:135`) for arms trained 3000 (`looped.py:269`), which favors the larger lr. K has two values and tied on pendulum. "Many inner loops" was therefore never trained.
9. **Oracle gate is tier-level only, and its cutoff is unregistered.** `thread_a.py:287-295` stores only the mean, so per-task gating is impossible. `thread_c.py:914` hardcodes 0.5, which `HYPOTHESES.md` does not state. I treated pendulum extrap (0.33) as invalid either way.
10. **Clip bound dominates blow-up numbers** (`models.py:131-132`, `looped.py:563`). A 13x ratio on puck reflects observation bounds, not a degree of badness. Use blow-up fractions there.
11. **H4's reference rule has no force at large `lambda`.** With `q` about 0.06, half the reference is 0.03. And the `val_calibration` route "supports" H4 on a model 8x worse than frozen (`thread_b.py:366-370`).
12. **One trajectory draw is a large unmodeled noise source.** `thread_a.py:279` draws 3 trajectories and cuts to K, while `:179` draws 1, so the same RLS on the same environments gets different data. Pendulum interp is 0.124 in one and 0.150 in the other, with per-seed changes up to 2x (0.072 to 0.145). So H1 and H3 numbers are not interchangeable.
13. **Coverage gaps against the protocol.** H7 grid is interp only, with 84 rollouts per seed (`thread_c.py:150-151`). Closed-loop in Thread C is interp and a subset of arms, with no carry across real steps. H3, H4, and H10 have no closed-loop, as registered for H4 and H10. H3 is the unregistered gap.
14. **Minor.** Python 3.13.17 and 3.13.18 are mixed across jobs. Thread C pendulum selection scores differ across jobs with the same choices, so that step is not bit-reproducible. `thread_a.py:19` still says `HYPOTHESES.md` names no primary horizon. Pendulum oracle gates here (extrap 0.24 to 0.43) differ from the figures in `MISSION_STATE.md` (0.38 to 0.48).

Checks that passed: source hashes, bootstrap reproduction, RLS order invariance, `check_ablations` (no forward no-op), zero skipped training updates in looped arms, zero numerical failures in H10, no selection on test tiers found in the code read.

## What this implies

### Results (what the data show)

- On pendulum, a latent wins inside the training family, an RLS write wins outside it, and each is robust on every check. This is one family with a working model.
- Latent-then-RLS with validation-chosen `lambda` is worse than RLS out-of-family and adds nothing elsewhere.
- Tuned gradient writes match RLS in-family at about 20 times the writes. RLS is ahead with one pass, and on shifted pendulum data even against tuned.
- `phi' P phi` ranks one-step error very well on pendulum in all tiers (about 0.84). It does not give correct magnitudes under shift.
- Per-episode reset beats plain forgetting everywhere switching is detectable in these streams.
- The fixed-point residual is near chance as a detector.
- Normalization is required for stable carried rollouts in this core. Action re-injection has no effect at K = 4 and a large effect at 16 or more loops.

### Inference (my reading, not measured)

- Capacity matching remains a live explanation, with one supporting family. It has not been tested on a second system where both methods work.
- The strongest driver of RLS quality in Thread A looks like data coverage rather than conditioning (3.2x from iid-state data, with collinearity unchanged). This is post hoc and points at probing, the subject of H5, which should be pre-registered in those terms.
- Reset beating forgetting may come from returning to a good prior, not from discarding old data. A forgetting variant that keeps the prior would test this.
- At K = 4 the loops do no work a same-size MLP does not do. Both reach the same floor because neither identifies the environment. Thread C's arms only separate above trained depth, so a fair test needs deeper training first.

### Order of next steps (recommendation)

1. Fix the puck model before spending anything on puck verdicts. Check contact fit and whether a latent can reduce loss at all, then gate on a minimum closed-loop rate before re-running.
2. Correct the outcome fields for H1, H2, H4, H7 so stored verdicts equal registered rules, in their own commit.
3. Register a real collinearity manipulation for H3, for example controlling N against F, with a check that the measures move, or retire that clause.
4. H5 and H6 can proceed on pendulum only, on ranking, with the coverage caveat above.

Changes to any threshold or prediction need their own commit per `MISSION_STATE.md`.

## Thread P: pendulum on pixels (run 41841227508, seeds 10-19)

Fingerprint bfee057fec6212ac, commit 83be5d1, on GitHub Actions CPU runners, not a GPU, with no setting changed. Raw records are in
`results/thread_p/pendulum_41841227508/`. Own renderer and small models, not a published benchmark. Pendulum only: drone and puck failed Gate GP and give no
verdict. Gate GP passes on every seed (frozen 0.807, oracle 0.995, reduction 37.9%). An independent audit recomputed 105 comparisons from the per-seed records
and found no numeric or verdict disagreement with `summary.json`.

Decoded physical error at horizon 10, K = 1 decides. a vs b, lower is better; reduction is relative to the worse side.

| Hypothesis | Deciding comparison | a | b | 95% CI of a - b | Reduction | Outcome |
|---|---|---|---|---|---|---|
| H1p | latent vs better RLS, interp | 0.251 | 0.401 | [-0.163, -0.138] | 37.3%, latent, 10/10 seeds | Killed: latent wins both tiers |
| H1p | latent vs better RLS, oof | 0.718 | 0.817 | [-0.110, -0.087] | 12.1%, latent, 10/10 seeds | |
| H12p | hybrid vs better single, interp | 0.253 | 0.251 | [+0.000, +0.003] | 0.6%, hybrid worse | Supported, by the 10% threshold only |
| H12p | same, extrap | 0.496 | 0.536 | [-0.048, -0.035] | 7.6%, hybrid better | |
| H12p | same, oof | 0.778 | 0.718 | [+0.046, +0.076] | 7.8%, hybrid worse, 0/10 seeds better | |
| H4p | P vs phi on 1 - Spearman, interp | 0.766 | 0.775 | [-0.020, +0.000] | 1.1% | Killed: does not add beyond the norm |
| H3p | RLS vs tuned gradient, interp, extrap, oof | | | | 2.3%, 2.4%, 2.7%, RLS | Not killed; the expected RLS win on shift did not appear |
| H5p | D-optimal vs random, interp, extrap, oof | | | | 1.2% better, 2.0% worse, 2.8% worse | Killed: coverage moved, no error gain |
| H14p | error-fed vs no-error, interp | 0.455 | 0.477 | [-0.031, -0.016] | 4.8% | Killed |

Caveats that limit these verdicts:
- **The round-one crossover did not replicate on pixels at K = 1.** At K = 3 latent is still better on oof in 10/10 seeds (0.702 vs 0.757, 7.2%), under the threshold, so that
  label is no crossover, not a kill. RLS never wins.
- **RLS is close to frozen at K = 1:** 1.4% on interp and 0.7% on oof, both no detectable difference, with 40 transitions for a 64-feature readout. H1p on oof is nearly latent vs frozen.
- **Open loop is not closed loop.** Interp success: oracle 0.995, frozen 0.807, RLS 0.783, latent 0.733. Oof: oracle 0.971, frozen 0.310, RLS 0.310, latent 0.260.
  Latent's lower open-loop error does not produce better control, and the cause is unknown. Differences from RLS are under the effect rule.
- **The oof tier keeps training-range hidden-parameter variation,** so latent can gain there through ordinary inference. By mechanism (descriptive): torque bias 17.7%, action flip 8.9%. No arm separates these effects.
- **H12p is the weakest verdict:** hybrid is worse on oof in all 10 seeds, and worse than frozen on action flip (1.078 vs 1.022).
- **H14p's gain looks like a training difference:** most of the gap to no-error also appears without any belief. All error-fed arms are worse than frozen on interp, with a different core and fewer steps.
- **H3p's gradient grid** sits at an edge in 7 of 10 seeds, which can only handicap gradient.
- **Metric saturation** compresses shifted tiers at horizon 10. It does not hide an RLS win: latent leads at horizons 1 and 10 in all seeds.
