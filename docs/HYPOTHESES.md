# Pre-registered hypotheses

Written before any experiment ran. The commit history is the record: if a prediction or threshold
changes after results exist, the change gets its own commit and a note saying why.

Sources for every cited claim: `docs/lit/test_time.md`, `docs/lit/looped.md`, `docs/lit/cross_domain.md`.

## The connecting idea

Recent test-time methods look different but differ mainly in one choice: how many degrees of freedom
get adapted, and how.

| Method | What adapts | Size | Update |
|---|---|---|---|
| T2-Z | environment latent | 32 numbers | 40 gradient steps |
| Sandwich-Residuals | residual adapters | ~100K | 1 gradient step per replan |
| AdaJEPA / OpTTA | last predictor block | subset of weights | 1 gradient step per replan |
| World-T3 | fast-weight readout | one matrix per layer | closed-form RLS |
| LaCT | fast weights | up to 40% of params | Muon |

The notes record an unresolved contradiction: T2-Z finds 32 numbers beat fast weights, while LaCT finds
more fast-weight state is consistently better. Proposed reconciliation: **capacity should match what the
data identifies and what the shift requires.** A low-dimensional latent can only reach shifts inside the
family seen in training. A readout write can reach outside that family but needs more data.

Second observation: the RLS write in World-T3 is the exact posterior-mean update of Bayesian linear
regression on the fast-weight features. With prior `W ~ N(W0, sigma^2/lambda I)` and iid noise, `P` is
the posterior covariance up to `sigma^2`, so `phi' P phi` is epistemic variance at a query. World-T3 tracks
`P` only as an update coefficient. If it is calibrated, one matrix addresses three gaps in the notes: no
paper calibrates adapted-model uncertainty, nobody knows which probe action is informative (T2-Z), and
OpTTA's optimism has no bound and no infeasible-goal test.

This identity is not new: ALPaCA (1807.08912) states it, and Neural-Fly (2205.06908) and Davydov
(2411.00107) use it. What appears to be new is applying it to the recent test-time world-model methods and
comparing it directly with latent optimization. Full check in `docs/lit/prior_art.md`.

### Status after the prior-art check (revised before any results existed)

| Hypothesis | Status | Effect on the plan |
|---|---|---|
| H1 capacity crossover | open | main contribution |
| H2 hybrid | open | main contribution |
| H3 RLS vs gradient | partly answered | keep; the collinearity-controlled test is the new part |
| H4 calibration | formula known | keep; calibration under each shift tier is what is measured |
| H5 active probing | method exists | replication; the comparison against random probes is the new part |
| H6 bounded optimism | parts exist separately | keep; the combination in test-time adaptation is untried |
| H7 loops x rollout | partly answered | keep; no ablation found |
| H8 action injection | design exists | keep; the only controlled ablation found is in language models |
| H9 loop residual signal | open | must beat ARCADE's predictive-likelihood changepoint signal |
| H10 forgetting | mostly answered | replication against ARCADE (2512.14331); plain forgetting vs reset vs persist is the new part |

## Protocol for every thread

These respond directly to 2610.19375, which reports untouched baselines varying by up to 26 points across
evaluation seeds.

- At least 5 training seeds; at least 20 held-out environments per condition; evaluation seeds paired
  across methods.
- Report the paired difference with a bootstrap 95% CI over training seeds, plus per-seed spread.
- **Effect threshold:** a method "wins" only if the paired CI excludes 0 and the error reduction is at
  least 10%. Otherwise the result is reported as no detectable difference.
- Include a matched continued-training control wherever a method adds training-time cost.
- Choose hyperparameters (`lambda`, learning rates, step counts) on validation environments only, never on
  test shifts.
- Report both open-loop rollout error and closed-loop control success, since the notes show they can move
  apart.
- Report compute (wall-clock, forward and backward passes) with every comparison.
- Runs happen in the cloud only. Nothing is run locally.
- Scope limit: results are from small state-based simulators on CPU. They do not by themselves support
  claims about pixels, large models, or real robots, and write-ups must say so.

### Implementation decisions fixed before results

- **Primary metric:** open-loop rollout error at horizon 10. Horizons 1 and 30 and closed-loop success are
  reported but do not decide verdicts on their own.
- **H3 collinearity conditions:** shuffling trajectories leaves `Phi'Phi` unchanged, so it cannot reduce
  collinearity. H3 instead compares on-trajectory data against random-action and iid-state data, reporting a
  regularized condition number and participation ratio.
- **Gradient writes** use normalized steps (NLMS), which favors the gradient side, so an RLS win under H3 is
  conservative.
- **Latent size** is 8 rather than T2-Z's 32, since these systems have far fewer hidden parameters.
- **Latent-free RLS:** `rls` runs on the mean training latent, which may handicap it, so `rls@nolatent` is also
  reported. H1 is judged on the better of the two, chosen on validation environments.
- **Sanity gates:** if the oracle planner cannot solve a task, closed-loop results on that task are invalid and
  are not reported as evidence.
- **Known weakness:** the out-of-family tier mixes mechanisms, so per-mechanism breakdowns have no confidence
  intervals and are descriptive only. Test environments differ per training seed.
- **Bootstrap resolution:** 5 seeds allow only 126 distinct resamples, so the CI is coarse. If a verdict is
  sensitive to this, add seeds rather than reinterpreting.

H4 and H10, fixed before any run of them:

- **H4 thresholds revised.** The original raw Spearman thresholds (0.5 target, 0.3 kill) are not meaningful alone.
  Squared error is chi-square noisy even for a perfectly specified model, so those values may be unreachable for
  any method. Judge against `well_specified_reference`, the Spearman expected if the Bayesian model were true.
  - Kill, under interpolation, if within-environment Spearman is below half that reference, or if `P` adds nothing
    beyond the `||phi||^2` baseline.
  - The raw values are still reported, including against 0.3 and 0.5.
- **H4 primary:** within-environment ranking. Across-environment ranking is structurally weak because `P` never
  sees targets, so a poor result there is expected and does not kill H4.
- **Primary metric exception:** H4 and H10 verdicts use one-step error, since that is what the variance and
  prequential scores refer to. Horizon 10 is reported for H10. Neither runs closed-loop.
- **H10 validity:** a win that comes with numerical failures does not count. `P` size is reported and forced resets
  are counted, because forced resets can flatter forgetting.

Thread C, fixed before any Thread C run:

- **H7 "stable":** at horizon 100, less than 2x the error of the full recipe, with under 5% of rollouts non-finite
  or clipped. Rollout error is bounded, so a 2x gap may be unreachable; `two_x_reachable` records that, and a
  saturated case counts as inconclusive, not as support for either side.
- **H9 scope:** tests detection only. Allocating loops and gating writes are untested until detection holds.
- **H9 baseline:** a simplified predictive-likelihood signal (fixed Gaussian plus CUSUM), not ARCADE's full
  Bayesian decoder. Beating it does not mean beating ARCADE, and write-ups must say so.
- **Known limit:** training carries state for only 4 steps, so long rollouts also extrapolate state age.

## Shared benchmark

Small vectorized simulators with hidden physical parameters:

- `pendulum`: smooth dynamics; hidden mass, length, damping.
- `puck`: agent pushes a puck with contact; hidden puck mass, friction.

Three shift tiers:

- **Interpolation:** hidden parameters inside the training range, unseen values.
- **Extrapolation:** parameters outside the training range.
- **Out-of-family:** a mechanism never varied in training, such as wind, rotated actions, or anisotropic
  friction.

## Hypotheses

### Thread A: what should adapt

**H1. Capacity crossover.** With adaptation data of K = 1 trajectory:
- Interpolation: latent optimization has lower rollout error than the RLS readout write.
- Out-of-family: the RLS write has lower rollout error, because the latent saturates.
- Kill: if one method wins in both tiers, capacity matching is the wrong explanation.

**H2. Hybrid.** Optimizing the latent, then RLS-writing the residual, is no worse than the better single
method in every tier and beats both in at least one.
- Kill: if the hybrid is worse than the better single method in any tier.

**H3. RLS versus gradient writes.** On smooth trajectories (near-collinear features), RLS beats gradient
writes given the same data and one pass over it, and the gap shrinks when feature collinearity is reduced.
This is the rationale World-T3 asserts; the notes found no controlled ablation.
- Kill: if tuned gradient writes match RLS, or the gap does not depend on collinearity.

### Thread B: use the uncertainty

**H4. Calibration.** `phi' P phi` predicts held-out one-step error (Spearman at least 0.5) under
interpolation and extrapolation. I expect this to weaken out-of-family because the features are
misspecified, and that is itself worth measuring.
- Kill: Spearman below 0.3 under interpolation ends H5 and H6 as stated.

**H5. Active probing.** Probes that maximize expected information (greedy D-optimal on predicted features)
reach the same rollout error as random probes with fewer trajectories, and improve error at K = 1 by at
least 10%.
- Kill: no detectable gain over random probes.

**H6. Bounded optimism.** OpTTA-style unbounded optimism degrades when the goal is infeasible. Optimism
bounded by the RLS confidence set keeps most of OpTTA's gain on feasible goals and does not fall below the
frozen model on infeasible ones. OpTTA's TwoRoom decline is not evidence for this, because AdaJEPA, which
has no optimism term, declines there too.
- Kill: if unbounded optimism does not degrade on infeasible goals, the bound has nothing to fix.

### Thread C: looped dynamics

**H7. The product of loops and rollout.** LoopWM designs for carried state and shows no ablation in the
pages read; TTF (2610.19552) tests loops against 32-128-step rollouts without carried hidden state. So many
inner loops with many rollout steps and carried state has no controlled test that we found. A looped dynamics core stays stable under that product only with all of
Solv3R's recipe: input re-injection, state normalization, and training on mixed refinement depths. Removing
any one degrades long-rollout error by at least 2x.
- Note the live contradiction: 2610.31943 finds removing normalization extrapolates better. Either outcome
  here is informative.
- Kill: if a plain weight-tied core is already stable under the product.

**H8. Action injection.** Re-injecting the action at every loop beats prelude-only injection on contact
dynamics. Untested in the notes.
- Kill: no detectable difference.

**H9. Loop residual as a surprise signal.** The per-step fixed-point residual `||z_{k+1} - z_k|| / ||z_k||`
rises at contact events and under dynamics shift, which would make it usable both to allocate loops
adaptively and to gate test-time writes. Solv3R showed this correlation only on static multi-view scenes.
- Baseline: ARCADE's predictive-likelihood changepoint signal (2512.14331).
- Kill: AUROC below 0.7 for detecting contact or shift onset, or no detectable gain over the baseline.

### Thread D: persistence and switching

**H10. Forgetting.** Under regime switches, RLS with a forgetting factor beats both per-episode reset and
persist-forever, and persist-forever is worst just after a switch. The notes report persistence and reset
each winning in different papers, and switching untested everywhere.
- Kill: reset matches forgetting in every switching schedule.

### Thread E: weight-space programming

**H11. Agent weight edits against a real baseline.** The Weight-Space Programming blog used a closed model
and reports fixes on a Kinetix policy. Test: give an LLM agent a small dynamics network that fails in one
regime, plus a fixed forward-pass budget. Compare against a ridge fix of the last layer and a gradient fix
using the same budget. My expectation is that the agent does **not** beat the ridge fix on prediction error,
but its edits may be sparser and easier to explain.
- This thread is exploratory, with few runs and an agent-dependent method, so results are reported as case
  studies rather than effect estimates.

## Second round (post hoc: formed after seeing the first results)

These come from reading `docs/RESULTS.md`, so they are not independent of it. Each is tested on data not used to form it:
new seeds 5-9 and, where stated, a new family. The protocol and effect threshold above still apply.

### Benchmark repair, required before any further puck verdict

**Gate G1.** A puck model must reach closed-loop success of at least 0.5 on interp with no adaptation, and its latent must
lower training loss by at least 10% against no latent. Until both hold, puck results are not evidence for or against any
method. Report what fixed it, or that nothing did within the budget.

**Second working family.** Add one smooth system with hidden parameters, unrelated to the pendulum's equations, so that no
claim rests on a single system. It must pass the same gates: oracle success at least 0.5 on interp, and the latent lowering
training loss by at least 10%.

### Thread F: replication and capacity on demand

**H1r. Replication.** H1's crossover holds on the second family and, if G1 passes, on puck.
- Kill: one method wins both tiers on a family that passes its gates.

**H12. Empirical-Bayes write strength.** Choosing RLS `lambda` and noise scale per environment by maximizing the Bayesian
marginal likelihood of the adaptation data, using no validation set, makes the write as strong as that environment needs.
Predictions on latent-then-RLS with this rule:
- It beats H2's validation-chosen hybrid out-of-family.
- It is no worse than `latent_opt` on interp and no worse than RLS out-of-family, so one method is no worse than the better
  single method in both tiers. This is what H2 failed to deliver.
- On a family where writes do not help, it picks weak writes by itself and is no worse than frozen.
- Kill: worse than the better single method in any tier on a family that passes its gates.
- Risk to check: with K = 1 the evidence may be poorly identified or overconfident. Report the selected values per tier.

**H13. Prior-preserving forgetting.** Forgetting toward `W0` (decaying the posterior back to the prior rather than letting the
prior itself decay) matches or beats reset under switching, and beats plain forgetting, without needing to know when
switches happen. Compare with per-episode reset, plain forgetting, and surprise reset. Prior work exists (ARCADE, Neural-Fly),
so this is largely a replication on this benchmark.
- Kill: no better than plain forgetting.

### Thread B continued: coverage

**H5 (restated, pendulum and the second family only).** In the first round, data coverage moved RLS by 3.2x with collinearity
unchanged, so probing is registered in terms of coverage. Probes chosen by greedy D-optimal design on `phi' P phi`, using the
model's own predictions, get closer to the iid-state ceiling than the default probe and random actions at the same K.
- Kill: no detectable gain over random-action probes at K = 1.

**H6** runs next if H5 holds, on the same families, with the original kill criterion.

### Thread E: a loop that sees its own error

**H14. Error-fed loops as amortized adaptation.** In the first round, carried state was inert and the residual near chance,
plausibly because the state is fully observed and the loop never sees its own prediction error. Feed the previous step's
prediction error into the loop's injection, so carried state can act as an online filter for hidden physics. Train over long
enough windows for this to matter. Predictions:
- The state becomes non-inert, with carry sensitivity at least 5x higher than the first round's `full`.
- After the same 40 adaptation transitions, it matches `latent_opt` on interp, with no test-time gradients.
- It fails out-of-family as the latent does, losing to RLS. So it is an amortized latent rather than a new capacity class.
- H9 revisited: its residual or error signal beats the simplified baseline at detecting shift. The first round's baseline
  saw prediction error and the residual did not, so this is a fair retest.
- Kill: no better than the first round's `full` on interp after adaptation, or state stays inert.

### Thread P: pixels on a rented GPU

Do the state-based findings survive when the model must see images? Small models and our own batched renderer, chosen for
speed and exact control of hidden physics. This is not a published benchmark, and write-ups must say so.

Fixed design:
- **Families:** pendulum, drone, puck, rendered to small images. Velocity is only visible across frames.
- **Seeds:** 10-19, unused so far. Ten seeds, because five gave a coarse bootstrap.
- **Metric comparable across models:** rollout error in physical state, decoded by a probe trained on training environments only.
  Primary: horizon 10, K = 1. Embedding-space error is not used for verdicts, because its scale differs per model.
- **Collapse check:** a model whose embedding does not predict state gets no verdict.
- **Gate GP, per family:** frozen closed-loop success on interp at least 0.5, oracle at least 0.5, and the latent lowering
  prediction loss by at least 10%. A family failing the gate is not evidence for or against any method.
- Protocol, effect rule, validation-only selection, and same adaptation data across methods all apply.

Predictions:
- **H1p:** the crossover holds on each family passing GP: latent wins interp, RLS wins out-of-family.
  Kill: one method wins both tiers on a passing family.
- **H12p:** latent-then-evidence-chosen RLS is no worse than the better single method in both tiers.
  Kill: worse than the better single method in any tier on a passing family.
- **H4p:** `phi' P phi` ranks one-step error above half its well-specified reference and adds beyond `||phi||^2`.
  Kill: either fails under interp on a passing family.
- **H16 appearance shift (exploratory).** Add a tier changing appearance only (colours, background), with physics unchanged.
  Writes to the dynamics readout cannot repair a shifted encoder, so expect neither the latent nor RLS to recover. Reported as
  descriptive, with no verdict.
- **H14p (changed before any H14 result).** The state-based run was dropped to focus on this thread, so H14 is tested here
  directly: an error-fed recurrent core, with no test-time gradients, against `latent_opt`, a no-error control, and a
  shuffled-error control. Predictions: it matches `latent_opt` on interp, beats both controls, and loses to RLS out-of-family.
  Kill: no better than the no-error control on interp.
- **H5p:** after the same number of adaptation transitions, D-optimal probes beat random-action probes for RLS at K = 1, and
  the coverage measure moves as intended, with the iid-state ceiling reported. A result with coverage unmoved is
  inconclusive. Kill: coverage moves, with no detectable error gain.
- **H3p:** RLS vs tuned gradient writes on the same data. Tuned means chosen on validation. Round-one expectation: tuned
  gradient matches in-family, RLS wins on shifted data, at much lower write cost. Kill: tuned gradient wins any tier.
- **H17 capacity curve (descriptive).** Latent size 2, 8, 32, on three seeds per family. Tests the capacity account directly:
  larger latents should not rescue out-of-family error. Reported with no verdict.
- Implementation choices fixed before any pixel run:
  - **H4p error:** physical one-step error decides, for consistency with the metric rule above. The embedding-space result is
    stored and flagged, and does not decide.
  - **Collapse rule:** a model gets no verdict if its probe explains too little of the state; threshold 0.5, as coded.
  - **Pilot:** seed 1010 is used only for the preflight pilot and reads validation data only. It is outside seeds 10-19 and is
    not evidence.
- The state-based Thread F and H14 runs are stopped. Their code remains in the repo, unused by this thread where noted.

### Parked

- **H3's collinearity clause:** needs a real manipulation, such as controlling N against F with a check that the measures move.
- **H7:** trained loops were never high. Retest only if H14 makes loops useful.
- **H8:** the post hoc effect above trained depth stands unregistered.
- **H11 agent weight edits:** needs a runtime for forward passes. Run on a family that passes its gates.

## Order of work

1. Benchmark, models, and Thread A, since everything else reuses them.
2. Thread B, which depends on H4 surviving.
3. Thread C, which can run in parallel once the benchmark exists.
4. Threads D and E.
