# Prior-art check for the ten test-time-adaptation hypotheses

Written 2026-11-04. Method: alphaXiv discovery searches, then targeted full-text reads of the closest papers.
Every arXiv ID below was returned by a tool call in this session.

Reading levels used below:
- **FULL** = I read returned pages of the paper body (not always every page; the tool returns the most relevant pages).
- **SNIPPET** = I saw only the search-result abstract fragment. Treat any claim about these papers as unverified.
- **CITED** = named inside a paper I read, but not retrieved myself. Not a citation I can vouch for; listed only as a lead.

"Stated" means the paper says it. "Inference" means my reading, not the paper's claim.

Numbering. The ten items follow the task summary. `docs/HYPOTHESES.md` numbers them differently:

| Summary item | HYPOTHESES.md |
|---|---|
| 1 RLS = Bayesian last layer, phi'P phi | premise of H4 (also H5, H6) |
| 2 latent vs readout crossover | H1 |
| 3 latent then RLS residual | H2 |
| 4 RLS vs gradient, collinear features | H3 |
| 5 D-optimal probes | H5 |
| 6 bounded optimism | H6 |
| 7 loops x rollout stability | H7 |
| 8 action re-injection | H8 |
| 9 loop residual as surprise | H9 |
| 10 forgetting under switches | H10 |

## Verdict table

| # | Status | Scooped by | What is left |
|---|---|---|---|
| 1 | **Answered** | ALPaCA 1807.08912; Davydov 2411.00107 | Only calibration of phi'P phi on a World-T3-style jointly trained fast-weight readout under shift |
| 2 | **Open** (weak partial evidence) | none head-to-head | The whole comparison |
| 3 | **Open** | none found | The whole hybrid |
| 4 | **Partly** | ALPaCA (asserted); Gated KalmaNet 2511.21016 (covariance-aware vs identity-covariance writes) | Controlled collinearity sweep in a dynamics world model |
| 5 | **Mostly answered** in mechanism | Davydov 2411.00107 (hardware) | Comparison to random probes, K=1 sample efficiency, world-model TTA setting |
| 6 | **Partly** | H-UCRL 2006.08684 (bounded optimism, regret); OpTTA 2610.35636 (unbounded) | Bounded optimism inside test-time adaptation of a pretrained deep world model; infeasible-goal test |
| 7 | **Partly** | LoopWM 2606.18208 (design, theory); TTF 2610.19552 (loops x rollout, no carried state) | Ablating each ingredient with carried state on contact dynamics |
| 8 | **Partly** | LoopWM (design, no ablation seen); Looped LM Part I 2610.23036 (LM ablation) | Ablation on contact dynamics |
| 9 | **Open** | none | Everything; plus a stronger baseline to beat |
| 10 | **Mostly answered** | ARCADE 2512.14331; Neural-Fly 2205.06908 | Plain forgetting factor vs reset vs persist on a frozen deep-WM readout, pre-registered |

Two corrections to premises in HYPOTHESES.md are collected at the end.

---

## 1. RLS write of a last-layer readout is Bayesian last-layer regression, so phi'P phi is epistemic variance

**Status: answered.** This is a textbook identity and is already used in dynamics learning.

**ALPaCA, 1807.08912 (FULL).**
- Stated: Bayesian linear regression on the last layer of a meta-learned network; matrix-normal prior; posterior precision `Lambda_tau = Phi'Phi + Lambda_0`.
- Stated: posterior predictive covariance `Sigma_{tau+1} = (1 + phi' Lambda^-1 phi) Sigma_eps` (Eq. 8).
- Stated: the recursive Woodbury/RLS update (Algorithm 2, Eqs. 13-14), cited to Ljung and Soderstrom.
- Stated: confidence intervals "appear to be well-calibrated" on a step-function toy. This is qualitative, from plots.
- Features and prior are meta-trained jointly through the posterior, so the features are shaped to make the posterior useful.

**Davydov et al., 2411.00107 (FULL).**
- Same model on a Toyota Supra: one-step predictive variance `sigma^2 (1 + phi' Lambda^-1 phi)` (Eq. 7b) and the recursive update (Eq. 6).
- The variance is used as an uncertainty for planning.

**Neural-Fly, 2205.06908 (FULL).**
- Stated: the adaptation law is a Kalman-Bucy filter with covariance P, plus tracking-error and regularisation terms. It is written with learned basis phi and linear coefficients a.

**Others.**
- 2512.14331 ARCADE (FULL): BLR decoder on frozen latent features, closed-form posterior, uncertainty used in MPC cost modulation.
- 2601.09178 Vision-Conditioned VBLL (FULL): VBLL dynamics with predictive posterior `phi' S phi + Sigma`; compares against "VBLL + RLS adaptation".

**World-T3, 2610.17434 (FULL, pages 1, 2, 4-6, 10, 18, 22, 25 only).**
- The RLS write is Eq. 4. It tracks `P_j = (sum w_i phi_i phi_i' + lambda I)^-1`, `g = P phi / (1/w + phi' P phi)`, `W2 <- W2 + (v - W2 phi) g'`, `P <- P - g phi' P`. Stated: lambda = 1e-3, P0 = I/lambda.
- In the pages I saw, P is used only as the update preconditioner. No uncertainty read-out. This matches HYPOTHESES.md.

**Inference, not stated by any paper above.** Three caveats on the identity as used for World-T3:
- World-T3 has per-sample learned weights `w_i`. The Bayesian reading needs `w_i` to be inverse noise variances.
- The features and W1 are trained end-to-end through the RLS recursion. The Bayesian reading holds only conditional on the features, as in ALPaCA.
- The "iid noise" assumption is likely false for smooth trajectories.

**Remaining.** Whether phi'P phi rank-predicts held-out error on a World-T3-like fast-weight readout under interpolation, extrapolation and out-of-family shift (HYPOTHESES.md H4). Downgrade the identity claim itself to "known".

**Warning for H4.** 2604.25416 "Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent Dynamics Models" (SNIPPET only) is about limits of epistemic uncertainty in latent dynamics models. I did not read it. It is a lead for what can go wrong under misspecification.

---

## 2. Low-dimensional latent beats the readout in-family; readout beats the latent out-of-family (capacity crossover)

**Status: open.** No paper I found runs this comparison. There is weak partial evidence for each half.

**T2-Z, 2610.22364 (FULL).**
- Stated: 32-D environment latent, optimised for 40 steps on K = 1 or 2 test trajectories, frozen world model (Wan2.2-5B based).
- Stated, Table 1 (7 simulated tasks, average): physical error (mean) 4.80 px vs 9.35 (LoRA TTA), 12.92 (TTT-KQV, "fast weights within the world model"), 12.86 (no adaptation). Action success 74.2% vs 44.8%, 42.1%, 37.9%.
- Stated, Table 2 (4 real tasks, average): action success 70.8% vs 44.0%, 36.2%, 26.5%.
- Stated, Table 12: latent dimension 8/16/32/64/128 gives action success 44/64/72/64/64% on Friction (K = 1). So there is a non-monotone capacity optimum, which fits the "match capacity" story but is not an out-of-family test.
- Stated, K ablation (Table 11): K = 1 -> 4 helps modestly (Mass Collision success 61.1 -> 72.2%).
- The paper claims the latent "is not restricted to the range of factor values during training". Its test environments are held-out values of factors varied in training, plus discrete rules.
- No closed-form (ridge/RLS) readout baseline. The fast-weight baseline is gradient-based and has no uncertainty.
- No tier where a mechanism was never varied in training.

**Evidence for the readout side.**
- Neural-Fly 2205.06908 (FULL): linear coefficients over a meta-learned basis, trained on winds 0-22 km/h. Stated: it beats nonlinear, L1 and INDI baselines "including the 30.6 km/h, 43.6 km/h, and sinusoidal wind speeds all of which exceed the wind speed in the training data". Improvements stated as 66% over the nonlinear controller, 42% over L1, 35% over INDI. This is an extrapolation result for a linear-in-features write, but there is no latent-optimisation comparator.
- Neural-Fly's own representation is deliberately wind-invariant (adversarial discriminator), so wind-specific information sits only in the linear coefficients. That design choice is itself a capacity argument.

**Relevant but not read.**
- 2610.06651 "Considering Context: When World Models Need Context Encoders" (SNIPPET).
- RMA 2107.04034, CaDM 2005.06800, Dynamics-Aligned Latent Imagination 2508.20294 (SNIPPET): amortised context encoders, not test-time latent optimisation vs readout writes.
- 2609.12278 Amortized Low-Rank Adaptation for MBRL (SNIPPET).

**Remaining.** The within-family vs out-of-family crossover at fixed K, with a closed-form readout and a latent optimiser on the same backbone. The adversarial caveat: T2-Z already beats gradient fast weights by about 2x in mean error, so the readout arm must be RLS or ridge to be interesting, and a tuned gradient readout is a weak comparator.

---

## 3. Latent optimisation, then RLS residual (hybrid)

**Status: open.** I found no paper that optimises an environment latent and then RLS-writes a residual readout.

Closest structure (all offline latent features plus online linear update, not latent optimisation at test time):
- ARCADE 2512.14331 (FULL): frozen encoder gives latent features z; online BLR decoder.
- Davydov 2411.00107 (FULL): physics model plus BLL residual.
- Neural-Fly (FULL): learned basis plus online linear coefficients.
- 2609.39179 LocoWM "World-Model-Guided Residual Adaptation" (SNIPPET only).

**Remaining.** The specific sequence, and the kill condition in HYPOTHESES.md. Note a design coupling: after the latent step, the readout features depend on the latent, so P must be reset or the features frozen. Prior art does not address this.

---

## 4. RLS vs gradient writes under collinear features

**Status: partly answered. Asserted and indirectly supported, not controlled.**

**ALPaCA (FULL).**
- Stated: "moving from gradient updates to least squares yields a simpler algorithm", and "in the case of linear filtering, recursive least squares approaches have been shown to outperform gradient-based approaches", citing Cioffi and Kailath 1984 (CITED, not retrieved).
- Their experiments compare to MAML with 1 and 5 inner steps on a sinusoid, a step function, pendulum and hopper (figures; numbers not in the text I got). This is a comparison to a meta-learned initialisation with gradient steps, not a collinearity ablation.

**Gated KalmaNet, 2511.21016 (FULL).**
- Stated: DeltaNet, Gated DeltaNet and Kimi Delta Attention are "approximations to the KF recurrence under an identity error covariance assumption", i.e. gradient-like delta-rule writes. GKA keeps the full covariance and the exact Kalman gain, and is a steady-state ridge regression.
- Stated: GKA beats Mamba2 and Gated DeltaNet on short-context tasks and by more than 10% relative on long-context RAG/LongQA up to 128k. This is the closest evidence that covariance-aware writes beat delta-rule writes at scale. It is language modelling, and not isolated against collinearity.
- Stated: regularisation strength interacts with condition number; too small a ridge hurts through numerics (a = 0.01 vs 0.02 vs 0.05 average scores 58.43, 58.89, 57.32).

**ARCADE (FULL), indirect.**
- Stated: an MLP baseline fine-tuning the last layer by SGD "exhibits a sharp speed-stability trade-off". Cartpole online CRMSE: 17.76 (MLP last layer) vs 5.83 for their BLR method. The comparison is confounded by architecture and the changepoint machinery, so I would not cite it as an RLS-vs-SGD result.

**FAAST, 2605.04651 (FULL).**
- Stated: closed-form pseudoinverse fast weights match or beat a backprop-trained linear probe (CIFAR-10 5-shot 73.8 vs 71.9; full data 86.7 vs 88.3). Theorem B.1 says gradient descent from zero with small steps converges to the same minimum-norm solution. So the benefit is about the number of steps and cost, not the optimum.

**Inference (background knowledge, not from a retrieved paper).** The dependence of LMS/gradient convergence on eigenvalue spread of the input covariance, versus RLS being invariant to it, is classical adaptive-filtering theory. That makes the sign of the H3 prediction near certain. What is untested is its size for one pass over short trajectories in a learned world model, and what a tuned multi-step or preconditioned gradient write achieves. HYPOTHESES.md already says the World-T3 authors assert collinearity as the rationale; in the pages I read I also saw no ablation (pages 10, 18 and 22 were returned; I did not see a table comparing write rules).

**Remaining.** A controlled collinearity sweep with tuned gradient, momentum and Adam writes at equal passes. Treat as a replication of known theory with a new setting.

---

## 5. D-optimal probe selection using P

**Status: mostly answered in mechanism; the evidence is thin on comparison to random probes.**

**Davydov et al., 2411.00107 (FULL).**
- Stated: information-gathering objective `l_info = 1/2 sum_k sum_i log(1 + phi_ik' Lambda_{i,0}^-1 phi_ik)`, derived as one-step mutual information from the BLL representation (attributed to Lew et al. 2022, CITED, not retrieved).
- Stated: using the updated `Lambda_{i,k}` instead gave no substantial difference in trajectories.
- Stated, hardware (Toyota Supra, 3 attempts each), Figure-8 drift, RMS lateral deviation, best / median / worst, metres: Once-OCP-Model 2.31 / 3.19 / 7.35; Once-Info-Model 1.18 / 1.31 / 3.64; Twice-Info-Model 0.61 / 0.83 / 1.29. Only Twice-Info completed the full manoeuvre.
- Stated: online adaptation alone gave large transients (about 150 m to initiate a drift).
- Their baseline is nominal-MPC data, not random excitation. No comparison to random probes. n = 3.
- Inference: `log(1 + phi'Pphi)` is exactly the greedy marginal log-det gain for a Bayesian linear model, so "greedy D-optimal on P" is the same quantity.

**Others.**
- Optimistic Active Exploration of Dynamical Systems 2306.12371 (SNIPPET): active exploration of dynamics.
- 2609.19378 Task-Oriented Active Learning of Residual Dynamics for MPPI; 2604.12542 Goal-oriented safe active learning with Bayesian RNNs; 2601.21988 Generalized Information Gathering Under Dynamics Uncertainty (all SNIPPET).

**Remaining.** (a) A head-to-head against random probes. (b) K = 1 in a test-time-adaptation world-model setting with a pretrained deep model. (c) The H5 threshold of 10% at K = 1. Adversarially: expect random probes to be hard to beat on smooth low-dimensional simulators because most actions are informative.

---

## 6. Optimism bounded by the RLS confidence set, vs OpTTA's unbounded optimism

**Status: partly answered. Both halves exist separately; the combination in test-time adaptation is untried.**

**H-UCRL, 2006.08684 (FULL).**
- Stated: optimism over the plausible set `{f : |f - mu| <= beta sigma}`, parameterised with hallucinated controls `eta in [-1,1]^p` on the one-step epistemic uncertainty. Sublinear regret for well-calibrated GP models. Works with deep ensembles.
- Stated: needs the model to be well-calibrated for one-step predictions only.
- Empirics: inverted-pendulum swing-up with action penalties 0, 0.1, 0.2 (figure only).
- Bounded optimism over a calibrated confidence set therefore exists. The BLL/RLS set `|f - mu| <= beta sqrt(phi'Pphi)` is a special case (inference).

**OpTTA, 2610.35636 (FULL).**
- Stated: loss `L_pred + alpha_k L_plan`, no confidence set, derived from reward-biased MLE. Planning term evaluated on 200 CEM candidates.
- Stated: authors explicitly say uncertainty-based optimism (they cite Curi et al. 2020 and Sukhija et al. 2025) needs calibrated uncertainty, "difficult to obtain under distribution shift". So the motivation for H6 is anticipated and argued against by the authors.
- Stated, Table 1: final planning success over frozen, predictor-only: OpTTA +13.5 pp on average vs +6.9 for AdaJEPA (8 settings); predictor and encoder +13.5 vs +7.5 (7 settings). 14 of 15 shifted settings beat AdaJEPA.
- Stated, ablation: pessimism (rho = -0.1) gave final success 32.7% on TS-PushT, 60.7% on Reacher, below AdaJEPA (44.7%, 70.7%). The gains come from optimism.
- Stated, Table 2: adaptation costs about +20 ms per replan over AdaJEPA on LeWM/TS (5-16% longer replan).
- Not seen: the "gradient-balancing rule" (Eq. 8) which OpTTA uses. So "unbounded optimism" in HYPOTHESES.md is a simplification: there is a `sqrt(k/Kmax)` ramp and a balancing rule. I did not read Eq. 8.

**Correction to the H6 premise.** In OpTTA's TwoRoom, both AdaJEPA and OpTTA fall below frozen (LeWM TwoRoom visual corruptions: frozen 61.0, AdaJEPA 57.6, OpTTA 57.2; in distribution 93.6, 93.2, 91.2). The authors call it a task-dependence they do not understand. It is not evidence that optimism degrades on infeasible goals. No infeasible-goal experiment appears in the pages I read. The H6 kill condition ("if unbounded optimism does not degrade on infeasible goals, the bound has nothing to fix") is therefore a live risk.

**Remaining.** An infeasible-goal test; a bounded-optimism variant inside the OpTTA loop; whether a BLL confidence set stays calibrated under out-of-family shift (links to H4). Related optimistic MBRL, SNIPPET only: SOMBRL 2511.20066, Optimistic World Models 2602.10044 (same Texas A&M group as OpTTA), Optimistic Thompson Sampling 2410.04988.

---

## 7. Stability of inner loops x rollout steps with carried state

**Status: partly answered.** The claim "no paper tests this" in HYPOTHESES.md is not safe.

**Looped World Models (LoopWM), 2606.18208 (FULL, pages 1-8, 11, 12).**
- Stated: looped dynamics core with prelude / recurrent block / coda; `h(t+1) = A h(t) + B e + R(h(t), e)`; stability by `A = diag(-exp(a))`, discretised so entries lie in (0, 1) (Parcae-style, citing Prairie et al., CITED 2604.12946).
- Stated: "cross-timestep state propagation": the terminal hidden state at env step k initialises the loop at k+1. They claim both loops remain bounded and rollouts are "provably stable across arbitrary rollout lengths".
- Stated: Poisson-sampled loop count during training, per sequence; layer norm on the conditioning signal; truncated backprop through loops; learned exit gate for adaptive depth.
- So all three ingredients named in H7 (input re-injection, normalisation, mixed depths) are present in the design.
- Evidence in the pages I saw: a ScienceWorld text world-modelling table, 1B-parameter model, 5-action rollouts: LoopWM exact match 68.4% vs 47.2% for a frontier LLM baseline, F1 85.3 vs 72.8 (Tables 2-3). A "deferred decoding" section says it helps more as rollouts accumulate. There is no ablation of any ingredient, no continuous dynamics, no contact benchmark in the pages returned (pages 9-10 were not returned, so I cannot rule out more experiments). The paper states its disclosure is "intentionally selective".
- Caveat on the source: this is an unusual report (organisation-level authorship, text-world benchmark, 100x parameter-efficiency claim). Weight accordingly.

**Test-Time Forecasting (TTF), 2610.19552 (FULL, partial).**
- Stated: recurrent-depth one-step forecaster (implicit: attention recurrence; explicit: inner gradient descent on fast weights), trained one-step, evaluated autoregressively. Generalises to out-of-distribution ODE/PDE systems.
- Stated: loop scaling reduces error on a 128-step rollout at the hardest advection-diffusion setting (Pe = 1e8), and on 32-step KdV and KS rollouts (Figures 4C, 5; numbers in figures; KS table: nRMSE 1.107 at 1 inner step to 0.597-0.764 at 16 steps depending on batch).
- Stated: for advection-diffusion, fast weights are adapted on the first rollout step and reused for the rest. In other experiments, the loops are re-run at each rollout step ("essentially refit on each rollout step"). Hidden state is not carried between steps in the way H7 requires.
- So "many loops with many rollout steps" has been tested for one family; "with carried state, and ablating normalisation, re-injection and mixed depths" has not.

**Looped LM stability literature.**
- Part I, 2610.23036 (FULL): no stability-vs-rollout analysis. 2606.18206 Fixed-Point Reasoners, 2610.06833 Part II, 2610.18174, 2609.26487, 2610.10996, 2609.37432 Looped Actor (all SNIPPET).
- The contradiction cited in HYPOTHESES.md (2610.31943, "removing normalisation extrapolates better") and Solv3R were not retrieved by me. I cannot confirm them.

**Remaining.** Ablate each ingredient with state carried across rollout steps, on contact dynamics, with the 2x long-rollout-error threshold. Note LoopWM claims a theoretical guarantee for the contractive part; the empirical question of whether nonlinear `R` plus carried state stays stable is open.

---

## 8. Action re-injection at every loop

**Status: partly answered by design and by language-model evidence; contact-dynamics ablation open.**

**LoopWM 2606.18208 (FULL, partial).** The action embedding `u_k` enters the prelude (Eq. 4: `e = LN(P([h_{k-1}; e_k; u_k]))`), and `e` is re-injected via `B e` at every inner iteration (Eq. 5). Section 3.5 writes the inner update as `B [u_k; h_k]` and says the action is injected at each action step. So action re-injection is in the architecture. No ablation against prelude-only injection appeared in the pages I read.

**Looped LM Part I, 2610.23036 (FULL).**
- Stated, controlled ablation at 730M parameters, 336B tokens: persistent injection of the prelude state before every recurrent pass, vs no injection. Middle loop: BBH-CoT +6.63, MMLU +2.53, HumanEval+ +5.49, MBPP+ +4.23, DROP +1.39; but MATH500 -3.60 and GSM8K -2.51. Full-stack Ouro with raw-token injection: gains on the same groups, smaller.
- Stated reading: injection helps tasks that must retrieve input context, can hurt iterative quantitative reasoning.
- Inference: a dynamics model's action is exactly the input that must stay accessible, which favours H8; but the maths results show a possible downside when the state should drift away from the input.

**Remaining.** Dynamics-specific ablation. Not scooped.

---

## 9. Fixed-point loop residual as a surprise signal for contact or shift

**Status: open.**

- LoopWM (FULL, partial) argues in prose that adaptive depth would allocate more iterations to "collisions, contact events", using a learned exit gate on `h(t)`. It does not report the residual as a detector, and in the pages I saw does not test it on contact.
- LoopWM cites Pappone et al. 2025 for acceleration-based exit rules (`||h(t) - 2h(t-1) + h(t-2)||`) (CITED, not retrieved).
- TTF (FULL, partial) says inner loss can be used as an early-stopping signal and that adaptive loop schedules "based on the stiffness of the dynamics" are future work. Duffing eTTF used a threshold.
- Language-model looped work uses convergence for halting (SNIPPETS only: 2610.18174, 2609.26487, 2610.06833).
- I found no paper that uses the loop residual to detect dynamics shift or contact onset and reports AUROC.

**Competing baseline that is already published (important).** ARCADE 2512.14331 (FULL) computes a changepoint posterior from the Bayesian predictive likelihood of the residual (Eqs. 8-9), and its detected changepoints "align closely with intervention times" on cartpole (figure; no AUROC). So a surprise signal from the readout's standardised innovation (`delta' (phi'Pphi + sigma^2)^-1 delta`) is free and probably stronger than a fixed-point residual. If H9 is run, it should report the loop residual against that innovation baseline, not against chance. Davydov 2411.00107 also proposes (future work) using the BLL uncertainty to detect out-of-distribution behaviour.

---

## 10. Forgetting factor under regime switches

**Status: mostly answered.** The core claim (forgetting or reset beats persist-forever after a switch) is established; the specific three-way comparison with a plain forgetting factor is not.

**ARCADE 2512.14331 (FULL, partial).**
- Stated: frozen encoder, BLR decoder, changepoint-aware posterior tempering (prior precision scaled by `gamma in {1, beta^2}`), beam of changepoint hypotheses. Theory: bounded predictive variance under changepoints; adaptive regret logarithmic in time and linear in the number of shifts.
- Stated, cartpole (10 trajectories, shifts at steps 0, 40, 80, including a 10% mass reduction): CRMSE 5.83 +/- 1.18 (full) vs 10.28 +/- 2.07 without changepoints (persist), GP 12.53, MLP last layer 17.76, offline only 24.03.
- Stated, quadrotor with payload drops: open-loop CRMSE 5103 vs 5152 (Proto-MPC), 6367 (MLP), 6804 (GP), 9863 (SysID). Beam-size ablation: RMSE 0.128 m (K = 5) to 0.118 m (K = 20-30).
- This is persist-vs-reset/temper on switching schedules. A plain exponential forgetting factor and a per-episode reset were not among the baselines I saw. The comparison is single-domain, with a smallish number of trajectories (10).

**Neural-Fly 2205.06908 (FULL).**
- Stated: the covariance recursion `P <- (1 - lambda dt)^2 P + Q dt` and coefficient damping `(1 - lambda dt) a` (Eqs. 15-16). That is a forgetting/process-noise mechanism. Time-varying wind (30.6 + 8.6 sin(t) km/h) was tested. The paper interprets `lambda` as "how quickly the environment returns to nominal" and `Q` as "how quickly the environment changes".

**Others.**
- ALPaCA reference list includes Kulhavy, "Restricted exponential forgetting in real-time identification" and Al-Shedivat et al. (continuous adaptation in non-stationary environments) (CITED in ALPaCA's bibliography; not retrieved).
- SNIPPET only: 2609.18950 Changepoint-Aware World Models (forgetting stale replay), 2609.18167 Replay Retention Under Dynamics Shift, 2604.02260 MBRL under time-varying dynamics, 2609.23932 directional forgetting in adaptive control, 2606.15317 covariance-regulated recursive Koopman learning, 2412.02972 online adaptive Koopman MPC, 2610.10340 Data Reuse in Non-Stationary Learning.

**Remaining.** A pre-registered three-way (forgetting-factor RLS, reset, persist) on a frozen deep world-model readout with controlled switch schedules; the "persist is worst just after a switch" prediction is nearly certain. Expect this to be a replication, not a finding.

---

## Search coverage and gaps

Searched with results: ALPaCA / Bayesian last layer, Neural-Fly, meta-learned adaptive control, VBLL dynamics, active system identification, information gain, H-UCRL / optimism in MBRL, RMA / CaDM / context encoders, forgetting factor / changepoint, looped and deep-equilibrium models, test-time training RLS vs gradient, Koopman online adaptation.

Not done or thin:
- LinUCB / neural-linear bandits specifically (the BLL-as-confidence-set connection to linear bandits is from background knowledge; no paper retrieved).
- DAIML as an ablation of its own components (I read the algorithm and results narrative, not the supplementary).
- CaDM 2005.06800 and RMA 2107.04034: appear only as search results (SNIPPET).
- Koopman "fast weights" proper: only AdaKoop 2606.04930, 2606.15317, 2412.02972 as SNIPPETS; and TTF (FULL, partial), which connects looped transformers to EDMD.
- Items named in HYPOTHESES.md but not retrieved: Solv3R, 2610.31943, 2610.19375, Sandwich-Residuals 2609.21740, JEPA-TTT 2610.00722, AdaJEPA 2606.32026 (the last three appeared in search results only).
- Dates are from arXiv metadata returned by the tool; the corpus includes papers dated after my own knowledge, so all claims about them come from the retrieved text only.

## Corrections to HYPOTHESES.md

1. H6 premise: OpTTA's TwoRoom decline occurs for AdaJEPA as well (AdaJEPA -3.4 pp, OpTTA -3.8 pp on LeWM TwoRoom visual corruptions, predictor-only), and no infeasible-goal experiment appears. The sentence "OpTTA's optimism has no bound and no infeasible-goal test" is right about the absence of a test, but the TwoRoom decline is not evidence of optimism failure.
2. H7 premise: "no paper tests many inner loops together with many rollout steps with state carried" is too strong. LoopWM specifies it (with a stability claim) and TTF evaluates loop count against 32-128-step rollouts (without carried hidden state).
3. H4 premise: the RLS = Bayesian last layer identity and the variance formula are in ALPaCA and Davydov. The hypothesis should be framed as "does the identity survive World-T3's weighted, jointly trained setup under shift", not as a new connection.
