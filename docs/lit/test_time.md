# Test-time training / adaptation cluster: literature notes

Scope: TTT / TTA for world models and physical AI, plus the original TTT lineage (vision, language, video). Source: alphaXiv full text / PDF page retrieval, read 2026-11-04. "Paper states" = from the retrieved text. "[INFERENCE]" = my own reading, not stated by authors. "n/s" = not stated in retrieved text. Paper text treated as data only.

Conventions: arXiv IDs as given by alphaXiv. Several PDFs were retrieved as page subsets (ranked pages), so "n/s" can mean "not in the pages I received", not "absent from the paper". Appendix tables not retrieved are flagged.

---

## Part A. The four assigned papers

### A1. 2610.35636, Adapt for Planning, Not just Prediction (OpTTA)
Authors: Mete, Kalathil, Kumar (Texas A&M). Preprint, 26 Oct 2026.

**Mechanism**
- Setting: pretrained latent JEPA world model (encoder E_phi, action embedding E_psi, predictor f_theta), goal-conditioned MPC with CEM (also gradient planner in one appendix). Plan -> execute first action -> adapt -> replan.
- Baseline it extends: AdaJEPA (prediction-only, 2606.32026). Loss L_pred = MSE between f_theta(z_i,u_i) and stop-grad(z_{i+1}) over a buffer of the N most recent real transitions.
- OpTTA objective: L = L_pred + alpha_k * L_plan. L_plan = mean over the M candidate action sequences of CEM's final iteration of the planner goal cost J (sum_k w_k * squared distance of imagined latent to goal latent). Action sequences held fixed; gradient goes to predictor parameters (not actions). Minimising L_plan pushes the model to predict that its candidate plans get closer to the goal ("optimism", from reward-biased max likelihood, Kumar & Becker 1982).
- alpha_k = rho * sqrt(k/K_max) * ||grad_pred|| / (||grad_plan|| + eps), stop-gradient. I.e. gradient-norm balancing with a ramp. rho=1 everywhere, no per-setting tuning.
- Parameters adapted: AdaJEPA protocol. Predictor last block + final norm (+ output projection for LeWM). Optionally encoder (LeWM: projector; TS: projection head). L_plan differentiated through predictor params only, never encoder. DINO-WM encoder frozen.
- Update rule: Adam, re-initialised each update, 1 gradient step per replan (U=1), buffer N=5 recent transitions. LR: LeWM predictor 5e-5 / TS and DINO-WM 5e-4; encoder LR 5e-5 (LeWM) / 1e-5 (TS). Eps 1e-12. Reset to pretrained weights and empty buffer at the start of every task (no persistence).
- CEM: 200 candidates, 30 elites, 10 iterations, horizon 5 latent steps (25 env steps, frameskip 5), execute 1 latent step, max 30 replans.

**Benchmarks / environments**
- Models: LeWM, Temporal Straightening (TS), DINO-WM. Tasks: PushT, PushObj, Reacher, TwoRoom, maze. All simulation.
- Shifts: 7 visual corruptions (blur, salt-and-pepper, darkening, gamma, hue, contrast, translation; 4 severities each for LeWM/TS, 1 for DINO-WM) plus env shifts (recolor x3, held-out shapes x3, maze low mass, maze high damping). Severities chosen using only the frozen model.
- Seeds: 5 seeds x 50 episodes (LeWM, TS); 3 seeds x 50 (DINO-WM).

**Measured numbers (final success %, Table 1, predictor adapted; Frozen -> AdaJEPA -> OpTTA)**
- Headline: gain over Frozen averaged across 8 predictor-only shifted settings: OpTTA +13.5 pp vs AdaJEPA +6.9 pp. Predictor+encoder (7 settings): +13.5 vs +7.5. OpTTA beats AdaJEPA in 14 of 15 shifted settings.
- LeWM PushT visual (7 corruptions): 58.5 -> 62.4 -> 68.5
- LeWM Reacher visual: 52.8 -> 60.3 -> 70.8
- LeWM TwoRoom visual: 61.0 -> 57.6 -> 57.2 (both methods WORSE than frozen)
- TS PushT visual: 47.8 -> 58.3 -> 74.0
- DINO-WM PushT visual: 45.3 -> 57.1 -> 58.8
- TS PushT recolors: 38.1 -> 40.8 -> 55.2; PushObj shapes: 34.7 -> 55.2 -> 58.3; maze dynamics: 85.6 -> 87.6 -> 89.2
- In-distribution: LeWM PushT 80.0 -> 88.8 -> 92.0; TS PushT 68.4 -> 81.6 -> 91.6; LeWM TwoRoom 93.6 -> 93.2 -> 91.2.
- Gradient-based planner (TS PushT, 20 replans, rho=1 no ramp): in-dist 59.2 -> 74.4 -> 84.4.
- Ablations (TS PushT darkening 0.7 / LeWM Reacher darkening 0.85): rho in {0.5,1,2} gives 76.0-80.0 / 75.3-76.7 final, all above AdaJEPA (44.7 / 70.7). Negative rho (-0.1, pessimism) gives 32.7 / 60.7, below AdaJEPA, so gain is attributed to optimism. Using only top-1 candidate rollout is worst (58.7 / 65.3); all 200 candidates best on mean-over-replans. Removing sqrt ramp does not reduce final success (79.3 / 78.0).
- Diagnostic (Fig 4): OpTTA's prediction loss on its own executed transitions is not consistently lower than AdaJEPA's; it is lower on an off-policy goal-reaching trajectory (Reacher) and the model becomes more optimistic about distance-to-goal.
- Per-shift note: PushT "red block" shift: Frozen 16.0, AdaJEPA 10.8, OpTTA 18.0 (low for all).

**Compute**
- One RTX PRO 6000. Per-replan ms (planning + adapt): LeWM PushT 118 frozen / 141 AdaJEPA / 163 OpTTA; TS PushT 379 / 389 / 410; DINO-WM PushT 2955 / 2971 / 5233 (OpTTA adaptation +2278 ms; 349 ms if restricted to 30 CEM elites). OpTTA adds ~20 ms on LeWM/TS (+5-16% per replan), claimed offset by fewer replans.

**Stated limits / failure modes**
- TwoRoom: neither AdaJEPA nor OpTTA improves over pretrained; best parameters to update vary by task; "understanding this task dependence remains open".
- Extra per-replan compute; default config "need not be optimal"; alpha design "practical but not necessarily optimal".

**Code**: paper shows "Webpage" and "Code" links but URLs not in retrieved text. States it builds on public AdaJEPA codebase and public LeWM codebase, and that pretrained checkpoints are public.

---

### A2. 2610.22364, T2-Z: Test-Time Optimization of Environment Latents for Robotic World Models
Authors: Zhou, Hong, ... Fei-Fei, Wu, Guibas, Choi (Stanford, CMU, TRI, USC). Preprint, 15 Oct 2026.

**Mechanism**
- Backbone: BWM-based video diffusion transformer initialised from Wan2.2, action-conditioned. Adds a learnable 32-D environment latent z_e, not output by any encoder, no physical-parameter supervision. Two learned projectors map z_e to model width: one into the cross-attention conditioning sequence (environment token), one into the modulation pathway. Original action paths retained.
- Training: separate learnable latent per training environment; alternates (a) world-model update with latents fixed, (b) latent update with world model frozen. Same flow-matching loss in both. Schedule: 300 world-model updates, then blocks of 200 model / 200 latent updates. Model LR 1e-5, latent LR 0.03, latents clipped to [0,1], independent initialisation.
- Test time: freeze world model, initialise a new latent, optimise only the latent on K observed interaction trajectories of the new environment with the same flow-matching loss. 40 optimisation steps, coarse-to-fine LR schedule (exact values n/s in retrieved pages). Then hold latent fixed and predict videos for candidate actions; pick the best candidate by predicted outcome and execute.
- K = 1 trajectory for 7 tasks, 2 for others (Table 7: Friction, Gravity, Mass Collision, Mass x Friction, Diverse Backgrounds K=1; Light Switch, Mass Balance K=2; real Door Close and Soft Pull ablated at K=1,2,4).

**Benchmarks / environments**
- Sim: custom tasks in a LIBERO-based simulator, Franka Panda, 224x224 at 20 fps, 14-channel action. 7 tasks: Friction, Gravity, Mass Collision, Light Switch (discrete rule), Mass Balance, Mass x Friction, Diverse Backgrounds.
- Real: Franka + RealSense D435i, 4 tasks: Door Close (spring latch), Ball Friction (steel beads inside ball), Stick Balance (adjustable end masses), Soft Pull (soft tubes of different stiffness/curvature).
- Action selection: candidate action set always contains at least one action that satisfies the target; model's predicted videos choose; chosen action executed.
- Baselines (same backbone, same test-time trajectories): Standard WM; LoRA TTA (rank 8, alpha 8, attention Q/K/V/O, e.g. 50 AdamW updates LR 1e-4 on Friction); DINOv2 Context (frozen DINOv2-B/14 + GRU -> 32-D context via MLP, no test-time gradient); TTT-KQV (fast-weight 2-layer GELU MLP modules after 8 attention blocks, inner LR base 0.1, trained as in Dalal et al.).

**Measured numbers (Table 1/2)**
- Sim average (7 tasks): Standard WM / LoRA TTA / DINOv2 / TTT-KQV / T2-Z
  - Physical error mean (px): 12.86 / 9.35 / 8.15 / 12.92 / 4.80
  - Final error: 21.98 / 17.53 / 14.80 / 22.55 / 7.67
  - LPIPS: 0.0574 / 0.0584 / 0.0492 / 0.0585 / 0.0430; PSNR: 31.19 / 30.18 / 31.65 / 30.82 / 31.76
  - Action success: 37.9% / 44.8% / 51.8% / 42.1% / 74.2%
- Selected sim tasks (action success): Friction 32 / 48 / 48 / 32 / 72; Gravity 23.5 / 47.1 / 76.5 / 35.3 / 82.4; Mass Collision 38.9 / 33.3 / 38.9 / 55.6 / 61.1; Mass x Friction 35.0 / 26.7 / 60.8 / 33.3 / 69.7; Diverse Backgrounds 45.8 / 58.3 / 58.3 / 58.3 / 66.7.
- Gravity physical error (mean): T2-Z 7.34 vs DINOv2 5.93 (DINOv2 better on error there, T2-Z still higher action success 82.4 vs 76.5). Authors say T2-Z does not lead every visual/error metric on every task.
- Real average: Standard 19.30 mean err / 38.24 final / 26.47% success; LoRA 19.27 / 34.42 / 43.97%; DINOv2 17.50 / 32.04 / 31.86%; TTT-KQV 18.10 / 35.85 / 36.24%; T2-Z 10.85 / 16.66 / 70.80%.
- Real per-task success (Std -> T2-Z): Door Close 22.2 -> 55.6; Ball Friction 0 -> 62.5; Stick Balance 20.0 -> 83.3; Soft Pull 63.6 -> 81.8. Soft Pull LPIPS worse for T2-Z (0.0758 vs 0.0721).
- Ablations (Friction): latent dim 8/16/32/64/128 -> success 44/64/72/64/64% (32 best; ADE 8.92/7.63/3.85/6.86/7.08). Table 3: joint model+latent training 68% (err 7.03), direct 3072-D token 24% (err 94.5, LPIPS 0.76, i.e. diverges), random init range [-0.05,0.05] 60%, alternating 100/100 56%, 400/400 68%, default 200/200 72%. Steps: improves to 40 then saturates (10-60 tested). Interaction count: Light Switch K=1 50% -> K=2 87.5%; Door Close 22.2 -> 55.6 -> 55.6; Mass Collision 61.1 -> 66.7 -> 72.2 (K=1,2,4).
- Latent PCA: 2-D structure reflecting friction and mass in Mass x Friction; consistent across visual backgrounds. Authors say this is association, not proof of disentanglement.

**Compute (one H200)**
- Test-time, K=1: 18.09 s including latent adaptation (40 steps) and five 25-denoising-step query videos, vs 32.73 s LoRA TTA, 37.57 s cached TTT-KQV, ~6.1 s non-adaptive baselines.
- Training per-update (six loss-bearing chunks): world-model step 4.17 s, latent step 1.60 s (vs 1.89 standard, 2.02 DINOv2, 9.08 TTT-KQV); peak memory: 54.3 GiB WM step, 30.4 GiB latent step, 125.3 GiB TTT-KQV.

**Stated limits**
- Custom tasks, one backbone; needs interaction trajectories with known actions and observed outcomes; depends on informativeness of probing action; PCA not proof of disentanglement; further robot platforms/safety needed. Test-time optimisation costs compute.

**Code**: https://github.com/T2-z-latent-optimization/T2-z ; data https://huggingface.co/datasets/huagailuowen/T2-z-datasets ; site https://t2-z-latent-optimization.github.io/ (task-specific adaptation configs "released with the code").

---

### A3. 2610.17434, World-T3: Test-Time Training as Persistent Memory for Real-Time Video World Models
Authors: Wu, Li et al. (Meta Reality Labs, UIC, Brown). 12 Oct 2026.

**Mechanism** (note: this is TTT-as-memory, written during rollout, not adaptation to a distribution shift)
- Backbone: Wan2.2-TI2V-5B (30 DiT blocks, width 3072, 24 heads), camera-controlled (Plucker raymaps), block-causal autoregressive: bounded context = persistent sink block + K recent clean blocks + current block; blocks of B=3 latent frames (2,640 tokens at 1280x704).
- Memory: each transformer layer has a 2-layer MLP "fast weight": fixed W1 maps pose key k to phi(k)=gelu(W1 k); only the readout W2 is updated. Keys from camera raymaps only (not noise); values from clean hidden states of the block evicted from the context window.
- Update rule: not SGD. Closed-form recursive least squares on a weighted ridge regression: min_W2 sum_i w_i ||v_i - W2 phi_i||^2 + lambda||W2 - W2^(0)||_F^2; with P tracked as inverse regularised feature covariance: g = P phi / (1/w + phi^T P phi); W2 += (v - W2 phi) g^T; P -= g phi^T P. lambda=1e-3, P_0 = I/lambda. One write per eviction per layer. Claimed exact minimiser at each step, so invariant to observation order. Learned scalar weight w_j per write; W2^(0) learned base readout (outer loop). Read: query by target camera pose; retrieved feature injected into DiT block via gated residual.
- Rationale: smooth camera trajectories give near-collinear pose keys; plain gradient writes would repeatedly reinforce the dominant direction; RLS whitens.
- Training: 3 stages. S1 bidirectional camera-conditioned teacher (LoRA on attention + FFN projections + camera module). S2 block-causal AR with masked teacher forcing, backbone and memory trained jointly. S3 dual-teacher distribution matching distillation to 4 steps (bidirectional and memory teachers). Data: 908K camera-annotated clips (poses from VGGT-Omega).

**Benchmark / baselines**
- Revisit benchmark: 120 out-and-back clips, 160 frames (10 s @16 FPS), 1280x704, camera reverses at frame 80. Revisit pairs (8,152),(16,144),(24,136),(32,128), all beyond the 21-latent-frame window. Baselines (official checkpoints): Matrix-Game 3.0, HY-World 1.5, DreamX-AR, LingBot-Fast, SANA-WM.

**Measured numbers (Table 1)**
- Revisit PSNR (dB): MG3.0 11.15; HY-World 10.84; DreamX-AR 11.66; LingBot-Fast 10.59; SANA-WM 13.52; World-T3 Stage 2 (50 steps) 16.69; Stage 3 (4 steps) 18.31.
- DINOv2 semantic similarity: SANA-WM 0.9200 -> S2 0.9476 -> S3 0.9610 (GT 0.9982). Absolute RotErr: SANA-WM 3.344 deg -> S3 1.075 deg (GT-VAE-decoded pose estimator noise 0.194 deg).
- VBench (7-dim avg): GT 75.69; HY-World 75.32; MG3.0 74.72; S2 75.15; S3 74.52; SANA-WM 72.87. Paired sign tests favour MG3.0 and HY-World over S3 on VBench. Local Memory score: S2 0.9442 -> S3 0.9258 (DreamX/HY-World higher). Authors state a trade-off: distillation raises revisit fidelity but lowers local perceptual quality.
- Sign tests: S3 beats SANA-WM on DINOv2 revisit distance in 111/120 clips, PSNR 112/120. S3 vs teacher: PSNR 91/120 wins, DINOv2 only 72/120 (p=0.035).
- Sweep tiers (DINOv2 sim, T1/T2/T3): World-T3 S3 0.9582/0.9644/0.9603 flat; MG3.0 0.8743 -> 0.7737 degrades; DreamX-AR 0.9074 -> 0.8289.
- Speed: 20.6 FPS at 1280x704 on 1x B200 with online memory updates (Table 9: SANA-WM 14.2, DreamX 11.0, MG3.0 7.1, HY-World 4.5, LingBot 1.6 FPS single GPU); 56.2 FPS on 8 GPUs. Peak memory 54 GB (SANA-WM 48, HY-World 110, LingBot 144). 30 s video latency 22.59 s.

**Stated limits**
- Relies on 3D foundation model (VGGT-Omega) for 6-DoF poses; pose tracking noise/drift degrades pose-keyed retrieval and may accumulate. (Appendix F continues beyond the retrieved page; any further items not seen.)
- Explicit trade-off: lower VBench / Local Memory after distillation.
- Ablations of RLS vs gradient-descent writes were NOT in the retrieved pages. n/s. (Main text asserts RLS rationale; I did not see a controlled ablation number.)

**Code**: project page https://adreamwu.github.io/World-T3/ ; code release n/s.

---

### A4. 2610.29088, Causal World Models under Physical Parameter Shifts
Authors: Chu Lin, Yi Sun (CCNY). 21 Oct 2026.

**IMPORTANT: this paper contains NO test-time training or test-time adaptation.** All weights are fixed after training; it is included in the cluster by topic (shift robustness) only. Related online-ID methods (UP-OSI, CaDM, RMA, Context-Aware DeLaN) are cited only as related work.

**Mechanism**
- (i) Synthetic: frozen dependency mask from paired interventions: E_ij = mean |y_cf - y_base| per intervened input i, column-normalised, thresholded at 1e-3 of max; mask frozen before fitting predictor.
- (ii) CausalWorld (TriFinger) link-mass extrapolation. Per-finger local models: mass enters inertia M(q,m)=L0L0^T + sum mu_r L_r L_r^T + 1e-4 I and bias h(q,qdot,m) via explicit affine coefficients mu_r = m_r/0.03 kg; actions act either through a learned torque net or the known simulator PD controller (Kp=10, tau_max=0.36 Nm); neural residual for contact does not get masses as input. Object state by separate MLP. Semi-implicit integration with tanh limits.
- Training: one-step (Adam, batch 256, LR 1e-3, 80 epochs) then rollout fine-tuning (horizon 10 then 50, 1,500 updates). Controls: separately trained model with mass inputs removed/fixed at training mean.
- Assumes link masses and low-level control law are available (known), not estimated.

**Benchmark / numbers**
- Synthetic (10 seeds): OOD RMSE fully-connected 1.769 -> intervention mask 0.697 (generator-oracle 0.694); 60.6% reduction.
- CausalWorld pushing + picking, 2 tasks x 5 seeds = 10 runs. Train masses u~U(0,0.4) of range, OOD test U(0.6,1) (mean test/train mass ratio ~1.85); 1,200 light episodes (864/96/240), 300 heavy OOD episodes; 150 steps @250 Hz.
- Known-PD mass-affine: median 150-step joint R2 0.609 (fixed-mass control) -> 0.727 (actual masses); better in 10/10 runs (sign test p~0.002). Median within-run benefit 0.121 OOD vs 0.009 ID. Learned-torque control gets 0.733 (control is stronger than PD control: 0.609 vs 0.733).
- Shared 100-step forecast (Table 4): RWM(Ntr=8) none 0.718, actual masses -0.648; RWM(Ntr=50) none 0.752, actual -0.540; mass-affine learned torque mean 0.768, actual 0.755 (mass gain -0.010, positive in 2/10 runs); known PD mean 0.677, actual 0.746 (+0.071, 10/10). Hold-state reference -0.320. Object RMSE ~2.65-2.82 cm for best models.
- PETS-style ensemble A->B transfer: skill 0.1407 -> 0.5335 with graph input selection; 0.6122 intervention-masked predictor.

**Stated limits**: requires known link masses and (for PD variant) known controller; forecasts only 0.4-0.6 s; dependency structure hand-specified (finger-local routing) not learned; 2 tasks x 5 seeds; longer deployments, uncertain parameter estimates and downstream control untested. Authors note comparison across mechanism families confounds scaling/training settings. Authors give one "plausible explanation" (not tested) for why PD control is weak.
**Code**: n/s (cites official RWM repo github.com/leggedrobotics/rsl_rl_rwm).
**Compute**: n/s.

---

## Part B. Closely related TTA-for-world-model papers found by search

### B1. AdaJEPA, 2606.32026 (Wang, Bounou, LeCun, Ren; NYU/AMI Labs, 30 Jun 2026)
- Mechanism: plan -> execute first action chunk -> adapt on newly observed transition -> replan. L_ada = MSE(f_theta(z_i, E_a(a_i)), sg(z_{i+1})) over replay buffer (recent-5). Updates subset Omega of encoder/predictor params. Default: predictor last block + final LayerNorm + encoder last stage (projection head), 1 gradient step per MPC step, training learning rates (pred 5e-4, enc 1e-5), buffer 5 recent. Episode-local (each episode starts from the same pretrained model).
- Benchmarks: PushT (incl. PushObj shape shifts, visual shifts blur/snp/dark/red agent/red anchor/red block), PointMaze (low mass x0.2, high damping x20, layout shifts). 3 test seeds x 50 episodes, max 20 MPC steps (30 in some plots).
- Numbers: Table 2 (PushT in-dist, GD / CEM, Frozen -> Adapt): TS global feat 84.0 -> 85.3 / 74.0 -> 81.3; TS spatial 91.3 -> 92.0 / 89.3 -> 93.3; DINO-WM 68.0 -> 70.0 / 86.7 -> 90.0. Latency +0.01-0.03 s per replan (H200). Hyperparam ablation (PushObj): T (seen) 50% -> up to 88%, square (unseen) 20% -> up to 51%; lr 5x with 1 step OK but overshoot with more steps; replay buffer matters little (none 84.7 vs recent5 86.7 on T). LoRA (rank 8, alpha 16) also improves but not consistently better than direct selected-layer updating. Training data diversity matters: 4 shapes x 4k trajectories gives 51.9% unseen with AdaJEPA.
- Limits stated: modest gains under red-anchor / red-block shifts (colour used to distinguish objects; suggests augmentation/invariance); frozen model already strong under dynamics shift (attributed to 3-frame history).
- Code: project page https://agenticlearning.ai/adajepa (code release n/s).

### B2. JEPA-TTT, 2610.00722 (Zhang et al., Honda Research Institute USA / JHU, 30 Sep 2026)
Read via the alphaXiv generated report only (not PDF pages), treat details as lower-confidence.
- Mechanism: persistent self-supervised updates of the dynamics predictor (action encoder + autoregressive predictor + prediction projection) with frozen encoder and fixed reward head; MSE to detached target latents; dense replay (windows at every valid offset, growing buffer, uniform minibatch), at most one update per replanning step; predictor, optimiser state and buffer persist across episodes. CEM planner with a fixed reward head fitted pre-deployment (no goal image, no online reward during test).
- Benchmark: PushT, Two-Room, Reacher, OGBench-Cube, 2 dynamics shifts each (rotated actions, altered contacts, nonlinear cross-axis coupling, delayed actuation). Pretrain 3,000 episodes; deploy 500 episodes under one fixed shifted regime; evaluate every 50 episodes on 100 held-out episodes; 3 deployment runs.
- Numbers (report): mean best held-out score frozen 0.267 -> JEPA-TTT 0.678; normalised AUC 0.267 -> 0.571; 5-block latent prediction error down on all shifts (mean -83%, range ~45-99%). Outperforms PPO-TTT (uses online rewards) and AdaJEPA on all shifts. Ablation: persistence vs per-episode reset: best score 0.729 vs 0.289, AUC 0.637 vs 0.273. Sparse 0.497/0.368, compute-matched sparse 0.625/0.495, dense 0.671/0.548, dense+replay 0.678/0.571 (best/AUC).
- Limits stated: one fixed dynamics shift only, no visual/reward shift; forgetting/retention across multiple shifts untested; simulation only; variability from 3 runs of one pretrained model; larger gap vs offline reference on the two Reacher shifts.
- Code: n/s.

### B3. Sandwich-Residuals, 2609.21740 (Soni, Sehgal, Dave, Rueckert; Leoben, 18 Sep 2026)
Read via alphaXiv report only.
- Mechanism: all pretrained encoders/predictor frozen; trainable residual modules around the predictor: linear action-embedding residual conditioned on [action; proprio]; visual input residual MLP (64 hidden, GELU), visual output residual MLP, proprio output residual (linear). 99,520 trainable params (~1.0% of AdaJEPA's 9.93M in maze; 2.7% of 3.66M in push). One Adam step per plan-execute cycle on the same AdaJEPA latent loss over 5 recent transitions; LR 1e-2 for linear residuals, 5e-4 for visual MLP residuals; linear init zero; reset each episode.
- Numbers: 21 primary conditions: frozen 49.2%, Sandwich 65.0%, AdaJEPA predlast+enclast 63.4%, predfirst+enclast 68.2%. 7 compound shifts: ~66.0% vs frozen ~34.5%. Medium maze 50x damping: 46.4 -> 66.0 (AdaJEPA variants 52.0 / 59.2). DINO-WM OGBench-Cube (4 conditions): 60.0% vs 58.7% frozen (modest). Ablation: output residuals dominant in maze (removing: 65.6 -> 56.4); in PushT all pathways contribute (removing action residual -10, output -15.6, input -16.4 points).
- Limits stated: red-block appearance <15% for all methods; cannot recover information lost in the frozen encoder; gradients still flow through the frozen predictor (param count != compute); simulation only; residuals reset between episodes (no cross-episode retention).
- Code: n/s.

### B4. WorldAgen, 2609.08162 (Wan et al., Northwestern, 8 Sep 2026)
- Mechanism: shared Qwen3 transformer backbone with a world-model head and an agent (policy) head; test-time: random exploratory rollouts in the new environment (language relabelled "no-lang"), then LoRA fine-tuning on the observation-reconstruction loss L_o of the world model only; policy head frozen. LoRA on q/k/v/o and gate/up/down projections.
- Numbers: CALVIN avg len 3.87 -> 3.93 with TTT (SOTA Seer-Large 3.83); LIBERO-10 avg 75.5 -> 79.0. Pre-TTT baseline already competitive, so TTT gain is +0.06 length / +3.5 pp. LoRA rank 16-256 insensitive (3.918-3.930). TTT data: 6 samples 3.871, 90: 3.922, 204: 3.928, 340: 3.917 (degrades past ~200 samples; overfitting to world-modeling). Full fine-tune 3.85 (below no-TTT 3.87), LoRA 3.93. Without world modelling at all: CALVIN 2.96, LIBERO 46.5%.
- Compute: ~8 min per task (CALVIN), ~2 min per scene (LIBERO) on RTX 4090.
- Limits stated: single model size/architecture; only world-model component adapted. "Sim-to-real" check is just Gaussian noise (std 0.1) on observations, not a real robot.
- Code: https://worldagen.github.io (code n/s).

---

## Part C. Original and foundational TTT work

### C1. Test-Time Training with Self-Supervision, 1909.13231 (Sun, Wang, Liu, Miller, Efros, Hardt; ICML 2020)
- Mechanism: Y-shaped network, shared feature extractor theta_e, main head, self-supervised head (rotation prediction, 4-way). Joint training with L = L_main + L_ssl. At test time minimise L_ssl(x) over theta_e on a single test sample (augmented copies), then predict with updated weights.
  - Standard: 10 SGD steps, LR 0.001 (last-epoch LR), momentum and weight decay 0, theta reset after each sample.
  - Online: 1 step per new image, parameters carried over the stream.
  - ResNet-26 (CIFAR-10), ResNet-18 (ImageNet), Group Norm (BatchNorm hurts with single-image batches; with BN, only hard-example mining works for standard TTT; online does not work with BN).
- Numbers (CIFAR-10-C level 5 error %, baseline B / joint training JT / TTT / TTT-Online): Gaussian noise 50.5 / 49.4 / 45.6 / 25.8; shot 47.2 / 45.3 / 41.8 / 22.6; impulse 56.1 / 53.4 / 50.0 / 30.6; pixelate 55.8 / 51.6 / 47.2 / 18.1; contrast 25.0 / 25.3 / 23.9 / 15.6; original 8.9 / 8.1 / 7.9 / 8.2. ImageNet-C level 5 accuracy (ResNet-18; B / TTT / TTT-Online): Gaussian noise 1.3 / 3.1 / 26.3; fog 15.3 / 20.0 / 47.6; original 68.9 / 69.0 / 68.8. Online TTT beats UDA-SS (which sees the whole test set) on 13 of 15 corruptions.
- Theory: for convex models, positive gradient inner product between main and self-supervised losses guarantees improvement; empirical gradient inner product vs improvement correlation 0.93 (standard) / 0.89 (online).
- Stated limits: online assumes same or gradually changing shift; BN incompatibility; "forgetting" the training distribution is not harmful in their setting.
- Code: project website https://test-time-training.github.io/ (as stated).

### C2. Learning to (Learn at Test Time): TTT layers, 2407.04620 (Sun, Li, Dalal et al.; Stanford, UCSD, Berkeley, Meta)
- Mechanism: hidden state is the weights W of an inner model f; update rule is a gradient step on a self-supervised reconstruction loss. W_t = W_{t-1} - eta * grad l(W_{t-1}; x_t), l = ||f(theta_K x; W) - theta_V x||^2, output z_t = f(theta_Q x_t; W_t). theta_K,V,Q are outer-loop (slow) parameters, W is the inner (fast) state.
  - TTT-Linear: f(x) = x + LN(W x); TTT-MLP: 2-layer MLP (hidden 4x, GELU) with LN and residual.
  - Inner mini-batch b=16 tokens (b=1 online GD, b=T batch GD). Dual form (matmul-based) for speed, gradient checkpointing through time.
  - Learnable W_0 (needed for stability), learnable token-dependent eta = eta_base * sigmoid(theta_lr . x); eta_base 1 (Linear), 0.1 (MLP). Mamba backbone (conv before TTT) helps.
- Numbers: models 125M-1.3B, Pile 2k/8k and Books3 1k-32k, Chinchilla recipe from Mamba paper. Ablation at 125M (ppl): linear attention 15.91 -> improved 15.23 -> +LN & residual 14.05 -> +mini-batch TTT 12.35 -> +learnable eta 11.99 -> +Mamba backbone 11.09 (TTT-Linear final). Biggest gains from mini-batch TTT (-1.70) and LN+residual in f (-1.22). At 8k context TTT roughly equals Mamba; at long context (to 32k) TTT-Linear and TTT-MLP keep reducing perplexity while Mamba plateaus after 16k. Exact perplexity values per scale are in figures, not retrieved.
- Stated limits: TTT-MLP still has memory I/O problems and higher wall-clock than FLOPs suggest; at 2k Books, Mamba slightly better than TTT-Linear; Mamba2 and Gated DeltaNet came after the experiments (not compared in the original runs). Authors: "remains to be seen" whether instantiations beat wall-clock limits. Perplexity worsens at very long context for models trained from scratch.
- Compute: not itemised beyond model sizes (training tokens 2.5B-26B per Table 3); kernels in JAX and PyTorch; "Code available in JAX and PyTorch" (URL n/s in retrieved text).

### C3. One-Minute Video Generation with Test-Time Training, 2504.05298 (Dalal et al.; NVIDIA, Stanford, UCSD; CVPR 2025)
- Mechanism: add TTT-MLP layers (same design as C2, inner LR eta=0.1, inner mini-batch b=64) after each attention layer of pretrained CogVideoX-5B; gate tanh(alpha) with alpha init 0.1; bidirectional by running a reversed copy (TTT'(X)=rev(TTT(rev(X)))) sharing theta_K,V,Q but separate gates. Self-attention restricted to 3 s segments; TTT global across the full sequence (>300k tokens for 63 s). Five-stage context extension 3 s -> 9 -> 18 -> 30 -> 63 s (steps 5000/5000/1000/500/250; LR 1e-4 TTT, 1e-5 rest in stage 1; 1e-5 after, TTT + local attn QKVO trained). On-chip tensor-parallel Hopper kernel (ThunderKittens) with DSMEM all-reduce.
- Data: ~7 hours Tom and Jerry (81 episodes), human-annotated storyboards. Compute: ~50 hours on 256 H100s.
- Numbers (human Elo, 63 s videos, 100 videos per method): TTT-MLP average 1033 vs Sliding window 999, Gated DeltaNet 991, Mamba2 978 (+34 over second best). Per-axis: temporal consistency 1042 (vs 1004 GDN), motion naturalness 1039 (vs 1000), aesthetics 1037, text following 1014. Speed: TTT-MLP inference 2.5x and training 3.8x time of local attention at 63 s (full attention 11x/12x); TTT-MLP is 1.4x (inference) and 2.1x (training) slower than Gated DeltaNet.
- Stated limits: at 18 s Gated DeltaNet is best (avg 1032 vs TTT-MLP 1004; 28 Elo ahead); wall-clock worse than GDN/Mamba2; artifacts (object morphing at 3-s segment boundaries, floating objects, lighting inconsistent) attributed to the limited 5B base model; only one-minute, one-domain (cartoon) tested.
- Code: sample videos and code at https://test-time-training.github.io/video-dit (as stated).

### C4. Test-Time Training Done Right (LaCT), 2505.23884 (Zhang et al.; MIT, Adobe)
- Mechanism: very large chunk TTT (2K to 1M tokens per update) instead of per-token or 16-64 token updates. Fast weight = SwiGLU MLP W={W1,W2,W3} without bias, f_W(x)=W2[SiLU(W1 x) * (W3 x)]. Loss = negative dot product -f_W(k)^T v. Update: gradient of the sum over the chunk of per-token-lr-weighted losses; then optimiser step with Muon (orthogonalised gradient, via Newton-Schulz) with L2 weight renormalisation. Window attention within chunk for local structure. Apply/update ordering controlled by block-causal masks.
- Efficiency claims: up to 70% GPU utilisation on A100 (original TTT <5%); state size up to 40% of model parameters (prior 0.1-5%); pure PyTorch, "few dozen lines".
- Tasks: novel view synthesis (up to 128 images at 960x536, 1M tokens, claims to beat 3D Gaussian Splatting under that scale; 312M model, 84M fast weights), language modelling (760M and 3B, chunk 2048-4096, to 32K), autoregressive video diffusion (adapted 14B video DiT, 56K visual tokens, chunk = 3 frames).
- Ablations: larger state consistently better (state 12 d^2 per block = 40% of params); Muon > momentum > vanilla GD; nonlinear SwiGLU fast weights beat linear even at smaller state; large-chunk beats per-token Mamba2 for NVS at equal state; for language, linear large-chunk underperforms GLA at equal state but exceeds per-token methods with big nonlinear state + Muon. Specific PSNR/perplexity numbers are in figures/tables not retrieved.
- Stated limits: no rotation invariance; only three tasks; language reasoning and parameter scaling unexplored (compute); video metric hard to discriminate; unposed reconstruction untested. 
- Code: authors say they open-sourced code and weights; site https://tianyuanzhang.com/projects/ttt-done-right/.

### C5. Titans: Learning to Memorize at Test Time, 2501.00663 (Behrouz, Zhong, Mirrokni; Google)
- Mechanism: neural long-term memory M (deep MLP) updated at test time by gradient of an associative-memory loss ("surprise"). Update with momentum and decay: S_t = eta_t S_{t-1} - theta_t grad l(M_{t-1}; x_t); M_t = (1 - alpha_t) M_{t-1} + S_t (weight decay = forgetting gate), data-dependent eta, theta, alpha. Mini-batch/tensorised parallel training. Combined with attention as memory-as-context (MAC), gate (MAG) or layer (MAL), plus persistent (data-independent) memory tokens.
- Numbers: 340M-760M params, 15-30B FineWeb-Edu tokens. At 760M: neural memory alone (LMM) Wiki ppl 20.04 and 51.56 avg accuracy; MAC 19.93 / 52.51; MAG 18.61 / 52.50; vs Gated DeltaNet 21.18 / 49.69, Transformer++ 25.21 / 48.69, TTT 24.17 / 47.32. S-NIAH-Needle (16k): LMM 80.2, MAC 97.4 vs TTT 4.4, Mamba2 0.0; S-NIAH-W 16k: LMM 80.6, MAC 95.2 vs 0.0. Ablation (avg acc / ppl): w/o weight decay 45.11/29.04, w/o momentum 45.49/28.98, linear memory 46.97/28.49 vs full LMM 47.83/27.01. Claims >2M context. BABILong beats GPT-4 and others in fine-tuning setting (numbers only in figures).
- Stated limits: this is the first version; "working on finalizing larger models". Throughput of the memory module slightly slower than Mamba2 / Gated DeltaNet. Code: "intend to release soon" (n/s URL).
- [INFERENCE] The ablation attributing most of the gain to forgetting (weight decay) and momentum is relevant to World-T3's separate choice of RLS (no forgetting term other than ridge prior to base readout).

### C6. The Surprising Effectiveness of Test-Time Training for Few-Shot Learning, 2411.07279 (Akyurek et al.; MIT; ICML 2025)
- Mechanism: per-task LoRA adapters trained at inference on a synthetic dataset built from the task's own demonstrations: leave-one-out in-context tasks (each demo becomes a synthetic test with others as demonstrations, with permutations), plus invertible geometric augmentations for ARC (rotations, flips, transpose, resolution up-scaling, colour/example shuffles). Loss: LM loss on test output plus demonstration outputs ("all outputs"). LoRA rank 128, alpha 16 (ARC), AdamW, LR 5e-5 (1B/3B) / 1e-4 (8B), 2 epochs, batch 1-2, up to 250 examples per task; BBH uses rank 64 and 40 shuffles. Inference: augmented greedy decoding with hierarchical voting.
- Numbers: ARC (80-task validation subset, 1B model, pass@2) FT 5 tasks -> FT+TTT 29 tasks (about 6x). Full 8B on ARC public validation: FT 18.3% -> TTT 47.1%; with BARC's model 53.0%; ensembling with program synthesis 58.5% / 61.9% (avg human 60.2%). Semi-private evaluation: 47.5% (authors speculate larger distribution shift). Ablations (tasks solved of 80): no transformations 13, direct I/O 18, shared adapter 22, no demo loss 26, full 29. Hierarchical voting 29 vs flat 27 vs oracle 32. BBH (Llama-3.1-8B, 10-shot): ICL 50.5% -> TTT 57.8% (+7.3), zero-shot 40.9%; direct I/O 51.5%; no permutations 55.7%; test-output loss only 54.4%; shared adapter 59.8% (shared helps on BBH, hurts on ARC).
- Compute: whole TTT + inference ~12 hours for 100 validation tasks on one A100 (per the appendix).
- Stated limits: scaling after TTT unclear (1B and 3B identical after TTT); semi-private drop; per-task adapters mean N tasks = N adapters.
- Code: https://github.com/ekinakyurek/marc (ARC), https://github.com/adamzweiger/Fewshot-TTT (BBH), as stated.

---

## Part D. Other TTT papers surfaced by search but NOT read closely (title and one-line only; details n/s)
- 2307.05014, Test-Time Training on Video Streams (Berkeley/Meta/UCSD, 2023).
- 2606.26515, Forget, Anticipate and Adapt: TTT for Long Videos (UCF).
- 2610.04701, Test-Time Training as Residual Memory for Robot Policies (UIC/Cisco).
- 2607.06988, WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time (PKU/Galbot).
- 2608.07110, Modular TTT; 2604.06169, In-Place TTT (ByteDance, LLM weights adapting to streams); 2512.01643, ViT^3 TTT in vision; 2610.38255 ParaTTT; 2609.34174 GradLev (token-parallel TTT); 2610.05484 Universal TTT; 2610.16268 Error Correction in Trained TTT Layers.
- Cited in the assigned papers but not retrieved: RoboTTT 2607.15275; TTT-VLA 2606.03127; Test-time training for visual foresight VLA 2605.08215; Test-time mixture of world models (ICLR 2026); World model implanting (ICML 2025); Retrieve-then-steer 2605.10094; Agentic-VLA 2605.22896; Optimistic world models 2602.10044 (the RBMLE-RL precursor of OpTTA); TTT3R 3D reconstruction as TTT (ICLR 2026).

---

## Cross-paper comparison table (stated numbers only)

| Paper | What is adapted at test time | Loss/update | Steps | Persistence | Needs reward/goal | Params touched |
|---|---|---|---|---|---|---|
| TTT 2020 (1909.13231) | shared feature extractor | rotation-prediction CE, SGD lr 1e-3 | 10 (standard), 1 (online) | per-sample, or stream | no | conv trunk |
| TTT layers (2407.04620) | inner model W (hidden state) | recon MSE, GD, b=16, learned eta | 1 per 16 tokens | within sequence | no | fast weights |
| Video TTT (2504.05298) | inner MLP W | same, b=64 | 1 per 64 tokens | within video | no | fast weights |
| LaCT (2505.23884) | SwiGLU fast weights | -f(k).v, Muon | 1 per 2K-1M tokens | within sequence | no | fast weights up to 40% of params |
| Titans (2501.00663) | deep memory MLP | surprise gradient + momentum + decay | per token/chunk | within sequence | no | memory |
| TTT-ARC (2411.07279) | LoRA | LM loss on leave-one-out | 2 epochs, <=250 examples | per task | no (uses demos) | LoRA r=128 |
| AdaJEPA (2606.32026) | last pred block + enc stage | latent MSE, Adam | 1 per replan | per episode | goal for planning only | small subset |
| OpTTA (2610.35636) | same | latent MSE + alpha * planning cost | 1 per replan | per episode | yes (goal cost) | small subset |
| JEPA-TTT (2610.00722) | predictor | latent MSE, dense replay | <=1 per replan | across 500 episodes | no | predictor |
| Sandwich-Residuals (2609.21740) | residual adapters | latent MSE | 1 per replan | per episode | goal for planning | ~99.5K |
| T2-Z (2610.22364) | 32-D environment latent | flow matching | 40 | per environment | no | 32 numbers |
| WorldAgen (2609.08162) | LoRA on backbone | obs reconstruction | n/s (approx 8 min on 4090) | per task/scene | no | LoRA |
| World-T3 (2610.17434) | W2 readout of pose-keyed MLP | closed-form RLS | 1 per block eviction per layer | within rollout | no | fast weights |
| Causal WM (2610.29088) | none | none | 0 | n/a | n/a | n/a |

---

## Open gaps

**Things not tested in these papers**
1. No cross-comparison on a common benchmark. T2-Z (video diffusion, hidden physics, action selection among candidates), OpTTA/AdaJEPA/JEPA-TTT/Sandwich (latent JEPA, goal-conditioned CEM), and World-T3 (revisit consistency) share no environment, metric or model. Numbers across them are not comparable.
2. Real-robot evidence is thin: only T2-Z has real-robot experiments (4 tasks, few trials: success percentages such as 55.56%, 62.5%, 83.33%, 81.82% imply denominators of order 6-12 per task; exact counts n/s). All JEPA-TTA papers (AdaJEPA, OpTTA, JEPA-TTT, Sandwich) are simulation-only. WorldAgen's "sim-to-real" is added observation noise.
3. Closed-loop control with the adapted video world model was not run in T2-Z: success is "pick the best of a candidate set that always contains a valid action, then execute", i.e. selection/ranking, not planning or policy learning. The candidate set guarantee is stated by the authors.
4. Multiple or switching shifts, forgetting, and return to an earlier regime: untested everywhere except implicitly. JEPA-TTT says retention across shifts is not evaluated; AdaJEPA/OpTTA/Sandwich reset every episode so persistence is never tested; T2-Z resets per environment.
5. Combining the mechanisms: no paper combines a hidden-environment latent (T2-Z) with optimistic planning loss (OpTTA), or with persistent memory (World-T3), or measures them against each other. T2-Z's TTT-KQV baseline is a fast-weight memory adapted from Dalal et al. and fares poorly there (42.1% sim, 36.2% real action success), but World-T3 never compares RLS fast-weight memory against environment-latent optimisation for dynamics identification.
6. World-T3: no ablation of RLS vs gradient-descent writes in retrieved pages; no test of memory under dynamic scenes (benchmark is camera revisit of mostly static content; revisit pairs compare generated outbound vs return); no non-camera actions (robot actions); pose-keyed retrieval needs accurate poses. Whether this memory can carry physical state (object moved by the robot) is untested. [INFERENCE: the revisit benchmark rewards remembering static appearance, and a pose-keyed store would by construction return the first-seen appearance even if the scene changed; not tested.]
7. Test-time compute scaling curves: T2-Z shows steps saturate at 40 and K saturates at 1-2; OpTTA/AdaJEPA study 1-5 steps; none study whether more adaptation steps overfit or destabilise at longer horizons beyond what is shown. WorldAgen shows performance falling when TTT data exceeds ~200 samples.
8. Safety/uncertainty: none of the papers calibrates uncertainty of the adapted model; OpTTA explicitly avoids uncertainty estimates and adds optimism; there is no evaluation of failure cost (e.g. collisions) when the optimistic model is wrong.
9. Language-side TTT (ARC, BBH) was not connected to physical AI by any of the world-model papers except citation; AdaJEPA cites Akyurek et al. only as related work. Transferability of leave-one-out + augmentation data synthesis to world-model adaptation (e.g. leave-one-transition-out, symmetries of physical scenes) is not tested anywhere I read. [INFERENCE for relevance; the absence of tests is factual for the retrieved text.]

**Contradictions / tensions between papers**
1. Is more adaptation capacity better? Sandwich-Residuals (99.5K params) gets ~95% of the best AdaJEPA variant; T2-Z adapts only 32 numbers and beats LoRA TTA and a fast-weight TTT module on its benchmark; WorldAgen finds full fine-tune worse than LoRA and LoRA rank irrelevant; LaCT finds larger fast-weight state (up to 40% of params) consistently better. Different regimes (per-environment identification vs long-context memory), but the papers do not reconcile this.
2. Prediction loss vs task performance. OpTTA shows lower own-trajectory prediction loss is not what drives gains; T2-Z shows LPIPS/PSNR and action success can move opposite ways (Soft Pull; also Gravity physical error better for DINOv2 context while T2-Z has higher success); JEPA-TTT reports prediction error down 83% with planning improved; World-T3's distillation raises revisit PSNR while lowering VBench. Metric choice changes the conclusion; no common standard.
3. Fast-weight TTT works for memory but not as environment adaptation in T2-Z: TTT-KQV (Dalal-style) is near or below the standard model on several metrics there (sim average physical error 12.92 vs 12.86), whereas World-T3 and Titans/LaCT present fast-weight test-time updates as the winning mechanism. Authors of T2-Z note that TTT-KQV replays observed interactions at each denoising timestep; the discrepancy may be due to how it is trained or the very small K, which no paper tests. [INFERENCE for cause.]
4. Persistence: JEPA-TTT reports persisting across episodes gives best score 0.729 vs 0.289 with reset; AdaJEPA, OpTTA, Sandwich all use per-episode reset by design. OpTTA claims "no harm" in-distribution; JEPA-TTT's no-shift control finds no consistent benefit. Sandwich lists lack of cross-episode retention as a limitation.
5. Does TTA help or hurt under visual shift? OpTTA and AdaJEPA both degrade below frozen on LeWM TwoRoom visual corruptions (OpTTA 57.2, AdaJEPA 57.6 vs frozen 61.0) yet help elsewhere; the red-block shift remains low for all. Contradiction with the "TTA is safe in-distribution" claim in AdaJEPA: OpTTA TwoRoom in-distribution 93.6 -> 91.2.
6. Frozen model strength under dynamics shift differs: AdaJEPA notes the frozen model already performs strongly on new maze dynamics (attributed to 3-frame history), OpTTA maze dynamics frozen is 85.6%, whereas JEPA-TTT frozen model scores 0.267 mean best score on dynamics shifts it designs itself. Shift severity is not standardised, so "TTA helps under dynamics shift" is not comparable across papers.
7. The causal-models paper (2610.29088) argues for structure plus known physical parameters rather than adaptation, and finds that supplying out-of-range masses to MLP/GRU world models strongly degrades prediction (RWM R2 0.752 -> -0.540). This conflicts with the T2-Z premise only if latent inputs were similarly ill-behaved; T2-Z clips latents to [0,1] and its ablation shows a raw 3072-D conditioning token diverges (physical error 94.5, success 24%). Both are evidence that unconstrained conditioning inputs extrapolate badly; neither tests the other's remedy.

**Weak or untested assumptions**
1. T2-Z assumes test-time interaction trajectories with known actions and observed outcomes from the same environment, in effect a labelled probing phase, and that the probing action is informative. Light Switch needs K=2 (50% -> 87.5%), Door Close needs K=2 for gains; there is no analysis of which probe actions are informative or how to choose them.
2. T2-Z latent range: latents clipped to [0,1] with test-time latents from a different optimisation budget than training latents (training uses alternating 200-step blocks; test uses 40 steps with coarse-to-fine schedule). Whether test-time latents land inside the training-latent manifold for truly novel environments is not tested; PCA plots overlay a handful of test latents.
3. OpTTA assumes the planner's candidate rollouts (CEM final iteration) are informative samples of the plausible action space and that a goal cost is available at test time; optimism term can push model to believe unreachable goals are reachable. The pessimism sign-flip ablation is the only check that the sign matters; no test where the goal is infeasible. Hyperparameters (gradient-balanced rho=1) set once, with "not necessarily optimal".
4. OpTTA's comparison fixes AdaJEPA's one-step update protocol; the authors note AdaJEPA is sensitive to which block is adapted (first vs last block changes AdaJEPA final success 44.7 -> 68.7 on PushT). So the OpTTA margin partly depends on a sub-optimal baseline configuration. OpTTA itself is insensitive to block choice (76.0 vs 72.7).
5. World-T3: assumes accurate 6-DoF camera poses as memory keys (acknowledged); assumes revisit returns to an unchanged scene; memory is overwritten via ridge regression with a single fixed lambda=1e-3 and learned scalar weights, no capacity or interference analysis (e.g. how many distinct poses before recall degrades) in retrieved pages; the revisit benchmark has only 4 pose pairs per clip at 24-36 latent-frame gaps (10 s clips). Long-horizon (30 s) results are qualitative only.
6. JEPA-TTT assumes a pre-fitted reward head and a single fixed dynamics shift lasting 500 episodes; no goal; relies on 3,000 pretraining episodes per environment; variability from 3 runs on a single pretrained model (authors flag).
7. Statistical power: OpTTA uses 5 seeds x 50 episodes (3 seeds for DINO-WM); AdaJEPA 3 seeds x 50; T2-Z real-world tasks have small trial counts with no confidence intervals reported in retrieved text; World-T3 uses paired sign tests over 120 clips (good) but a single benchmark constructed by the authors.
8. Benchmarks are custom or author-constructed in T2-Z (custom LIBERO-based tasks, authors' own real tasks), World-T3 (revisit set), and the causal paper (two CausalWorld tasks, 5 seeds, 0.4-0.6 s forecasts); results may not transfer to standard benchmarks.
9. Causal-models paper assumes link masses and (PD variant) the low-level controller are exactly known; unknown or noisy parameters, which is where TTT/system-identification would sit, are explicitly not tested.
10. Several PDFs were read through ranked page subsets; for JEPA-TTT and Sandwich-Residuals only the alphaXiv-generated summary was read, so their numbers are less verified than those taken from tables above.
