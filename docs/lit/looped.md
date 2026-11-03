# Looped / recurrent-depth models: literature notes for world models and physical AI

Scope: looped / recurrent-depth transformers from any domain. Source: alphaXiv full-text queries, 2026-11-04.
Convention: "stated" = in the paper text I read. "[INFERENCE]" = mine. "n/s" = not stated in the text retrieved. "Excerpt" means alphaXiv returns only the most relevant pages, so "n/s" can mean "not in the pages I saw". Where it matters I say so.
Notation: T = loops, mu_rec = mean training loops (Poisson), TBPTT = truncated backprop through loops, prelude/recurrent/coda = P/R/C.

---------------------------------------------------------------------
## 0. One-screen summary (stated facts only unless marked)

| Paper | Domain | Evidence level for "world model / physical AI" use |
|---|---|---|
| 2606.18208 LoopWM | text world model (ScienceWorld, AlfWorld), ~1B | Only world-model paper. No loop counts, no ablations, no stability or rollout-length experiment, no code. |
| 2609.37379 Looped Transformers as Optimizers | LM pretraining, MoE 5B/10B | Loop-transition design. Single seed. No test-time loop extrapolation tested. |
| 2610.07591 Recurrent Looped Transformer | synthetic algorithmic tasks, 25-29M | Cross-token recurrence (not depth loop). 3 seeds, CPU, tiny. |
| 2610.23036 Looped Models Done Right I | LM pretraining, 730M dense, 8BA0.8B MoE | Best controlled ablation of envelope / injection / state init. Training depth only; no loop extrapolation. |
| 2611.02857 Solv3R | multi-view 3D geometry, 117M | Closest to physical AI. Stationary operator, stable to 1024 calls, stop rule, streaming. Code "will be released". |

---------------------------------------------------------------------
## 1. The five assigned papers

### 1.1 2606.18208 Looped World Models (LoopWM), FaceMind Research Asia
URL: https://arxiv.org/abs/2606.18208 (v1 16 Jun 2026)

**Architecture (stated)**
- Modules: observation encoder E (conv or ViT) -> e_k; action embedder A -> u_k; looped dynamics core L; MLP heads (obs or latent target, reward, continuation).
- Core is prelude / recurrent / coda. Prelude: L_P layers on concat [h_{k-1}; e_k; u_k], then LayerNorm -> conditioning signal e. Recurrent block: L_R layers, weights shared over T loops. Coda: L_C layers, not shared, plus a projection.
- Recurrence: h^(t+1) = A_bar h^(t) + B_bar e + R_bar(h^(t), e). A_bar = exp(Delta * diag(-exp(a))), diagonal with entries in (0,1) (Parcae parameterization). B_bar = Delta * B, unconstrained, with LN on e.
- State: h^(0) ~ N(0, sigma^2 I), or, for temporal rollouts, the previous env step's final hidden state ("cross-timestep state propagation"; inner loop = refinement, outer loop = time).
- Imagination rollouts: encoder bypassed, h_{k+1} = L(h_k, 0, u_k).
- Loop count at train: T ~ Poisson(mu_rec) with learnable mu_rec, sampled per sequence. At test: adaptive early exit via a 1-layer MLP+sigmoid gate g(t) > threshold tau, plus entropy regulariser on the gate. Tmax can exceed mu_rec ("test-time compute scaling").
- Gradient: TBPTT truncated at ceil(mu_rec/2) loops.
- Deferred Decoding (DD): K latent transitions with no intermediate decode, decode only h_K. Loss = terminal loss + latent-consistency loss (projection g_w(h_k) vs stop-grad frozen-encoder embedding of o_k) + a hinge penalty on cumulative ||h_{k+1}-h_k||. Curriculum K = min(Kmax, 1 + floor(step/Delta)). Authors say direct large K is unstable because gradients pass through K x T applications.
- Exact values of L_P, L_R, L_C, T, mu_rec, tau, K, d: n/s.

**Task and benchmark (stated)**
- ScienceWorld world-modelling: feed 5 consecutive actions, score the predicted text state. Metrics EM / token-F1 / BLEU-4 / Entity. AlfWorld similarly. Danmaku (live-comment) generation human eval. The 'observations' are text, not pixels or continuous physics.
- Model ~1B params. Base model / pretraining / data size / compute: n/s.

**Numbers (stated)**
- ScienceWorld overall (EM / F1 / BLEU / Entity): LoopWM 68.4 / 85.3 / 80.7 / 83.9. Claude-Opus-4-6-Max 47.2 / 72.8 / 64.4 / 72.3. Gemini-3-flash-preview-thinking 30.8 / 68.9 / 51.1 / 73.8. Qwen-3.5-flash 10.0 / 46.9 / 26.7 / 63.0.
- Per-task examples: Lifespan 100 EM vs Opus 0; LifeStages 0.0 EM for both LoopWM and Opus. LoopWM is lower than Opus on Thermometer F1 (85.3 vs 98.1) and Grow F1 (80.0 vs 81.6); Freeze EM is only 25.0 for LoopWM.
- AlfWorld: best BLEU (71.6) among 4 models, second on EM and F1; "entity scores relatively low" (error analysis). Table numbers beyond BLEU not retrieved.
- DD effect (Tables 5/6): average over tasks, relative EM gain vs Gemini-3-flash: +73.2% (step1), +54.5, +103.6, +82.9, +113.8% (step5). Boil task step4 +700.9% EM. Entity −8.0% at step 1 on Boil.
- "Up to 100x parameter efficiency": stated vs closed API models (Opus). Opus size not stated. The 100x is "more than 100x smaller" than the API model.
- Adaptive-exit FLOP savings ("~25x per simple step", "two orders of magnitude") are computed from a hypothetical 100-layer baseline (Sec 3.4). No measured FLOP reduction reported in anything I retrieved.

**Stated limits / failure modes**
- Authors state the paper "stops short" of a scaling-law characterisation and that training "can benefit from curriculum-like engineering strategies". Section 6 says the manuscript is "intentionally selective in disclosure scope" and the limitation is "presentation scope".
- No measured divergence with loop count, because there is no experiment varying loop count or rollout length that I could find. The "provably stable" claim is about the linear A_bar retention term only.

**Code**: none found.
**Compute**: n/s.

**[INFERENCE] flags**
- Claims in abstract/intro (stable long rollouts, adaptive compute, loop-count scaling) have no matching experiment in the text I read; the evidence reduces to the ScienceWorld/AlfWorld tables and DD tables.
- DD tables compare LoopWM-with-DD against Gemini, not against LoopWM-without-DD, so DD's own effect is not isolated.
- No non-looped ~1B baseline is reported, so the contribution of looping vs just a fine-tuned 1B LM is not isolated.
- Eq. (4) feeds h_{k-1} into the prelude, while Sec 3.2 also uses h_{k-1} as h^(0): possible double use of the carried state; spec ambiguous.
- The (0,1) diagonal A_bar bounds only the linear part. Other papers below find boundedness needs more (see Parcae, Solv3R, FPRM).

---------------------------------------------------------------------
### 1.2 2609.37379 Looped Transformers as Optimizers (OperLoop), HKUST(GZ)/StepFun et al.
URL: https://arxiv.org/abs/2609.37379 (29 Sep 2026)

**Architecture (stated)**
- Middle-loop prelude / recurrent / coda; shared K blocks looped R times. Backbone: MoE decoder, SWA:full attention 3:1, width 1024, 256 experts top-8.
- Loop state is treated as a "fast weight" Y^l updated by a local gradient-style step: Y^{l+1} = (I - eta Lambda) Y^l - eta (H_in)^T f'(o^l), o^l = H_in Y^l. The paper treats the shared block's output as a target; training is end-to-end through all loops ("All recurrent steps are fully backpropagated").
- Input-map alignment: Parcae -> y_{l+1} = A_bar^T Blocks(A_bar y + B_bar e); HyperLoop -> tie H_post = H_pre (also reduces params).
- OperLoop: HyperLoop's 4 parallel streams (r=4) with delta (squared-error) objective, state-dependent input map / diagonal decay / step size from RMSNorm(flatten(Y)); step size causal: eta_l = sigma(.) * eta_{l-1} (monotone decreasing).
- Input injection: via the transition (bias e_l and the read/write maps). Initial state Y_0 = prelude output duplicated into 4 streams. Stopping rule: none (fixed loops).
- Loops: Loop3 (layout 4-4x3-2, 18 effective layers, 10 distinct blocks) and Loop6 (4-4x6-2, 30 effective layers), trained separately and evaluated at their trained depth. Larger scale: 4-8x3-4 (32 effective, 16 distinct).

**Task / benchmark**: LM pretraining on ~300B tokens (5B model, 67K updates) and ~500B tokens (10B, 120K updates), seq 4096. Eval: Wikitext/Lambada PPL, commonsense LL, GSM8K, HumanEval(+), MMLU, BBH.

**Numbers (stated)**
- Input-map alignment, 5B scale (commonsense avg): Parcae 59.14 -> 59.29; HyperLoop 58.93 -> 59.47. HyperLoop ARC-C normalized falls 38.40 -> 35.15.
- Table 4 (5B baseline 18L, EM avg of 6 generative tasks): baseline 23.89; HyperLoop Loop3 22.67 (-1.22); OperLoop Loop3 25.58 (+1.69); HyperLoop Loop6 29.75 (+5.86); OperLoop Loop6 31.90 (+8.01). Loop6 costs 1.67x FLOPs. Params 0.56x. GSM8K(8-shot CoT): OperLoop 17.13 -> 31.08 from Loop3 to Loop6.
- PPL/LL worse than baseline at matched FLOPs: Wikitext 13.90 vs 13.08; LL avg 59.77 vs 62.19. 10B scale: OperLoop EM 37.46 vs Parcae 36.28 vs HyperLoop 36.57; +1.04 over baseline.
- Ablations (Loop3): no delta objective 24.84 EM (vs 25.58); non-causal step size 24.02; unit step 24.40.

**Stated limits**: one model family, two scales, one seed for training and eval (no seed variance reported); the local objective is a design choice, not literally minimising the LM loss; no claim of transfer to other architectures.
**Code**: not stated in retrieved pages. **Compute**: nominal 281B and 503B tokens; Muon optimizer; GPU type/hours n/s.

**[INFERENCE]** Differences of ~0.5 to 1.5 EM points at a single seed are small relative to the unreported seed variance (RLT/What-Limits papers show seed SD of several points in other settings). Loop6 vs Loop3 compares different trained models, so it does not tell whether a Loop3 model improves if you run it for 6 loops at test time.

---------------------------------------------------------------------
### 1.3 2610.07591 Recurrent Looped Transformer (RLT), Princeton/Penn
URL: https://arxiv.org/abs/2610.07591 (v2 11 Oct 2026). Project page: https://github.com/yifanzhang-pro/recurrent-looped-tranformer (spelling as in the paper)

**Architecture (stated)**
- Not a depth loop: recurrence across tokens. Causal encoder (parallel over known tokens; builds KV memory) + recurrent decoder that, at each token, merges encoder output with the previous token's final decoder state via a learned gate (feedback scale alpha=0.1). Decoder blocks use sliding-window attention (window 8) plus cross-attention to encoder memory.
- Recurrent path length after t tokens = t * L_D decoder blocks; per-token cost fixed = L_E + L_D layers. 8 logical layers split 4+4, 5+3, 6+2, 7+1, 8+0; width 512, 4 heads.
- Variants: RLT-1 feedback every token; RLT-2 chunk4 (feedback updated every B=4 tokens, known tokens in a chunk run in parallel); RLT-0 no feedback.
- Gradient: BPTT over the unrolled decoder; "TBPTT128 covers every training sequence, so no gradients are truncated". Teacher forcing, next-token CE. Stopping rule: none.
- Params: RLT-1 26.10 to 28.73M vs 25.31M Transformer (not param/compute matched).

**Tasks and numbers (stated; 3 seeds, mean ± SD, best in-distribution-loss checkpoint)**
- Parity, trained on 3-40 bits, test 256 bits: RLT-1 5+3 and 7+1 = 100 ± 0; 4+4 66.76 ± 28.78; 6+2 84.05 ± 27.63; 8+0 68.91 ± 27.39; Transformer 50.07 ± 1.63.
- Swap-S5, trained 32 ops: 256 ops 4+4 97.30%, Transformer 0.85%. 512 ops 4+4 55.70 ± 25.78, 5+3 34.86, 6+2 22.14, Transformer 0.85. Standard S5 (120 perms): near chance for all (e.g. 1.24 vs 0.81 at 512).
- Flat mod-5 (63 tokens) 4+4: RLT-1 44.43 ± 43.81, RLT-0 21.84, chunk4 50.78 ± 38.02, GPT-8 33.20. Bracketed mod-5 (64): RLT-1 64.10, RLT-0 39.45, chunk4 62.76, GPT-8 46.71. At the longest lengths (255/256 tokens) all models are ~18 to 26%.
- Addition: all models reach 100% in-distribution; at 32-digit operands token accuracy ~15 to 17% for RLT and Transformer alike.
- Feedback ablation at 64: parity RLT-1 100, chunk4 98.99, RLT-0 50.23. Swap-S5 at 64 ops: RLT-1 100, chunk4 19.60, RLT-0 at chance (Table 9 at 48 ops: RLT-1 100, chunk4 68.29, RLT-0 7.26, GPT 22.10).
- Timing (seed 42, CPU): chunk4 training steps 2.27x faster than RLT-1; RLT-0 4.17 to 4.30x.

**Stated limits / failure modes**: temporary regression (parity seed 42, 5+3 drops 100 -> 48.05% at step 600, recovers at 700); flat mod-5 highly seed-dependent (seeds 17.58, 19.53, 98.96 for 4+4); standard S5 unsolved; benefit at 512 ops not reliable; no LM pretraining or RL tested; not param/compute matched; CPU-only.
**Code**: URL above (project page). **Compute**: four-thread CPU FP32; 2,000 steps (5,000 mod-5), batch 512.

**[INFERENCE]** This is the only paper in the set whose recurrence axis is time (per-token state carry), which is the axis a world model uses. Evidence is for discrete synthetic state tracking with tiny models.

---------------------------------------------------------------------
### 1.4 2610.23036 Towards Looped Models Done Right, Part I (Topology, Input Injection, Recurrent-State Design), IFM/USC/CMU
URL: https://arxiv.org/abs/2610.23036 (16 Oct 2026). Code: https://github.com/ifm-ai/xllm-loop

**Architecture (stated)**: controlled Ouro->Huginn transformation, matched 730M stored / 2.9B unrolled params, logical depth 112 block executions.
- Ouro-style: full 28-block stack tied, T=4. Huginn-style: 8-layer prelude + 12-block tied core x T=8 + 8-layer coda (8+12x8+8=112); input write W(z,e) = alpha*z + delta*W_in e (Parcae diagonal operator, delta=softplus, alpha=exp(-delta*exp(a))), e = prelude output; random init z0 ~ N(0, I/d).
- H/L variant: H=2 cycles, L=3 updates/cycle, one shared body, 8 body applications total.
- Loop count at test: same as train (8 / 4). Stopping rule: none. Gradient method: not described in the retrieved pages [n/s].
- Training: TxT360, seq 8192, batch 4.19M tokens/step, AdamW, LR sweep {4,6,8}e-4, 336B tokens (460 tokens per param). MoE: 8BA0.8B (32BA3B unrolled), 500B tokens, top-2.

**Numbers (stated; 730M, 336B tokens)**
- Input injection, middle loop: BBH-CoT +6.63, MMLU +2.53, HumanEval+ +5.49, MBPP+ +4.23; MATH500 -3.60, GSM8K -2.51. Full-stack Ouro with raw-token write: BBH +1.80, HumanEval+ +2.44, MBPP+ +4.50, MATH500 -1.60, GSM8K +0.53.
- Random init vs z0=e (middle loop+injection): ARC-C +3.34, GSM8K +1.22, MMLU -1.29, MATH500 -1.80, HumanEval+ -1.83, MBPP+ -2.11. Direct init better on 6 of 10.
- Shared H/L vs single state: ARC-C +2.23, DROP +1.80, MATH500 -7.20, GSM8K -2.19, MBPP+ -2.64.
- Complete Huginn beats Ouro on all 10 benchmarks (dense); MoE: Huginn beats Ouro on nearly all; vs 112-layer feedforward MoE (same FLOPs, 75% fewer resident params): wins DROP, GSM8K, ties MATH500, competitive elsewhere.

**Stated limits**: findings limited to the matched regime; input injection hurts quantitative reasoning (mechanism not isolated); "our evaluations do not directly measure multi-start path independence or extrapolation to recurrent depths beyond those used during training"; shared-module H/L only, a separate-module HRM-Text-style variant is "still under evaluation"; single run per cell (seed variance n/s).
**Compute**: not given in pages retrieved.

---------------------------------------------------------------------
### 1.5 2611.02857 Solv3R: 3D Looped Transformer as a Multi-view Solver, UBC/Wayve/Sony
URL: https://arxiv.org/abs/2611.02857 (2 Nov 2026). Code: "will be made publicly available" at https://ubc-vision.github.io/solv3r/ (not confirmed live).

**Architecture (stated)**
- Fine-tuned from DejaView (shared transformer block, frame-wise + global attention, 117M params, input images long side 504 px). DejaView conditions each update on the iteration index tau_k; Solv3R removes it, so the update is stationary: z_{k+1} = T(z_k; z_0). Gates replaced by learned constants; updated state rescaled to fixed RMS (rho ~0.85, single scalar for the whole state); u = U(z_k + W LN(z_0)); z_{k+1} = rho u/RMS(u). W zero-initialised linear.
- State: per-view tokens; views may have received different numbers of calls.
- Frozen: image encoder and prediction heads (depth/ray/camera).
- Train loops: random 16 to 40 no-grad calls, then 8 differentiable calls supervised at final call only; 15% of iterations perturb state (Gaussian noise sigma in [0,0.5]*RMS, or reset up to half of views to z_0). Loss = geometry + 0.5 * distillation to DejaView's 16-step output. Gradient: truncated (last 8 calls).
- Test: fixed 16 calls, or adaptive stop when ||z_{k+1}-z_k||/||z_k|| < 0.02, max 64.

**Numbers (stated)**
- Stability ablation (Table 4, AUC@5 pose at 16 / 128 / 1024 calls; 3k-step fine-tunes except final): DejaView released 76.5 / 0.0 / 0.0; Solv3R final 77.2 / 79.1 / 78.8; w/o input injection 71.6 / 33.1 / 0.0; w/o state normalisation 74.3 / 70.5 / 0.0; w/o random-depth pre-calls 73.1 / 3.4 / 0.0; w/o perturbation 71.0 / 69.1 / 61.7 (but join/cache errors worse). Step-conditioned variants lose stability. Perturbation recovery (rot. error): Stability 0.22 deg vs 8.37 for DejaView; Join 0.54 vs 1.80; Cache 0.72 vs 55.5.
- Adaptive stopping: 144 of 148 windows reach threshold within 64 calls, mean 17.5 calls (median 17, IQR 15-19), all stop between 13 and 32. Median rotation error 0.44 deg at stop vs 0.43 at 64 calls. Spearman(stopping call, rot. error)=0.77. Threshold 0.01 -> ~31-55 calls; 0.05 -> ~12-14 calls and a loss of 4-8 AUC points on DTU, ETH3D, ScanNet++.
- Offline (Table 1, TFLOPs for 24 views): DejaView 75.9 TFLOPs; Solv3R fixed(16 calls) 76.5, adaptive 91.3. DTU AUC@3 83.3 (DejaView) vs 84.5 fixed vs 85.5 adaptive; ScanNet++ 79.5 vs 80.7 vs 81.2; 7-Scenes 13.9 / 13.8 / 14.2. In the same table the larger VGGT (1.26B) lists DTU AUC@3 96.5 (parsed from extracted table text; verify).
- Streaming (Table 3): Solv3R 95 ms/frame, 5.0 GB; TUM-dyn ATE 0.015 (vs LingBot-Map 0.032, CUT3R 0.170); Sintel ATE 0.157 (LingBot-Map 0.149); ScanNet ATE 0.077 (LingBot-Map 0.089).
- Initialization: from perturbed or swapped starts the median rotation differs ≤0.11 deg after 128 calls. New view initialised from the previous frame's solved state converges in ~10 calls.
- KV caching of solved views: 16x less storage than stage-specific caching in DejaView.

**Stated limits**: fixed-budget accuracy initially drops vs DejaView ("recovers with sufficient training"); "accurate fixed-budget predictions are insufficient for reusable geometric solving". Appendix mentions a failure case discussion (content not retrieved). Streaming and relocalization require the stationary property; step-conditioned variants fail.
**Compute**: ~1 week on 8xH100 (fine-tune; stage 1 ~70k iters batch 8, stage 2 ~20k iters batch 32). AdamW, LR 2e-5 (block) and 2e-4 (gates), EMA 0.999.

**[INFERENCE]** Solv3R is the only paper here that measures behaviour at loop counts far beyond training (to 1024), and it needed three things beyond weight sharing: injection of z_0, state RMS normalisation, and training on states of mixed refinement depth. Each ablation alone breaks at 1024 calls.

---------------------------------------------------------------------
## 2. Related work read (beyond the five)

### 2.1 Deep Equilibrium Models (Bai et al. 2019) and gradient approximations
Not retrieved from alphaXiv directly; facts below are as cited in the papers I read.
- DEQ finds z* = f(z*; x) by root finding (Broyden / Anderson), differentiates by the implicit function theorem; 1-step / Neumann-truncated gradients are used in practice (cited in HRM, TRM, Truncate-or-Not, FPRM).
- Reported difficulties: "DEQ models are difficult to optimize" (FPRM citing Bai 2021, Anil 2022, Jolicoeur-Martineau 2025). TRM tried TorchDEQ fixed-point iteration: slower and "worse generalization".

### 2.2 2506.21734 Hierarchical Reasoning Model (HRM), Sapient. Code: github.com/sapientinc/HRM (stated)
- Two 4-layer transformer modules f_L (fast), f_H (slow); N=2 L-steps per H-step, T=2 cycles; 27M params; post-norm; init states truncated normal, fixed. Gradient: one-step approximation of last L and last H step (IFT / Neumann first term), O(1) memory; deep supervision with detached state between segments; ACT via Q-learning (halt/continue), M_max segments, 1-segment minimum with prob.
- Numbers (stated): Sudoku-Extreme 55.0%, Maze-Hard 74.5%, ARC-AGI-1 40.3%, ARC-AGI-2 5.0%, ~1000 training examples each; CoT LLMs 0% on Sudoku/Maze.
- Inference-time scaling: models trained with M_max=8 improve when run with 16; extra compute gives minimal gains on ARC.

### 2.3 2510.04871 Tiny Recursion Models (TRM), Samsung SAIL Montreal
- One 2-layer network, latent z and answer y; n=6 latent updates per y update, T=3 recursion processes, T-1 without gradient then one with full backprop through the final process (no 1-step approximation), N_sup=16 deep-supervision steps, halting via a BCE head (no extra forward pass), EMA 0.999.
- Numbers: Sudoku-Extreme 87.4% (MLP-token-mixer, 5M); Maze-Hard 85.3% (attention, 7M); ARC-AGI-1 44.6% and ARC-AGI-2 7.8% (7M, 2 tries). HRM: 55.0 / 74.5 / 40.3 / 5.0. Ablation (Sudoku): ACT with extra pass 86.1; separate f_H/f_L 82.4; no EMA 79.9; 4 layers 79.5; self-attention 74.7; T=2,n=2 73.7; 1-step gradient 56.5.
- Failed ideas (stated): MoE (generalisation "decrease massively"); partial backprop through k=4 of 7 steps ("did not help"); no ACT; weight tying embedding/head; TorchDEQ fixed-point iteration.
- Limits: adding layers hurts (overfitting, small data); more recursions cause "massive slowdowns" and OOM for larger n; attention-free variant fails on 30x30 grids. 
- Compute: Sudoku 1 L40S <36 h; Maze 4 L40S <24 h; ARC 4 H100 ~3 days. Code: not in pages retrieved (public repo exists from my memory, not verified here).
- Cited analysis (ARC Prize): deep supervision doubled accuracy (19 -> 39%), recursion itself 35.7 -> 39.0 (stated by TRM).

### 2.4 2604.12946 Parcae: stable looped LMs (UCSD / Together)
- Dynamical-system view h_{t+1} = A h_t + B e + R(h_t, e). Prior injections: addition gives rho(A)=1 (marginal), concatenation gives unconstrained A (unstable). Parcae: A = exp(Delta * diag(-exp(.))) (ZOH), B Euler, LN on prelude output e (added after late-stage spikes at 1.3B after ~150k steps, state norm reaching ~1e7), per-sequence depth sampling, mu_bwd = ceil(mu_rec/2); h0 ~ N(0, sigma).
- Divergent runs learned rho(A) >= 1 (fig 3); Parcae converged at all LRs tested (2e-4 to 1e-3) while baseline converged only at 2e-4, residual-norm variant up to 4e-4.
- Results: up to 6.3% lower val PPL than prior RDMs; at 1.3B/100B tokens Core +2.99, Core-Ext +1.18 vs Transformer; up to 87.5% of a 2x-size Transformer. Scaling law: training compute optimum increases mu_rec and tokens together (power laws); test-time looping follows a saturating exponential L(T) = L_inf + Z exp(-zT) with "gains plateau near mu_rec", i.e. training depth sets the ceiling.
- Stated limits: small architectures only; as mu_rec increases the number of test-time steps needed grows.
- Code: n/s in retrieved pages.

### 2.5 2606.18206 Fixed-Point Reasoners (FPRM), ELLIS Tuebingen et al.
- Non-hierarchical looped transformer; pre-norm + residual scaling (layer-wise alpha1,beta1; per-iteration input mixing z_{i+1} = alpha2 z + beta2 x), with a boundedness theorem (Thm 1) and contraction condition alpha2*Lipschitz < 1 (Thm 2). Fixed-point halting: stop when relative residual < tau (0.1), damped updates with patience-based step decay. Training: TBPTT window k, deep supervision, builds on TRM codebase. Sudoku: train max_iter 12, eval max_iter 35,000; Maze train 24.
- Stated: post-norm looped model does not benefit from added loops; bare pre-norm diverges in activation norm at large loop counts; with residual scaling it trains. HRM/TRM ACT "can fail to provide adaptivity".
- Appendix Table 3: adding pre-norm + residual scaling to TRM itself lowers Sudoku accuracy: original TRM 63.98% / 72.60% (with/without conv); +pre-norm+scaling(alpha2 only) 14.98 / 13.73; (alpha1, alpha2) 39.46 / 19.77. Absolute FPRM headline numbers (Table 1) not retrieved. Fig 1 claims +10 accuracy points over TRM with ~27% less compute (text garbled; verify).
- Code: "implementation can be found here" (URL not in extracted text).

### 2.6 2605.21488 Equilibrium Reasoners (EqR), CMU. Code: https://github.com/locuslab/EqR
- TRM-style backbone plus (RI) randomised state initialisation and (NI) noise injection with damping: z_{k+1} = z_k + (1-lambda) r(z_k;x) + beta*eps (lambda=0.05, beta=0.01). Training capped at 16 iterations; test depth D up to 1024 iterations (~40,000 effective layers) and breadth B restarts with convergence-based selection.
- Numbers: Sudoku-Extreme 84.8 (TRM-style baseline) -> 86.4 (RI+NI) at D=16; D=64 -> 93.0; D=64,B=128 -> 99.8 (Transformer backbone: 72.0 -> 74.7 -> 95.9). Maze-Unique 44.9 -> 82.2 -> 88.9 -> 93.0. Mini-ARC 55.28 vs TRM 48.35 vs HRM 44.85. Feedforward 64-layer: 2.6% Sudoku; weight-tied 2-block: 32.6%. 5 seeds: baseline 84.33 ± 0.59, EqR 86.18 ± 0.44.
- Stated limits: original Maze-1k has non-unique shortest paths; "iterative models fail to exhibit stable test-time scaling" there; they built Maze-Unique and used a smaller model (1 layer, width 128) to make effects visible. Four "landscape regimes", incl. spurious attractors where depth does not help.

### 2.7 2604.15259 Stability and Generalization in Looped Transformers (Labovich). Code: github.com/ashlab11/generalization
- Fixed-point framework with three axes (reachability, input dependence, geometry). Proved: without recall (input injection) fixed points are countable (dimension zero) and input dependence is weak; recall + outer normalisation gives a regime satisfying all three. Single-layer looped transformers, 100 epochs; progressive loss with random N no-grad + K grad iterations out of T=30.
- Results: best normalised model beats best un-normalised by 12 points (Sudoku, hard) and 2 points (chess) on harder-than-train data; prefix-sums: only normalised models get non-zero hard accuracy at the largest LR.
- Stated limits: conditions "act far closer to necessary conditions than sufficient"; small single-layer models; best architecture differs per task.

### 2.8 2610.31943 Agreement rather than normalisation keeps depth reusable (Viakhirev et al.), code "released after review"
- On released Ouro-1.4B/2.6B, removing the loop-boundary normalisation drives accuracy to ~0 within 2-3 loops while state norm stays within 2.02x: readout collapse, not norm explosion. Trained from scratch (19-76M, 16 seeds/config), training without the boundary normalisation is never worse and extrapolates better (e.g. K=8x budget: 0.932 vs 0.584; K=4x 0.981 vs 0.618); the normalised arm peaks at the trained depth and degrades on both sides.
- Contradicts the "bounded state" account (FPRM, Parcae). Authors say they did not measure per-step update size; the "growth damps the loop" reading is offered as interpretation. Matched-overshoot comparison: looped vs stacked depth advantage "not detectable" (circuit) / reduced (hops) at 6 seeds.

### 2.9 2610.26030 Truncate or Not? Scaling study of looped transformers with TBPTT (HKU/Yale). Code: https://github.com/xjxyys/Truncate-or-Not
- ~600 runs, 58M to 919M non-embedding params, FineWeb-Edu, T up to 16; fitted law with effective parameters; (T,B) = forward/backward loops. Finding: with T prescribed truncation can help; with T and B free, full BPTT is optimal; at fixed model size most extra compute should go to tokens (optimal loops ~ C^0.24); two fully trained loops give lower validation loss than compute-optimal non-looped with 27% fewer params at +8% compute (2.36 vs 2.37 nats).
- Limits (stated): dense only, validation loss only (no downstream), T ≤ 16, "much deeper recurrence can exacerbate training instability".

### 2.10 2609.39967 What Limits Recursive Reasoning Models (RND NLP)
- Unified retraining of HRM/TRM/URM across Sudoku, Maze, Game of Life, Arithmetic (controlled OOD), ARC. Recipe (13.6M): shared block, (H,L) cycles (4,2), gradient through last (K_L,K_H)=(2,2), bounded + gated + normalised state update, recurrence-consistent dropout, relative noise, large physical batch, EMA.
- Numbers: Arithmetic OOD 71.16 (TRM 36.20, URM 28.50, HRM 13.80); Game of Life ID/OOD 66.08/65.80 (URM 55.42/55.26, TRM 7.11/7.14); Sudoku 98.41 (TRM 87.40); Maze 84.70 (URM 80.20); ARC-1 pass@2 59.50 (TRM 44.60). Gradient horizon is non-monotone: (1,3) ID 99.60 but OOD 17.88 vs (2,2) OOD 71.16. Physical batch 8192 -> OOD 63.16 vs 37.71 with 4x gradient accumulation. Seeds: base config Game-of-Life SD 3.32/2.90 reduced to 0.11/0.13 with the full stabilisation recipe.
- Limits: hierarchy no consistent benefit; bigger blocks no consistent gain; extra test-time ACT budget helps Arithmetic and Sudoku but little on Game of Life, Maze, ARC. Inference cost not small: 6T block applications vs 1 for dense.

### 2.11 2609.34488 FlexLoop: depth-elastic looped policies (Tsinghua), deep RL
- Builds on Ghugare et al. 2026 "On the role of computation in RL" (IRU: interpolation recurrent unit, a gated recurrent cell looped K times at one decision step). Key stated finding: pretrained looped policies show "severe recurrent-depth specialization": decisions are reliable only near the trained depth (fig 1: IRU-K performs well only at depth K).
- Method: post-train from a pretrained IRU-5 with RL loss + adjacent-depth KL distillation sg[pi_{k+1}] -> pi_k; stopping rule: stop when adjacent-depth policy discrepancy delta_k < epsilon.
- Numbers: 30 goal-conditioned long-horizon environments (BoxPick, LightsOut, 25 OGBench); average depth cut up to 43% at matched success (e.g. OGBench 4.19 avg depth, 0.410 vs 0.385 success; LightsOut-4x5 train 2.84 depth, 0.747 vs 0.732); up to 1.34x wall-clock (BoxPick-Exact-4: 133.0 vs 170.1 ms). 3 seeds. Proposition (appendix): a shared-weight policy can be near-optimal at depth K and near-worst at all shallower depths.
- Limits: gains larger for value-based (GC-DQN) than PPO/IQL; shallowest depths hardest; code: builds on Ghugare's codebase (URL n/s).

### 2.12 2610.19552 Test-Time Forecasting (TTF): recurrent depth for in-context dynamics prediction (UT Austin)
- One-step forecaster trained for next-state prediction, rolled out autoregressively. Implicit TTF: a looped (linear-attention + short-conv) block whose recurrence equals gradient descent on an EDMD / Koopman objective, with concatenation input injection. Explicit TTF: inner loop = AdamW steps on fast weights of a small forecaster (FNO) in latent space (parameter-space recurrence; needs Hessian-vector products).
- Training: loop counts from log-normal Poisson (mean 16 to 30), TBPTT tau=10 to 30, 50K iterations, PSGD-Kron/SOAP outer optimizer; inference loop count fixed per rollout step, may early-stop on inner loss.
- Results (qualitative from text): looped linear transformer shows linear scaling of spectral error with loops on unseen LDS; 17-parameter FNO eTTF recovers the Peclet number from its fast weights and error improves with loops on a 128-step rollout at Pe=1e8; KS eTTF (82,130 params) rolls out 64 steps on OOD domain length with nRMSE from 1.107 (L=1,B=1) to 0.597 (L=16,B=16) (table), compared with parameter-matched and 16-block FNO baselines (numbers not retrieved).
- Stated limits: more loops "over-energize higher wavenumbers" in KS; random fast-weight init fails on advection-diffusion; looped training can cost more than fixed depth; eTTF is not compile-friendly and memory-heavy; adaptive early exit left as future work.
- Code: "provided in the supplementary details" (no URL).

---------------------------------------------------------------------
## 3. Seen in search results but NOT read (titles only; treat as leads)
- 2607.13491 DeepLoop: depth scaling for looped transformers (Princeton/UCLA)
- 2610.35544 StateLoop: persistent state; 2610.35556 LoopFold: parallelizing recurrent depth; 2610.02185 Apple: decoding looped transformers "better for free"
- 2610.14030 Rotate the Loop: blockwise conformality; 2609.40316 Scaling laws for looped MoE (Meta)
- 2610.28871 How much gradient does truncated backprop lose; 2610.24135 mechanistic study of weight-tied transformers on depth-bound tasks
- 2609.22197 Dissecting HRM; 2610.26666 HRM/TRM/URM on ARC-AGI-1 (nested shared-operator recursion)
- 2610.34563 Recursive CLIP (vision-language, looped blocks); 2609.01924 Jacobian-lens look at looped transformers
- Cited-only by the papers above: Ouro (2510.25741), Huginn (2502.05171), Hyperloop Transformers, LoopFormer (2602.11451), Iso-depth scaling (2604.21106), Loop-Think-Generalize (2604.07822), Attractor Models (2605.12466), URM (2512.14693), ELT (looped transformers for visual generation), "Recurrent-FNO test-time accuracy-cost control in neural simulators" (Abdul Majid 2026), Ghugare "On the role of computation in RL".

---------------------------------------------------------------------
## 4. Cross-paper comparison (stated numbers only)

| Paper | Train loops | Test loops | Injection | State init | Stop rule | Gradient |
|---|---|---|---|---|---|---|
| LoopWM | Poisson(mu_rec) per sequence (value n/s) | adaptive gate; Tmax may exceed mu_rec (untested) | B_bar e each loop, e from prelude | N(0,sigma^2) or previous step's h(T) | learned exit gate | TBPTT ceil(mu_rec/2) |
| OperLoop | 3 or 6 (separate models) | same as train | in transition (bias e_l, read/write maps) | prelude out copied to 4 streams | none | full |
| RLT | n/a (per-token recurrence) | n/a | encoder memory + gated prior state | n/a | none | BPTT, not truncated |
| Looped Right I | 8 (Huginn), 4 (Ouro) | same | prelude state e, diagonal write | e or N(0,I/d) | none | n/s |
| Solv3R | 16-40 no-grad + 8 grad | 16 fixed, adaptive up to 64; tested to 1024 | z_0 added every call | z_0 (encoder output) | ||dz||/||z||<0.02 | truncated, last 8 |
| Parcae | Poisson(mu_rec), e.g. 8 | saturates near mu_rec | B e each loop | N(0,sigma) | none | TBPTT ceil(mu_rec/2) |
| FPRM | up to 12 (Sudoku) | up to 35,000 | alpha2 z + beta2 x | n/s | residual<0.1, damped | TBPTT k |
| TRM | n=6, T=3, N_sup=16 | N_sup=16 | x each step | learned | BCE halting head (train) | full through last process |
| EqR | <=16 | up to 1024 | x each step | random N(0,sigma) | residual-based selection | as TRM |
| TTF | Poisson mean 16-30 | fixed per step | concat Phi each loop | meta-learned / identity | optional loss threshold | TBPTT tau |

Compute classes: five orders of magnitude apart. Single CPU (RLT), 1 L40S <36 h (TRM), 1 A100 (FPRM), 8 H100 x ~1 week (Solv3R fine-tune), hundreds of billions of tokens (OperLoop 300-500B, Looped Right I 336-500B).

---------------------------------------------------------------------
## 5. Open gaps

### 5.1 Things these papers did not test
1. Looped World Models: no experiment on any of: long-horizon rollout error vs horizon, loop count vs error, spectral-constraint ablation, adaptive-exit FLOP reduction, a non-looped ~1B baseline, a looped-vs-fixed-depth parameter-matched comparison, continuous or pixel observations, contact-rich or physical dynamics, RL/planning use (Dreamer-style imagination or CEM/MPPI planning is described but not run). Base model, data, compute, loop counts, code are unstated.
2. No paper tests stable looping across both axes a world model needs at once: many inner loops per step AND many outer (rollout) steps with state carried across them. LoopWM proposes it; Solv3R tests stability over loop count on a static scene; RLT tests per-token carry over 40 to 512 steps without depth loops; no one tests the product.
3. Looped Right I, OperLoop, Parcae: no downstream results at test-time loop counts different from training (Looped Right I explicitly says so). Parcae finds a ceiling near mu_rec in LM; Solv3R gets to 1024 only with special training. Whether LM-style looped models extrapolate in loop count is therefore open.
4. OperLoop and Looped Right I: single seed, no variance; effects of 0.5 to 2 points on benchmarks cannot be ranked without it.
5. Stopping rules: only Solv3R (residual), FlexLoop (adjacent-depth policy KL) and FPRM (residual) validate a state-based stop signal against error. LoopWM's learned gate has no reported behaviour. The entropy-regularised gate collapse failure it guards against is untested.
6. Gradient method: nobody compares one-step / TBPTT / full BPTT for a rollout-trained dynamics model. Truncate-or-Not (LM loss only) says full BPTT wins when T is free; What-Limits (reasoning tasks) says intermediate horizon (2,2) wins OOD; TRM says full-recursion beats 1-step (87.4 vs 56.5) while its partial k=4 test showed no gain. Conditions differ, so no common answer.
7. Action conditioning: only LoopWM conditions on actions, and it feeds them into the prelude. Whether an action should be re-injected every loop (as input is in Solv3R, TRM, EqR) is not tested.
8. Stochastic or multimodal dynamics: EqR shows non-unique targets break stable test-time scaling (Maze-1k); TRM/HRM/FPRM train with single-target CE. World models usually have stochastic transitions; no paper loops a model with a stochastic latent (RSSM-style) or a generative (diffusion) head.
9. Robustness of the adaptive stop under distribution shift (new dynamics, contact events) is untested; Solv3R shows the stop call correlates with difficulty (Spearman 0.77) only on static multi-view windows.
10. Compute accounting: reported "parameter efficiency" ignores that looped models do not reduce FLOPs or latency. What-Limits notes TRM-like models cost 5.25T to 6T block applications per answer; FlexLoop wall-clock gains are 1.2 to 1.3x; Parcae notes more test-time steps are needed as mu_rec grows. LoopWM's 100x is vs API-model parameters, not FLOPs.

### 5.2 Contradictions and tensions between papers
1. Boundedness. LoopWM, Parcae and FPRM argue stable looping needs a bounded / contractive state (LoopWM via the Parcae A_bar; FPRM via pre-norm + residual scaling). 2610.31943 finds that training without a loop-boundary normalisation extrapolates better (0.932 vs 0.584 at 8x budget) and state norm growth (4.84x) coexists with stability; and shows readout collapse, not norm growth, breaks released Ouro. Solv3R, on the other hand, needs state RMS normalisation (without it: collapse at 1024 calls). These are different tasks, scales and readouts; the papers do not reconcile them.
2. Pre-norm + residual scaling: FPRM recommends it; applying it inside TRM (FPRM's own Table 3) reduces Sudoku accuracy sharply (72.60 -> 13.73 to 19.77 for the no-conv variant). Whether the benefit is architecture-specific is unclear.
3. Random initialisation: EqR reports RI helps (Maze 44.9 -> 68.6); Looped Right I finds no consistent cross-task advantage (2 gains, 4 losses >1 point) and notes its eval does not measure path independence; Parcae and LoopWM initialise h^(0) from noise by default.
4. Hierarchy / separate H and L states: TRM, FPRM, What-Limits and Looped Right I find no consistent benefit; EqR found "smaller additional gain"; TRM reported "w/ separate f_H, f_L" 82.4 vs 87.4 (worse). Looped Right I's H/L test shares weights and its separate-module variant is unfinished.
5. Input injection: Looped Right I finds it helps context tasks but hurts MATH500 (-3.60) and GSM8K (-2.51) in the middle-loop model; Solv3R, Labovich and TTF treat it as necessary (removal breaks stability/long-loop behaviour). Different metrics (accuracy at trained depth vs behaviour at extreme depth).
6. Gradient truncation: see 5.1(6). Also Parcae reports that setting the loop distribution by mu_bwd "significantly hurts extrapolation" (they use mu_rec alone), while LoopWM adopts mu_bwd = ceil(mu_rec/2) without extrapolation tests.
7. Do extra loops help at test time? Parcae: saturating plateau near mu_rec. HRM/EqR/FPRM: large gains far beyond training depth on puzzles. What-Limits: gains task-dependent (Arithmetic and Sudoku yes; Game of Life, Maze, ARC little). FlexLoop: looped RL policies specialise to the trained depth. Solv3R: DejaView collapses beyond its training range (AUC 76.5 -> 0.0 at 128 calls). Pattern is task- and training-recipe-dependent; no unified account.
8. Benchmark reproducibility: TRM's reported Sudoku 87.4 is retrained by EqR at 84.8 (best checkpoint, 50k steps) and 84.33 ± 0.59 (5 seeds); HRM Maze 74.5 (Maze-Hard) vs 0.3 on EqR's Maze-Unique; Maze-1k is stated by EqR to have non-unique shortest paths. Cross-paper comparisons on Maze are not like-for-like.
9. Looped vs stacked: Parcae and OperLoop report looped beating or matching larger/deeper baselines; 2610.31943 finds no detectable advantage at matched overshoot (6 seeds), and Truncate-or-Not finds that with free model size the law favours no looping (two loops come close with 27% fewer params). Perplexity gets worse and generative scores get better in OperLoop at matched FLOPs.

### 5.3 Assumptions that look weak
1. "Environment dynamics are repeated application of a shared law, therefore inner loops mirror physics" (LoopWM). The paper itself says the correspondence is "conceptual rather than exact" and that the inner loop is not physical time. No test of loop-count vs physical-step relation.
2. "(0,1) diagonal A_bar guarantees bounded dynamics regardless of rollout length." Holds for the linear term only; nonlinear R_bar and the unconstrained B_bar are unconstrained. Parcae itself needed an extra prelude-norm fix at 1.3B after 150k steps; Solv3R needed extra normalisation + perturbation training; 2610.31943 shows boundedness is not what breaks readouts.
3. "Adaptive exit saves ~25x to 100x FLOPs." Derived from an assumed 100-layer baseline and favourable transition mix; mean loops per step never measured. Compare FlexLoop's measured 43% depth cut / 1.34x wall-clock, with a stop signal (policy KL) that needed specialised post-training.
4. Single-seed LM comparisons (OperLoop, Looped Right I) treated as ranking evidence.
5. Text-token "world model" results (ScienceWorld/AlfWorld) are used to support claims about continuous physical simulation; the paper says applicability to continuous visual environments is a "suggestion".
6. Deferred decoding's benefit is shown relative to a different, much larger model's per-step outputs, with the "relative improvement" computed against Gemini on a metric where the Gemini baseline is low (e.g. EM 0 -> "—"/huge percentages).
7. Fixed-point assumption for 1-step gradients (HRM): TRM shows residuals do not reach near-zero at the point the approximation is applied; FPRM and EqR use residual-based halting/selection, but only on puzzles with unique solutions.
8. Checkpoint selection: RLT selects by in-distribution validation loss and reports test on longer lengths with n=3 seeds and wide SDs (e.g. 44.43 ± 43.81); headline "100%" results do not hold for every split (4+4 parity at 256 bits is 66.76 ± 28.78).
9. Training-test coupling of loop count: Parcae (plateau near mu_rec), FlexLoop (depth specialisation), 2610.31943 (readout calibrated to the training topology), Solv3R (step-conditioned gates fail) all point to the loop count/topology being baked into the readout. Papers that claim test-time loop scaling (LoopWM, EqR, HRM) mostly do not isolate this.
