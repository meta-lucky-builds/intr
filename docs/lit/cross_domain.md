# Cross-domain literature scout (as of 2026-11-04)

Scope: recent, transferable techniques for latent world models and robot controllers. Test-time training and looped transformers are excluded (other scouts).

Conventions:
- "Source" gives the document type. PREPRINT = arXiv, not peer reviewed. BLOG = author blog or project page. WORKSHOP = NeurIPS 2026 workshop paper.
- Everything under "Stated" comes from the source text I read. "Not stated" means the source did not give it, and I did not estimate.
- "INFERENCE" marks my own proposals for transfer. These are not claims from any source.
- Reading depth: full text for 2610.19683, 2609.04264, 2606.16533 (selected pages), 2610.18774, 2611.01677 (selected pages), 2603.19312, 2610.19375, 2610.15429, 2610.06805, 2610.07334. Web page only for the two blogs. Abstract only for 2609.38166, 2610.12194, 2610.42146.

## Corrections to the brief (read first)

1. The Reactive Optimizers page (xingjianbai.com/reactive-optimizers) says the agent was Claude Opus 5.5. It does not mention GPT-6 Astra and has no Kinetix car-on-ramp experiment.
2. GPT-6 Astra and the Kinetix car-on-ramp fix come from the Weight-Space Programming blog (rikiyat.github.io). They are two different works.
3. The hot feed tool returned only paper IDs, no titles. I checked 6 of the top ~100 IDs by abstract (2610.39302, 2611.02270, 2609.08130, 2611.01560, 2611.03178, 2610.42146). None of those six is a strong lead, except ID-WM (2610.42146), which is noted in Open gaps. The other ~94 were not triaged. I used targeted discovery searches instead.
4. LeWorldModel is arXiv 2603.19312 (not found by title in alphaXiv, found by ID).

---

## Part A. Weight-space work

### T1. Weight-Space Programming (an LLM edits network weights directly)
- Source: BLOG (rikiyat.github.io/blog/weight-space-programming/, alphaXiv 2611.weight-space-programming). Not peer reviewed. I read it via WebFetch summary, not the raw page.
- Mechanism (stated): a frontier LLM, "GPT-6 Astra" (closed model, few details given), reads and writes network weights by reasoning, not by gradient descent. It infers what each part computes and edits to match. Tools vary by experiment: none, a Python sandbox, a limited number of forward passes, or attention-pattern queries.
- Measured results (stated; baselines as given):
  - Exp 4, Kinetix car-on-ramp policy (171,664 weights): editing 3.5% of weights gives 1,024/1,024 successful episodes. PPO baseline: 0/256 successes.
  - Exp 5, MinAtar nets written from zero weights in one shot: Breakout 517.4, Seaquest 53.1, Asterix 53.2. PPO 100M steps: 15.8 / 5.6 / 25.8. PPO 500M steps: 25.3 / 88.9 / 55.2. So the edited net loses to PPO 500M on Seaquest.
  - Exp 6, Go 9x9 (1,480,274 weights): with 0 forward passes it wins over half of games vs two smaller PPO nets. With 50 forward passes it beats an equal-sized PPO net trained on 20M games.
  - Exp 8, Qwen3.5-0.8B, 131,072 weights edited (0.016%), 50 forward passes: MATH-500 7.9 to 48.1, GSM8K 13.8 to 67.1, IFEval 21.0 to 62.3. Token-limit hits 93.0% to 33.1%. At 10/25 forward passes: MATH-500 12.3/14.1.
  - Exp 7 (find the sycophantic Gemma 3 270M): from weights alone 0/4. With an attention tool 3/3 pairs.
  - Gemma 3 270M math: 6.8 to 7.7 at best. Authors report no meaningful gain.
- Stated limits: "very early". Edits remove an obstacle (looping) but cannot add a missing ability. Context length and harness design limit it. The post does not reconcile "two smaller PPO nets" and "equal-sized PPO net" for the Go baseline.
- Code/assets (stated): logs github.com/RikiyaT/lm_understand-logs; edited model huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming; benchmark/harness "in development, no link".
- Compute (stated): Exp 8 used a Python kernel with no GPU; Exp 6 CPU numpy/PyTorch analysis. Astra's total compute and cost not reported. Baseline PPO runs were 100M to 1B steps.
- INFERENCE (transfer): the Kinetix result is the only physical-control-adjacent evidence. For a robot controller, the closest analogue is a small targeted edit of a failing policy head, found by an outer LLM that gets a few rollout batches as feedback. For a latent world model, the same loop could patch a predictor layer that mispredicts a contact regime. Unknown whether it works on a policy not trained by PPO from scratch in a 171k-weight toy.
- Weakness: all evidence is from the authors' logs with a closed model; no independent replication.

### T2. Weight Oracles (an LLM reads raw weights to audit a network)
- Source: WORKSHOP/PREPRINT, arXiv 2610.07334 (Kabra, Venhoff, Schroeder de Witt, Oxford; NeurIPS 2026 workshop on NN artifacts as a data modality). Full text read.
- Mechanism (stated): Qwen2.5-3B-Instruct gets each weight-matrix row as a continuous token (learned linear projection R^8 to R^2048, plus a per-matrix-type embedding). Phase I: trained to do the dot-product primitive of a small transformer forward pass, with external deterministic code doing LayerNorm, GELU, residuals (5 LLM passes per target forward pass). Phase II: trained on natural-language diagnostic questions (17 types) about weight anomalies using only 10 benign pathologies, then tested zero-shot on backdoors.
- Measured (stated):
  - Phase I: 99.0% holdout accuracy on unseen real small transformers with the external chain of computation. Single-pass full MLP block plateaued at 45.1%, equal to a residual-only shortcut.
  - Phase II: AUROC 0.93 +/- 0.04 on attention-routed (pos0) backdoors; 0.81 across four threat types (range 0.74-0.89). Baselines: total L2 norm 0.445, best V/O-pathway composite 0.752. At 94% TPR the FPR is 20%.
  - Initializing from the Phase I simulator gave 0.73 vs 0.81 from scratch (negative transfer).
- Stated limits: targets are toy transformers of about 2K parameters; the number of weight tokens grows with parameter count and becomes infeasible beyond a few million; the oracle detects statistical correlates, not mechanisms (lastpos AUROC 0.74); single-token triggers only; no ablation of whether the LLM beats a small learned detector.
- Code: not stated in the text I read. Compute: single-GPU fine-tune of a 3B model, 29,000 steps (stated); GPU-hours not stated.
- INFERENCE: a weight-reading auditor could be used to flag a robot policy checkpoint with anomalous attention or head structure before deployment. Realistically blocked by scale (2K parameters tested). Low priority for the project.

### T3. OPAL, Reactive Optimizers (an LLM agent rewrites the optimizer during one run)
- Source: BLOG/project page (xingjianbai.com/reactive-optimizers). Not peer reviewed. WebFetch summary only. Agent: Claude Opus 5.5.
- Mechanism (stated): the agent edits two programs during training, `updater.py` (per-parameter-group update rule) and `analysis.py` (its own probes: gradient norms, update-to-weight ratios, gradient cosines). Zero-shot, nothing carried between runs.
- Measured (stated):
  - LightningDiT-XL/1 ImageNet FID-50K: with guidance 2.18 to 1.83; without 5.15 to 4.33. RAEv2-B 3.39 to 3.19.
  - Goldfish Amharic 125M: official recipe 83.48 to 81.11 bits/sentence; best static sweep 80.89 to 80.73.
  - 34 toy 2D landscapes, log10(f-f*): OPAL -11.6, best of 12 recipes -2.2, best of 382,080 swept recipes -7.4.
  - CIFAR-10 10% label noise: OPAL loss 0.685 / acc 78.6%; AdamW final 2.493 / 73.5%; AdamW early-stopped 1.010 / 72.9%; AdamW EMA 0.832 / 77.5%.
  - modded-nanogpt Track 3: "outperforms or matches the best baseline"; exact OPAL values are only in a figure (not available to me).
- Cost (stated): LightningDiT run about 100 agent decisions, about $6.42 at list price, about 74 H100-hours (about $510 on-demand). Wall-clock overhead 1.01-1.14x if training does not wait, 1.30-4.39x if it does.
- Stated limits: smaller gains on short runs or heavily tuned recipes; mostly single runs (few three); no sharded large-scale support yet.
- Code (stated): github.com/xingjian-bai/opal, `pip install opal-optimizer`, `opal.wrap(opt, model, scheduler, total_steps, goal, max_cost_usd)`.
- INFERENCE: the corruption/label-noise result suggests value for world-model training on messy robot logs (noisy action labels). Since the stated gains were on image/LM training, it is untested on JEPA objectives, and SIGReg-style losses have unusual gradient statistics. A cheap test: wrap the LeWM optimizer and compare against its AdamW recipe with paired seeds.

---

## Part B. World-model structure

Background (LeWM, primary source arXiv 2603.19312): encoder (ViT-tiny, about 5M params) and predictor (6-layer transformer, about 10M, AdamW-style action conditioning via AdaLN, zero-initialized) trained end to end with loss = next-embedding MSE + 0.1 * SIGReg (1024 random projections, Epps-Pulley normality test). 15M params, single GPU in a few hours. Planning: CEM, 300 samples, 30 iterations, 30 elites. Stated results: planning up to 48x faster than DINO-WM (0.98 s vs 47 s); success PushT 96%, Reacher 86%, OGBench-Cube 74%, Two-Room 87% (DINO-WM 100%). Stated limits: weak on very low-dimensional data (Two-Room); needs action labels; needs coverage of the offline data. Predictor dropout 0.1 matters: Push-T 78% at 0.0, 96% at 0.1, 66.7% at 0.5. Latent trajectories straighten without any term for it.

### T4. Fourier-symmetrized auxiliary head (Spectral-Target Physical Latent Structuring)
- Source: PREPRINT arXiv 2609.04264 (Kaliber Labs, v3). Full text read.
- Mechanism (stated): a 2-layer MLP head, training-time only, predicts spectral features y = sum_i [cos(W^T b_i), sin(W^T b_i)] of object bounding boxes b_i=(x,y,w,h), with W a fixed random frequency matrix. The sum over objects makes the target permutation invariant. The loss is applied to encoded latents and to predicted latents (head trained only on the encoded term). The head is removed at inference. Kept shallow on purpose ("laziness onset with deeper heads").
- Problem it addresses (stated): "physical representation laziness". With SIGReg, the latent is not collapsed but ignores the controlled ball (Spearman rho_ctrl = -0.0008, rho_env = 0.40).
- Measured (stated): success-rate gains in the n-ball task of +91.3, +78.5, +65.3 points (1, 3, 6 environment balls, H=4). Modest but consistent gains on Push-T, Two-Room, OGBench-Cube. Helps most in low-data regimes. The excerpt I read does not give absolute success rates for the baseline or for Push-T/Cube.
- Stated limits: no real-world tests; only LeWM tested as a base; needs object positions or boxes (simulator, detector, or tracker); gives worse results combined with VISReg than with SIGReg; too little data makes the target underdetermined; the correlation with the physical property is not monotone with success on Reacher/Cube.
- Code and compute: not stated in the pages I read; inference cost zero.
- INFERENCE: cheapest known patch for lazy latents. For a robot, a wrist/gripper bounding box or proprioceptive vector can serve as the target. Pair with T12 (below) because the third-party control paper reports that a similar state-grounding head was indistinguishable from continued training.

### T5. SpecWM, spectral alignment (teach the latent a chosen similarity kernel)
- Source: PREPRINT arXiv 2610.15429 (Stanford, ETH, Brown; Balestriero is a co-author). Full text read (main pages and appendix excerpt).
- Mechanism (stated): replace SIGReg with a loss L_spec = ||C S C - K||_F^2 where S is the Gram matrix of (norm-rescaled) embeddings in a trajectory clip and K a target kernel. The kernel can be supervised (state, depth, proprioception distance) or self-supervised temporal (Laplacian kernel on |i-j| frame index, "requires only the ordering of observations"), or a learned quasimetric (QRL/IQE). Predictor trained with a cosine loss, planned with CEM using cosine similarity to the goal. For CALVIN they add 8 distant frames per clip and a two-bandwidth kernel.
- Measured (stated; CEM, 100 tasks per env, mean of 3 checkpoints x 3 seeds): OGBench Cube 52 to 68 (temporal) / 71 (quasimetric); OGBench Scene 49 to 77 / 74; CALVIN 29 to 51 / 47. Random-policy baseline 13/21/17. Average gain +21 points.
- Important caveats (stated): the training data included extra "failure" rollouts with noised actions (1M extra frames per env). Ablation on play data only (1 seed): SpecWM temporal falls from 68 to 48 (Cube), 77 to 48 (Scene), 51 to 38 (CALVIN); the paper says failure data helps both models and does not explain the gap. The play-only LeWM numbers are not given in the text I read (the figure labels are ambiguous), so I record none. Probing R^2 does not predict planning (e.g. Scene LeWM 0.09 vs temporal 0.78 is consistent, but CALVIN quasimetric R^2 0.07 yet success 47). Stop-gradient on the prediction target lowered planning in most cases. Optimizer: MuonAdamW. Training: 33,375 updates. Per-update frames: SpecWM 552 (744 on CALVIN) vs LeWM 552.
- Stated limits: next step is to find kernels for large-scale training; code at anonymous.4open.science/r/SpecWM/ (anonymous mirror).
- INFERENCE: temporal kernel needs only frame order, so it is usable on unlabeled robot video if the action-conditioned predictor is trained on a labeled subset. It may be an alternative or complement to SIGReg. Note the gain was shown only on one codebase and the failure-data confound is large.

### T6. MemWM (associative memory slots conditioning a JEPA predictor)
- Source: PREPRINT arXiv 2610.19683 (Shanghai Academy of AI for Science, Fudan; v2). Full text read.
- Mechanism (stated): M=10 learnable slots (dim 192) with paired learnable keys. Retrieval: softmax((z W_Q) Xi^T / (sqrt(D) tau)), tau=0.5, giving m_t = sum A_i Gamma_i. Conditioning is sequential AdaLN with parameter sharing: memory first, then action. Two memory losses: slot diversity (sum of squared pairwise cosine similarities between slots) and temporal diversity (1 - normalized entropy of mean slot usage over the horizon). Total loss = MSE + 0.09 * SIGReg + L_div + lambda_tem * L_tem. CEM planning as in LeWM.
- Measured (stated; mean of 3 seeds, 50 trajectories): average success 89.9% (PushT 96.7, TwoRoom 90.7, Reacher 90.7, OGBCube 81.3) vs DINO-WM 89.3 (75M trainable params + frozen 1.1B DINOv2), LeWM 85.8, PLDM 79.5. 19M params. Ablations: no memory loss 83.5; no temporal loss 86.7; no param sharing 82.9; 5/15/20 slots 86.5/86.0/85.7.
- Stated limits: needs action labels; evaluation budget was 50 steps. The paper notes LeWM's original TwoRoom budget was 100 and says LeWM would degrade to 40% at 50 steps (this is the authors' statement, which I could not check).
- Weak points: the +4.1 point gain over LeWM is of similar size to the seed noise reported by T12 (up to 26 points across evaluation seeds on OGBench-Cube for an untouched baseline). Reported intervals are only 0.9-2.5 points over 3 seeds on the same 50 trajectories. Slot-role interpretations ("approach", "push") are qualitative.
- Code: not stated in the pages read.
- INFERENCE: the slots may act as a cheap task-phase variable. Could be tested as an input to a robot controller for phase-dependent behavior.

### T7. H-JEPA (stack of JEPAs with distinct latent spaces; top-down planning)
- Source: PREPRINT arXiv 2610.06805 (NYU, AMI Labs, INRIA, Brown; v2, 1 Nov 2026). Full text read.
- Mechanism (stated): level l encodes pooled windows of the level below (stride 2, window 1), has its own predictor, loss = prediction + SIGReg (+ IDM term on real robot video). Plan top-down: the top level optimizes macro-actions toward the goal in its own latent space, and its first predicted state becomes the subgoal for the next level down. On DROID, an inverse-dynamics loss on consecutive latents is required (without it the model collapses to a slow scene-identity feature and performance is zero).
- Measured (stated; SE over 3 seeds): Visual AntMaze 18% to 73% (3 levels, flat LeWM vs H-JEPA), at lower planner compute. Changing only the cost space on AntMaze: native L1 cost 23.3 (2 levels), hierarchical 39.3 (2 levels), 73.3 (3 levels). On Push-T the 3-level model is poor because it sees only 58% of LeWM's transitions per epoch. DROID: the IDM gives about +34-40 points of Frechet fidelity; a second level adds a further +1 to +5 (read from a figure, approximate).
- Stated limits: selective abstraction appears only when fast and slow factors have a frequency gap of 2x or more (AntMaze, Humanoid, FourRoom), not in manipulation datasets; DROID evaluation is offline open-loop planning fidelity, not closed-loop robot success ("a next step is to test... closed-loop control on a physical robot"). Compute: H200 GPUs; code link on the project page (not read).
- INFERENCE: for tabletop manipulation the paper itself reports little benefit, so expect the gain to appear for locomotion or mobile tasks with fast body dynamics and slow position. A cheap precondition test: compute the entity frequency gap on the project's data before building levels.

### T8. Kairos hybrid temporal attention (local window + dilated window + gated delta rule)
- Source: PREPRINT arXiv 2606.16533 (Kairos Team, v3, July 2026). Selected pages read (about 10 of a very long report).
- Mechanism (stated): LinearDiT backbone mixing sliding-window attention (local), dilated sliding-window attention (d in {6, 12}, mid-range), and Gated DeltaNet as the only global pathway (update S_t = alpha_t S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T). Video DiT and Action DiT are coupled by mixed attention. At deployment the video branch can be switched off ("action-only mode").
- Measured (stated): Kairos-4B 5 s 720P generation: 2.3 PFlops, 23.5 GB, 43 s (1 GPU) / 9 s (4 GPUs), vs Wan2.2-5B 16.6 PFlops / 201 s / 85 s and Cosmos-Predict2.5-14B 156.5 PFlops / 2526 s / 687 s. PAI-Bench overall 80.8 (Cosmos-Predict2.5 2B/14B: 81.0). Ablations: human-centric pretraining 83.0 to 89.0; joint video+action training vs action-only prediction 65.8 to 89.0; adding explicit future imagination 89.0 to 90.8. 4-step distilled 480P model: 11.7 s on A800 (1 GPU), 11.4 s on RTX 5090.
- Stated limits (important): the report says closed-loop real-robot validation (imagined-vs-real correlation, failure prediction, safety filtering) is future work; "quantitative latency savings for the action-only path are not yet reported"; distillation failure modes (mode collapse, motion diminution) are described qualitatively, not measured. Code: github.com/kairos-agi/kairos.
- Weak points: the multi-timescale story is supported by a theorem about architectural sufficiency, not by an ablation I saw that isolates GLA vs SWA vs DSWA for control. Not verified: whether that ablation exists in unread pages.
- INFERENCE: for a latent world model the GDN state is a fixed-size recurrent memory that could replace a growing context in the predictor. Competes with MemWM (T6) as a memory mechanism.

### T9. Xiaomi Auto World Model 2.0 (View-Temporal Attention + gated linear temporal attention + streaming DMD + explicit Gaussian scene memory)
- Source: PREPRINT arXiv 2611.01677 (Xiaomi, 1 Nov 2026). Method pages read; the experiment tables were not in the excerpt I got, so no downstream numbers are recorded here.
- Mechanism (stated): (a) View-Temporal Attention alternates temporal-only and view-only attention layers (complexity O(T N (N+T)) instead of O(N^2 T^2)). (b) Gated linear attention used only in temporal layers, with a 4:1 ratio (one softmax anchor at blocks 4n+1) that "matches the full-softmax baseline in our experiments"; the per-layer temporal cache is cut by over 97% vs a softmax KV cache. (c) Consistency distillation, then streaming DMD with Random Gradient Sampling (keep gradients only for the last chunk plus a few randomly chosen earlier chunks) to fit up to 7 views in memory. (d) WorldRec 2.0 reconstructs static and dynamic 3D Gaussians in one forward pass and feeds their rendering through the control channel as scene memory.
- Not available: benchmark numbers, ablation tables, code (a project URL is given: xiaomiautoworldmodel.github.io/CausalWorldModel).
- INFERENCE: the 4:1 linear/softmax hybrid ratio and Random Gradient Sampling are cheap, portable tricks for training long-horizon autoregressive latent rollouts without backprop through all chunks.

### T10. GreenWAM recipe (shared anatomical action space, action expert split, consistency distillation)
- Source: PREPRINT arXiv 2610.18774 (Sber Robotics Center, Oct 2026). Pages 1-3, 7, 19, 26 read.
- Mechanism (stated): per-robot "codec" maps native joints to a shared 79-slot anatomical action vector (legs 12, body 23, arms 18, hands 26); empty slots are masked in the loss; a single shared input projection. Seven training modes (visual, action-conditioned, world-action) on a Cosmos 3 Edge 4B base. Policies split an action expert off the generator at the start of adaptation; world-model modes are kept during policy adaptation. A one-step policy is obtained by consistency distillation of four guided steps (EMA 0.995 target, plus an action flow-matching term).
- Measured (stated): real humanoid in-distribution instruction success 95.6% vs 88.9% for a policy adapted from the parent model and 84.4% for pi0.5 (one humanoid, one packing task). Latency on one RTX 4090: distilled policy 35.4 ms (28.3 Hz), 3.8x faster than the optimized teacher (134.6 ms), 12.3x faster than unoptimized. Controlled comparisons: robot specialization +3.0 points trajectory accuracy, -23% video prediction error, +5.4% policy success (+8.4% with action expert); action expert split +3.3%; keeping world-model modes +8.0% in randomized RoboTwin scenes.
- Stated limits: sensitive to task geometry changes (table height, initial robot configuration); error accumulates over long rollouts; each benchmark uses its own post-training; real-robot evaluation covers one humanoid and one task. Code and weights released (project page; link not read).
- INFERENCE: the codec/slot idea is directly reusable for any multi-embodiment controller; a cheap form is a fixed slot layout with a loss mask.

### T11. Compact future targets for world-action models (MiniWAM)
- Source: PREPRINT arXiv 2610.12194 (NUS/NTU/A*STAR). Abstract only.
- Mechanism (stated): predict compact future representations from a frozen encoder (PRISM, trained with inverse-dynamics supervision plus feature reconstruction) instead of native visual features.
- Measured (stated in abstract): 65x fewer future feature tokens, up to 8x faster world-action training, 0.25B parameters "competitive with substantially larger WAMs" on LIBERO, LIBERO-Plus, RoboTwin 2.0 (simulation). Baseline numbers not read. Code: only a project page (j1dan.github.io/MiniWAM).
- INFERENCE: independent support for the idea that action-relevant compact targets beat reconstructing everything, consistent with T4/T5.

---

## Part C. Hardware-aware and evaluation techniques

### T12. Evaluation caution: one-step fidelity is a poor proxy for planning (and baselines are noisy)
- Source: PREPRINT arXiv 2610.19375 (Cheung, Zhang He). Full text read. One architecture (LeWM, 15M), 3 envs, 3 training seeds.
- Stated findings: on PushT, the first epoch of fine-tuning cuts one-step error 32% and adds +4.4 points planning; over the next 15 checkpoints a further 15% fidelity gain has correlation r = -0.010 with planning success. A 93% fidelity improvement on TwoRoom gives -0.7 points of planning; 40% on PushT gives +4.9 [+1.1, +8.9]. Untouched baselines vary by up to 26 points across evaluation seeds (OGBench-Cube sd 13.1, PushT sd 6.0). An auxiliary state-grounding loss was indistinguishable from five more epochs of ordinary training. Low-data-support states plan about 18-20 points worse than high-support states.
- Recommendation (stated): paired multi-seed evaluation, a matched continued-training control, report seed spread.
- Why it matters here: it directly weakens the claim strength of T4, T5, T6 (all small differences on 50-100 tasks over 3 seeds).
- Code: not stated.

### T13. Low-bit recurrent-state quantization for linear-attention models (LeapQuant)
- Source: PREPRINT arXiv 2609.38166 (Berkeley, UW, MIT, Perplexity, NVIDIA). Abstract only (WebFetch).
- Mechanism (stated): quantize the Gated DeltaNet state once per token window (buffer high-precision updates inside the window), keep the largest outliers as high-precision "Compensator Tokens", smooth residuals before quantizing. Training-free.
- Measured (stated): near-lossless at 8-bit; kernel speedup 2.05-3.70x, end-to-end 1.47x on average; Qwen, Kimi, GLM families; B200, RTX PRO 6000, RTX 5090. No code link on the page. Limitations not stated on the page.
- Related, unread (titles only): 2610.39931 (low-bit GDN state, global gradient-aware allocation), 2610.15139 (scale-state feedback in recurrent-state quantization), 2610.12444 (4-bit AdamW states), 2610.41408 (SACQ ultra-low-bit VLA quantization), 2611.00524 (Reflex: reuse and correct past action chunks to cut policy-serving latency).
- INFERENCE: relevant only if the world model adopts GDN/GLA recurrent state (T8, T9). Note LLM-tested, not tested on video or control states, whose outlier structure may differ.

### T14. Streaming and distillation for real-time rollouts (shared pattern in T8, T9, T10)
- Stated common elements: few-step consistency distillation, then DMD or consistency refinement; EMA target; a loss on both video and action streams (GreenWAM); streaming DMD with random gradient sampling (Xiaomi). Failure modes reported (Kairos, qualitative): mode collapse, motion diminution, visual homogenization.
- INFERENCE: if the project's world model is a latent JEPA (no sampling), this does not apply directly. It applies to any diffusion/flow policy head.

---

## Open gaps (contradictions and weak assumptions)

1. Attribution mix-up in the brief: Reactive Optimizers page shows Claude Opus 5.5 and no Kinetix. Kinetix and GPT-6 Astra belong to the Weight-Space Programming blog. Someone should not cite the optimizer page for the control fix.
2. Weight-Space Programming evidence is a blog with a closed model. The Go baseline is described two ways (two smaller PPO nets vs an equal-sized PPO net on 20M games). The Kinetix baseline (PPO, 0/256) has no stated training budget in what I read. The result that edits fix looping but not ability (Gemma) suggests the Kinetix fix may be of the same kind (removing an obstacle in a policy that already nearly works). Not verified.
3. Weight Oracles scale from about 2K parameters. No evidence at the size of any world model or policy. The paper itself says this is the main open challenge.
4. Noise floor vs claimed gains. T12 reports untouched LeWM baselines vary by 6-13 points sd across evaluation seeds. MemWM's gain over LeWM is +4.1 points on 50 trajectories and 3 seeds. SpecWM's +16 to +28 points are larger than that noise but confounded by 1M extra failure frames. The Fourier-head paper reports big gains only on a custom n-ball task.
5. The state-grounding result (T12) and the Fourier head (T4) are close cousins: one is reported null against a matched control, the other has no matched continued-training control in the text I read. This is an open contradiction, not a resolved one.
6. Probing does not track planning. SpecWM reports probe R^2 and planning disagree; T4 reports planning improved on Reacher/Cube while the latent correlation with the physical property was lost; T12 reports fidelity and planning decouple. Any hypothesis that "more physical info in the latent means better control" is weakly supported.
7. MemWM re-evaluates LeWM at a 50-step budget and states it would fall to 40% on TwoRoom, while its own table copies LeWM's published 87% (all four LeWM entries match the LeWM paper). So MemWM's TwoRoom comparison mixes budgets, and the 40% figure is the authors' unverified statement.
8. Hierarchy helps where fast and slow factors separate (H-JEPA); the paper's manipulation environments show little abstraction and one (Push-T) gets worse at 3 levels. Another hierarchy paper (ID-WM, 2610.42146, abstract only, not read) claims 38.5% gains over policy/diffusion planners and 135% over single-level world-model MPC without baseline details in the abstract. Treat as unverified.
9. Kairos is explicit that no real-robot closed-loop evidence exists for its control-sufficiency claim, and that action-only latency savings are unmeasured. GreenWAM has one real robot and one task.
10. Hot feed coverage: about 94 of ~100 IDs were not examined (the tool returned IDs only).
11. Not read in full: Xiaomi experiments, Kairos ablations on temporal attention, alphaXiv-only items T14's source papers (Flash-WAM 2606.05254 and others referenced in GreenWAM), and any code repositories.
