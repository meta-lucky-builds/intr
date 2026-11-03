# Hot feed triage

Source: alphaXiv get_hot_feed, limit 100. The feed returned 101 entries by count: 96 arXiv IDs and 5 slug-style IDs.
Titles and first sentences were fetched from https://arxiv.org/abs/<id> with WebFetch. 96 fetched, 0 failed.
Slug-style IDs were not fetched and are marked "not an arXiv id". Their tag is "other" because they were not checked.

Sentences marked * were returned by WebFetch as a partial quote or a paraphrase, because of its quote-length limit. They are not verbatim.

| id | title | first sentence | tag |
|---|---|---|---|
| 2611.weight-space-programming | not an arXiv id | - | other |
| 2611.mistral-large-4 | not an arXiv id | - | other |
| 2610.39302 | Proving and Improving the Platonic Representation Hypothesis with JEPAs | Whether models trained on different modalities converge to the same representation of the world is central to AI. | world-model |
| 2611.02270 | Hinted Scaling Laws | Scaling laws predict the outcome of large, expensive training runs by extrapolating trends from smaller, cheaper runs. | other |
| 2610.13732 | LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos | Scaling generalist robotic policies is limited by the high cost of collecting embodied interaction data. | robotics |
| 2611.automating-ai-rd-for-molecular-discovery | not an arXiv id | - | other |
| 2611.01560 | Democratizing Robot-Use Agents: A Unified Harness with Open Models | "Robot-use Agents (RUAs) have shown promising zero-shot manipulation capabilities,"* | robotics |
| 2611.03178 | CoworkOS: Scaling Agent Coworkers for Real-world Automation | "Rapid advances in cowork agents are shifting human-AI collaboration toward real-world automation,"* | other |
| 2610.42146 | Hierarchical World Models with Implicit Dynamics | "Planning with world models enables test-time adaptation for embodied agents,"* | world-model |
| 2611.03648 | Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation | "General-purpose mobile manipulation requires a combination of capable robot hardware, scalable demonstration collection,"* | robotics |
| 2611.03126 | CUA-Master: Scaling Computer-use Agents through Open-World Recursive Improvements | Constraints on interaction data quality are the bottleneck for computer-use agents.* | other |
| 2611.02833 | Reimagining computational quantum research with AI scientists | "AI scientists are expanding automated research across scientific domains,"* | other |
| 2611.03442 | Subpolynomial-Query Proximal Sampling Beyond Log-Concavity | "We study the gradient query complexity of sampling from a density,"* | optimization |
| 2611.03646 | EgoActionBench: Benchmarking Dense Action Captioning for Egocentric Human Videos | "Learning from human data offers a scalable path toward embodied intelligence, yet it requires models"* | robotics |
| 2611.03647 | Pixel-UMI: Learning Environment-Aware Whole-Body Manipulation from Sensorized Human Demonstrations | "Sensorized human demonstrations offer scalable data for mobile manipulation,"* | robotics |
| 2611.02720 | Inverse optimization for divisor methods of apportionment | The apportionment problem is to allocate a fixed number of seats among regions in proportion to their populations. | optimization |
| 2611.03092 | Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents | "Personal AI agents are increasingly capable of handling long-horizon tasks across applications,"* | other |
| 2611.03622 | D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models | "Vision-Language Models (VLMs) hold promise for zero-shot video task-progress estimation in robot learning."* | robotics |
| 2611.03618 | Strategist: Evolving Reasoning with Agents | A model is only half of an AI agent system. | other |
| 2611.02102 | NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale | "Scaling computer-use agents from individual actions to complex software work requires experience,"* | other |
| 2611.02846 | SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding | "Speculative decoding (SD) accelerates large language model inference by reducing the sequential cost"* | other |
| 2611.01860 | Advances in Agentic Computer Graphics | Computer graphics is built on code. | other |
| 2611.03584 | TAMPAgent: Agentic Belief-Space Planning for Mobile Manipulation | Open-world robotic mobile manipulation requires acting in physical environments with only partial information. | robotics |
| 2611.03602 | Reviso: Not All Visual Evidence Deserves Equal Attention in Video Captioning | "Multimodal large language models (MLLMs) can generate detailed video captions, yet they may overemphasize minor details"* | other |
| 2611.02487 | RecRSI: Mechanism-Guided Discovery of Recommendation Algorithms | "Improving recommendation algorithms for the Web requires coordinating interacting design choices,"* | other |
| 2611.03633 | Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap | "Muon has recently emerged as a promising optimizer for pretraining large language models" | optimization |
| 2611.01565 | Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining | "In modern robot learning, generative robot policies are first pretrained through supervised learning from demonstrations,"* | robotics |
| 2611.03520 | PanoTac: A Dense Three-Axis Tactile Sensor for Robust Touch-Centric Manipulation | Contact-rich manipulation benefits from tactile feedback that captures both detailed contact geometry and interaction forces. | robotics |
| 2611.02171 | GoalDex: Learning Image-Goal-Conditioned Policies for Dexterous In-Hand Manipulation | "Dexterous in-hand manipulation is commonly formulated around explicit object pose estimation,"* | robotics |
| 2611.02236 | Iterative Gumbel Planning for Continuous Control | "Search-based planning remains a powerful policy improvement paradigm in reinforcement learning,"* | other |
| 2611.02436 | M3Former: Fine-Grained Cross-Layer Mixing for Transformers | "Residual connections give each Transformer block access to all earlier layers only through one fixed sum,"* | other |
| 2611.02691 | Closing the Dynamics Gap through Self-Directed Robotic Interaction | "Agentic Real2Sim2Real systems can reconstruct a scene in simulation from camera observations, but contact-rich dynamics"* | robotics |
| 2611.03228 | AnchorComplete: Recovering Complete Scene Meshes from Partial Point Clouds | "Most image-based 3D reconstructors primarily recover visible surfaces, producing partial point clouds with large holes,"* | other |
| 2611.02497 | RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild | "General-purpose robot policies must work across robots, tasks, and environments,"* | robotics |
| 2611.03517 | Ask, Don't Equip: Question-Driven Coding Agents for Deep Kernel Optimization | LLM agents for GPU kernel optimization sit at two ends of a structure-freedom tradeoff. | other |
| 2611.02476 | GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data | "Dialectal Arabic speech recognition remains constrained by the scarcity of dialect-resolved transcribed speech,"* | other |
| 2611.03294 | ForceLDM: A Force-aware Latent Dynamics Model for Contact-Rich Manipulation | "Contact-rich manipulation requires fine-grained control and complex physical interactions with the environment,"* | robotics |
| 2611.03412 | DeliverBench: Benchmarking AI Agents on Delivering Realistic Multi-Stage Professional Projects | LLM agents are judged by what they deliver, and in professional work what gets delivered is a project, not a task. | other |
| 2611.01484 | CrossSWE: Benchmarking Cross-Repository Joint Development for Coding Agents | "As coding agents achieve increasingly high success rates on software engineering benchmarks, recent efforts have sought"* | other |
| 2611.01729 | ODE Policy: Robot Learning via Action Dynamics, Not Actions | "Most robot policies, whether trained by imitation learning (IL) or reinforcement learning (RL), map observations to action"* | robotics |
| 2610.41711 | Hierarchical Residuals: A Minimalist Decoupling of Residual Width from Compute Width | "Residual connections underpin Transformers, but they implicitly tie the residual width carried through depth"* | other |
| 2611.03604 | Near-Optimal Space and Time Cut Sparsification in Turnstile Streams | "A seminal result of Ahn, Guha, and McGregor (PODS'12) showed that a cut sparsifier of an unweighted graph"* | other |
| 2611.03634 | WebHarbor: Docking Real Websites for Evolving GUI Agent Environments | GUI web agents require realistic and interactive environments that remain stable during both evaluation and training. | other |
| 2611.03394 | TaoMate-HD: Progressive-Resolution Autoregressive Audio-Video Generation | Streaming digital humans produce detailed video while continuing speech as new text arrives. | other |
| 2611.03272 | Diverse-Teacher On-Policy Distillation for Generative Recommendation | Generative recommendation generates the semantic ID (SID) of the next item from a user's interaction history. | other |
| 2611.03501 | When Robots Outweigh Objects: How Inertia Mismatch Shapes Robot Policy Learning | "Conventional robot arms can be extremely heavy compared with the everyday objects they manipulate,"* | robotics |
| 2611.03371 | EO-Jev: Decision Model for Earth Observation | "Earth observation (EO) workflows require selecting among known outcomes, estimating confidence, and processing large volumes"* | other |
| 2611.03033 | GPU-Native Real-Time Textured Mesh Reconstruction with Seam-Aware Multi-View Radiometric Correction | "We present a GPU-native volumetric-video system for real-time reconstruction of graphics-compatible"* | other |
| 2611.03543 | Can LLM Agents Improve Imperfect Earth System Model Parameterizations? | Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection.* | other |
| 2611.02613 | MAPO: Mixture of Experts with Alternating Policy Optimization for Universal Humanoid Motion Tracking | Universal humanoid motion tracking is essential for enabling humanoid robots to perform diverse tasks in human environments. | robotics |
| 2611.01659 | HuG: Learning Humanoid Whole-Body Affordance-Aware Grasping with Active Perception | "Acquiring objects in human environments is hard for humanoids: targets are partially occluded under egocentric viewpoints,"* | robotics |
| 2611.02981 | Utility-Aware Generative Recommendation: Utility-Enriched Semantic Codes and Utility-Guided Decoding | "Generative recommendation has been widely adopted in large-scale industrial recommendation systems,"* | other |
| 2611.02600 | W2A: Distilling World Priors into Action Models for Real-Time Robotic Control | "World Action Models (WAMs) have recently shown great promise in robotic control,"* | robotics |
| 2611.03142 | DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction | "Scaling depth is a key route to stronger large language models,"* | other |
| 2611.02516 | Cumulative and Audited Guardrail Tool Evolution for LLM Agent Safety | "Large language model (LLM) agents increasingly operate in open-world environments,"* | online-adaptation |
| 2611.02857 | Solv3R: 3D Looped Transformer as a Multi-view Solver | "Classical structure-from-motion reconstruction proceeds iteratively,"* | recurrence-or-memory |
| 2610.42147 | Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention | "Agentic workflows are increasingly shifting towards long-horizon tasks, such as open-ended scientific work,"* | other |
| 2611.03498 | SNAP: Semantic Navigation Among Pedestrians | "Navigating crowded urban environments requires robots to avoid pedestrians while adhering to social norms"* | robotics |
| 2610.40564 | Training Foundation Models with a Single Algorithm | "Foundation models are typically trained in stages: pretraining, midtraining, supervised fine-tuning (SFT),"* | other |
| 2611.03097 | DR Pixel: Page-Native Deep Research with Vision-Language Models | "Complex documents such as scientific papers contain rich information in tables, figures, and equations; however,"* | other |
| 2611.03510 | I want to Break Free: The Positive Side of Symmetry Breaking | We study the role of internal field-space symmetries in Effective Field Theories constrained by positivity bounds. | other |
| 2611.01718 | Delivered, Not Used: Evaluating Agent Memory at Its Point of Use | "Task success, retrieval quality, and cost do not show whether an LLM agent used a delivered memory,"* | recurrence-or-memory |
| 2611.02943 | Exact Fusion of Softmax Attention Heads Can Require Exponentially Many Atoms | "Fusing independently normalized attention heads can increase the number of explicit key-value entries, or atoms"* | other |
| 2611.02830 | Think with Skills: Pseudocode Planning Before Execution for Advanced Reasoners | Language model agents often struggle with complex tasks even when equipped with expert-written skills. | other |
| 2611.01736 | RSI-Env: An Open-Ended Environment for Recursive Self-Improvement of Agents | "Recursive self-improvement (RSI) envisions agent systems that autonomously improve their own models and harnesses"* | other |
| 2611.pedestrian-harm-poverty-exposure-new-york | not an arXiv id | - | other |
| 2611.02633 | Measuring AI Agents as Teammates with Collaborative Intelligence | Progress in AI is often measured by what a single model can do on its own. | other |
| 2611.02515 | Summing over Seifert Manifolds in Narain Ensemble Averages | The ensemble average of Narain conformal field theories over their moduli space is a real-analytic Eisenstein series.* | other |
| 2611.03043 | VideoEditor-MoE: Scaling Complex Video Editing with Mixture of Experts | Complex video editing requires understanding complex instructions while jointly fulfilling multiple editing requests. | other |
| 2611.ced-moe-reversible-service-tiers | not an arXiv id | - | other |
| 2611.03533 | CoLa3D: Composable Latent 3D Decomposition | "We introduce CoLa3D, a simple and effective model for composable 3D scene decomposition that"* | other |
| 2611.03645 | Episodic Star Formation - II. The Origin of the Fundamental Metallicity Relation | "The Fundamental Metallicity Relation (FMR) extends the mass-metallicity relation (MZR)"* | other |
| 2611.03600 | Four Bridges: Measuring Deception Propensity in a Multi-Agent Game | "AI agents are increasingly deployed in settings where they interact and coordinate with other agents,"* | other |
| 2611.03595 | DynaSolver: A Neural Operator for Autoregressive 4D Flow Prediction with Dynamic Boundaries | Neural operators serve as efficient stand-ins for computational fluid dynamics (CFD), but most are built for static geometries.* | other |
| 2611.03150 | FlowBatt: Flow Matching for Probabilistic Battery Degradation Prediction | Battery degradation remains a central challenge in the development and broad application of sustainable energy technologies. | other |
| 2611.03143 | Visual-to-Executable Procedural Reconstruction of Buildings | "Procedural modeling offers compact, editable, and reusable representations for architectural content"* | other |
| 2611.03116 | From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents | "Large language models (LLMs) are driving autonomous agents toward real-world environments,"* | other |
| 2611.02629 | DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation | "Long-horizon dexterous manipulation remains a fundamental challenge for robot learning,"* | robotics |
| 2611.02782 | History Forcing: Distilling History-Dependent Continuations for Long-Horizon Video World Models | "Few-step autoregressive video diffusion enables efficient generation, yet maintaining consistency over long rollouts"* | world-model |
| 2611.02540 | PixelBench: Can LLMs Understand and Generate Images Pixel by Pixel? | "Large language models (LLMs) can generate visual assets through text-based formats such as SVG and CAD,"* | other |
| 2611.03233 | Axion 1.0: Open Agent Foundation Models, Simplified | "We introduce Axion 1.0, a fully open agent foundation model that combines strong and broad agentic capabilities"* | other |
| 2611.03424 | IdeaTaste: Do LLMs Share the Research Taste of Human Researchers? | "Research taste, the judgment of what to read, which problems to pursue, and which methods to try,"* | other |
| 2611.02105 | Faster Semi-Streaming and Dynamic Approximate Maximum Weight Matchings in General Graphs: A Unified Framework | "We study the (1-epsilon)-approximate maximum weight matching problem in general graphs."* | optimization |
| 2611.03626 | Isekai3D SDG: A Platform for Diverse Synthetic Gameplay Visual Data | "We present Isekai3D Synthetic Data Generation (SDG), a platform for simulating and capturing synthetic 3D gameplay"* | other |
| 2611.03571 | Nonlinear gauge maps and the structure of Goldstone loops in canonical inflation | "We clarify the nonlinear relation between the spatially flat scalar and Goldstone descriptions of canonical single-field"* | other |
| 2611.03601 | DiabAgent: Toward Agentic Physiological Control | As pretraining scales, medical language models have advanced rapidly in clinical question answering. | other |
| 2611.02697 | Two Fast Radio Burst Candidates Detected in the RATAN-600 Blind Survey | "We report the detection of two single dispersed pulses with widths of about 6.7 ms and 15.6 ms."* | other |
| 2611.03489 | Exploring Proximity and Tactile Sensing for Robotic Manipulation | "Pre-touch sensing, or sensing information about an object before contact, is an attractive modality for robot hands"* | robotics |
| 2611.02797 | SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents | Researchers increasingly see recursively self-improving systems as a promising route to artificial general intelligence.* | other |
| 2611.01967 | Better Answers, Weaker Evidence Control: Evidence Bypass in On-Policy Distillation | On-policy distillation (OPD) can improve answer accuracy while weakening responsiveness to supporting evidence. | other |
| 2611.00859 | Wave Forcing: Towards Speed-of-Light Streaming Video Generation | Real-time long-video generation requires continuous output with consistent appearance and natural motion. | other |
| 2611.01701 | Just-in-Time Skill Retrieval via Long-Horizon State Tracking | LLM agents increasingly rely on external skill libraries to support complex task execution. | recurrence-or-memory |
| 2611.03410 | OpenZilo: An Open Smart-Ring Platform for Human-Centric Sensing and Interaction with Robots | Wearable robot interfaces must connect user input to timely physical action. | robotics |
| 2611.02985 | Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model | The scarcity of human-annotated literary corpora limits the development of domain-specific sentiment analysis models. | other |
| 2611.03624 | Lee-Yang theorems for q-deformed oscillators and doubly-scaled SYK | "We show that the vacuum partition function of the q-deformed oscillator"* | other |
| 2611.03347 | Smoothing the Bouncing-Geodesic Singularity in AdS/CFT | Black hole singularities remain a central open problem in quantum gravity. | other |
| 2611.03112 | Many-Body Correlations in Phase-Space Microscopes | "The recently proposed phase-space microscope (PSM) extends quantum gas microscopes"* | other |
| 2611.03577 | Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training | Most parameters of a Mixture-of-Experts (MoE) model belong to its experts.* | optimization |
| 2610.41903 | Statistical Foundations of Reinforcement Learning: A Non-Asymptotic Perspective | Reinforcement learning (RL) embodies a foundational paradigm for sequential decision-making in the face of uncertainty. | other |
| 2611.03565 | The Virasoro Bootstrap: from the c~1 desert to coupled Potts | "We develop a numerical conformal bootstrap for two-dimensional CFTs with c>1 based on Virasoro conformal blocks"* | other |
| 2611.02667 | UPSHIFT: User Preference Steering from Human-AI Interaction Feedback Traces | Personalizing large language model assistants requires adapting their responses to individual users. | online-adaptation |
