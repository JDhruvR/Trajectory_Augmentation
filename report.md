# Phase 2 Final Presentation Report Draft

## 1. Innovation and Novelty
- **Action Invertibility:** We developed an automated pipeline to inject 6D inverse-kinematics physical noise into expert demonstrations and mathematically invert the actions to gracefully recover. This introduces critical out-of-distribution (OOD) states, forcing downstream policies to learn robust recovery behaviors rather than simply cloning perfect, deterministic trajectories.
- **Experimentation:** Our innovation is backed by rigorous experimentation. We successfully validated the invertibility mathematical proofs in the MuJoCo simulator, handled complex state-dimensionality shifts across suites (e.g., 110-dim to 79-dim), and resolved low-level OpenGL/EGL rendering artifacts (the 180-degree camera flip issue) to ensure perfect visual fidelity.
- **Trajectory-Level Perturbation:** State-of-the-art benchmarks like LIBERO-Plus reveal that current VLAs fail drastically on initial state distributions (performance collapsing from >95% to <30%). While existing methodologies focus purely on scene-level augmentations (lighting, textures), we uniquely address this by perturbing the robot's actual state trajectory. This algorithmically expands the dataset to previously unseen, physically valid configurations. Furthermore, this structured trajectory expansion unlocks immense potential for future applications in curriculum learning and RL-based few-shot fine-tuning.
- **Standardization & LeRobot Compliance:** Most manipulation datasets are still isolated in native, complex formats like HDF5. We architected a highly parallelized conversion pipeline to map the LIBERO state vectors directly into the Hugging Face **LeRobot v3.0 format** (Parquet + SVT-AV1 compressed video). This makes the dataset universally accessible and "plug-and-play" ready, making it extremely easy to use in any modern VLA development pipeline.

## 2. Architecture & Open Datasets
- **Source Datasets:** We built upon the standard open-source **LIBERO HDF5 datasets**, leveraging the native expert demonstrations for the `libero_goal`, `libero_spatial`, and `libero_object` benchmark suites.
- **Architectural Workflow:**
  1. Parse original HDF5 demonstrations to locate critical pre-grasp timesteps.
  2. Inject bounded, physically valid Gaussian noise using a headless MuJoCo simulator.
  3. Step back toward the original path using the mathematically proven inverse action (reducing reconstruction error drastically from 0.613 with naive math to 0.060 with simulator replay).
  4. Convert the robust, augmented trajectories into the LeRobot format for high-throughput VLA training.
- **Why LeRobot?** LeRobot is the rapidly emerging gold standard in the Hugging Face robotics ecosystem. By standardizing our output to LeRobot, our dataset integrates natively with Hugging Face's `LeRobotDataset` API, enabling instant data loading for architectures like SmolVLA, ALOHA, and ACT.


## 3. Final Deliverables
- **The Dataset:** A fully augmented, LeRobot-compatible dataset encompassing multiple LIBERO task suites, designed specifically to train robust recovery behaviors.
- **Scalable Generation:** While our published dataset provides 13,000 newly augmented trajectories (2 augmentations per expert demonstration), our open-source codebase is entirely parameterized. Researchers can seamlessly scale this to generate an unlimited number of unique OOD recovery trajectories per expert demo to fit their specific training scale and compute budget.
- **Link:** [https://huggingface.co/datasets/jdhr/libero_trajectory_augmented](https://huggingface.co/datasets/jdhr/libero_trajectory_augmented)

## 4. Future Work
- **Training & Evaluation Scope:** The natural next step is to fine-tune compact VLAs (such as SmolVLA or Qwen-based VLAs) exclusively on this augmented dataset to quantify trajectory-diversity gains without requiring new human demonstrations.
- **Targeting Benchmarks:** By evaluating on robust splits like LIBERO-Plus and LIBERO-PRO, future studies can concretely measure the reduction in brittleness when models face mid-trajectory disturbances and unseen robotic start configurations.
- **Curriculum Learning Framework:** We can develop a curriculum-based training pipeline where VLAs are initially taught using easier, shorter trajectories originating near the target grasp point. As the agent's success rate improves, we can incrementally introduce trajectories starting further away, eventually chaining them with perturbations of increasing magnitude (e.g., perturbing from an already perturbed state).
- **Sim-to-Real (Sim2Real) Transfer Validation:** Real-world robotic deployments suffer heavily from motor jitter, sensor noise, and slight spatial discrepancies. Because our trajectory augmentation physically simulates these exact types of noise, a critical future step is deploying VLAs trained on our dataset onto physical robotic arms (e.g., a physical Franka Panda) to see if this method effectively closes the sim-to-real gap.

## 5. AI & Agentic Tool Usage
We relied heavily on AI coding assistants (Gemini 3.1 Pro and Claude Opus 4.6) through agentic interfaces like **Antigravity** and **AlphaXiv** to accelerate development. 

### Reasoning & Planning
The most successful workflow we discovered was the **Plan-Then-Execute** pattern. By forcing the agent to generate a structured implementation plan and wait for human approval before writing code, we prevented it from going down counterproductive rabbit holes.

### What Worked Well
- **Asynchronous Tool Chaining:** The agent flawlessly orchestrated terminal commands, spun up `tmux` sessions, and handled long-running batch conversions (processing ~4,500 trajectories) autonomously in the background.
- **Stack Trace Parsing:** The agent excelled at deciphering cryptic MuJoCo and Robosuite stack traces, saving hours of manual debugging.
- **Dynamic Code Exploration:** The agent autonomously navigated the dense LIBERO source code to discover undocumented state dimensionality changes across suites without explicit human guidance.
- **Agent Skills Integration:** We utilized strict behavioral prompts (adapted from mattpocock/skills) like `tdd` for test-driven development, `improve-codebase-architecture` for modularity, and `caveman` to dramatically compress token overhead. We intentionally kept our `AGENT.md` guidelines under 150 lines to prevent context degradation.

### Challenges & Human Intervention (What Didn't Work)
- **Math Across Files:** The agent struggled with tracking mathematical operations (like our 180-degree image flip bug) across multiple scripts, initially attempting to apply FFmpeg band-aids rather than fixing the root matrix math in the rendering pipeline.
- **Sycophancy & Rabbit Holes:** The agent occasionally exhibited "sycophantic" loops—blindly repeating failed bash commands to please the user instead of stepping back to diagnose the failure. It also struggled to "zoom out" of failing approaches without explicit human overrides.
- **Context Loss:** Over long sessions, the agent would occasionally forget global rules (like always using headless `egl` rendering), which required us to track engineering decisions in dedicated markdown files as a traceable record.
- **Spontaneous Commits:** The agent would occasionally commit changes to git without asking, requiring strict behavioral controls via `git-guardrails` to block dangerous operations before execution.

**Takeaway:** Agentic coding tools dramatically sped up our development—condensing weeks of manual debugging into days. However, the best results came from treating the agent as a highly capable junior engineer: excellent at execution, but requiring strict guardrails, architectural direction, and human check-ins to stay on track.
