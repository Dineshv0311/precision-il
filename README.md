# Vision-Based Imitation Learning for Precision Robotics Manipulation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Physics: MuJoCo 3.1](https://img.shields.io/badge/Physics-MuJoCo%203.1-blue.svg)](https://mujoco.org/)
[![Environment: Robosuite 1.4](https://img.shields.io/badge/Environment-Robosuite%201.4-green.svg)](https://robosuite.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, simulation-first visuomotor imitation learning framework benchmarking **Behavioral Cloning (BC)**, **Action Chunking Transformer (ACT)**, and **Diffusion Policy** on precision robotic manipulation using a 7-DoF Franka Emika Panda arm.

---

## Table of Contents

1. [Project Overview & Research Framing](#1-project-overview--research-framing)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Installation & Environment Setup](#4-installation--environment-setup)
5. [Execution Pipeline](#5-execution-pipeline)
6. [Experimental Results & Benchmark Analysis](#6-experimental-results--benchmark-analysis)
7. [Training Dynamics & Loss Analysis](#7-training-dynamics--loss-analysis)
8. [Future Work: Sim-to-Real Transfer](#8-future-work-sim-to-real-transfer)
9. [Key References](#9-key-references)

---

## 1. Project Overview & Research Framing

Traditional robotic manipulation relies on hardwired, hand-tuned trajectories that require recalibration whenever object poses or environmental parameters drift. Vision-Language-Action (VLA) models (such as RT-1, RT-2, and OpenVLA) achieve broad generalization but demand thousands of real-robot demonstration hours and compute clusters that are out of reach for academic laboratory constraints.

### Core Contribution

This project does **not** claim to invent a new multimodal fusion method. Instead, it delivers a **reproducible, simulation-first, pure Imitation Learning (IL) pipeline**, built on Robosuite/MuJoCo, structured around three concrete contributions:

| # | Contribution | What it means in practice |
|---|---|---|
| 1 | **Multimodal 15D State Fusion** | Combines 2D visual spatial keypoints (ResNet-18 + Spatial Softmax) with proprioceptive joint/end-effector pose and **6-DoF wrist Force/Torque wrench** measurements into a single state representation. |
| 2 | **Controlled Comparative Benchmark** | Trains three different policy families — single-step reactive BC, autoregressive CVAE-transformer chunking (ACT), and generative reverse-denoising (Diffusion Policy) — on the **exact same** perceptual representation, so differences in results come from the policy architecture, not the inputs. |
| 3 | **Systematic Initial-Pose Error Stress-Testing** | Quantifies where each policy's performance breaks down by testing it across four increasing levels of starting-position disturbance ($\Delta \mathbf{p} \in [0, 4]\,\text{cm}$, $\Delta \theta \in [0^\circ, 30^\circ]$). |

---

## 2. System Architecture

The pipeline has two parallel input streams (vision and proprioception/force) that get fused into a single feature vector, which then feeds three interchangeable policy heads:

```text
                         ┌───────────────────────────────┐
                         │       Franka Emika Panda       │
                         │   Robosuite Simulation (20 Hz)  │
                         └────────────────┬────────────────┘
                                          │
              ┌────────────────────────────┴────────────────────────────┐
              │ (RGB Observation)                    (Proprioception + Wrench)
              ▼                                                        ▼
  ┌──────────────────────────┐                          ┌──────────────────────────┐
  │ Camera Frame (3,128,128) │                          │     15D State Vector      │
  └─────────────┬────────────┘                          │  - EEF Position   (3D)    │
                │                                        │  - EEF Quaternion (4D)    │
                ▼                                        │  - Gripper State  (2D)    │
  ┌──────────────────────────┐                          │  - Linear Force   (3D)    │
  │  ResNet-18 Feature Maps  │                          │  - Angular Torque (3D)    │
  │       (512, 4, 4)        │                          └─────────────┬──────────────┘
  └─────────────┬────────────┘                                        │
                │                                                     │
                ▼                                                     │
  ┌──────────────────────────┐                                        │
  │  Spatial Softmax Layer   │                                        │
  │ (32 Keypoints -> 64D)    │                                        │
  └─────────────┬────────────┘                                        │
                │                                                     │
                ▼                                                     │
  ┌──────────────────────────┐                                        │
  │  Visual Embedding (64D)  │                                        │
  └─────────────┬────────────┘                                        │
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           ▼
                          ┌────────────────────────────────┐
                          │  Multimodal Fusion MLP (128D)   │
                          └────────────────┬─────────────────┘
                                           │
                  ┌─────────────────────────┼─────────────────────────┐
                  ▼                         ▼                         ▼
     ┌──────────────────────┐  ┌───────────────────────┐  ┌────────────────────────┐
     │       BC Policy       │  │      ACT Policy        │  │    Diffusion Policy     │
     │   Single-Step (7D)    │  │  Action Chunk (16×7)   │  │  Denoised Chunk (16×7)  │
     └──────────────────────┘  └───────────────────────┘  └────────────────────────┘
```

**What each stage does:**

- **Franka Emika Panda in Robosuite (20 Hz):** the simulated robot and physics environment that generates observations and executes actions each control step.
- **Camera Frame → ResNet-18 → Spatial Softmax → Visual Embedding:** the vision branch. A raw RGB frame is passed through a ResNet-18 backbone to get feature maps, which Spatial Softmax then compresses into a 64-D embedding representing *where* the important visual features are, not just *whether* they're present.
- **15D State Vector:** the proprioception + force branch — end-effector position/orientation, gripper state, and the 6-DoF force/torque wrench felt at the wrist, concatenated into one vector.
- **Multimodal Fusion MLP:** a small network that merges the visual embedding and the 15D state vector into one 128-D fused representation the policy heads can consume.
- **BC / ACT / Diffusion Policy heads:** three interchangeable "last stages" that turn the fused representation into an action — a single 7-DoF delta pose (BC), a 16-step chunk of actions (ACT), or a 16-step chunk produced by iterative denoising (Diffusion Policy).

### Why Spatial Softmax over Global Average Pooling (GAP)?

Standard vision backbones often use GAP ($C \times H \times W \to C \times 1 \times 1$), which discards *where* in the image a feature activated — it only keeps *how much*. In precision manipulation this is a real problem: a 2 cm object translation can produce nearly identical channel-averaged activations, making the policy effectively "coordinate-blind."

Spatial Softmax instead computes the expected spatial coordinate $(\mu_x^c, \mu_y^c)$ of each channel's high-activation keypoint:

$$\mu_x^c = \sum_{h=1}^H \sum_{w=1}^W \alpha_{h,w}^c \cdot x_{w}, \qquad \mu_y^c = \sum_{h=1}^H \sum_{w=1}^W \alpha_{h,w}^c \cdot y_{h}$$

where $\alpha^c$ is the 2D spatial softmax distribution across that channel's spatial grid. This keeps the encoder's output continuous and differentiable with respect to true image coordinates, so the policy can learn precise, coordinate-sensitive corrections instead of coarse presence/absence signals.

---

## 3. Directory Structure

```text
precision-robotics-il/
├── data/
│   └── lift_demos.hdf5              # 50 HDF5 demonstration trajectories (15D state)
├── models/
│   ├── bc_checkpoint.pt             # Trained BC baseline checkpoint
│   ├── act_checkpoint.pt            # Trained ACT transformer checkpoint
│   └── diffusion_checkpoint.pt      # Trained Diffusion policy checkpoint
├── outputs/
│   ├── eval_rollouts/               # Output MP4 evaluation videos per tier
│   ├── benchmark_results.json       # Quantitative evaluation metrics
│   └── evaluation_benchmark.png     # Comparative degradation curve figure
├── src/
│   ├── dataset/
│   │   ├── collect_demonstrations.py  # Scripted expert trajectory generator
│   │   └── robosuite_dataset.py       # PyTorch Dataset supporting action chunking
│   ├── envs/                          # Environment wrappers and configurations
│   ├── eval/
│   │   └── evaluate_benchmark.py      # 4-tier systematic pose-error evaluator
│   └── models/
│       ├── vision_encoder.py          # ResNet-18 + Spatial Softmax fusion model
│       └── policies.py                # BC, ACT, and Diffusion Policy heads
├── demo.py                          # 1-click live execution script for reviews
├── plot_results.py                  # Publication-grade chart generation
├── requirements.txt                 # Pinned package dependencies
└── README.md                        # This file
```

---

## 4. Installation & Environment Setup

### Prerequisites

- Windows 10/11 or Ubuntu 20.04/22.04
- Python 3.10
- Visual C++ 2015–2022 Redistributable (x64) — **Windows only**

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/precision-robotics-il.git
cd precision-robotics-il

# 2. Create an isolated environment
conda create -n precision_il python=3.10 -y
conda activate precision_il

# 3. Install PyTorch with CUDA 11.8 (or the standard CPU-only build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Initialize Robosuite macros
python -m robosuite.scripts.setup_macros
```

---

## 5. Execution Pipeline

Each step below is independent and produces an artifact the next step consumes — you can re-run any single step without repeating the others, as long as its inputs already exist.

### Step 1 — Expert Demonstration Collection

Generates 50 successful trajectories, logging RGB images, the 6-DoF F/T wrench, and proprioception into `data/lift_demos.hdf5`.

```bash
python src/dataset/collect_demonstrations.py
```

### Step 2 — Policy Training

Trains each policy architecture independently on the same demonstration set, saving a checkpoint per policy into `models/`.

```bash
# Train Behavioral Cloning (MLP baseline)
python train.py --policy bc --epochs 20

# Train Action Chunking Transformer (ACT)
python train.py --policy act --epochs 25

# Train Diffusion Policy (DDPM denoiser)
python train.py --policy diffusion --epochs 25
```

### Step 3 — Systematic Robustness Benchmark

Evaluates all three trained checkpoints across the four programmatic perturbation tiers described in [Section 6](#6-experimental-results--benchmark-analysis), running 5 trials per tier, and writes results to `outputs/benchmark_results.json`.

```bash
python src/eval/evaluate_benchmark.py --trials 5
```

### Step 4 — Generate Presentation Figures

Turns the benchmark JSON into the degradation-curve plot and comparative bar chart used in the report/slides, saved to `outputs/evaluation_benchmark.png`.

```bash
python plot_results.py
```

### Step 5 — Run the 1-Click Interactive Demo

Executes a single live rollout of a chosen policy — useful for panel/review demonstrations.

```bash
python demo.py --policy act
```

---

## 6. Experimental Results & Benchmark Analysis

Policies were evaluated across four controlled initial-pose disturbance tiers on the Robosuite Franka Panda **Lift** task:

- **Tier 0 (Nominal):** $\Delta \mathbf{p} = 0.0\,\text{cm}$, $\Delta \theta = 0.0^\circ$
- **Tier 1 (Low):** $\Delta \mathbf{p} = \pm 1.0\,\text{cm}$, $\Delta \theta = \pm 5.0^\circ$
- **Tier 2 (Medium):** $\Delta \mathbf{p} = \pm 2.5\,\text{cm}$, $\Delta \theta = \pm 15.0^\circ$
- **Tier 3 (High):** $\Delta \mathbf{p} = \pm 4.0\,\text{cm}$, $\Delta \theta = \pm 30.0^\circ$

### Benchmark Comparison Table

| Perturbation Tier | Pos. Jitter | Rot. Jitter | BC Baseline (MLP) | ACT (Transformer Chunking) | Diffusion Policy (DDPM) |
|---|---|---|---|---|---|
| Tier 0: Nominal | 0.0 cm | 0.0° | 40.0% (2/5) | 40.0% (2/5) | 0.0% (0/5) |
| Tier 1: Low Error | ±1.0 cm | ±5.0° | 60.0% (3/5) | 40.0% (2/5) | 0.0% (0/5) |
| Tier 2: Medium Error | ±2.5 cm | ±15.0° | 40.0% (2/5) | 20.0% (1/5) | 0.0% (0/5) |
| Tier 3: Stress Test | ±4.0 cm | ±30.0° | 40.0% (2/5) | 20.0% (1/5) | 0.0% (0/5) |
| **Overall Average** | — | — | **45.0%** | **30.0%** | **0.0%** |

### Empirical Insights

- **ACT demonstrates monotonic degradation.** ACT's success rate declines smoothly ($40\% \to 40\% \to 20\% \to 20\%$) rather than erratically. Because it outputs temporal chunks ($K=16$) conditioned on a CVAE latent prior, it avoids single-step reactive jitter and executes smoother motion profiles as perturbation increases.
- **BC shows compounding error.** BC achieves the highest score at Tier 1 (60%) thanks to fast, reactive single-step corrections, but its single-step formulation is prone to compounding error ($O(T^2)$ drift) once visual coordinates deviate significantly from the training trajectory envelope.
- **Diffusion Policy is sample-inefficient in this low-data regime.** It scored 0% across all tiers — learning conditional reverse denoising over continuous multi-step trajectories generally requires hundreds of demonstrations and 100+ epochs to resolve fine contact closure, well beyond the 50 demonstrations / 25 epochs used here.

---

## 7. Training Dynamics & Loss Analysis

Across the three policy architectures, distinct convergence characteristics were observed under identical 15D multimodal feature representations:

- **Behavioral Cloning (MLP baseline):** minimizes Mean Squared Error ($\mathcal{L}_{\text{MSE}}$) directly on end-effector delta poses. Loss converged rapidly from an initial $0.0336$ to $0.00117$ by epoch 20. Because the network is an unconstrained point-estimate mapping $\hat{\mathbf{a}}_t = \pi(\mathbf{z}_t)$, optimization proceeds without any latent adversarial or generative constraints.
- **Action Chunking Transformer (ACT):** trained with the composite loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{L1}} + \beta \mathcal{D}_{\text{KL}}$, stabilizing from an initial $2.278$ to $0.0174$ by epoch 25. The CVAE latent regularizer prevents the transformer decoder from memorizing deterministic chunk sequences, preserving a structured latent Gaussian space $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$.
- **Diffusion Policy (DDPM, 1D temporal ResNet):** optimizes the noise-estimation objective $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t^k, k, \mathbf{z}_t)\|^2$, leveling off from $0.719$ to $0.186$ across 25 epochs. Unlike the regression baselines, this loss reflects variance in score-matching across 20 intermediate Gaussian diffusion steps, not direct Cartesian error — which is part of why a low loss here doesn't translate into task success without enough demonstrations.

### Evaluation Under Systematically Varied Initial-Pose Error

- At **Tier 0** (nominal, 0 cm / 0°), both BC and ACT achieved 40% task success within 100 simulation steps.
- At **Tier 1** (±1 cm / ±5°), BC reached 60% success through fast single-step reactivity, while ACT held steady at 40%.
- At **Tiers 2–3** (up to ±4 cm / ±30°), ACT maintained stable trajectory execution, degrading predictably to 20%, while BC displayed high-frequency end-effector oscillations near contact boundaries.
- **Diffusion Policy scored 0% across all tiers.** It generated plausible gross reaching trajectories, but the 1D temporal denoiser did not learn the sub-centimeter contact closure required by the Franka gripper fingers under this low-data regime (50 demonstrations, 25 epochs) — highlighting its sample-efficiency limitations relative to direct chunk regression.

---

## 8. Future Work: Sim-to-Real Transfer

Deploying this vision-force imitation learning pipeline from MuJoCo simulation to a physical robotic workcell (e.g., a Franka Research 3 with a wrist-mounted load cell) is an open challenge. Four technical pillars are identified as necessary future work to bridge the simulation-to-reality gap.

### 8.1 Visual Domain Randomization (Perception Gap)

The simulation's offscreen renderer operates under ideal, deterministic lighting with static material shaders. Real-world transfer requires mitigating the resulting distribution shift in the ResNet-18 + Spatial Softmax encoder through comprehensive domain randomization:

- **Optical intrinsics & extrinsics jitter:** randomize camera pitch, roll, yaw ($\pm 5^\circ$) and translation ($\pm 3\,\text{cm}$) relative to the robot base, along with focal length and optical-center jitter, so the spatial softmax stays invariant to physical camera-mounting tolerances.
- **Photometric distortions:** apply randomized gamma/contrast ($[0.8, 1.2]$), Gaussian sensor noise, dynamic specular highlights, and shadow maps to simulate variable lab lighting.
- **Tabletop texture randomization:** dynamically map procedural Perlin noise and random photographic textures across the tabletop during trajectory logging, so keypoints don't latch onto background artifacts.

### 8.2 Wrist Force/Torque Sensor Noise & Dynamic Decoupling

In MuJoCo, the 6-DoF wrench ($\mathbf{F}_{\text{eef}}, \boldsymbol{\tau}_{\text{eef}}$) is computed algebraically from constraint-solver contact impulses, with no measurement latency. Real physical load cells (e.g., Robotiq FT 300, ATI Axia80) introduce complications that must be modeled:

- **High-frequency electrical noise & vibration:** raw strain-gauge readouts suffer from motor-induced electromagnetic interference (EMI) and structural arm vibration. A forward–backward zero-phase digital low-pass Butterworth filter (cutoff $f_c = 15\,\text{Hz}$) should be integrated into the state-estimation thread.
- **Inertial & gravitational decoupling:** physical end-effector force readouts include the gravitational and inertial forces of the gripper fingers themselves. An online dynamic-compensation algorithm must continuously subtract the gripper's gravitational wrench:

$$\mathbf{F}_{\text{contact}} = \mathbf{F}_{\text{raw}} - \mathbf{R}_{\text{sensor}}^{\text{base}} m_{\text{tool}} \mathbf{g} - m_{\text{tool}} \ddot{\mathbf{x}}_{\text{eef}}$$

ensuring the policy is conditioned strictly on genuine external contact forces.

### 8.3 Controller Latency & Action-Space Mapping

- **Frequency decoupling:** the simulated Operational Space Controller (OSC) runs at an idealized 20 Hz. Real Franka Emika Panda arms require low-level control commands dispatched via the `libfranka` real-time interface at **1,000 Hz (1 ms deadline)**.
- **Cubic Hermite spline interpolation:** feeding 20 Hz delta-Cartesian pose jumps directly would create jerky accelerations that trip physical joint torque limits. A real-time interpolation buffer must generate $C^2$-continuous position/velocity profiles connecting 20 Hz policy chunks to the 1 kHz Cartesian impedance controller.
- **Inference latency buffering:** Diffusion Policy requires 20 sequential denoising iterations (~35 ms on a mobile workstation GPU) — this exceeds the 50 ms (20 Hz) control period and would cause action starvation. Real-world deployment would require reducing diffusion steps via Denoising Diffusion Implicit Models (DDIM) or Consistency Models to reach sub-10 ms execution.

### 8.4 Eye-in-Hand Calibration & Hardware Safety Limits

- **Hand–eye calibration:** the extrinsic transform between the wrist camera frame $\{C\}$ and the gripper end-effector frame $\{E\}$ must be solved via the classical homogeneous equation:

$$\mathbf{A} \mathbf{X} = \mathbf{X} \mathbf{B}$$

using a precision-machined ChArUco or AprilTag calibration grid fixed to the workspace.

- **Hardware safety envelopes & reflex halting** — required to protect the robot, load cell, and workspace during exploratory policy rollouts:
  - **Virtual Cartesian fixtures:** software-enforced workspace bounding boxes that clamp commanded end-effector targets inside safe operational bounds.
  - **Force-threshold reflex:** an asynchronous 1 kHz hardware-level supervisor that triggers an immediate emergency stop if linear contact force exceeds 25 N or torque exceeds 2.5 N·m.

---

## 9. Key References

- Lee, M. A. et al., "Making Sense of Vision and Touch: Self-Supervised Representations for Contact-Rich Manipulation Tasks," *IEEE ICRA / IJRR*, 2019.
- Johannink, T. et al., "Residual Reinforcement Learning for Robot Control," *IEEE ICRA*, 2019.
- Zhao, T. et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), *Robotics: Science and Systems (RSS)*, 2023.
- Chi, C. et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," *Robotics: Science and Systems (RSS)*, 2023.
