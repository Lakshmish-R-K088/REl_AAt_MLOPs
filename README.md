# 🛸 Autonomous Search & Rescue (SAR) Drone Project — `REl_AAt_MLOPs`

An enterprise-grade, containerized MLOps pipeline implementing **Reinforcement Learning (RL)** for an autonomous **Search and Rescue (SAR)** drone mission.

The project models an advanced **Partially Observable Markov Decision Process (POMDP)** in which an autonomous drone navigates a dynamically changing environment to locate survivors under strict battery constraints.

---

# 🏗️ System Architecture

```plaintext
REl_AAt_MLOPs/
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml     # Automated GitHub Actions CI/CD Pipeline
│
├── sim/
│   ├── __init__.py
│   └── visual_env.py              # 20x20 SAR POMDP Gym Environment
│
├── models/
│   └── policy_sar_ppo.zip         # Saved trained PPO model artifact
│
├── experiments/
│   └── logs/                      # TensorBoard runs & evaluation CSV logs
│
├── Dockerfile                     # Containerized execution environment
├── requirements.txt               # Unified project dependencies
├── train.py                       # PPO training + MLflow/TensorBoard integration
├── test_policy.py                 # Human-in-the-loop Pygame evaluation
└── ci_sanity_check.py             # Headless structural smoke test
```

---

# 🧠 Environment Formulation — POMDP

The environment is modeled as a **Partially Observable Markov Decision Process (POMDP)**.

Unlike fully observable environments, the drone does **not** have access to the complete map initially. Instead, it operates with a limited sensor-based field of view, simulating real-world autonomous deployment conditions.

---

# 📡 Observation Space (State Space)

The PPO agent uses a `MultiInputPolicy` dictionary that emulates real drone telemetry streams.

## Components

### `drone_pos`

A 2D coordinate vector representing the drone’s current GPS position:

[
[X, Y]
]

---

### `battery`

A scalar value representing remaining operational energy capacity.

Initial battery level:

[
200
]

---

### `explored_map`

A memory-aware:

[
20 \times 20
]

grid storing localized environmental knowledge.

## Grid Encoding

| Value | Meaning                               |
| ----- | ------------------------------------- |
| `-1`  | Unexplored region (Fog of War)        |
| `0`   | Safe navigable path                   |
| `1`   | Wall / obstacle / impassable boundary |
| `2`   | Survivor location                     |

---

# ⚖️ Reward Engineering

The reward system is carefully balanced to encourage efficient exploration while penalizing unsafe or wasteful behavior.

| Parameter      | Value           | Purpose                                  |
| -------------- | --------------- | ---------------------------------------- |
| Hustle Penalty | `-0.5` per step | Prevents inefficient wandering           |
| Fatal Penalty  | `-50.0`         | Punishes collisions or battery depletion |
| Rescue Reward  | `+50.0`         | Reward for rescuing a survivor           |
| Mission Bonus  | `+100.0`        | Bonus for rescuing all 5 survivors       |

---

# ⚙️ PPO Hyperparameter Registry

The environment uses **Proximal Policy Optimization (PPO)** with custom tuning for sparse-reward, partially observable exploration.

| Hyperparameter                   | Value       | Purpose                      |
| -------------------------------- | ----------- | ---------------------------- |
| Total Timesteps                  | `2,000,000` | Long-horizon policy learning |
| Learning Rate                    | `0.0003`    | Stabilized convergence       |
| Entropy Coefficient (`ent_coef`) | `0.05`      | Encourages exploration       |
| Batch Size                       | `128`       | Stable gradient estimation   |

---

# 🐳 Containerized Execution (Docker)

The entire system is fully containerized to ensure deployment parity across machines and cloud runtimes.

## Build the Docker Image

```bash
docker build -t sar-drone-env:latest .
```

## Launch the Container

```bash
docker run -it \
  -p 5000:5000 \
  -p 6006:6006 \
  -v "$(pwd):/workspace" \
  sar-drone-env:latest
```

---

# 🏋️ Training the PPO Agent

Inside the running container:

```bash
python train.py
```

## Training Pipeline Features

* PPO-based reinforcement learning
* TensorBoard metric streaming
* MLflow experiment tracking
* Automated log persistence
* Model artifact generation

Generated outputs include:

* `progress.csv`
* TensorBoard event files
* MLflow metadata
* Trained `.zip` policy binaries

---

# 📊 Experiment Tracking & Observability

## Launch TensorBoard

```bash
tensorboard \
  --logdir=./experiments/logs/tensorboard/ \
  --host 0.0.0.0 \
  --port 6006
```

Access:

```plaintext
http://localhost:6006
```

### Monitor Metrics Such As

* `ep_rew_mean`
* `entropy_loss`
* `policy_gradient_loss`
* rollout statistics

---

## Launch MLflow UI

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000
```

Access:

```plaintext
http://localhost:5000
```

### MLflow Tracks

* Experiment runs
* Hyperparameters
* Reward curves
* Model artifacts
* Performance comparisons

---

# 🎮 Human-in-the-Loop Evaluation

To visually inspect exploration behavior using Pygame rendering:

```bash
python test_policy.py
```

This launches a real-time simulation showing:

* Drone movement
* Fog-of-war exploration
* Survivor detection
* Obstacle avoidance
* Battery consumption behavior

---

# 🚀 CI/CD Pipeline — GitHub Actions

The project includes a fully automated GitHub Actions workflow:

```plaintext
.github/workflows/mlops_pipeline.yml
```

---

# 🔄 Pipeline Flow

```plaintext
[ Push / Pull Request ]
          │
          ▼
┌─────────────────────────────────┐
│ 1. Continuous Integration (CI)  │
│                                 │
│ • Spins up clean Ubuntu runner  │
│ • Installs dependencies         │
│ • Validates Gymnasium API       │
│ • Executes smoke tests          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Continuous Deployment (CD)   │
│                                 │
│ • Builds Docker container       │
│ • Verifies deployment parity    │
│ • Detects broken system configs │
│ • Confirms environment integrity│
└─────────────────────────────────┘
```

---

# ✅ Continuous Integration (CI)

The CI stage performs:

* Dependency installation
* Syntax validation
* Environment bootstrapping
* Automated smoke testing using:

```bash
python ci_sanity_check.py
```

Purpose:

* Validate Gymnasium compliance
* Catch shape mismatches
* Detect broken observation spaces
* Avoid expensive failed training runs

---

# 🚢 Continuous Deployment (CD)

The CD stage validates the root-level Docker environment.

It ensures:

* Dependency compatibility
* Container reproducibility
* System isolation integrity
* Cloud deployment readiness

This prevents broken containers from reaching production or collaborative environments.

---

# 🌍 Core Research Focus

This project explores the intersection of:

* Reinforcement Learning
* Autonomous Navigation
* POMDP-based decision systems
* Sparse reward optimization
* Fog-of-war exploration
* MLOps infrastructure
* Containerized AI deployment

---

# 🔬 Key Technical Highlights

✅ Partial observability environment
✅ Sparse-reward PPO optimization
✅ Dynamic exploration memory mapping
✅ TensorBoard + MLflow integration
✅ Dockerized reproducibility
✅ GitHub Actions CI/CD automation
✅ Human-in-the-loop visualization
✅ Enterprise-style MLOps workflow
✅ Structured experiment tracking
✅ Scalable autonomous SAR simulation
