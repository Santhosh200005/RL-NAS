# RL-NAS

### Reinforcement Learning–Driven Neural Architecture Search

**An experimental Neural Architecture Search (NAS) framework that uses Reinforcement Learning to discover efficient CNN architectures while balancing predictive performance and computational complexity.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

---

## Overview

RL-NAS is a research-oriented Neural Architecture Search platform designed to automate the discovery of Convolutional Neural Network (CNN) architectures.

Instead of manually designing neural networks, the system explores different architecture configurations using multiple search strategies.

The primary objective is to investigate whether Reinforcement Learning can discover architectures that achieve a competitive balance between:

* Validation accuracy
* Number of model parameters
* Floating-point operations (FLOPs)
* Computational efficiency

The framework implements and compares four search strategies:

1. Random Search
2. Bayesian Optimization
3. Reinforcement Learning (RL)
4. Reinforcement Learning with Accuracy-Only Reward (RL-Acc)

An interactive Next.js dashboard, powered by a FastAPI backend, provides experiment execution, live progress monitoring, architecture inspection, and visualization of search results.

---

## Research Objectives

* Develop an automated CNN architecture search framework.
* Implement an RL-based architecture generation controller.
* Apply policy-gradient learning to optimize architecture selection.
* Design a multi-objective reward function balancing accuracy and model complexity.
* Compare RL against Random Search and Bayesian Optimization.
* Investigate the impact of reward design through an accuracy-only RL ablation.
* Visualize architecture performance and optimization progress through an interactive dashboard.

---

## System Architecture

```text
                  RL-NAS PLATFORM
                         |
          +--------------+--------------+
          |                             |
    SEARCH ENGINE                 WEB DASHBOARD
          |                             |
    +-----+------+                Next.js 14
    |     |      |                     |
  Random  BO    RL / RL-Acc       React Components
                 |                    |
          RL Controller          REST API Client
                 |                    |
          Architecture             FastAPI
             Generator                 |
                 |                 Experiment
          CNN Evaluation           Management
                 |                    |
          Reward Calculation    Results & Metrics
                 |
          Search History
                 |
          Analysis & Pareto
             Optimization
```

### Main Components

| Component         | Responsibility                                             |
| ----------------- | ---------------------------------------------------------- |
| Search Engine     | Executes architecture search algorithms                    |
| RL Controller     | Generates CNN architectures using policy-gradient learning |
| Search Space      | Defines valid neural architecture configurations           |
| Evaluator         | Trains and evaluates candidate architectures               |
| Reward Function   | Balances accuracy and computational complexity             |
| Analysis Module   | Generates summaries, learning curves and Pareto fronts     |
| FastAPI Backend   | Provides REST endpoints for experiment management          |
| Next.js Dashboard | Provides an interactive experiment interface               |

---

## Search Algorithms

### 1. Random Search

Randomly samples architectures from the defined search space.

It serves as a baseline for evaluating the effectiveness of more sophisticated optimization strategies.

### 2. Bayesian Optimization

Uses a Gaussian Process surrogate model with a Matérn kernel and Expected Improvement acquisition strategy.

It uses previously evaluated architectures to guide the selection of promising candidates.

### 3. Reinforcement Learning

Uses a neural controller to generate candidate architectures.

The controller is optimized using a REINFORCE-style policy-gradient approach with:

* Policy-gradient optimization
* Moving-average reward baseline
* Entropy regularization
* Adam optimizer
* Gradient clipping

The controller learns to generate architectures that maximize the search reward.

### 4. RL-Acc: Accuracy-Only Ablation

Uses the same RL search approach but focuses on validation accuracy rather than the multi-objective reward.

This provides an ablation experiment for studying the effect of incorporating model-complexity penalties into the reward function.

---

## Multi-Objective Reward Function

The main RL search strategy uses a reward function that considers validation accuracy and model complexity.

Conceptually:

$$
R = Accuracy -
\lambda_p \left(\frac{Parameters}{P_{ref}}\right)
-
\lambda_f \left(\frac{FLOPs}{F_{ref}}\right)
-
\lambda_l \left(\frac{Latency}{L_{ref}}\right)
$$

Where:

* \(Accuracy\): Validation accuracy.
* \(Parameters\): Number of trainable model parameters.
* \(FLOPs\): Computational complexity.
* \(Latency\): Inference latency.
* \(\lambda_p,\lambda_f,\lambda_l\): Penalty coefficients.
* \(P_{ref},F_{ref},L_{ref}\): Reference values.

The latency penalty is disabled in the current default configuration.

The objective is to investigate architectures that offer a useful trade-off between predictive performance and computational cost.

---

## Experimental Configuration

The primary experiments use the CIFAR-10 dataset.

| Parameter                     | Configuration             |
| ----------------------------- | ------------------------- |
| Dataset                       | CIFAR-10                  |
| Training Samples              | 10,000                    |
| Validation Samples            | 5,000                     |
| Training Epochs per Candidate | 3                         |
| Batch Size                    | 128                       |
| Learning Rate                 | 0.003                     |
| Search Budget                 | 100                       |
| Random Seeds                  | 0, 1, 2                   |
| Search Algorithms             | Random, Bayes, RL, RL-Acc |
| RL Controller Hidden Size     | 64                        |
| RL Controller Learning Rate   | 0.005                     |
| RL Batch Size                 | 4                         |
| Entropy Coefficient           | 0.01                      |

### Experiment Scale

The full experiment evaluates:

* 4 search strategies
* 3 independent random seeds
* 100 architecture evaluations per strategy and seed

**Total: 1,200 candidate architecture evaluations.**

---

## Technology Stack

### Machine Learning and Research

* Python
* PyTorch
* NumPy
* Pandas
* SciPy
* Scikit-learn
* PyYAML

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Next.js 14
* React 18
* TypeScript
* CSS

### Development and Infrastructure

* Git
* GitHub Actions
* Docker
* PowerShell

---

## Project Structure

```text
RL-NAS/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
│
├── experiments/
│   ├── __init__.py
│   ├── run.py
│   ├── analyze.py
│   └── final_eval.py
│
├── frontend/
│   ├── app/
│   │   ├── api/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   │
│   ├── components/
│   │   ├── ArchitectureCard.tsx
│   │   ├── ExperimentForm.tsx
│   │   ├── LineChart.tsx
│   │   ├── MetricCard.tsx
│   │   └── ParetoChart.tsx
│   │
│   ├── lib/
│   │   └── api.ts
│   │
│   ├── package.json
│   └── README.md
│
├── rlnas/
│   ├── analysis.py
│   ├── controller.py
│   ├── data.py
│   ├── evaluators.py
│   ├── models.py
│   ├── reward.py
│   ├── search.py
│   ├── search_space.py
│   └── trainer.py
│
├── tests/
│   └── test_core.py
│
├── results/
│   └── .gitkeep
│
├── Dockerfile
├── Dockerfile.worker
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

* Python 3.11
* Node.js and npm
* Git
* NVIDIA GPU with compatible CUDA-enabled PyTorch installation (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/Santhosh200005/RL-NAS.git
cd RL-NAS
```

### 2. Create a Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your CUDA environment if required.

### 4. Run a Search Experiment

The default configuration:

```powershell
python -m experiments.run --config configs\default.yaml --methods random bayes rl rl_acc --seeds 0 1 2 --budget 100 --out results\full
```

For a smaller experiment, use the smoke configuration:

```powershell
python -m experiments.run --config configs\smoke.yaml
```

### 5. Analyze Results

```powershell
python -m experiments.analyze results\full
```

The analysis module generates:

* Per-seed summary CSV
* Aggregated method summary CSV
* Validation accuracy learning curves
* Reward learning curves
* Pareto-front visualization

Generated experiment outputs are stored locally in the `results/` directory.

---

## Interactive Dashboard

RL-NAS includes a web-based dashboard for interacting with the search framework.

### Dashboard Features

* Start architecture search experiments.
* Select search algorithms.
* Configure search budgets and random seeds.
* Monitor experiment progress.
* View validation accuracy and reward curves.
* Inspect architecture configurations.
* Explore search history.
* Visualize Pareto-optimal architectures.

### Start the Backend

From the project root:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Backend health endpoint:

```text
http://localhost:8000/health
```

### Configure the Frontend

Open another terminal:

```powershell
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
RLNAS_API_URL=http://localhost:8000
```

An API key is optional for local development. If authentication is enabled on the backend, configure the same `RLNAS_API_KEY` value in the backend environment and frontend environment.

### Start the Dashboard

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

If port 3000 is already occupied, Next.js may use another available port.

---

## Evaluation Metrics

The framework tracks several metrics to compare search strategies.

| Metric                   | Purpose                                     |
| ------------------------ | ------------------------------------------- |
| Best Validation Accuracy | Highest validation accuracy discovered      |
| Best Reward              | Best multi-objective reward                 |
| Model Parameters         | Model size and complexity                   |
| FLOPs                    | Computational cost                          |
| Unique Architectures     | Number of distinct architectures explored   |
| Search Evaluation Time   | Recorded candidate evaluation/training time |

The analysis module also generates Pareto-front visualizations to examine the trade-off between model complexity and validation accuracy.

---

## Reproducibility

Experiments support multiple random seeds to reduce dependence on a single search run.

The experiment configuration is maintained in YAML files.

Results are saved as JSONL records, allowing individual architecture evaluations to be analyzed later.

The repository excludes generated datasets, experiment outputs, Python caches, frontend dependencies, and environment files from version control.

---

## Current Limitations

* The experiments use a limited CIFAR-10 training and validation subset.
* Each candidate receives a relatively small training budget.
* Search performance may vary with random seed.
* Latency optimization is not enabled in the default reward configuration.
* The dashboard currently maintains experiment state in backend memory.
* The current experimental results should not be interpreted as a universal ranking of NAS algorithms.

---

## Future Improvements

* Expand experiments to larger datasets.
* Investigate additional search-space configurations.
* Introduce hardware-aware latency optimization.
* Explore more advanced RL controllers.
* Improve experiment persistence and checkpoint recovery.
* Add additional NAS baselines.
* Perform statistical significance testing.
* Evaluate discovered architectures with longer training schedules.
* Investigate multi-objective evolutionary optimization.

---

## Author

**Nenavath Santhosh**

B.Tech – Computer Science and Engineering

GitHub: [Santhosh200005](https://github.com/Santhosh200005)

LinkedIn: [Nenavath Santhosh](https://www.linkedin.com/in/santhosh-nenavath-35895a376)

---

## Acknowledgements

This project is an independent research-oriented implementation exploring Reinforcement Learning, Neural Architecture Search, and multi-objective deep learning optimization.

---

**RL-NAS — Automating Neural Architecture Discovery Through Reinforcement Learning.**
