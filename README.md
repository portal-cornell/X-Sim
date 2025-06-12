<div align="center">
  <h1><b> X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real </b></h1>
</div>

<div align="center">

### [Prithwish Dan](https://pdan101.github.io/)<sup>\*</sup>, [Kushal Kedia](https://kushal2000.github.io/)<sup>\*</sup>, Angela Chao, Edward W. Duan, Maximus A. Pace, [Wei-Chiu Ma](https://www.cs.cornell.edu/~weichiu/), [Sanjiban Choudhury](https://sanjibanc.github.io/)

<sup>*</sup> Equal Contribution<br/>
Cornell University

[![arXiv](https://img.shields.io/badge/arXiv-2505.07096-b31b1b.svg)](https://arxiv.org/abs/2505.07096)
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://portal-cornell.github.io/X-Sim/)

</div>



---
<div align="center">
  <video width="800" controls>
    <source src="docs/overview.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Pipeline Overview](#-pipeline-overview)
- [Quick Start](#-quick-start)
- [Available Tasks](#-available-tasks)
- [Detailed Usage](#-detailed-usage)
- [Citation](#-citation)

## 📁 Project Structure

```
X-Sim/
├── simulation/
│   ├── ManiSkill/                    # ManiSkill simulation environment
│   ├── scripts/                     # RL training and data generation scripts
├── diffusion_policy/
│   ├── scripts/                    # Diffusion policy training and evaluation
│   ├── cfgs/                      # Configuration files
│   ├── utils/                     # Shared utilities for diffusion policy
├── run_pipeline.py               # Automated pipeline execution script
├── setup.sh                      # Installation script
└── README.md                     # Project documentation
```

## 🛠 Installation

### Environment Setup
```bash
bash setup.sh
# Create conda env & install packages & download assets
```

## 🔄 Pipeline Overview

X-Sim's pipeline consists of three main phases:

### Phase 1: Real-to-Sim
- **Real-to-Sim** *(Code Coming Soon)*: Construct photorealistic simulation and track object poses from human videos

### Phase 2: RL Training in Sim
- **RL Training**: Learn robot policies with object-centric rewards

### Phase 3: Sim-to-Real
- **Synthetic Data Collection**: Generate RGB demonstration trajectories using trained state-based policies
- **Diffusion Policy Training**: Train image-conditioned policies on synthetic data
- **Auto-Calibration**:
  - **Auto-Calibration Data**: Deploy policy on real robot and obtain paired sim rollouts
  - **Training with Auxiliary Loss**: Fine-tune with calibration auxiliary loss

## 🚀 Quick Start

### Full Pipeline

Run the complete X-Sim pipeline for any task with a single command:

```bash
python run_pipeline.py --env_id "Mustard-Place"
```

**What this does:**
1. **RL Training**: Trains policies with object-centric rewards
2. **Synthetic Data Generation**: Collects demonstration trajectories
3. **Image-Conditioned Diffusion Policy**: Trains on synthetic data
4. **Auto-Calibration Data**: Converts real trajectories into corresponding sim trajectories (Requires real robot deployment ⚠️)
5. **Calibrated Training**: Trains with auxiliary loss using paired real-to-sim data

**Output:** All results saved to `experiments/pipeline/<task_name>/`

## 🎯 Available Tasks

X-Sim supports the following manipulation tasks:

| Task Name | Environment ID | Description |
|-----------|----------------|-------------|
| Mustard Place | `Mustard-Place` | Place mustard on left side of kitchen |
| Corn in Basket | `Corn-in-Basket` | Place corn into basket |
| Letter Arrange | `Letter-Arrange` | Arrange letters next to each other |
| Shoe on Rack | `Shoe-on-Rack` | Place shoe onto shoe rack |
| Mug Insert | `Mug-Insert` | Insert mug into holder |

---
To add your own tasks, refer to files in ```simulation/ManiSkill/mani_skill/envs/tasks/xsim_envs```

## 📖 Detailed Usage

### Step 1: RL Training

Train reinforcement learning policies with object-centric rewards:

```bash
cd simulation
python -m scripts.rl_training \
    --env_id="<TASK_NAME>" \
    --exp-name="<EXPERIMENT_NAME>" \
    --num_envs=1024 \
    --seed=0 \
    --total_timesteps=<TIMESTEPS> \
    --num_steps=<STEPS> \
    --num_eval_steps=<EVAL_STEPS>
```

### Step 2: Synthetic Data Collection

Generate demonstration trajectories using the trained RL policies:

```bash
cd simulation
python -m scripts.data_generation_rgb \
    --evaluate \
    --num_trajectories=<NUM_TRAJ> \
    --trajectory_length=<TRAJ_LENGTH> \
    --randomize_init_config \
    --checkpoint="<PATH_TO_RL_CHECKPOINT>" \
    --env_id="<TASK_NAME>-Eval" \
    --randomize_camera
```

### Step 3: Image-Conditioned Diffusion Policy Training

Train diffusion policies on the synthetic demonstration data:

```bash
cd diffusion_policy
python -m scripts.dp_training_rgb \
    --config_path=cfgs/sim2real.yaml \
    --dp.use_aux_loss=0 \
    --save_dir=<SAVE_DIRECTORY> \
    --dataset.paths=["<PATH_TO_SYNTHETIC_DATA>"] \
    --eval.env_id="<TASK_NAME>-Eval" \
    --eval_freq=5 \
    --eval.num_episodes=10 \
    --num_epoch=60 \
    --epoch_len=10000
```

### Step 4: Auto-Calibration Data Generation
Create real-sim paired RGB dataset using real rollout data and replaying it in sim:

```bash
cd diffusion_policy
python -m scripts.auto_calibration \
    --input_dir="<PATH_TO_REAL_ROLLOUTS>" \
    --env_id="<TASK_NAME>-Eval"
```

*Note*: You should adapt ```diffusion_policy/scripts/eval_dp.py``` to your robot hardware for real-world deployment.


### Step 5: Calibrated Policy - Training with Auxiliary Loss

Fine-tune the policy with calibration auxiliary loss:

```bash
cd diffusion_policy
python -m scripts.dp_training_rgb \
    --config_path=cfgs/sim2real.yaml \
    --dp.use_aux_loss=1 \
    --dp.aux_loss_weight=0.1 \
    --dp.distance_type="contrastive_cosine" \
    --save_dir=<SAVE_DIRECTORY> \
    --dataset.paths=["<PATH_TO_SYNTHETIC_DATA>"] \
    --dataset.real_pairing="<PATH_TO_REAL_DATA>" \
    --dataset.sim_pairing="<PATH_TO_SIM_PAIRING>" \
    --eval.env_id="<TASK_NAME>-Eval" \
    --eval_freq=5 \
    --eval.num_episodes=10 \
    --epoch_len=10000 \
    --num_epoch=60
```

### Evaluation

Evaluate trained diffusion policies:

```bash
cd diffusion_policy
python -m scripts.eval_dp \
    --checkpoint_path="<PATH_TO_DP_CHECKPOINT>" \
    --env_id="<TASK_NAME>-Eval" \
    --save-videos
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{dan2025xsim,
    title={X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real}, 
    author={Prithwish Dan and Kushal Kedia and Angela Chao and Edward Weiyi Duan and Maximus Adrian Pace and Wei-Chiu Ma and Sanjiban Choudhury},
    year={2025},
    eprint={2505.07096},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2505.07096}
}
```

---

<div align="center">
  <p>For more information, visit our <a href="https://portal-cornell.github.io/X-Sim/">project page</a>.</p>
</div>