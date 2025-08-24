# Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository is an extension for legged robot reinforcement learning based on Isaac Lab, which allows to develop in an isolated environment, outside of the core Isaac Lab repository. The RL algorithm is based on a [forked RSL-RL library](https://github.com/zitongbai/rsl_rl/tree/feature/amp). 

**Key Features:**

- `RL` Support vanilla RL for legged robots, including Unitree G1, Go2.
- `AMP` Adversarial Motion Priors (AMP) for humanoid robots, including Unitree G1. We suggest retargeting the human motion data by [GMR](https://github.com/YanjieZe/GMR).

## Demo

* Adversarial Motion Priors for Unitree G1:

https://github.com/user-attachments/assets/ed84a8a3-f349-44ac-9cfd-2baab2265a25

## News & Updates

- 2025/08/24: Support using more steps observations and motion data in AMP training.
- 2025/08/22: Compatible with Isaac Lab 2.2.0.
- 2025/08/21: Add support for retargeting human motion data by [GMR](https://github.com/YanjieZe/GMR).

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/zitongbai/legged_lab

# Option 2: SSH
git clone git@github.com:zitongbai/legged_lab.git
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/legged_lab
```

- Clone the forked RSL-RL library separately from the Isaac Lab installation and legged_lab repository (i.e. outside the `IsaacLab` and `legged_lab` directories):

```bash
# Option 1: HTTPS
git clone https://github.com/zitongbai/rsl_rl.git
# Option 2: SSH
git clone git@github.com:zitongbai/rsl_rl.git

cd rsl_rl
git checkout feature/amp
```

- Install the forked RSL-RL library:

```bash
# in the rsl_rl directory
python -m pip install -e .
```

- Download the usd model for Unitree G1:

```bash
# in legged lab directory
gdown "https://drive.google.com/drive/folders/1rlhZcurMenq4RGojZVUDl3a6Ja2hm-di?usp=drive_link" --folder -O ./source/legged_lab/legged_lab/data/Robots/
```

- Download the motion data for Unitree G1:

```bash
# in legged lab directory
gdown "https://drive.google.com/drive/folders/1tXtyjgM_wwqWNwnpn8ny5b4q1c-GxZkm?usp=sharing" --folder -O ./source/legged_lab/legged_lab/data/
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=LeggedLab-Isaac-Velocity-Rough-G1-v0 --headless
```

## Train

### RL

To train the reinforcement learning for Unitree G1, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task=LeggedLab-Isaac-Velocity-Rough-G1-v0 --headless
```

For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

### AMP

To train the AMP algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --max_iterations 10000
```

If you want to train it in a non-default gpu, you can pass more arguments to the command:

```bash
# replace `x` with the gpu id you want to use
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --max_iterations 10000 --device cuda:x agent.device=cuda:x
```

## Play

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-G1-Play-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

The video will be saved in the `logs/rsl_rl/experiment_name/run_name/videos/play` directory.

## TODOS

- [ ] Add more legged robots, such as Unitree H1
- [x] Self-contact penalty in AMP
- [x] Asymmetric Actor-Critic in AMP
- [ ] Symmetric Reward
- [ ] Sim2sim in mujoco

## Acknowledgement

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [GMR](https://github.com/YanjieZe/GMR)

