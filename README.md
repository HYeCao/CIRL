# CIRL: Towards Efficient Real-World Human-in-the-Loop Reinforcement Learning with Causal Influence

![](./docs/images/task_banner.gif)

CIRL (Causal-Influence Reinforcement Learning) provides a set of libraries, environment wrappers, and examples to train Reinforcement Learning policies using a combination of demonstrations, causal masking intervention, and human corrections to perform robotic manipulation tasks with near-perfect success rates.

**Table of Contents**
- [CIRL: Causal-Influence Reinforcement Learning](#cirl-causal-influence-reinforcement-learning)
  - [Installation](#installation)
  - [Overview and Code Structure](#overview-and-code-structure)
  - [Running the CIRL Pick and Place Task](#running-the-cirl-pick-and-place-task)
  - [Create a New Task](#create-a-new-task)
  - [Citation](#citation)

## Installation
1. **Setup Conda Environment:**
    Create an environment with:
    ```bash
    conda create -n cirl python=3.10
    conda activate cirl
    ```

2. **Install Jax as follows:**
    - For CPU (not recommended):
        ```bash
        pip install --upgrade "jax[cpu]"
        ```

    - For GPU:
        ```bash
        pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    - For TPU:
        ```bash
        pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ```
    - See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

3. **Install the Launcher and Robot Infra Packages:**
    - Install `serl_launcher`:
        ```bash
        cd serl_launcher
        pip install -e .
        pip install -r requirements.txt
        cd ..
        ```
    - Install `serl_robot_infra`:
        ```bash
        cd serl_robot_infra
        pip install -e .
        cd ..
        ```

## Overview and Code Structure

CIRL provides a set of common libraries for users to train RL policies for robotic manipulation tasks. The main structure of running the RL experiments involves having an actor node and a learner node, both of which interact with the robot gym environment. Both nodes run asynchronously, with data being sent from the actor to the learner node via the network using [agentlace](https://github.com/youliangtan/agentlace). The learner will periodically synchronize the policy with the actor. This design provides flexibility for parallel training and inference.

## Running the CIRL Pick and Place Task

This tutorial guides you through training and evaluating the Pick and Place Banana task using CIRL.

### 1. Reset Spacemouse and Verify Robot Control
Before collecting data or training, test the connection and spacemouse inputs:
```bash
conda activate cirl
python examples/test_reset_spacemouse.py \
    --exp_name pick_place_banana \
    --steps 200 \
    --print_pose_every 5
```

### 2. Interactive Camera Perspective Cropping
Define the crop region of interest (ROI) for the robot cameras:
```bash
export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

python serl_robot_infra/ur_env/camera/interactive_crop.py \
    --experiment pick_place_banana \
    --output /tmp/ur7e_camera_crops.json
```

### 3. Demonstration Data Collection
Collect human demonstration trajectories via the space mouse or interface:
```bash
cd examples/experiments/pick_place_banana

export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

python ../../record_demos.py \
    --exp_name pick_place_banana \
    --successes_needed 10
```

### 4. Policy Training (New Run)
Start policy training with your collected demo dataset and causal-guided intervention enabled:
```bash
cd examples/experiments/pick_place_banana

# Initialize new run variables
export NEW_RUN=1
export RUN_ID=banana_$(date +%Y%m%d_%H%M%S)
export DEMO_PATH=/path/to/pick_place_banana_demos.pkl

export WANDB_API_KEY=your_wandb_api_key
export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

./run_tmux.sh
```

### 5. Resuming Policy Training
To resume training from an existing checkpoint:
```bash
cd examples/experiments/pick_place_banana

unset NEW_RUN
unset CHECKPOINT_PATH

export RUN_ID=banana_20260516_114232
export RESUME_TRAINING=1
export DEMO_PATH=/path/to/pick_place_banana_demos.pkl

export WANDB_API_KEY=your_wandb_api_key
export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

./run_tmux.sh
```

### 6. Policy Inference and Evaluation
To evaluate the trained policy on the physical robot:
```bash
cd examples/experiments/pick_place_banana

export RUN_ID=banana_your_run_id
export EVAL_CHECKPOINT_STEP=10000
export EVAL_N_TRAJS=10

export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

./run_eval.sh
```

## Create a new task
1. **Create a new folder under examples/experiments, with the files:**
    - config.py
    - wrappers.py
    - run_actor.sh
    - run_learner.sh
2. **The actor and learner scripts can just be copy-pasted with the respective directory and task names.**
3. **The config should specify hyperparameters for the environment, inheriting the default environment config.**
4. **Define a class inheriting the training config that contains the `create_env` method. This method should initialize the robot environment and set up all required wrappers.**
5. **The wrappers should include a task-specific environment class inheriting the base robot environment. It also incorporates gymnasium wrappers, such as reward classifier wrappers, rotation representation wrappers, and/or gripper penalty wrappers.**
6. **Register the new task in mappings.py under examples/experiments.**
7. **To register a new robot, create a new gymnasium environment and a controller interface to send control actions to the physical robot.**

