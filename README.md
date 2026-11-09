# LeRobot Dobot CR5/O6 Adapter

This repository contains my Dobot CR5/O6 integration work for LeRobot: an ACT and Pi05 robot workflow, a ROS 2 bridge, and an OpenPI-compatible deployment path.

For general LeRobot setup and framework documentation, see the [upstream LeRobot repository](https://github.com/huggingface/lerobot) and [official docs](https://huggingface.co/docs/lerobot). For native OpenPI framework documentation, see [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi). This README covers only the project-specific adapter and workflows.

## Project-specific components

- Robot integration and interfaces: [`src/lerobot/robots/dobot_cr5_o6`](src/lerobot/robots/dobot_cr5_o6)
- ACT deployment: `deploy_dobot_cr3_act.py` and `deploy_dobot.py`
- Pi05 deployment: `deploy_dobot_cr3_pi05.py` and `deploy_dobot_pi05.py`
- OpenPI-compatible deployment: `deploy_dobot_openpi.py`
- ROS 2 bridge: `cr5_ros_node.py`

## ACT pick-and-place demonstration

[![ACT pick-and-place demonstration](%E8%A7%86%E9%A2%91/covers/act_pick_and_place.jpg)](https://moumde.github.io/lerobot_dobot_adapter/#act-demo)

This video shows the ACT policy recognizing the motor and completing the pick-and-place task.

## Train ACT

Run this from the LeRobot environment and replace dataset paths as needed:

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=cr5_o6_20250917 \
    --dataset.root=/home/je/code/lerobot/dataset/cr5_o6_20250917 \
    --output_dir=/home/je/code/lerobot/dataset/cr5_o6_20250917_act_result \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_lr_backbone=5e-6 \
    --steps=100000 \
    --batch_size=32 \
    --save_freq=10000
```

## ROS 2 interface

ROS 2 Humble uses Python 3.10, while the LeRobot environment requires Python 3.12 or newer. The NRC Python interfaces used here require Python 3.8. Run ROS-related processes outside the LeRobot Conda environment.

Start the ROS service node:

```bash
cd ~/code/ros2_ws/
source ~/code/ros2_ws/install/setup.bash
ros2 launch nrc_interface_ros2 nrc_driver.launch.py
```

Start the LeRobot-side ROS client in a separate terminal:

```bash
source ~/code/ros2_ws/install/setup.bash
python3 src/lerobot/robots/dobot_cr5_o6/cr5_ros_node.py
```

## Run ACT and Pi05 on the robot

Activate the LeRobot environment and add the LinkerHand SDK to `PYTHONPATH`:

```bash
conda activate lerobot
export PYTHONPATH=/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH
```

Check the hand and cameras:

```bash
python3 src/lerobot/robots/dobot_cr5_o6/test_camera_hand.py
```

Run ACT inference:

```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_cr3_act.py
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot.py
```

Run Pi05 inference:

```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_cr3_pi05.py
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_pi05.py
```

## OpenPI-compatible deployment

Activate the OpenPI environment:

```bash
source /home/je/code/openpi/.venv/bin/activate
```

Set the project source paths, updating them if the repositories are in different locations:

```bash
export PYTHONPATH=/home/je/code/lerobot/src:/home/je/code/openpi/src:/home/je/code/openpi/packages/openpi-client/src:/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH
```

Start deployment:

```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_openpi.py
```

## Upstream frameworks

- [LeRobot repository](https://github.com/huggingface/lerobot) | [LeRobot documentation](https://huggingface.co/docs/lerobot)
- [OpenPI repository](https://github.com/Physical-Intelligence/openpi)