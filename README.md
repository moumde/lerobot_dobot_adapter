# 代码部署
项目文件夹置于lerobot/src/lerobot/robots/下

# 训练
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

# ros环境启动
注意ros2 humble环境为3.10，而lerobot需要的最低python版本为3.12，nrc相关python接口仅支持3.8版本
故ros相关操作要在非lerobot conda环境下进行

## 启动ros服务节点
```bash
cd ~/code/ros2_ws/
source ~/code/ros2_ws/install/setup.bash 
ros2 launch nrc_interface_ros2 nrc_driver.launch.py
```

## 启动ros监听客户端
```bash
source ~/code/ros2_ws/install/setup.bash 
python3 src/lerobot/robots/dobot_cr5_o6/cr5_ros_node.py
```


# lerobot环境下正常推理部署流程
## 启动虚拟环境
```bash
conda activate lerobot
```

## 将灵巧手的sdk路径添加至python环境
```bash
export PYTHONPATH=/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH
```

## 测试手部和相机功能正常
```bash
python3 src/lerobot/robots/dobot_cr5_o6/test_camera_hand.py
```

## 启动ACT推理
```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_cr3_act.py
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot.py
```

## 启动PI05推理
```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_cr3_pi05.py
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_pi05.py
```


# openpi兼容部署
## 启动openpi虚拟环境
```bash
source /home/je/code/openpi/.venv/bin/activate
```

## 导入环境变量
```bash
export PYTHONPATH=/home/je/code/lerobot/src:/home/je/code/openpi/src:/home/je/code/openpi/packages/openpi-client/src:/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH
```

## 启动推理
```bash
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_openpi.py
```
