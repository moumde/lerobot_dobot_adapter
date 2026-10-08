# 代码部署
先拉取lerobot代码框架
将本文件夹置于lerobot/src/lerobot/robots/下


# ros环境启动
注意ros2 humble环境为3.10，而lerobot需要的最低python版本为3.12，nrc相关python接口仅支持3.8版本
故ros相关操作要在非lerobot conda环境下进行

## 启动ros服务节点
cd ~/code/ros2_ws/
source ~/code/ros2_ws/install/setup.bash 
ros2 launch nrc_interface_ros2 nrc_driver.launch.py

## 启动ros监听客户端
source ~/code/ros2_ws/install/setup.bash 
python3 src/lerobot/robots/dobot_cr5_o6/cr5_ros_node.py 


# lerobot环境下正常推理部署流程
## 启动虚拟环境
conda activate lerobot

## 将灵巧手的sdk路径添加至python环境
export PYTHONPATH=/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH

## 测试手部和相机功能正常
python3 src/lerobot/robots/dobot_cr5_o6/test_camera_hand.py

## 启动ACT推理
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot.py

## 启动PI05推理
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_pi05.py



# openpi兼容部署
## 启动openpi虚拟环境
source /home/je/code/openpi/.venv/bin/activate

## 导入环境变量
export PYTHONPATH=/home/je/code/lerobot/src:/home/je/code/openpi/src:/home/je/code/openpi/packages/openpi-client/src:/home/je/code/linkerhand-python-sdk/LinkerHand:$PYTHONPATH

## 启动推理
python3 src/lerobot/robots/dobot_cr5_o6/deploy_dobot_openpi.py