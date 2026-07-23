# CIRL Robot Infra (serl_robot_infra)
![](../docs/images/robot_infra_interfaces.png)

This package contains the robot infrastructure interfaces and environments used by CIRL (Causal-guided Intervention Reinforcement Learning). All robot code is structured as follows: a Flask server sends commands to the robot via ROS, and a gym environment communicates with the Flask server via POST requests.

- `robot_server`: hosts a Flask server which sends commands to the robot via ROS.
- `ur_env`: gym environment for the Universal Robot (UR) arm.
- `franka_env`: gym environment for the Franka arm.

### Installation

1. Install `libfranka` and `franka_ros` (if using Franka arm).
2. Install this package and its dependencies:
    ```bash
    conda activate cirl
    pip install -e .
    ```

### Usage
To start using the robot, first power on the robot control box. Calibrate the end-effector payload to ensure accuracy of the impedance controller.

The following commands are used to start the impedance controller and robot server:
```bash
cd robot_servers
conda activate cirl

# Source the ROS workspace setup script
source </path/to/catkin_ws>/devel/setup.bash

# Set ROS master URI
export ROS_MASTER_URI=http://localhost:<ros_port_number>

# Start the http server and ros controller
python franka_server.py \
    --gripper_type=<Robotiq|Franka|None> \
    --robot_ip=<robot_IP> \
    --gripper_ip=<[Optional] Robotiq_gripper_IP> \
    --reset_joint_target=<[Optional] robot_joints_when_robot_resets> \
    --flask_url=<url_to_serve> \
    --ros_port=<ros_port_number> \
```

The HTTP server is used to communicate between the ROS controller and gym environments. Possible HTTP requests include:

| Request | Description |
| --- | --- |
| startimp | Start the impedance controller |
| stopimp | Stop the impedance controller |
| pose | Command robot to go to desired end-effector pose given in base frame (xyz+quaternion) |
| getpos | Return current end-effector pose in robot base frame (xyz+rpy)|
| getvel | Return current end-effector velocity in robot base frame |
| getforce | Return estimated force on end-effector in stiffness frame |
| gettorque | Return estimated torque on end-effector in stiffness frame |
| getq | Return current joint position |
| getdq | Return current joint velocity |
| getjacobian | Return current zero-jacobian |
| getstate | Return all robot states |
| jointreset | Perform joint reset |
| activate_gripper | Activate the gripper (Robotiq only) |
| reset_gripper | Reset the gripper (Robotiq only) |
| get_gripper | Return current gripper position |
| close_gripper | Close the gripper completely |
| open_gripper | Open the gripper completely |
| move_gripper | Move the gripper to a given position |
| clearerr | Clear errors |
| update_param | Update the impedance controller parameters |

These commands can also be called via terminal:
```bash
curl -X POST <flask_url>:5000/activate_gripper # Activate gripper
curl -X POST <flask_url>:5000/close_gripper # Close gripper
curl -X POST <flask_url>:5000/open_gripper # Open gripper
curl -X POST <flask_url>:5000/getpos # Print current end-effector pose
curl -X POST <flask_url>:5000/jointreset # Perform joint reset
curl -X POST <flask_url>:5000/stopimp # Stop the impedance controller
curl -X POST <flask_url>:5000/startimp # Start the impedance controller
```
