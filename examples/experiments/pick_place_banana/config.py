import os

import numpy as np

from experiments.config import DefaultTrainingConfig
from experiments.pick_place_banana.wrapper import (
    GripperPenaltyWrapper,
    UR7ERAMEnv,
    XYZGraspActionWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from ur_env.envs.ur5_env import DefaultEnvConfig
from ur_env.envs.wrappers import GripperCloseEnv, Quat2MrpWrapper, SpacemouseIntervention


class EnvConfig(DefaultEnvConfig):
    # Hardware values mirrored from:
    #   /home/pine/openpi/deploy_pi05_haoyuan_spacemouse_record_mp4.py
    ROBOT_IP = os.getenv("UR_ROBOT_IP", os.getenv("ROBOT_IP", "192.168.1.10"))
    CONTROLLER_HZ = 200

    HOME_TCP_POSE = np.array(
        [0.05421, -0.42701, 0.25496, -0.0149, 3.12013, -0.02528],
        dtype=np.float32,
    )
    RESET_Q = np.array(
        [[0.9519, -1.7670, 1.9762, -1.7274, -1.5715, -0.5454]],
        dtype=np.float32,
    )
    RESET_HEIGHT = 0.12
    RESET_LIFT_Z = float(os.getenv("RESET_LIFT_Z", "0.03"))
    RESET_TIMEOUT_S = float(os.getenv("RESET_TIMEOUT_S", "2.0"))
    RESET_POS_TOL = float(os.getenv("RESET_POS_TOL", "0.0015"))
    RESET_ROT_TOL = float(os.getenv("RESET_ROT_TOL", "0.03"))
    RESET_WAIT_DT = float(os.getenv("RESET_WAIT_DT", "0.005"))
    RESET_SPEED_MULTIPLIER = float(os.getenv("RESET_SPEED_MULTIPLIER", "4.0"))
    RESET_FORCE_LIMIT_SCALE = float(os.getenv("RESET_FORCE_LIMIT_SCALE", "4.0"))
    RESET_TARGET_Z_MAX = float(os.getenv("RESET_TARGET_Z_MAX", "0.35"))
    RESET_MAX_TRANSLATION = float(os.getenv("RESET_MAX_TRANSLATION", "0.2"))
    RESET_MAX_ROT_DELTA = float(os.getenv("RESET_MAX_ROT_DELTA", "0.35"))
    RESET_NOISE_ENABLED = os.getenv("RESET_NOISE_ENABLED", "1") != "0"
    RESET_XYZ_NOISE = np.array(
        [
            float(os.getenv("RESET_NOISE_X", "0.003")),
            float(os.getenv("RESET_NOISE_Y", "0.003")),
            float(os.getenv("RESET_NOISE_Z", "0.001")),
        ],
        dtype=np.float32,
    )
    MOVE_HOME_ON_CLOSE = False
    RELEASE_GRIPPER_ON_CLOSE = False
    Z_MAX = float(os.getenv("BANANA_Z_MAX", os.getenv("CARROT_Z_MAX", "0.40")))

    # Banana pick-place needs a wider lateral workspace than pick_carrot.
    # Defaults keep the same home pose while allowing movement to a place region.
    X_NEG_RANGE = float(os.getenv("BANANA_X_NEG_RANGE", "0.05"))
    X_POS_RANGE = float(os.getenv("BANANA_X_POS_RANGE", "0.05"))
    Y_NEG_RANGE = float(os.getenv("BANANA_Y_NEG_RANGE", "0.05"))
    Y_POS_RANGE = float(os.getenv("BANANA_Y_POS_RANGE", "0.05"))
    Z_LOW = float(os.getenv("BANANA_Z_LOW", "0.15"))
    Z_POS_RANGE = float(os.getenv("BANANA_Z_POS_RANGE", "0.05"))

    ABS_POSE_LIMIT_LOW = np.array(
        [HOME_TCP_POSE[0] - X_NEG_RANGE, HOME_TCP_POSE[1] - Y_NEG_RANGE, Z_LOW, -3.19859, -3.24117, -3.10],
        dtype=np.float32,
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [HOME_TCP_POSE[0] + X_POS_RANGE, HOME_TCP_POSE[1] + Y_POS_RANGE, min(float(HOME_TCP_POSE[2] + Z_POS_RANGE), Z_MAX), 3.19859, 3.24117, 3.10],
        dtype=np.float32,
    )
    ABS_POSE_RANGE_LIMITS = np.array(
        [-max(X_NEG_RANGE, Y_NEG_RANGE), max(X_POS_RANGE, Y_POS_RANGE)],
        dtype=np.float32,
    )

    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)

    # Keep pick-place exploration gentler around the banana and place target.
    ACTION_SCALE = np.array([float(os.getenv("BANANA_ACTION_XYZ_SCALE", "0.04")), 0.1, 1.0], dtype=np.float32)
    # The low-level UR env still receives [x, y, z, rx, ry, rz, gripper],
    # but this experiment exposes only [x, y, z, gripper] to the policy.
    EXECUTED_ACTION_MASK = np.array([1, 1, 1, 0, 0, 0, 1], dtype=np.float32)
    POLICY_ACTION_MASK = np.array([1, 1, 1, 1], dtype=np.float32)
    SPACEMOUSE_INVERT_AXES = np.array([-1, 1, -1, -1, 1, -1], dtype=np.float32)
    TOOL_ROLL_ONLY = False
    TOOL_ROLL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    MAX_EPISODE_LENGTH = 200

    # Lower stiffness than the historical UR5 defaults, so contacts are compliant.
    CONTROLLER_KP = float(os.getenv("CONTROLLER_KP", "2400.0"))
    CONTROLLER_KD = float(os.getenv("CONTROLLER_KD", "180.0"))
    CONTROLLER_ROT_KP = float(os.getenv("CONTROLLER_ROT_KP", "55.0"))
    CONTROLLER_ROT_KD = float(os.getenv("CONTROLLER_ROT_KD", "5.5"))
    ERROR_DELTA = float(os.getenv("ERROR_DELTA", "0.04"))
    TRUNCATE_FORCE_N = 80.0
    FORCEMODE_DAMPING = float(os.getenv("FORCEMODE_DAMPING", "0.04"))
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array(
        [
            float(os.getenv("FORCEMODE_XY_SPEED", "0.5")),
            float(os.getenv("FORCEMODE_XY_SPEED", "0.5")),
            float(os.getenv("FORCEMODE_Z_SPEED", "0.45")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.20")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.20")),
            float(os.getenv("FORCEMODE_ROT_SPEED", "1.20")),
        ],
        dtype=np.float32,
    )

    # Robotiq gripper via the UR controller socket server, matching OpenPI.
    GRIPPER_ENABLED = True
    GRIPPER_COMMUNICATION = "socket"
    GRIPPER_PORT = int(os.getenv("GRIPPER_PORT", "63352"))
    GRIPPER_SPEED = 90
    GRIPPER_FORCE = 10
    GRIPPER_TIMEOUT = 500
    GRIPPER_AUTO_CALIBRATE = False
    RESET_GRIPPER_ACTION = os.getenv("RESET_GRIPPER_ACTION", "release")
    RESET_GRIPPER_SETTLE_S = float(os.getenv("RESET_GRIPPER_SETTLE_S", "0.15"))
    SPACEMOUSE_GRIPPER_ENABLED = True

    # RealSense cameras from the provided OpenPI command:
    # hand/wrist camera: 218622270687; third-person/external: 409122274280.
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": os.getenv("HAND_SERIAL", "218622270687"),
            "dim": (int(os.getenv("HAND_WIDTH", "640")), int(os.getenv("HAND_HEIGHT", "480"))),
            "fps": int(os.getenv("CAPTURE_FPS", "15")),
            "exposure": int(os.getenv("HAND_EXPOSURE", "15000")),
        },
        "external": {
            "serial_number": os.getenv("EXTERNAL_SERIAL", "254622076156"),
            "dim": (int(os.getenv("EXTERNAL_WIDTH", "640")), int(os.getenv("EXTERNAL_HEIGHT", "480"))),
            "fps": int(os.getenv("CAPTURE_FPS", "15")),
            # "exposure": int(os.getenv("EXTERNAL_EXPOSURE", "400")),
        },
    }
    # Optional crop ranges generated by:
    #   python -m ur_env.camera.interactive_crop --experiment pick_place_banana
    # Format is [y0, y1, x0, x1]. Leave empty to use the default center square crop.
    IMAGE_CROP = {
        "wrist": [74, 480, 88, 627],
        "external": [42, 340, 101, 384],
    }



class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "external"]
    classifier_keys = ["wrist", "external"]

    proprio_keys = [
        "tcp_pose",
        "tcp_force",
        "gripper_state",
    ]

    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper" # Options: "single-arm-learned-gripper", "single-arm-learned-gripper", "dual-arm-fixed-gripper", "dual-arm-learned-gripper"
    max_traj_length = EnvConfig.MAX_EPISODE_LENGTH
    buffer_period = 5000
    checkpoint_period = 5000
    steps_per_update = 50
    causal_reward_mode = "outcome_return_to_go"
    causal_rtg_gamma = 0.98
    causal_update_interval = 500
    causal_sample_size = 500
    causal_action_indices = [0, 1, 2]  # Use xyz; gripper is handled by the grasp critic.
    causal_sampling_strategy = "phase_mixed"
    causal_online_ratio = 0.25
    causal_intervention_ratio = 0.35
    causal_demo_ratio = 0.4
    causal_recent_trajectory_files = 48
    causal_pre_intervention_window = 20
    causal_policy_tail_window = 20
    causal_demo_match_candidate_multiplier = 10
    causal_demo_match_use_actions = False
    use_grasp_causal_bias = False
    grasp_causal_beta = 0.05
    grasp_causal_update_interval = causal_update_interval
    grasp_causal_sample_size = causal_sample_size
    grasp_causal_bias_sync_period = 50

    use_reward_classifier = False
    classifier_ckpt_path = os.path.abspath("classifier_ckpt/")

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = UR7ERAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            max_episode_length=EnvConfig.MAX_EPISODE_LENGTH,
            hz=10,
            camera_mode="rgb",
        )

        if not fake_env:
            env = SpacemouseIntervention(env)

        env = XYZGraspActionWrapper(env)

        if self.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
            env = GripperCloseEnv(env, fixed_gripper_action=0.0)

        env = Quat2MrpWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
