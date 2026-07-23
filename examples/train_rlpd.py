#!/usr/bin/env python3

import glob
import time
from collections.abc import Mapping
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
import threading
import select
import sys
import termios
import tty
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

# Prefer this checkout even when the active environment has a stale editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_SERL_LAUNCHER = os.path.join(REPO_ROOT, "serl_launcher")
LOCAL_ROBOT_INFRA = os.path.join(REPO_ROOT, "serl_robot_infra")
for source_root in (REPO_ROOT, LOCAL_ROBOT_INFRA, LOCAL_SERL_LAUNCHER):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.utils.causal_mask_utils import (
    add_encoded_features,
    augment_latent_features,
    compute_cmi_masks_jax,
    create_causal_state,
    evaluate_causal_model,
    select_lowest_score_mask,
    causal_mask_counterfactuals,
    train_causal_model,
    train_causal_model_from_buffers,
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import DataStoreBase, QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
import math

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "ur5e_aruco_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_boolean(
    "resume_training",
    False,
    "Resume from an existing checkpoint_path. Defaults to starting a new run.",
)
flags.DEFINE_boolean(
    "allow_existing_checkpoint_path",
    False,
    "Allow actor to attach to an active run directory without restoring from disk.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("use_causal_entropy", True, "Use causal entropy for exploration.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


GRASP_CAUSAL_PAYLOAD_KEYS = (
    "matrix",
    "state_mean",
    "state_std",
    "fallback_bias",
    "mean_bias",
    "active_matrix",
)


def normalize_grasp_causal_payload(payload):
    if isinstance(payload, dict):
        normalized = {}
        for key in GRASP_CAUSAL_PAYLOAD_KEYS:
            if key in payload:
                normalized[key] = np.asarray(payload[key], dtype=np.float32)
        return normalized if normalized else None
    if payload is None:
        return None
    bias = np.asarray(payload, dtype=np.float32).reshape(-1)
    if bias.shape[0] <= 0:
        return None
    return bias[:3]


def summarize_grasp_causal_payload(payload):
    if isinstance(payload, dict):
        active = float(np.asarray(payload.get("active_matrix", [0.0])).reshape(-1)[0])
        key = "mean_bias" if active > 0.5 and "mean_bias" in payload else "fallback_bias"
        return np.asarray(payload.get(key, np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
    if payload is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(payload, dtype=np.float32).reshape(-1)[:3]


class ManualEpisodeLabeler:
    def __init__(self):
        self.success = False
        self.failure = False
        self.pressed_keys = set()
        self.listener = None
        self.keyboard = None
        self.stdin_fd = None
        self.stdin_settings = None

    def start(self):
        try:
            if sys.stdin.isatty():
                self.stdin_fd = sys.stdin.fileno()
                self.stdin_settings = termios.tcgetattr(self.stdin_fd)
                tty.setcbreak(self.stdin_fd)
        except Exception as exc:
            print(f"Manual stdin labels disabled: {exc}")

        try:
            from pynput import keyboard as pynput_keyboard
            self.keyboard = pynput_keyboard
            self.listener = self.keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self.listener.start()
        except Exception as exc:
            print(f"Manual pynput labels disabled: {exc}")

        print("Manual actor labels: SPACE/s = success and reset, f/ESC = failure and reset.")

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
        if self.stdin_fd is not None and self.stdin_settings is not None:
            try:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_settings)
            except Exception:
                pass

    def consume(self):
        self._poll_stdin()
        if self.success:
            self.success = False
            self.failure = False
            return "success"
        if self.failure:
            self.failure = False
            return "failure"
        return None

    def _mark_char(self, char):
        if char in (" ", "s", "S"):
            self.success = True
        elif char in ("f", "F", "\x1b"):
            self.failure = True

    def _poll_stdin(self):
        if self.stdin_fd is None:
            return
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                self._mark_char(sys.stdin.read(1))
        except Exception:
            pass

    def _on_press(self, key):
        if self.keyboard is not None and key == self.keyboard.Key.space:
            token = "space"
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char(" ")
            return
        if self.keyboard is not None and key == self.keyboard.Key.esc:
            token = "esc"
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char("\x1b")
            return
        try:
            token = key.char
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char(token)
        except AttributeError:
            pass

    def _on_release(self, key):
        if self.keyboard is not None and key == self.keyboard.Key.space:
            self.pressed_keys.discard("space")
            return
        if self.keyboard is not None and key == self.keyboard.Key.esc:
            self.pressed_keys.discard("esc")
            return
        try:
            self.pressed_keys.discard(key.char)
        except AttributeError:
            pass


def _iter_env_chain(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _call_env_chain(env, method_name):
    for current in _iter_env_chain(env):
        method = getattr(current, method_name, None)
        if callable(method):
            method()


def prepare_manual_reset(env, action_filter):
    action_filter.reset()
    _call_env_chain(env, "clear_intervention")



def align_transition_action(transition, action_space):
    if "actions" not in transition:
        return transition

    target_shape = tuple(action_space.shape)
    target_dim = int(np.prod(target_shape))
    action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)

    if action.shape[0] == 7 and target_dim == 4:
        # Banana exposes [x, y, z, gripper]; older UR demos are
        # [x, y, z, rx, ry, rz, gripper].
        action = action[[0, 1, 2, 6]]
    elif action.shape[0] > target_dim:
        action = action[:target_dim]
    elif action.shape[0] < target_dim:
        action = np.pad(action, (0, target_dim - action.shape[0]), constant_values=0.0)

    transition["actions"] = action.reshape(target_shape).astype(np.float32)
    return transition


def validate_value_shape(value, space, field_path):
    if hasattr(space, "spaces"):
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_path} must be a mapping, got {type(value).__name__}")
        missing_keys = sorted(set(space.spaces) - set(value))
        if missing_keys:
            raise ValueError(f"{field_path} is missing keys: {missing_keys}")
        for key, child_space in space.spaces.items():
            validate_value_shape(value[key], child_space, f"{field_path}.{key}")
        return

    expected_shape = tuple(space.shape)
    actual_shape = tuple(np.asarray(value).shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{field_path} has shape {actual_shape}, expected {expected_shape}"
        )


def validate_loaded_transition_schema(transition, env, source):
    try:
        validate_value_shape(transition["observations"], env.observation_space, "observations")
        validate_value_shape(
            transition["next_observations"], env.observation_space, "next_observations"
        )
        validate_value_shape(transition["actions"], env.action_space, "actions")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Incompatible transition schema in {source}: {exc}. "
            "Use demos and resume buffers collected with the current environment schema."
        ) from exc


def prepare_loaded_transition(transition, env, include_grasp_penalty, source="loaded data"):
    transition = copy.deepcopy(transition)
    transition = align_transition_action(transition, env.action_space)
    if "infos" in transition and "grasp_penalty" in transition["infos"]:
        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
    elif include_grasp_penalty and "grasp_penalty" not in transition:
        transition["grasp_penalty"] = 0.0
    if "causal_rewards" not in transition:
        transition["causal_rewards"] = float(np.asarray(transition.get("rewards", 0.0)).reshape(()))
    validate_loaded_transition_schema(transition, env, source)
    return transition


def infer_episode_outcome(trajectory, fallback_outcome=None):
    if fallback_outcome is not None:
        return float(fallback_outcome)
    if not trajectory:
        return 0.0
    for transition in reversed(trajectory):
        info = transition.get("infos", {})
        if isinstance(info, dict) and "succeed" in info:
            return float(bool(info["succeed"]))
        reward = float(np.asarray(transition.get("rewards", 0.0)).reshape(()))
        done = bool(transition.get("dones", False))
        if done:
            return float(reward > 0.5)
    final_reward = float(np.asarray(trajectory[-1].get("rewards", 0.0)).reshape(()))
    return float(final_reward > 0.5)


def relabel_causal_rewards(trajectory, gamma=0.98, mode="outcome_return_to_go", fallback_outcome=None):
    trajectory = [copy.deepcopy(t) for t in trajectory]
    if not trajectory:
        return trajectory

    mode = str(mode).lower()
    outcome = infer_episode_outcome(trajectory, fallback_outcome=fallback_outcome)
    rewards = np.array(
        [float(np.asarray(t.get("rewards", 0.0)).reshape(())) for t in trajectory],
        dtype=np.float32,
    )

    if mode in ("episode_outcome", "outcome"):
        causal_rewards = np.full(len(trajectory), outcome, dtype=np.float32)
    elif mode in ("return_to_go", "rtg"):
        causal_rewards = np.zeros(len(trajectory), dtype=np.float32)
        running = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            running = float(rewards[i]) + float(gamma) * running
            causal_rewards[i] = running
    elif mode in ("outcome_return_to_go", "outcome_rtg", "episode_outcome_rtg"):
        causal_rewards = np.array(
            [outcome * (float(gamma) ** (len(trajectory) - 1 - i)) for i in range(len(trajectory))],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown causal_reward_mode: {mode}")

    for transition, causal_reward in zip(trajectory, causal_rewards):
        transition["causal_rewards"] = float(causal_reward)
        info = copy.deepcopy(transition.get("infos", {}))
        if isinstance(info, dict):
            info["causal_episode_outcome"] = outcome
            info["causal_reward_mode"] = mode
            transition["infos"] = info
    return trajectory


def split_trajectories(transitions):
    trajectory = []
    for transition in transitions:
        trajectory.append(transition)
        if bool(transition.get("dones", False)):
            yield trajectory
            trajectory = []
    if trajectory:
        yield trajectory


def outcome_from_path(path):
    name = os.path.basename(path).lower()
    if "success" in name:
        return 1.0
    if "failure" in name:
        return 0.0
    return None


def latest_step_from_path(path, prefix):
    if not path:
        return None
    name = os.path.basename(path.rstrip(os.sep))
    name = os.path.splitext(name)[0]
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    digits = []
    for ch in suffix:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def transition_is_intervention(transition):
    if bool(transition.get("intervention", False)):
        return True
    info = transition.get("infos", {})
    return isinstance(info, dict) and bool(info.get("intervention", False))


def transition_is_successful_episode(transition):
    info = transition.get("infos", {})
    if isinstance(info, dict):
        if "causal_episode_outcome" in info:
            return float(info["causal_episode_outcome"]) > 0.5
        if "succeed" in info:
            return bool(info["succeed"])
    return float(np.asarray(transition.get("causal_rewards", 0.0)).reshape(())) > 0.0


class SuccessfulInterventionFanoutDataStore(DataStoreBase):
    """Route intervention data to RL demo buffer and successful-causal-mask buffer."""

    def __init__(self, primary_store, success_store):
        super().__init__(getattr(primary_store, "capacity", 0))
        self.primary_store = primary_store
        self.success_store = success_store

    def insert(self, data):
        self.primary_store.insert(data)
        if transition_is_successful_episode(data):
            self.success_store.insert(copy.deepcopy(data))

    def batch_insert(self, batch_data):
        for data in batch_data:
            self.insert(data)

    def latest_data_id(self):
        return self.primary_store.latest_data_id()

    def get_latest_data(self, from_id):
        return self.primary_store.get_latest_data(from_id)

    def __len__(self):
        return len(self.primary_store)


def _sample_list(items, count, rng):
    if count <= 0 or not items:
        return []
    replace = len(items) < count
    indices = rng.choice(len(items), size=count, replace=replace)
    return [items[int(i)] for i in np.asarray(indices).reshape(-1)]


def _transition_to_causal_fields(transition):
    obs = transition.get("observations", {})
    if (isinstance(obs, dict) or hasattr(obs, "items")) and "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32)
    else:
        state = np.asarray(obs, dtype=np.float32)
    action = np.asarray(transition["actions"], dtype=np.float32)
    reward = float(np.asarray(transition.get("rewards", 0.0)).reshape(()))
    causal_reward = float(np.asarray(transition.get("causal_rewards", reward)).reshape(()))
    return state, action, reward, causal_reward


def _stack_causal_transitions(transitions):
    if not transitions:
        return None
    states, actions, rewards, causal_rewards = zip(
        *[_transition_to_causal_fields(transition) for transition in transitions]
    )
    return {
        "observations": {"state": np.stack(states, axis=0).astype(np.float32)},
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "causal_rewards": np.asarray(causal_rewards, dtype=np.float32),
    }


def _reduce_causal_batch(batch):
    obs = batch["observations"]
    if (isinstance(obs, dict) or hasattr(obs, "items")) and "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32)
    else:
        state = np.asarray(obs, dtype=np.float32)
    rewards = np.asarray(batch.get("rewards", np.zeros(state.shape[0])), dtype=np.float32)
    causal_rewards = np.asarray(batch.get("causal_rewards", rewards), dtype=np.float32)
    return {
        "observations": {"state": state},
        "actions": np.asarray(batch["actions"], dtype=np.float32),
        "rewards": rewards.reshape(state.shape[0], -1)[:, 0],
        "causal_rewards": causal_rewards.reshape(state.shape[0], -1)[:, 0],
    }


def _index_causal_batch(batch, indices):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return {
        "observations": {"state": np.asarray(batch["observations"]["state"])[indices]},
        "actions": np.asarray(batch["actions"])[indices],
        "rewards": np.asarray(batch["rewards"])[indices],
        "causal_rewards": np.asarray(batch["causal_rewards"])[indices],
    }


def _concat_causal_batches(batches):
    batches = [batch for batch in batches if batch is not None and batch["actions"].shape[0] > 0]
    if not batches:
        return None
    return {
        "observations": {
            "state": np.concatenate([np.asarray(batch["observations"]["state"]) for batch in batches], axis=0)
        },
        "actions": np.concatenate([np.asarray(batch["actions"]) for batch in batches], axis=0),
        "rewards": np.concatenate([np.asarray(batch["rewards"]).reshape(-1) for batch in batches], axis=0),
        "causal_rewards": np.concatenate(
            [np.asarray(batch["causal_rewards"]).reshape(-1) for batch in batches], axis=0
        ),
    }


def _causal_feature_matrix(batch, use_actions=False):
    states = np.asarray(batch["observations"]["state"], dtype=np.float32)
    states = states.reshape(states.shape[0], -1)
    if not use_actions:
        return states
    actions = np.asarray(batch["actions"], dtype=np.float32)
    actions = actions.reshape(actions.shape[0], -1)
    return np.concatenate([states, actions], axis=-1)


def _match_demo_batch_to_anchors(
    demo_buffer,
    anchor_batch,
    count,
    rng,
    candidate_multiplier=6,
    use_actions=False,
):
    if count <= 0 or len(demo_buffer) <= 0:
        return None
    if anchor_batch is None or anchor_batch["actions"].shape[0] == 0:
        candidate_count = min(len(demo_buffer), count)
        demo_random = _reduce_causal_batch(demo_buffer.sample(batch_size=candidate_count))
        if candidate_count >= count:
            return demo_random
        extra = rng.choice(candidate_count, size=count - candidate_count, replace=True)
        indices = np.concatenate([np.arange(candidate_count), extra], axis=0)
        return _index_causal_batch(demo_random, indices)

    candidate_count = min(
        len(demo_buffer),
        max(count * int(candidate_multiplier), count),
    )
    demo_candidates = _reduce_causal_batch(demo_buffer.sample(batch_size=candidate_count))
    demo_features = _causal_feature_matrix(demo_candidates, use_actions=use_actions)
    anchor_features = _causal_feature_matrix(anchor_batch, use_actions=use_actions)

    all_features = np.concatenate([demo_features, anchor_features], axis=0)
    mean = np.mean(all_features, axis=0, keepdims=True)
    std = np.std(all_features, axis=0, keepdims=True) + 1e-6
    demo_features = (demo_features - mean) / std
    anchor_features = (anchor_features - mean) / std

    # Pick demo samples whose state is closest to the current online/intervention stage.
    distances = np.linalg.norm(
        demo_features[:, None, :] - anchor_features[None, :, :],
        axis=-1,
    )
    nearest = np.min(distances, axis=1)
    order = np.argsort(nearest)
    if order.shape[0] >= count:
        indices = order[:count]
    else:
        extra = rng.choice(order, size=count - order.shape[0], replace=True)
        indices = np.concatenate([order, extra], axis=0)
    return _index_causal_batch(demo_candidates, indices)


def build_phase_mixed_causal_batch(config, checkpoint_path, demo_buffer, sample_size, step):
    rng = np.random.default_rng(int(step) + int(FLAGS.seed))
    max_files = int(getattr(config, "causal_recent_trajectory_files", 48))
    pre_window = int(getattr(config, "causal_pre_intervention_window", 20))
    tail_window = int(getattr(config, "causal_policy_tail_window", 20))
    candidate_multiplier = int(getattr(config, "causal_demo_match_candidate_multiplier", 6))
    match_use_actions = bool(getattr(config, "causal_demo_match_use_actions", False))

    online_ratio = float(getattr(config, "causal_online_ratio", 0.3))
    intervention_ratio = float(getattr(config, "causal_intervention_ratio", 0.3))
    demo_ratio = float(getattr(config, "causal_demo_ratio", 0.4))
    ratio_sum = online_ratio + intervention_ratio + demo_ratio
    if ratio_sum <= 0:
        raise ValueError("causal sampling ratios must sum to a positive value")

    online_count = int(round(sample_size * online_ratio / ratio_sum))
    intervention_count = int(round(sample_size * intervention_ratio / ratio_sum))
    demo_count = sample_size - online_count - intervention_count

    buffer_dir = os.path.join(checkpoint_path, "buffer") if checkpoint_path else None
    trajectory_files = []
    if buffer_dir:
        trajectory_files = natsorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
    recent_files = trajectory_files[-max_files:]

    pre_intervention = []
    intervention = []
    policy_tails = []

    for path in recent_files:
        try:
            with open(path, "rb") as handle:
                trajectory = pkl.load(handle)
        except Exception as exc:
            print(f"Skipping causal trajectory file {path}: {exc}", flush=True)
            continue
        if not trajectory:
            continue

        intv_indices = [idx for idx, transition in enumerate(trajectory) if transition_is_intervention(transition)]
        if intv_indices:
            first_idx = intv_indices[0]
            pre_intervention.extend(trajectory[max(0, first_idx - pre_window):first_idx])
            intervention.extend([trajectory[idx] for idx in intv_indices])
        else:
            # This is transition-level online data. Do not gate it by episode
            # outcome; success/failure is already represented by causal_rewards.
            policy_tails.extend(trajectory[-tail_window:])

    online_candidates = pre_intervention + policy_tails
    online_batch = _stack_causal_transitions(_sample_list(online_candidates, online_count, rng))
    intervention_batch = _stack_causal_transitions(_sample_list(intervention, intervention_count, rng))
    anchor_batch = _concat_causal_batches([online_batch, intervention_batch])
    demo_batch = _match_demo_batch_to_anchors(
        demo_buffer=demo_buffer,
        anchor_batch=anchor_batch,
        count=demo_count,
        rng=rng,
        candidate_multiplier=candidate_multiplier,
        use_actions=match_use_actions,
    )
    mixed_batch = _concat_causal_batches([online_batch, intervention_batch, demo_batch])
    sample_total = 0 if mixed_batch is None else int(mixed_batch["actions"].shape[0])
    demo_fill_batch = None
    if sample_total < sample_size:
        demo_fill_batch = _match_demo_batch_to_anchors(
            demo_buffer=demo_buffer,
            anchor_batch=anchor_batch,
            count=sample_size - sample_total,
            rng=rng,
            candidate_multiplier=candidate_multiplier,
            use_actions=match_use_actions,
        )
        mixed_batch = _concat_causal_batches([mixed_batch, demo_fill_batch])

    sample_demo = 0 if demo_batch is None else int(demo_batch["actions"].shape[0])
    sample_demo_fill = 0 if demo_fill_batch is None else int(demo_fill_batch["actions"].shape[0])
    info = {
        "recent_files": len(recent_files),
        "online_candidates": len(online_candidates),
        "pre_intervention_candidates": len(pre_intervention),
        "intervention_candidates": len(intervention),
        "failure_tail_candidates": 0,
        "policy_tail_candidates": len(policy_tails),
        "target_online": online_count,
        "target_intervention": intervention_count,
        "target_demo": demo_count,
        "sample_online": 0 if online_batch is None else int(online_batch["actions"].shape[0]),
        "sample_intervention": 0 if intervention_batch is None else int(intervention_batch["actions"].shape[0]),
        "sample_demo": sample_demo + sample_demo_fill,
        "sample_demo_fill": sample_demo_fill,
        "sample_total": 0 if mixed_batch is None else int(mixed_batch["actions"].shape[0]),
    }
    return mixed_batch, info


def get_resume_step(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    resume_steps = []

    buffer_files = natsorted(glob.glob(os.path.join(checkpoint_path, "buffer", "transitions_*.pkl")))
    for buffer_file in buffer_files:
        step = latest_step_from_path(buffer_file, "transitions_")
        if step is not None:
            resume_steps.append(step + 1)

    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    step = latest_step_from_path(latest_ckpt, "checkpoint_")
    if step is not None:
        resume_steps.append(step + 1)

    return max(resume_steps, default=0)


def split_complete_trajectories(transitions):
    last_done = -1
    for idx, transition in enumerate(transitions):
        if bool(transition.get("dones", False)):
            last_done = idx

    if last_done < 0:
        return [], transitions
    return transitions[:last_done + 1], transitions[last_done + 1:]


def atomic_pickle_dump(obj, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_transition_file(path):
    try:
        with open(path, "rb") as f:
            obj = pkl.load(f)
    except EOFError:
        with open(path, "rb") as f:
            data = f.read()
        obj = pkl.loads(data + b"ue.")
        print(f"Recovered truncated pickle while loading {path}")
    if isinstance(obj, dict) and {"observations", "actions", "next_observations", "rewards", "masks", "dones"}.issubset(obj.keys()):
        return [obj]
    if isinstance(obj, list):
        return [
            x for x in obj
            if isinstance(x, dict)
            and {"observations", "actions", "next_observations", "rewards", "masks", "dones"}.issubset(x.keys())
        ]
    return []


def dump_completed_transitions(transitions, path):
    completed, remainder = split_complete_trajectories(transitions)
    if not completed:
        return remainder, 0
    atomic_pickle_dump(completed, path)
    return remainder, len(completed)


##############################################################################

class EMAActionFilter:
    def __init__(self, hz: float, cutoff_hz: float = 2.0, filter_rot=True, filter_gripper=True):
        self.dt = 1.0 / float(hz)
        tau = 1.0 / (2.0 * math.pi * float(cutoff_hz))
        self.alpha = self.dt / (tau + self.dt)
        self.filter_rot = filter_rot
        self.filter_gripper = filter_gripper
        self.prev = None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).copy()

        if self.prev is None:
            self.prev = a.copy()
            return a

        # Filter continuous arm action. Standard single-arm actions are
        # [x, y, z, rx, ry, rz, gripper]; banana exposes [x, y, z, gripper].
        if a.shape[0] == 4:
            idx_end = 3
            gripper_idx = 3
        else:
            idx_end = min(6, a.shape[0])
            gripper_idx = 6 if a.shape[0] > 6 else None
        self.prev[:idx_end] = self.prev[:idx_end] + self.alpha * (a[:idx_end] - self.prev[:idx_end])

        # Only touch gripper if it exists
        if gripper_idx is not None:
            if self.filter_gripper:
                self.prev[gripper_idx] = self.prev[gripper_idx] + self.alpha * (a[gripper_idx] - self.prev[gripper_idx])
            else:
                self.prev[gripper_idx] = a[gripper_idx]

        return np.clip(self.prev, -1.0, 1.0).astype(np.float32)
    def reset(self):
        self.prev = None


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    action_filter = EMAActionFilter(hz=10, cutoff_hz=2)
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        manual_labeler = ManualEpisodeLabeler()
        manual_labeler.start()
        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                done = False
                truncated = False
                start_time = time.time()
                while not (done or truncated):
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key
                    )
                    actions = np.asarray(jax.device_get(actions))
                    actions = action_filter(actions)
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    manual_label = manual_labeler.consume()
                    if manual_label == "success":
                        reward = 1.0
                        done = True
                        truncated = False
                        print("manual eval success marked; resetting environment")
                    elif manual_label == "failure":
                        reward = 0.0
                        done = True
                        truncated = False
                        print("manual eval failure marked; resetting environment")

                    if done or truncated:
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)
                        action_filter.reset()
                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            print(f"average time: {np.mean(time_list) if time_list else float('nan')}")
        finally:
            try:
                manual_labeler.stop()
            except Exception:
                pass
            try:
                env.close()
            except Exception as exc:
                print(f"eval cleanup failed: {exc}")
        return  # after done eval, return and exit
    
    start_step = (
        get_resume_step(FLAGS.checkpoint_path)
        if (FLAGS.resume_training or FLAGS.allow_existing_checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=500,
    )

    pending_network = {"params": None, "count": 0, "applied": 0}
    pending_network_lock = threading.Lock()

    def update_params(params):
        with pending_network_lock:
            pending_network["params"] = params
            pending_network["count"] += 1

    def apply_pending_network(force=False):
        nonlocal agent
        if not force and step % config.steps_per_update != 0:
            return
        with pending_network_lock:
            params = pending_network["params"]
            count = pending_network["count"]
            pending_network["params"] = None
        if params is not None:
            agent = agent.replace(state=agent.state.replace(params=params))
            skipped = max(0, count - pending_network["applied"] - 1)
            pending_network["applied"] = count
            if skipped:
                print_green(f"Applied latest learner params; skipped {skipped} stale updates")
            else:
                print_green("Applied latest learner params")

    client.recv_network_callback(update_params)

    current_trajectory = []
    trajectory_index = 0
    actor_stats = {
        "saved_trajectories": 0,
        "saved_success_trajectories": 0,
        "saved_failure_trajectories": 0,
        "saved_intervention_trajectories": 0,
        "saved_policy_trajectories": 0,
        "saved_transitions": 0,
        "saved_intervention_samples": 0,
        "saved_intervention_time_s": 0.0,
    }
    last_buffer_dump_step = start_step
    last_seen_step = start_step
    actor_sync_period = max(1, int(os.getenv("ACTOR_SYNC_PERIOD_STEPS", "10")))
    last_actor_sync_step = start_step
    use_grasp_causal_bias = FLAGS.use_causal_entropy and bool(
        getattr(config, "use_grasp_causal_bias", False)
    )
    grasp_causal_beta = float(getattr(config, "grasp_causal_beta", 0.05))
    grasp_causal_bias_sync_period = max(
        1,
        int(getattr(config, "grasp_causal_bias_sync_period", 50)),
    )
    grasp_causal_bias = None
    grasp_causal_bias_step = None
    grasp_causal_bias_mtime = None
    grasp_causal_bias_file = (
        os.path.join(FLAGS.checkpoint_path, "grasp_causal_bias.pkl")
        if FLAGS.checkpoint_path
        else None
    )

    def maybe_load_grasp_causal_bias(step, force=False):
        nonlocal grasp_causal_bias, grasp_causal_bias_step, grasp_causal_bias_mtime
        if not use_grasp_causal_bias or grasp_causal_bias_file is None:
            return
        if not force and step % grasp_causal_bias_sync_period != 0:
            return
        try:
            mtime = os.path.getmtime(grasp_causal_bias_file)
            if not force and grasp_causal_bias_mtime == mtime:
                return
            with open(grasp_causal_bias_file, "rb") as f:
                payload = pkl.load(f)
            bias = normalize_grasp_causal_payload(payload.get("bias"))
            if bias is None:
                return
            grasp_causal_bias = bias
            grasp_causal_bias_step = payload.get("step")
            grasp_causal_bias_mtime = mtime
            summary_bias = summarize_grasp_causal_payload(grasp_causal_bias)
            print_green(
                f"Loaded grasp causal bias step={grasp_causal_bias_step}: "
                f"{summary_bias}"
            )
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"failed to load grasp causal bias: {exc}", flush=True)

    def mark_transition_intervention(transition, intervention, trajectory_had_intervention=False):
        transition["intervention"] = bool(intervention)
        transition["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        info = copy.deepcopy(transition.get("infos", {}))
        info["intervention"] = bool(intervention)
        info["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        transition["infos"] = info
        return transition

    def dump_trajectory(trajectory, step):
        nonlocal trajectory_index
        if FLAGS.checkpoint_path is None or not trajectory:
            return 0
        trajectory_had_intvn = any(bool(t.get("intervention", False)) for t in trajectory)
        reward_value = float(np.asarray(trajectory[-1].get("rewards", 0.0)).reshape(()))
        done_value = bool(trajectory[-1].get("dones", False))
        success = bool(reward_value > 0.5 and done_value)
        label = "success" if success else "failure"
        source = "intvn" if trajectory_had_intvn else "policy"
        trajectory = [
            mark_transition_intervention(copy.deepcopy(t), t.get("intervention", False), trajectory_had_intvn)
            for t in trajectory
        ]
        trajectory = relabel_causal_rewards(
            trajectory,
            gamma=float(getattr(config, "causal_rtg_gamma", 0.98)),
            mode=getattr(config, "causal_reward_mode", "outcome_return_to_go"),
        )
        filename = f"transitions_{step}_{trajectory_index:06d}_{label}_{source}.pkl"
        buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer", filename)
        atomic_pickle_dump(trajectory, buffer_path)
        intervention_samples = [
            copy.deepcopy(t) for t in trajectory if bool(t.get("intervention", False))
        ]
        if intervention_samples:
            # Intervention steps are not necessarily contiguous. Preserve each
            # sample's own image stack when adding it to the demo buffer.
            for transition in intervention_samples:
                transition["_reset_frame_stack"] = True
            demo_filename = f"interventions_{step}_{trajectory_index:06d}_{label}.pkl"
            demo_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer", demo_filename)
            atomic_pickle_dump(intervention_samples, demo_path)
            for transition in intervention_samples:
                intvn_data_store.insert(copy.deepcopy(transition))
        actor_stats["saved_trajectories"] += 1
        actor_stats["saved_success_trajectories"] += int(success)
        actor_stats["saved_failure_trajectories"] += int(not success)
        actor_stats["saved_intervention_trajectories"] += int(trajectory_had_intvn)
        actor_stats["saved_policy_trajectories"] += int(not trajectory_had_intvn)
        actor_stats["saved_transitions"] += len(trajectory)
        actor_stats["saved_intervention_samples"] += len(intervention_samples)
        trajectory_intervention_time_s = 0.0
        for transition in trajectory:
            info = transition.get("infos", {})
            if isinstance(info, dict):
                trajectory_intervention_time_s += float(info.get("intervention_duration_s", 0.0))
        actor_stats["saved_intervention_time_s"] += trajectory_intervention_time_s
        stats_payload = {
            "buffer/actor_saved_trajectories": actor_stats["saved_trajectories"],
            "buffer/actor_saved_success_trajectories": actor_stats["saved_success_trajectories"],
            "buffer/actor_saved_failure_trajectories": actor_stats["saved_failure_trajectories"],
            "buffer/actor_saved_intervention_trajectories": actor_stats["saved_intervention_trajectories"],
            "buffer/actor_saved_policy_trajectories": actor_stats["saved_policy_trajectories"],
            "buffer/actor_saved_transitions": actor_stats["saved_transitions"],
            "buffer/actor_saved_intervention_samples": actor_stats["saved_intervention_samples"],
            "buffer/actor_saved_intervention_time_s": actor_stats["saved_intervention_time_s"],
            "human_intervention/total_time_s": actor_stats["saved_intervention_time_s"],
            "buffer/last_trajectory_length": len(trajectory),
            "buffer/last_trajectory_success": int(success),
            "buffer/last_trajectory_intervention": int(trajectory_had_intvn),
            "buffer/last_trajectory_intervention_samples": len(intervention_samples),
            "buffer/last_trajectory_intervention_time_s": trajectory_intervention_time_s,
            "human_intervention/last_trajectory_time_s": trajectory_intervention_time_s,
        }
        try:
            client.request("send-stats", stats_payload)
        except Exception as exc:
            print(f"actor stats log failed: {exc}", flush=True)
        trajectory_index += 1
        print(
            f"saved trajectory step={step} len={len(trajectory)} label={label} "
            f"intervention={trajectory_had_intvn}",
            flush=True,
        )
        return len(trajectory)

    def dump_actor_buffers(step, final=False):
        nonlocal last_buffer_dump_step
        # Completed trajectories are flushed immediately at episode end.
        last_buffer_dump_step = step

    def sync_actor_data(step, force=False):
        nonlocal last_actor_sync_step
        if not force and step - last_actor_sync_step < actor_sync_period:
            return
        try:
            if client.update():
                last_actor_sync_step = step
        except Exception as exc:
            print(f"actor datastore sync failed: {exc}", flush=True)


    maybe_load_grasp_causal_bias(start_step, force=True)

    manual_labeler = ManualEpisodeLabeler()
    manual_labeler.start()

    obs, _ = env.reset()
    action_filter.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    trajectory_had_intervention = False
    intervention_count = 0
    intervention_steps = 0
    intervention_time_s = 0.0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            last_seen_step = step
            timer.tick("total")
            apply_pending_network()
            maybe_load_grasp_causal_bias(step)

            manual_label = manual_labeler.consume()
            if manual_label is not None:
                prepare_manual_reset(env, action_filter)
                reward = 1.0 if manual_label == "success" else 0.0
                info = {
                    "succeed": manual_label == "success",
                    "episode": {
                        "intervention_count": intervention_count,
                        "intervention_steps": intervention_steps,
                        "intervention_time_s": intervention_time_s,
                    },
                }
                zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
                transition = dict(
                    observations=obs,
                    actions=zero_action,
                    next_observations=obs,
                    rewards=reward,
                    causal_rewards=reward,
                    masks=0.0,
                    dones=True,
                    infos=copy.deepcopy(info),
                )
                if config.setup_mode in ("single-arm-learned-gripper", "dual-arm-learned-gripper"):
                    transition["grasp_penalty"] = 0.0
                transition = mark_transition_intervention(transition, False, trajectory_had_intervention)
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                dump_trajectory(current_trajectory, step)
                current_trajectory = []
                sync_actor_data(step, force=True)
                running_return += reward
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["intervention_time_s"] = intervention_time_s
                print(f"manual {manual_label} marked; resetting environment", flush=True)
                print("[actor] calling env.reset", flush=True)
                obs, _ = env.reset()
                print("[actor] env.reset returned", flush=True)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                intervention_time_s = 0.0
                already_intervened = False
                trajectory_had_intervention = False
                action_filter.reset()
                timer.tock("total")
                continue

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    sample_action_kwargs = {}
                    if use_grasp_causal_bias and grasp_causal_bias is not None:
                        sample_action_kwargs.update(
                            {
                                "grasp_causal_bias": grasp_causal_bias,
                                "grasp_causal_beta": grasp_causal_beta,
                            }
                        )
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                        **sample_action_kwargs,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    actions = action_filter(actions)

            # Step environment
            with timer.context("step_env"):
                step_env_start_time = time.time()
                next_obs, reward, done, truncated, info = env.step(actions)
                step_env_time_s = time.time() - step_env_start_time
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    print("\n\n INTERVENING: ", actions)
                    intervention_steps += 1
                    intervention_time_s += step_env_time_s
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                    current_intervention = True
                    trajectory_had_intervention = True
                else:
                    actions = info.pop("executed_action", actions)
                    already_intervened = False
                    current_intervention = False

                manual_label = manual_labeler.consume()
                if manual_label == "success":
                    prepare_manual_reset(env, action_filter)
                    reward = 1.0
                    done = True
                    truncated = False
                    info["succeed"] = True
                    info.setdefault("episode", {})
                    print("manual success marked; resetting environment")
                elif manual_label == "failure":
                    prepare_manual_reset(env, action_filter)
                    reward = 0.0
                    done = True
                    truncated = False
                    info["succeed"] = False
                    info.setdefault("episode", {})
                    print("manual failure marked; resetting environment")

                terminal = bool(done or truncated)
                info["intervention_duration_s"] = step_env_time_s if current_intervention else 0.0
                info["episode_intervention_time_s"] = intervention_time_s
                if terminal:
                    info.setdefault("succeed", bool(reward > 0.0 and not truncated))
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["intervention_time_s"] = intervention_time_s
                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    causal_rewards=reward,
                    masks=1.0-float(done),
                    dones=terminal,
                    infos=copy.deepcopy(info),
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty'] = info['grasp_penalty']

                transition = mark_transition_intervention(
                    transition, current_intervention, trajectory_had_intervention
                )
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                obs = next_obs
                sync_actor_data(step)

                if done or truncated:
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["intervention_time_s"] = intervention_time_s
                    pbar.set_description(f"last return: {running_return}")
                    dump_trajectory(current_trajectory, step)
                    current_trajectory = []
                    sync_actor_data(step, force=True)
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    intervention_time_s = 0.0
                    already_intervened = False
                    trajectory_had_intervention = False
                    prepare_manual_reset(env, action_filter)
                    print("[actor] calling env.reset", flush=True)
                    obs, _ = env.reset()
                    print("[actor] env.reset returned", flush=True)
                    action_filter.reset()

            if config.buffer_period > 0 and step - last_buffer_dump_step >= config.buffer_period:
                dump_actor_buffers(step)

            timer.tock("total")

            # Keep the real-time actor loop free of blocking trainer requests.
            # Episode stats are sent after reset, where a short trainer timeout is less harmful.
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        try:
            dump_actor_buffers(last_seen_step, final=True)
        except Exception as exc:
            print(f"final buffer dump failed: {exc}")
        try:
            sync_actor_data(last_seen_step, force=True)
        except Exception as exc:
            print(f"final actor sync failed: {exc}")
        try:
            manual_labeler.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception as exc:
            print(f"actor cleanup failed: {exc}")


##############################################################################


def learner(
    rng,
    agent,
    replay_buffer,
    demo_buffer,
    causal_demo_buffer=None,
    causal_success_intervention_buffer=None,
    wandb_logger=None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = get_resume_step(FLAGS.checkpoint_path) if FLAGS.resume_training else 0
    step = start_step
    causal_entropy_policy_success_streak = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        nonlocal causal_entropy_policy_success_streak
        assert type == "send-stats", f"Invalid request type: {type}"
        payload = dict(payload)
        if (
            "buffer/last_trajectory_success" in payload
            and "buffer/last_trajectory_intervention" in payload
        ):
            last_success = bool(payload["buffer/last_trajectory_success"])
            last_intervention = bool(payload["buffer/last_trajectory_intervention"])
            if last_success and not last_intervention:
                causal_entropy_policy_success_streak += 1
            else:
                causal_entropy_policy_success_streak = 0
        payload["causal_entropy/policy_success_streak"] = (
            causal_entropy_policy_success_streak
        )
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Pretrain the causal model from collected demos before actor communication
    # begins. Once policy learning starts, keep refining it from interventions.
    causal_mask_enabled = bool(getattr(config, "causal_mask_enabled", True))
    causal_mask_threshold = float(getattr(config, "causal_mask_threshold", 0.1))
    causal_pretrain_steps = int(
        getattr(config, "causal_pretrain_steps", 1000)
    )
    causal_model_update_interval = int(
        getattr(config, "causal_model_update_interval", 1000)
    )
    causal_model_train_steps = int(getattr(config, "causal_model_train_steps", 200))
    causal_model_checkpoint_enabled = bool(
        getattr(config, "causal_model_checkpoint_enabled", True)
    )
    causal_model_checkpoint_subdir = str(
        getattr(config, "causal_model_checkpoint_subdir", "causal_model")
    )
    causal_model_skip_pretrain_if_loaded = bool(
        getattr(config, "causal_model_skip_pretrain_if_loaded", True)
    )
    causal_model_checkpoint_dir = (
        os.path.join(os.path.abspath(FLAGS.checkpoint_path), causal_model_checkpoint_subdir)
        if FLAGS.checkpoint_path is not None and causal_model_checkpoint_enabled
        else None
    )
    causal_model_online_demo_ratio = max(
        0.0, float(getattr(config, "causal_model_online_demo_ratio", 0.8))
    )
    causal_model_online_success_intervention_ratio = max(
        0.0,
        float(getattr(config, "causal_model_online_success_intervention_ratio", 0.2)),
    )
    causal_action_samples = int(getattr(config, "causal_action_samples", 64))
    causal_mask_inplace = bool(getattr(config, "causal_mask_inplace", False))
    causal_mask_online_batch = bool(getattr(config, "causal_mask_online_batch", True))
    causal_mask_demo_batch = bool(getattr(config, "causal_mask_demo_batch", True))
    causal_mask_ratio = max(
        0.0, min(1.0, float(getattr(config, "causal_mask_ratio", 1.0)))
    )
    causal_mask_baseline = str(getattr(config, "causal_mask_baseline", "mean")).lower()
    if causal_mask_baseline not in ("mean", "zero"):
        raise ValueError(f"Unsupported causal_mask_baseline: {causal_mask_baseline}")
    causal_mask_strength = max(
        0.0, min(1.0, float(getattr(config, "causal_mask_strength", 0.1)))
    )
    causal_mask_max_ratio = max(
        0.0, min(1.0, float(getattr(config, "causal_mask_max_ratio", 0.1)))
    )
    causal_mask_visual_only = bool(getattr(config, "causal_mask_visual_only", True))
    causal_mask_proprio_latent_dim = max(0, int(getattr(config, "causal_mask_proprio_latent_dim", 64)))
    causal_state = None
    causal_model_ready = False
    causal_model_loss = None
    causal_model_demo_loss = None
    causal_model_demo_val_nll = None
    causal_model_demo_val_mse = None
    causal_pretrain_completed = False
    causal_pretrain_val_mse = None
    causal_model_last_source_info = {}
    causal_checkpoint_loaded = False
    causal_checkpoint_step = None
    causal_checkpoint_last_save_step = None
    causal_mask_samples = 0
    causal_mask_samples_inplace = 0
    causal_mask_last_metrics = {}

    demo_frac = 0.5
    demo_bs = int(round(config.batch_size * demo_frac))
    demo_bs = max(1, min(config.batch_size - 1, demo_bs))
    online_bs = config.batch_size - demo_bs
    print_green(
        f"Initial sampling ratio: demo={config.batch_size}/{config.batch_size} (1.00), online=0/{config.batch_size} (0.00)"
    )
    print_green(
        f"Will switch to mixed sampling after online replay reaches {config.training_starts} transitions: "
        f"demo={demo_bs}/{config.batch_size}, online={online_bs}/{config.batch_size}"
    )

    replay_iterator = None
    using_online_replay = False

    demo_only_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": demo_bs,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    def next_train_batch(cmi_key_online=None, cmi_key_demo=None):
        nonlocal replay_iterator, using_online_replay
        nonlocal causal_mask_samples, causal_mask_samples_inplace, causal_mask_last_metrics
        online_ready = len(replay_buffer) >= config.training_starts and online_bs > 0
        if online_ready:
            if replay_iterator is None:
                replay_iterator = replay_buffer.get_iterator(
                    sample_args={
                        "batch_size": online_bs,
                        "pack_obs_and_next_obs": True,
                    },
                    device=sharding.replicate(),
                )
            if not using_online_replay:
                using_online_replay = True
                print_green(
                    f"Online replay ready: {len(replay_buffer)} transitions. Switching to mixed demo/online batches."
                )
            batch_online = next(replay_iterator)
            batch_demo = next(demo_iterator)
            if (
                causal_mask_enabled
                and causal_model_ready
                and cmi_key_online is not None
                and cmi_key_demo is not None
            ):
                # We need separate current/next images before extracting latents.
                batch_online = _unpack(batch_online)
                batch_demo = _unpack(batch_demo)
                z_online = agent.get_encoded_features(batch_online["observations"])
                z_demo = agent.get_encoded_features(batch_demo["observations"])
                z_next_online = agent.get_encoded_features(
                    batch_online["next_observations"]
                )
                z_next_demo = agent.get_encoded_features(batch_demo["next_observations"])
                paired_size = min(z_online.shape[0], z_demo.shape[0])
                paired_size = min(
                    paired_size, int(round(paired_size * causal_mask_ratio))
                )
                if paired_size <= 0:
                    return concat_batches(batch_online, batch_demo, axis=0)
                paired_online = jax.tree.map(lambda x: x[:paired_size], batch_online)
                paired_demo = jax.tree.map(lambda x: x[:paired_size], batch_demo)
                paired_z_online = z_online[:paired_size]
                paired_z_demo = z_demo[:paired_size]
                paired_z_next_online = z_next_online[:paired_size]
                paired_z_next_demo = z_next_demo[:paired_size]
                action_dim = batch_online["actions"].shape[-1]
                masks_online, scores_online = compute_cmi_masks_jax(
                    causal_state.params,
                    causal_state.apply_fn,
                    paired_z_online,
                    action_dim,
                    cmi_key_online,
                    threshold=causal_mask_threshold,
                    n_action_samples=causal_action_samples,
                    return_scores=True,
                )
                masks_demo, scores_demo = compute_cmi_masks_jax(
                    causal_state.params,
                    causal_state.apply_fn,
                    paired_z_demo,
                    action_dim,
                    cmi_key_demo,
                    threshold=causal_mask_threshold,
                    n_action_samples=causal_action_samples,
                    return_scores=True,
                )
                candidate_mask = jnp.logical_and(masks_online, masks_demo)
                augmentation_scores = jnp.maximum(scores_online, scores_demo)
                if causal_mask_visual_only and causal_mask_proprio_latent_dim > 0:
                    visual_latent_dim = max(
                        0, candidate_mask.shape[-1] - causal_mask_proprio_latent_dim
                    )
                    visual_mask = jnp.arange(candidate_mask.shape[-1]) < visual_latent_dim
                    candidate_mask = jnp.logical_and(candidate_mask, visual_mask)
                    ranked_visual_mask = select_lowest_score_mask(
                        augmentation_scores[:, :visual_latent_dim], causal_mask_max_ratio
                    )
                    ranked_augmentation_mask = jnp.zeros_like(candidate_mask, dtype=bool)
                    ranked_augmentation_mask = ranked_augmentation_mask.at[
                        :, :visual_latent_dim
                    ].set(
                        ranked_visual_mask
                    )
                else:
                    ranked_augmentation_mask = select_lowest_score_mask(
                        augmentation_scores, causal_mask_max_ratio
                    )
                visual_candidate_ratio = jnp.mean(candidate_mask.astype(jnp.float32))
                selected_latent_mask = ranked_augmentation_mask
                selected_count = jnp.maximum(
                    jnp.sum(selected_latent_mask.astype(jnp.float32)), 1.0
                )
                common_mask_metrics = {
                    "causal_mask/cmi_online_mean": jnp.mean(scores_online),
                    "causal_mask/visual_candidate_ratio": visual_candidate_ratio,
                    "causal_mask/selected_latent_ratio": jnp.mean(
                        selected_latent_mask.astype(jnp.float32)
                    ),
                    "causal_mask/selected_cmi_mean": jnp.sum(
                        jnp.where(selected_latent_mask, augmentation_scores, 0.0)
                    )
                    / selected_count,
                }
                if causal_mask_inplace:
                    online_mask = jnp.zeros_like(z_online, dtype=bool)
                    demo_mask = jnp.zeros_like(z_demo, dtype=bool)
                    if causal_mask_online_batch:
                        online_mask = online_mask.at[:paired_size].set(
                            selected_latent_mask
                        )
                    if causal_mask_demo_batch:
                        demo_mask = demo_mask.at[:paired_size].set(
                            selected_latent_mask
                        )
                    augmented_z_online = augment_latent_features(
                        z_online,
                        online_mask,
                        mask_baseline=causal_mask_baseline,
                        mask_strength=causal_mask_strength,
                    )
                    augmented_z_next_online = augment_latent_features(
                        z_next_online,
                        online_mask,
                        mask_baseline=causal_mask_baseline,
                        mask_strength=causal_mask_strength,
                    )
                    augmented_z_demo_full = augment_latent_features(
                        z_demo,
                        demo_mask,
                        mask_baseline=causal_mask_baseline,
                        mask_strength=causal_mask_strength,
                    )
                    augmented_z_next_demo = augment_latent_features(
                        z_next_demo,
                        demo_mask,
                        mask_baseline=causal_mask_baseline,
                        mask_strength=causal_mask_strength,
                    )
                    masked_rows = (
                        int(causal_mask_online_batch) + int(causal_mask_demo_batch)
                    ) * paired_size
                    active_mask_sources = max(
                        1,
                        int(causal_mask_online_batch) + int(causal_mask_demo_batch),
                    )
                    online_delta_l1 = jnp.mean(
                        jnp.abs(augmented_z_online - z_online)
                    )
                    demo_delta_l1 = jnp.mean(
                        jnp.abs(augmented_z_demo_full - z_demo)
                    )
                    causal_mask_samples_inplace += masked_rows
                    causal_mask_last_metrics = {
                        **common_mask_metrics,
                        "causal_mask/counterfactual_fraction_last_batch": 0.0,
                        "causal_mask/masked_fraction_last_batch": masked_rows
                        / (batch_online["actions"].shape[0] + batch_demo["actions"].shape[0]),
                        "causal_mask/masked_online_fraction_last_batch": (
                            paired_size / batch_online["actions"].shape[0]
                            if causal_mask_online_batch
                            else 0.0
                        ),
                        "causal_mask/masked_demo_fraction_last_batch": (
                            paired_size / batch_demo["actions"].shape[0]
                            if causal_mask_demo_batch
                            else 0.0
                        ),
                        "causal_mask/inplace_mask_active": float(masked_rows > 0),
                        "causal_mask/latent_delta_l1": (
                            int(causal_mask_online_batch) * online_delta_l1
                            + int(causal_mask_demo_batch) * demo_delta_l1
                        )
                        / active_mask_sources,
                    }
                    batch_online = add_encoded_features(
                        batch_online,
                        augmented_z_online,
                        augmented_z_next_online,
                        use_encoded=causal_mask_online_batch,
                    )
                    batch_demo = add_encoded_features(
                        batch_demo,
                        augmented_z_demo_full,
                        augmented_z_next_demo,
                        use_encoded=causal_mask_demo_batch,
                    )
                    return concat_batches(batch_online, batch_demo, axis=0)

                causal_mask_samples += paired_size
                augmented_z_demo = augment_latent_features(
                    paired_z_demo,
                    selected_latent_mask,
                    mask_baseline=causal_mask_baseline,
                    mask_strength=causal_mask_strength,
                )
                causal_mask_last_metrics = {
                    **common_mask_metrics,
                    "causal_mask/counterfactual_fraction_last_batch": paired_size
                    / (batch_online["actions"].shape[0] + batch_demo["actions"].shape[0] + paired_size),
                    "causal_mask/masked_fraction_last_batch": 0.0,
                    "causal_mask/inplace_mask_active": 0.0,
                    "causal_mask/latent_delta_l1": jnp.mean(jnp.abs(augmented_z_demo - paired_z_demo)),
                }
                batch_counterfactual = causal_mask_counterfactuals(
                    paired_demo,
                    paired_z_demo,
                    paired_z_next_demo,
                    selected_latent_mask=selected_latent_mask,
                    mask_baseline=causal_mask_baseline,
                    mask_strength=causal_mask_strength,
                )
                batch_online = add_encoded_features(
                    batch_online,
                    jnp.zeros_like(z_online),
                    jnp.zeros_like(z_next_online),
                    use_encoded=False,
                )
                batch_demo = add_encoded_features(
                    batch_demo,
                    jnp.zeros_like(z_demo),
                    jnp.zeros_like(z_next_demo),
                    use_encoded=False,
                )
                batch = concat_batches(batch_online, batch_demo, axis=0)
                return concat_batches(batch, batch_counterfactual, axis=0)
            return concat_batches(batch_online, batch_demo, axis=0)
        return next(demo_only_iterator)
    # replay_iterator = replay_buffer.get_iterator(
    #     sample_args={
    #         "batch_size": config.batch_size // 2,
    #         "pack_obs_and_next_obs": True,
    #     },
    #     device=sharding.replicate(),
    # )
    # demo_iterator = demo_buffer.get_iterator(
    #     sample_args={
    #         "batch_size": config.batch_size // 2,
    #         "pack_obs_and_next_obs": True,
    #     },
    #     device=sharding.replicate(),
    # )

    # Track causal-model update time separately from policy optimization.
    timer = Timer()

    def get_causal_demo_source_buffer():
        if causal_demo_buffer is not None and len(causal_demo_buffer) > 0:
            return causal_demo_buffer
        return demo_buffer

    def initialize_causal_state_fn(base_causal_buffer):
        nonlocal causal_state, rng
        if causal_state is not None:
            return True
        if len(base_causal_buffer) == 0:
            return False
        rng, causal_init_rng = jax.random.split(rng)
        dummy_batch = base_causal_buffer.sample(1)
        z_dummy = agent.get_encoded_features(dummy_batch["observations"])
        causal_state = create_causal_state(
            causal_init_rng,
            state_dim=z_dummy.shape[-1],
            action_dim=dummy_batch["actions"].shape[-1],
        )
        print_green(
            "Initialized causal model: "
            f"latent_dim={z_dummy.shape[-1]}, action_dim={dummy_batch['actions'].shape[-1]}"
        )
        return True

    def save_causal_checkpoint(step, *, reason):
        nonlocal causal_checkpoint_last_save_step
        if causal_model_checkpoint_dir is None or causal_state is None:
            return False
        os.makedirs(causal_model_checkpoint_dir, exist_ok=True)
        save_step = int(step)
        checkpoints.save_checkpoint(
            causal_model_checkpoint_dir,
            causal_state,
            step=save_step,
            keep=20,
            overwrite=True,
        )
        causal_checkpoint_last_save_step = save_step
        atomic_pickle_dump(
            {
                "step": save_step,
                "reason": reason,
                "pretrain_completed": bool(causal_pretrain_completed),
                "pretrain_steps": int(causal_pretrain_steps),
                "online_update_interval": int(causal_model_update_interval),
                "online_train_steps": int(causal_model_train_steps),
                "demo_val_nll": (
                    None
                    if causal_model_demo_val_nll is None
                    else float(np.asarray(causal_model_demo_val_nll))
                ),
                "demo_val_mse": (
                    None
                    if causal_model_demo_val_mse is None
                    else float(np.asarray(causal_model_demo_val_mse))
                ),
            },
            os.path.join(causal_model_checkpoint_dir, "metadata.pkl"),
        )
        print_green(
            f"Saved causal model checkpoint step={save_step} "
            f"reason={reason} path={causal_model_checkpoint_dir}"
        )
        return True

    def maybe_load_causal_checkpoint(base_causal_buffer):
        nonlocal causal_state, causal_model_ready, causal_pretrain_completed
        nonlocal causal_model_demo_val_nll, causal_model_demo_val_mse
        nonlocal causal_checkpoint_loaded, causal_checkpoint_step
        nonlocal causal_model_last_source_info
        if causal_model_checkpoint_dir is None:
            return False
        latest_ckpt = checkpoints.latest_checkpoint(causal_model_checkpoint_dir)
        if latest_ckpt is None:
            return False
        if not initialize_causal_state_fn(base_causal_buffer):
            return False
        causal_state = checkpoints.restore_checkpoint(
            causal_model_checkpoint_dir,
            causal_state,
        )
        causal_model_ready = True
        causal_pretrain_completed = True
        causal_checkpoint_loaded = True
        causal_checkpoint_step = latest_step_from_path(latest_ckpt, "checkpoint_")
        demo_val_metrics = evaluate_causal_model(
            causal_state,
            base_causal_buffer,
            encoder_fn=agent.get_encoded_features,
            batch_size=config.batch_size,
            sharding=sharding.replicate(),
        )
        causal_model_demo_val_nll = demo_val_metrics.get("nll")
        causal_model_demo_val_mse = demo_val_metrics.get("mse")
        causal_model_last_source_info = {
            "stage": -1.0,
            "old_demo_size": len(base_causal_buffer),
            "success_intervention_size": (
                0
                if causal_success_intervention_buffer is None
                else len(causal_success_intervention_buffer)
            ),
            "loaded_checkpoint": 1.0,
        }
        print_green(
            "Loaded causal model checkpoint: "
            f"{latest_ckpt}, step={causal_checkpoint_step}"
        )
        return True

    def update_causal_model(*, num_steps=None, stage="online"):
        nonlocal causal_state, causal_model_ready, causal_model_loss
        nonlocal causal_model_demo_loss
        nonlocal causal_model_demo_val_nll, causal_model_demo_val_mse
        nonlocal causal_pretrain_completed, causal_pretrain_val_mse
        nonlocal causal_model_last_source_info
        nonlocal rng
        train_steps = causal_model_train_steps if num_steps is None else int(num_steps)
        base_causal_buffer = get_causal_demo_source_buffer()
        if not causal_mask_enabled or len(base_causal_buffer) == 0 or train_steps <= 0:
            return False
        if not initialize_causal_state_fn(base_causal_buffer):
            return False
        causal_model_demo_loss = None
        causal_model_demo_val_nll = None
        causal_model_demo_val_mse = None
        causal_model_last_source_info = {
            "stage": 0.0 if stage == "pretrain" else 1.0,
            "old_demo_size": len(base_causal_buffer),
            "success_intervention_size": (
                0
                if causal_success_intervention_buffer is None
                else len(causal_success_intervention_buffer)
            ),
        }
        with timer.context("causal_model_update"):
            use_success_interventions = (
                stage != "pretrain"
                and causal_success_intervention_buffer is not None
                and len(causal_success_intervention_buffer) > 0
                and causal_model_online_success_intervention_ratio > 0.0
            )
            if use_success_interventions:
                causal_state, model_info = train_causal_model_from_buffers(
                    causal_state,
                    buffers=[base_causal_buffer, causal_success_intervention_buffer],
                    ratios=[
                        causal_model_online_demo_ratio,
                        causal_model_online_success_intervention_ratio,
                    ],
                    encoder_fn=agent.get_encoded_features,
                    num_steps=train_steps,
                    batch_size=config.batch_size,
                    sharding=sharding.replicate(),
                    return_info=True,
                )
                causal_model_last_source_info.update(
                    {
                        "using_success_interventions": 1.0,
                        "target_old_demo_ratio": causal_model_online_demo_ratio,
                        "target_success_intervention_ratio": (
                            causal_model_online_success_intervention_ratio
                        ),
                    }
                )
            else:
                causal_state, model_info = train_causal_model(
                    causal_state,
                    base_causal_buffer,
                    encoder_fn=agent.get_encoded_features,
                    num_steps=train_steps,
                    batch_size=config.batch_size,
                    sharding=sharding.replicate(),
                    return_info=True,
                )
                causal_model_last_source_info.update(
                    {
                        "using_success_interventions": 0.0,
                        "target_old_demo_ratio": 1.0,
                        "target_success_intervention_ratio": 0.0,
                    }
                )
            for key, value in model_info.items():
                if key == "loss":
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    causal_model_last_source_info[f"train_{key}"] = float(value)
            causal_model_demo_loss = model_info.get("loss")
            demo_val_metrics = evaluate_causal_model(
                causal_state,
                base_causal_buffer,
                encoder_fn=agent.get_encoded_features,
                batch_size=config.batch_size,
                sharding=sharding.replicate(),
            )
            causal_model_demo_val_nll = demo_val_metrics.get("nll")
            causal_model_demo_val_mse = demo_val_metrics.get("mse")
        causal_model_loss = causal_model_demo_loss
        causal_model_ready = True
        if stage == "pretrain":
            causal_pretrain_completed = True
            causal_pretrain_val_mse = causal_model_demo_val_mse
        return True

    if causal_mask_enabled and len(demo_buffer) > 0:
        causal_demo_source_buffer = get_causal_demo_source_buffer()
        print_green(
            "Causal model training source: "
            "pretrain=old demos; online=old demos + successful intervention transitions"
        )
        loaded_causal_ckpt = maybe_load_causal_checkpoint(causal_demo_source_buffer)
        if loaded_causal_ckpt and causal_model_skip_pretrain_if_loaded:
            print_green(
                "Using loaded causal model checkpoint. "
                "Skipping repeated demo pretraining."
            )
        else:
            print_green(
                "Pretraining causal model before policy learning: "
                f"steps={causal_pretrain_steps}, demos={len(causal_demo_source_buffer)}"
            )
            updated = update_causal_model(
                num_steps=causal_pretrain_steps,
                stage="pretrain",
            )
            if updated:
                save_causal_checkpoint(start_step, reason="pretrain")
        if causal_pretrain_completed:
            print_green("Causal model pretraining completed. Starting policy learning.")
        else:
            print_green("Causal model pretraining skipped. Starting policy learning.")
    elif causal_mask_enabled:
        print_green(
            "Causal model pretraining skipped because no demos are available. "
            "It will initialize after intervention data becomes available."
        )

    # Start actor communication only after the causal-model pretraining stage.
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    intervention_store = (
        SuccessfulInterventionFanoutDataStore(
            demo_buffer,
            causal_success_intervention_buffer,
        )
        if causal_success_intervention_buffer is not None
        else demo_buffer
    )
    server.register_data_store("actor_env_intvn", intervention_store)
    server.start(threaded=True)
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    def sample_train_batch():
        nonlocal rng
        if causal_mask_enabled and causal_model_ready:
            rng, cmi_key_online, cmi_key_demo = jax.random.split(rng, 3)
            return next_train_batch(cmi_key_online, cmi_key_demo)
        return next_train_batch()
    
    causal_weights = None
    causal_weight_failures = 0
    causal_update_interval = int(getattr(config, "causal_update_interval", 500))
    causal_sample_size = int(getattr(config, "causal_sample_size", 500))
    causal_action_indices = getattr(config, "causal_action_indices", None)
    policy_action_dim = getattr(config, "policy_action_dim", None)
    if causal_action_indices is None and config.setup_mode in (
        "single-arm-fixed-gripper",
        "dual-arm-fixed-gripper",
    ):
        if policy_action_dim is not None:
            causal_action_indices = list(range(int(policy_action_dim)))
    causal_sampling_strategy = str(getattr(config, "causal_sampling_strategy", "demo_only")).lower()
    causal_phase_mixed = causal_sampling_strategy in (
        "phase_mixed",
        "stage_mixed",
        "intervention_aware",
        "mixed_phase",
    )
    causal_source_info = {}
    use_grasp_causal_bias = FLAGS.use_causal_entropy and bool(
        getattr(config, "use_grasp_causal_bias", False)
    )
    grasp_causal_beta = float(getattr(config, "grasp_causal_beta", 0.05))
    grasp_causal_update_interval = int(
        getattr(config, "grasp_causal_update_interval", causal_update_interval)
    )
    grasp_causal_sample_size = int(
        getattr(config, "grasp_causal_sample_size", causal_sample_size)
    )
    grasp_causal_bias = None
    grasp_causal_bias_info = {}
    grasp_causal_bias_attempts = 0
    grasp_causal_bias_failures = 0
    grasp_causal_bias_last_step = None
    grasp_causal_bias_file = (
        os.path.join(FLAGS.checkpoint_path, "grasp_causal_bias.pkl")
        if FLAGS.checkpoint_path
        else None
    )
    
    if isinstance(agent, SACAgent):
        causal_agent_type = "continuous_sac"
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        causal_agent_type = "hybrid_sac"
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    if use_grasp_causal_bias and isinstance(agent, SACAgent):
        print_green("grasp causal bias is disabled for continuous fixed-gripper SAC.")
        use_grasp_causal_bias = False
    print_green(
        "Causal entropy setup: "
        f"agent={causal_agent_type}, setup_mode={config.setup_mode}, "
        f"policy_action_dim={policy_action_dim}, action_indices={causal_action_indices}"
    )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # Periodically compute causal weights
        # demo_only: keep the original conservative demo/intervention-only graph.
        # phase_mixed: use current-stage online anchors + interventions + matched demo samples.
        if (
            FLAGS.use_causal_entropy
            and step > 0
            and step % causal_update_interval == 0
            and (
                len(demo_buffer) >= causal_sample_size
                or (causal_phase_mixed and len(demo_buffer) > 0)
            )
        ):
            with timer.context("causal_weight_computation"):
                try:
                    if causal_phase_mixed:
                        from serl_launcher.utils.causal_weight import get_sa2r_weight_from_batch

                        causal_batch, causal_source_info = build_phase_mixed_causal_batch(
                            config=config,
                            checkpoint_path=FLAGS.checkpoint_path,
                            demo_buffer=demo_buffer,
                            sample_size=causal_sample_size,
                            step=step,
                        )
                        if causal_batch is None or causal_batch["actions"].shape[0] <= 0:
                            raise RuntimeError("phase_mixed causal sampler produced an empty batch")
                        causal_weights, _ = get_sa2r_weight_from_batch(
                            batch=causal_batch,
                            causal_method='DirectLiNGAM',
                            action_indices=causal_action_indices,
                        )
                    else:
                        from serl_launcher.utils.causal_weight import get_sa2r_weight_demo_only

                        causal_source_info = {
                            "sample_demo": causal_sample_size,
                            "sample_total": causal_sample_size,
                        }
                        causal_weights, _ = get_sa2r_weight_demo_only(
                            demo_memory=demo_buffer,
                            sample_size=causal_sample_size,
                            causal_method='DirectLiNGAM',
                            action_indices=causal_action_indices,
                        )
                except Exception as e:
                    causal_weight_failures += 1
                    print(f"Failed to compute causal weights at step {step}: {repr(e)}", flush=True)

        if (
            use_grasp_causal_bias
            and not isinstance(agent, SACAgent)
            and step > 0
            and step % grasp_causal_update_interval == 0
            and len(demo_buffer) > 0
        ):
            with timer.context("grasp_causal_bias_computation"):
                grasp_causal_bias_attempts += 1
                try:
                    from serl_launcher.utils.causal_weight import get_obs2grasp_bias_demo_only

                    grasp_causal_bias, grasp_causal_bias_info = get_obs2grasp_bias_demo_only(
                        demo_memory=demo_buffer,
                        sample_size=min(grasp_causal_sample_size, len(demo_buffer)),
                        causal_method="DirectLiNGAM",
                    )
                    grasp_causal_bias_last_step = int(step)
                    if grasp_causal_bias_file is not None:
                        atomic_pickle_dump(
                            {
                                "bias": normalize_grasp_causal_payload(grasp_causal_bias),
                                "step": grasp_causal_bias_last_step,
                                "beta": grasp_causal_beta,
                                "info": grasp_causal_bias_info,
                            },
                            grasp_causal_bias_file,
                        )
                    if wandb_logger:
                        bias_array = summarize_grasp_causal_payload(grasp_causal_bias)
                        bias_log = {
                            "grasp_causal_bias/enabled": int(use_grasp_causal_bias),
                            "grasp_causal_bias/active": 1,
                            "grasp_causal_bias/beta": grasp_causal_beta,
                            "grasp_causal_bias/update_interval": grasp_causal_update_interval,
                            "grasp_causal_bias/sample_size": grasp_causal_sample_size,
                            "grasp_causal_bias/last_step": grasp_causal_bias_last_step,
                            "grasp_causal_bias/compute_success": 1,
                            "grasp_causal_bias/compute_attempts": grasp_causal_bias_attempts,
                            "grasp_causal_bias/compute_failures": grasp_causal_bias_failures,
                            "grasp_causal_bias/open": float(bias_array[0]),
                            "grasp_causal_bias/stay": float(bias_array[1]),
                            "grasp_causal_bias/close": float(bias_array[2]),
                        }
                        for key, value in grasp_causal_bias_info.items():
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                bias_log[f"grasp_causal_bias/{key}"] = float(value)
                        wandb_logger.log(bias_log, step=step)
                except Exception as exc:
                    grasp_causal_bias_failures += 1
                    print(f"Failed to compute grasp causal bias at step {step}: {repr(exc)}", flush=True)
                    if wandb_logger:
                        wandb_logger.log(
                            {
                                "grasp_causal_bias/enabled": int(use_grasp_causal_bias),
                                "grasp_causal_bias/compute_success": 0,
                                "grasp_causal_bias/compute_attempts": grasp_causal_bias_attempts,
                                "grasp_causal_bias/compute_failures": grasp_causal_bias_failures,
                            },
                            step=step,
                        )

        if (
            causal_mask_enabled
            and step > 0
            and causal_model_update_interval > 0
            and step % causal_model_update_interval == 0
            and len(demo_buffer) > 0
        ):
            causal_demo_source_buffer = get_causal_demo_source_buffer()
            print_green(
                "Updating causal model during policy learning: "
                f"steps={causal_model_train_steps}, "
                f"old_demo={len(causal_demo_source_buffer)}, "
                "success_intervention="
                f"{0 if causal_success_intervention_buffer is None else len(causal_success_intervention_buffer)}"
            )
            update_causal_model(stage="online")

        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_train_batch"):
                batch = sample_train_batch()
                causal_update_kwargs = {}
                if causal_weights is not None:
                    causal_update_kwargs["causal_weights"] = causal_weights
                if use_grasp_causal_bias and grasp_causal_bias is not None:
                    causal_update_kwargs["grasp_causal_bias"] = grasp_causal_bias
                    causal_update_kwargs["grasp_causal_beta"] = grasp_causal_beta

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                    **causal_update_kwargs,
                )

        with timer.context("train"):
            batch = sample_train_batch()
            causal_update_kwargs = {}
            if causal_weights is not None:
                # Pass causal weights outside the batch so batch-shape checks stay valid.
                causal_update_kwargs["causal_weights"] = causal_weights
            if use_grasp_causal_bias and grasp_causal_bias is not None:
                causal_update_kwargs["grasp_causal_bias"] = grasp_causal_bias
                causal_update_kwargs["grasp_causal_beta"] = grasp_causal_beta
                
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
                **causal_update_kwargs,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            causal_log = {
                "causal_entropy/active": int(causal_weights is not None),
                "causal_entropy/policy_success_streak": (
                    causal_entropy_policy_success_streak
                ),
                "causal_entropy/weight_compute_failures": causal_weight_failures,
                "causal_mask/causal_model_ready": int(causal_model_ready),
                "causal_mask/inplace_mask": int(causal_mask_inplace),
                "causal_mask/mask_online_batch": int(causal_mask_online_batch),
                "causal_mask/mask_demo_batch": int(causal_mask_demo_batch),
                "causal_mask/causal_ckpt_loaded": int(causal_checkpoint_loaded),
            }
            if causal_checkpoint_step is not None:
                causal_log["causal_mask/causal_ckpt_loaded_step"] = int(
                    causal_checkpoint_step
                )
            if causal_checkpoint_last_save_step is not None:
                causal_log["causal_mask/causal_ckpt_last_save_step"] = int(
                    causal_checkpoint_last_save_step
                )
            if causal_model_loss is not None:
                causal_log["causal_mask/causal_model_loss"] = float(
                    np.asarray(causal_model_loss)
                )
            if causal_model_demo_val_nll is not None:
                causal_log["causal_mask/causal_val_nll"] = float(
                    np.asarray(causal_model_demo_val_nll)
                )
            if causal_model_demo_val_mse is not None:
                causal_log["causal_mask/causal_val_mse"] = float(
                    np.asarray(causal_model_demo_val_mse)
                )
            for key in (
                "masked_fraction_last_batch",
                "selected_latent_ratio",
                "selected_cmi_mean",
                "latent_delta_l1",
            ):
                value = causal_mask_last_metrics.get(f"causal_mask/{key}")
                if value is not None:
                    causal_log[f"causal_mask/{key}"] = float(np.asarray(value))
            using_success_interventions = causal_model_last_source_info.get(
                "using_success_interventions"
            )
            if using_success_interventions is not None:
                causal_log["causal_mask/causal_train_using_success_interventions"] = float(
                    np.asarray(using_success_interventions)
                )
            if causal_weights is not None:
                weight_array = np.asarray(causal_weights, dtype=np.float32).reshape(-1)
                action_weight_log = {}
                for dim, weight in enumerate(weight_array):
                    action_index = (
                        int(causal_action_indices[dim])
                        if causal_action_indices is not None
                        and dim < len(causal_action_indices)
                        else dim
                    )
                    action_weight_log[
                        f"causal_entropy/action_weight_dim_{action_index}"
                    ] = float(weight)
                    action_weight_log[
                        f"causal_entropy/effective_action_weight_dim_{action_index}"
                    ] = float(weight)
                if weight_array.size > 0:
                    causal_log.update(
                        {
                            "causal_entropy/weight_std": float(np.std(weight_array)),
                            "causal_entropy/computed_weight_std": float(
                                np.std(weight_array)
                            ),
                            **action_weight_log,
                        }
                    )
            if use_grasp_causal_bias:
                causal_log.update(
                    {
                        "grasp_causal_bias/active": int(grasp_causal_bias is not None),
                        "grasp_causal_bias/compute_failures": grasp_causal_bias_failures,
                    }
                )
            if grasp_causal_bias is not None:
                bias_array = summarize_grasp_causal_payload(grasp_causal_bias)
                causal_log.update(
                    {
                        "grasp_causal_bias/open": float(bias_array[0]),
                        "grasp_causal_bias/stay": float(bias_array[1]),
                        "grasp_causal_bias/close": float(bias_array[2]),
                    }
                )
            for key, value in grasp_causal_bias_info.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    causal_log[f"grasp_causal_bias/{key}"] = float(value)
            wandb_logger.log(causal_log, step=step)
            wandb_logger.log(
                {
                    "buffer/online_size": len(replay_buffer),
                    "buffer/demo_size": len(demo_buffer),
                    "buffer/online_ready": int(len(replay_buffer) >= config.training_starts),
                    "buffer/using_online_replay": int(using_online_replay),
                    "buffer/training_starts": config.training_starts,
                },
                step=step,
            )

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)
    config.policy_action_dim = int(np.prod(env.action_space.shape))

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.resume_training:
        if FLAGS.checkpoint_path is None or not os.path.exists(FLAGS.checkpoint_path):
            raise FileNotFoundError(
                "--resume_training requires an existing --checkpoint_path. "
                "Start a new run by omitting --resume_training or choose the saved run path."
            )
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        ckpt_step = latest_step_from_path(latest_ckpt, "checkpoint_")
        print_green(f"Loaded previous checkpoint at step {ckpt_step}.")
    elif (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and not FLAGS.allow_existing_checkpoint_path
    ):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer", "*.pkl"))
        demo_buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer", "*.pkl"))
        if latest_ckpt or buffer_files or demo_buffer_files:
            raise FileExistsError(
                f"Checkpoint path already has training state: {FLAGS.checkpoint_path}. "
                "Use --resume_training to continue it, or use a new --checkpoint_path."
            )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_causal_rewards=True,
        )
        # set up wandb and logging
        wandb_output_dir = None
        if FLAGS.checkpoint_path is not None:
            wandb_output_dir = os.path.abspath(FLAGS.checkpoint_path)
            os.makedirs(wandb_output_dir, exist_ok=True)

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=(
                f"{FLAGS.exp_name}_{os.path.basename(os.path.abspath(FLAGS.checkpoint_path))}"
                if FLAGS.checkpoint_path is not None
                else FLAGS.exp_name
            ),
            output_dir=wandb_output_dir,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_causal_rewards=True,
        )
        causal_demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_causal_rewards=True,
        )
        causal_success_intervention_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_causal_rewards=True,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            print("Demo path is: ", path, "Current working directory is: ", os.getcwd())
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                transitions = [
                    prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{path} transition {index}",
                    )
                    for index, transition in enumerate(transitions)
                ]
                for trajectory in split_trajectories(transitions):
                    trajectory = relabel_causal_rewards(
                        trajectory,
                        gamma=float(getattr(config, "causal_rtg_gamma", 0.98)),
                        mode=getattr(config, "causal_reward_mode", "outcome_return_to_go"),
                        fallback_outcome=outcome_from_path(path),
                    )
                    for transition in trajectory:
                        demo_buffer.insert(transition)
                        causal_demo_buffer.insert(copy.deepcopy(transition))
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"Causal old demo buffer size: {len(causal_demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.resume_training and FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable buffer file {file}: {exc}")
                    continue
                for index, transition in enumerate(transitions):
                    transition = prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{file} transition {index}",
                    )
                    replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.resume_training and FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in natsorted(glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            )):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable demo_buffer file {file}: {exc}")
                    continue
                transitions = [
                    prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{file} transition {index}",
                    )
                    for index, transition in enumerate(transitions)
                ]
                for transition in transitions:
                    transition["_reset_frame_stack"] = True
                for trajectory in split_trajectories(transitions):
                    trajectory = relabel_causal_rewards(
                        trajectory,
                        gamma=float(getattr(config, "causal_rtg_gamma", 0.98)),
                        mode=getattr(config, "causal_reward_mode", "outcome_return_to_go"),
                        fallback_outcome=outcome_from_path(file),
                    )
                    for transition in trajectory:
                        demo_buffer.insert(transition)
                        if transition_is_successful_episode(transition):
                            causal_success_intervention_buffer.insert(
                                copy.deepcopy(transition)
                            )
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )
            print_green(
                "Loaded previous successful intervention data. "
                f"Size: {len(causal_success_intervention_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        try:
            learner(
                sampling_rng,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                causal_demo_buffer=causal_demo_buffer,
                causal_success_intervention_buffer=causal_success_intervention_buffer,
                wandb_logger=wandb_logger,
            )
        finally:
            if wandb_logger is not None:
                wandb_logger.finish()

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    print(os.getcwd())
    app.run(main)
