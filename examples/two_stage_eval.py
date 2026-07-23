#!/usr/bin/env python3
"""
Run two trained HIL-SERL policies sequentially in ONE continuous episode.

Typical door-opening decomposition:
  Stage 1: grasp/secure the handle (learned gripper policy, 7D action).
  Stage 2: pull/open the door (often trained with GripperCloseEnv => 6D action).

This runner:
  - Creates ONE real environment (usually the stage-1 env config) and never resets between stages.
  - Loads two checkpoints and switches from policy-1 to policy-2 when a "stage-1 success" condition fires.
  - Optionally uses reward classifiers (same ones you use in your task wrappers) for:
      * stage-1 switch condition
      * stage-2 final success condition

It also supports:
  - Action EMA smoothing (optionally smoothing the gripper dim to reduce accidental toggles).
  - Human intervention via your existing SpacemouseIntervention wrapper (if your config enables it).

python3 run_two_stage_door_open.py \
  --exp_name_stage1=ur5e_door_grasp \
  --exp_name_stage2=ur5e_door_pull \
  --checkpoint_path_stage1=/absolute/path/to/grasp_ckpt_dir \
  --checkpoint_step_stage1=150000 \
  --checkpoint_path_stage2=/absolute/path/to/pull_ckpt_dir \
  --checkpoint_step_stage2=200000 \
  --n_trajs=5 \
  --max_steps_per_traj=400 \
  --ema_cutoff_hz=2.0 \
  --ema_filter_gripper=True \
  --stage1_switch_mode=gripper \
  --stage1_consecutive=3 \
  --stage2_success_mode=clf \
  --stage2_clf_threshold=0.90 \
  --stage2_clf_consecutive=2

python3 run_two_stage_door_open.py \
  --exp_name_stage1=ur5e_door_grasp \
  --exp_name_stage2=ur5e_door_pull \
  --checkpoint_path_stage1=/path/to/grasp_ckpt \
  --checkpoint_path_stage2=/path/to/pull_ckpt \
  --stage1_switch_mode=clf \
  --stage1_clf_ckpt=/path/to/grasp_classifier_ckpt \
  --stage1_clf_image_keys=wrist \
  --stage2_success_mode=clf \
  --stage2_clf_ckpt=/path/to/dooropen_classifier_ckpt \
  --stage2_clf_image_keys=shoulder
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Callable, Any

import numpy as np
from absl import app, flags

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm

# Reward classifier loader (used by your env wrappers as well)
from serl_launcher.networks.reward_classifier import load_classifier_func


FLAGS = flags.FLAGS

# Which two experiment configs to use (CONFIG_MAPPING keys)
flags.DEFINE_string("exp_name_stage1", None, "CONFIG_MAPPING key for stage-1 task (e.g., door_grasp).")
flags.DEFINE_string("exp_name_stage2", None, "CONFIG_MAPPING key for stage-2 task (e.g., door_pull).")

# Which experiment config to use to BUILD THE REAL ENV (defaults to stage-1)
flags.DEFINE_string(
    "exp_name_env",
    None,
    "CONFIG_MAPPING key used to build the real env. Defaults to exp_name_stage1. "
    "Use this if you made a combined env config that matches BOTH policies' observation structure.",
)

# Checkpoints
flags.DEFINE_string("checkpoint_path_stage1", None, "Checkpoint directory for stage-1 policy.")
flags.DEFINE_integer("checkpoint_step_stage1", 0, "Checkpoint step for stage-1 (0 = latest).")

flags.DEFINE_string("checkpoint_path_stage2", None, "Checkpoint directory for stage-2 policy.")
flags.DEFINE_integer("checkpoint_step_stage2", 0, "Checkpoint step for stage-2 (0 = latest).")

# Rollouts
flags.DEFINE_integer("n_trajs", 5, "How many rollouts to run.")
flags.DEFINE_integer("max_steps_per_traj", 500, "Hard cap on steps per rollout (safety).")
flags.DEFINE_boolean("save_video", False, "Whether to save videos (if your env supports it).")

# Action smoothing
flags.DEFINE_float("ema_cutoff_hz", 2.0, "EMA low-pass cutoff (Hz). Lower = more smoothing.")
flags.DEFINE_boolean(
    "ema_filter_gripper",
    True,
    "If True, EMA also filters action[6] (gripper). This often reduces accidental open/close toggles.",
)

# Stage-1 -> Stage-2 switching
flags.DEFINE_enum(
    "stage1_switch_mode",
    "gripper",
    ["gripper", "clf", "either"],
    "How to decide when stage-1 is 'done': "
    "gripper = use env.unwrapped.gripper_state, "
    "clf = use stage-1 reward classifier, "
    "either = OR of both.",
)
flags.DEFINE_integer("stage1_min_steps", 10, "Don't allow switching before this many steps.")
flags.DEFINE_integer("stage1_consecutive", 5, "How many consecutive 'success' checks to trigger switching.")
flags.DEFINE_float("stage1_closed_thresh", 0.7, "closed_norm threshold (0=open, 1=closed).")
flags.DEFINE_float("stage1_obj_thresh", 0.5, "object_detected threshold (0/1 in your controller).")

# Optional: stage-1 classifier overrides (if not provided, try config attrs)
flags.DEFINE_string("stage1_clf_ckpt", "", "Optional checkpoint path for stage-1 classifier (override).")
flags.DEFINE_multi_string(
    "stage1_clf_image_keys",
    None,
    "Optional image keys for stage-1 classifier (override). Can be repeated: --stage1_clf_image_keys=wrist ...",
)
flags.DEFINE_float("stage1_clf_threshold", 0.95, "Stage-1 classifier prob threshold.")
flags.DEFINE_integer("stage1_clf_consecutive", 5, "Stage-1 classifier consecutive frames.")

# Stage-2 success condition (usually classifier)
flags.DEFINE_enum(
    "stage2_success_mode",
    "clf",
    ["clf", "none"],
    "How to decide final success. 'clf' uses stage-2 classifier; 'none' never auto-terminates on success.",
)
flags.DEFINE_string("stage2_clf_ckpt", "", "Optional checkpoint path for stage-2 classifier (override).")
flags.DEFINE_multi_string(
    "stage2_clf_image_keys",
    None,
    "Optional image keys for stage-2 classifier (override). Can be repeated: --stage2_clf_image_keys=shoulder ...",
)
flags.DEFINE_float("stage2_clf_threshold", 0.95, "Stage-2 classifier prob threshold.")
flags.DEFINE_integer("stage2_clf_consecutive", 5, "Stage-2 classifier consecutive frames.")

# Determinism
flags.DEFINE_boolean("deterministic", True, "Use argmax=True for action sampling (recommended for real eval).")
flags.DEFINE_integer("seed", 42, "RNG seed.")


class EMAActionFilter:
    """
    First-order low-pass filter on actions.
    Useful for smoothing policy jitter. With thresholded/discrete gripper commands,
    filtering action[6] makes 'open/close' require persistence across multiple steps.
    """

    def __init__(self, hz: float, cutoff_hz: float = 2.0, filter_gripper: bool = False):
        self.hz = float(hz)
        self.cutoff_hz = float(cutoff_hz)
        self.filter_gripper = bool(filter_gripper)

        # Discrete-time 1st-order low-pass:
        # alpha = dt / (RC + dt), RC = 1/(2*pi*f_c)
        dt = 1.0 / max(self.hz, 1e-6)
        rc = 1.0 / max(2.0 * np.pi * self.cutoff_hz, 1e-6)
        self.alpha = float(dt / (rc + dt))

        self.prev: Optional[np.ndarray] = None

    def reset(self):
        self.prev = None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).copy()

        if self.prev is None:
            self.prev = a.copy()
            return a

        idx_end = min(6, a.shape[0])
        self.prev[:idx_end] = self.prev[:idx_end] + self.alpha * (a[:idx_end] - self.prev[:idx_end])

        if a.shape[0] > 6:
            if self.filter_gripper:
                self.prev[6] = self.prev[6] + self.alpha * (a[6] - self.prev[6])
            else:
                self.prev[6] = a[6]

        return np.clip(self.prev, -1.0, 1.0).astype(np.float32)


def _build_agent_from_config(config, sample_obs, sample_action):
    """
    Construct an agent object whose network shapes match `sample_obs` and `sample_action`,
    based on config.setup_mode (same logic as your training script).
    """
    setup_mode = getattr(config, "setup_mode", "single-arm-learned-gripper")

    if setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        return make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )

    if setup_mode == "single-arm-learned-gripper":
        return make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )

    if setup_mode == "dual-arm-learned-gripper":
        return make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )

    raise NotImplementedError(f"Unknown setup_mode={setup_mode!r} in config {type(config).__name__}")


def _restore_agent(agent, ckpt_dir: str, step: int):
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir,
        agent.state,
        step=None if step <= 0 else step,
    )
    return agent.replace(state=ckpt)


def _sigmoid_prob(logits: Any) -> float:
    # logits may be shape (1,) or (B,1); just flatten.
    x = jnp.ravel(jnp.asarray(logits))[0]
    return float(jax.device_get(jax.nn.sigmoid(x)))


def _maybe_load_classifier(
    *,
    config,
    sample_obs,
    override_ckpt: str,
    override_image_keys: Optional[Sequence[str]],
) -> Optional[Callable[[Any], Any]]:
    """
    Returns a callable clf_fn(obs)->logits or None.
    """
    ckpt_path = override_ckpt.strip() if override_ckpt is not None else ""
    if not ckpt_path:
        ckpt_path = str(getattr(config, "classifier_ckpt_path", "")).strip()

    if not ckpt_path:
        return None

    image_keys = list(override_image_keys) if override_image_keys else list(getattr(config, "classifier_keys", []))
    if not image_keys:
        raise ValueError(
            f"Classifier ckpt provided ({ckpt_path}) but no image_keys set. "
            f"Pass --*_clf_image_keys or set config.classifier_keys."
        )

    key = jax.random.PRNGKey(FLAGS.seed)
    clf_fn = load_classifier_func(
        key=key,
        sample=sample_obs,
        image_keys=image_keys,
        checkpoint_path=ckpt_path,
    )
    return clf_fn


def _pad_action_to_env(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    """
    If a policy outputs 6D (because it was trained with GripperCloseEnv),
    pad to 7D by appending a neutral gripper command (0).
    """
    a = np.asarray(action, dtype=np.float32)
    if a.shape == (env_action_dim,):
        return a
    if env_action_dim == 7 and a.shape == (6,):
        return np.concatenate([a, np.zeros((1,), dtype=np.float32)], axis=0)
    raise ValueError(f"Cannot pad action of shape {a.shape} to env dim {env_action_dim}.")


def main(_):
    # ---- Validate flags ----
    if FLAGS.exp_name_stage1 is None or FLAGS.exp_name_stage2 is None:
        raise ValueError("You must set --exp_name_stage1 and --exp_name_stage2.")

    if FLAGS.exp_name_stage1 not in CONFIG_MAPPING:
        raise KeyError(f"exp_name_stage1={FLAGS.exp_name_stage1!r} not found in CONFIG_MAPPING.")
    if FLAGS.exp_name_stage2 not in CONFIG_MAPPING:
        raise KeyError(f"exp_name_stage2={FLAGS.exp_name_stage2!r} not found in CONFIG_MAPPING.")

    if FLAGS.exp_name_env is None:
        FLAGS.exp_name_env = FLAGS.exp_name_stage1

    if FLAGS.exp_name_env not in CONFIG_MAPPING:
        raise KeyError(f"exp_name_env={FLAGS.exp_name_env!r} not found in CONFIG_MAPPING.")

    if FLAGS.checkpoint_path_stage1 is None or FLAGS.checkpoint_path_stage2 is None:
        raise ValueError("You must set --checkpoint_path_stage1 and --checkpoint_path_stage2.")

    # ---- Load configs ----
    cfg1 = CONFIG_MAPPING[FLAGS.exp_name_stage1]()
    cfg2 = CONFIG_MAPPING[FLAGS.exp_name_stage2]()
    cfg_env = CONFIG_MAPPING[FLAGS.exp_name_env]()

    # ---- Build REAL env (one env, no resets between stages) ----
    # IMPORTANT: classifier=False so the env doesn't terminate early on stage-1 success.
    env = cfg_env.get_environment(
        fake_env=False,
        save_video=FLAGS.save_video,
        classifier=False,
    )
    env = RecordEpisodeStatistics(env)

    # ---- Build FAKE envs for shape inference (no robot connections) ----
    env1_fake = cfg1.get_environment(fake_env=True, save_video=False, classifier=False)
    env2_fake = cfg2.get_environment(fake_env=True, save_video=False, classifier=False)

    # ---- Build agents from their own training shapes ----
    agent1 = _build_agent_from_config(cfg1, env1_fake.observation_space.sample(), env1_fake.action_space.sample())
    agent2 = _build_agent_from_config(cfg2, env2_fake.observation_space.sample(), env2_fake.action_space.sample())

    # Place agents on device (single- or multi-device). This matches your training pattern.
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    agent1 = jax.device_put(jax.tree.map(jnp.array, agent1), sharding.replicate())
    agent2 = jax.device_put(jax.tree.map(jnp.array, agent2), sharding.replicate())

    # ---- Restore checkpoints ----
    agent1 = _restore_agent(agent1, FLAGS.checkpoint_path_stage1, FLAGS.checkpoint_step_stage1)
    agent2 = _restore_agent(agent2, FLAGS.checkpoint_path_stage2, FLAGS.checkpoint_step_stage2)

    # ---- Optional classifier(s) ----
    sample_obs_env = env.observation_space.sample()

    stage1_clf_fn = _maybe_load_classifier(
        config=cfg1,
        sample_obs=sample_obs_env,
        override_ckpt=FLAGS.stage1_clf_ckpt,
        override_image_keys=FLAGS.stage1_clf_image_keys,
    )
    stage2_clf_fn = _maybe_load_classifier(
        config=cfg2,
        sample_obs=sample_obs_env,
        override_ckpt=FLAGS.stage2_clf_ckpt,
        override_image_keys=FLAGS.stage2_clf_image_keys,
    )

    if FLAGS.stage2_success_mode == "clf" and stage2_clf_fn is None:
        raise ValueError(
            "stage2_success_mode='clf' but no stage-2 classifier could be loaded. "
            "Set --stage2_clf_ckpt / --stage2_clf_image_keys or config.classifier_ckpt_path/classifier_keys."
        )

    # ---- Filters ----
    # We use env.unwrapped.hz if present; otherwise fall back to 10Hz.
    env_hz = float(getattr(env.unwrapped, "hz", 10.0))

    filt1 = EMAActionFilter(hz=env_hz, cutoff_hz=FLAGS.ema_cutoff_hz, filter_gripper=FLAGS.ema_filter_gripper)
    filt2 = EMAActionFilter(hz=env_hz, cutoff_hz=FLAGS.ema_cutoff_hz, filter_gripper=False)  # stage-2 usually fixed

    # RNG for action sampling
    rng = jax.random.PRNGKey(FLAGS.seed)

    # ---- Rollouts ----
    for ep in range(FLAGS.n_trajs):
        obs, _ = env.reset()
        filt1.reset()
        filt2.reset()

        stage = 1  # 1=grasp, 2=pull
        step_in_ep = 0

        # Stage-1 success streaks
        gripper_streak = 0
        clf1_streak = 0

        # Stage-2 success streak
        clf2_streak = 0

        t0 = time.time()
        print(f"\n=== Episode {ep+1}/{FLAGS.n_trajs} ===")

        while step_in_ep < FLAGS.max_steps_per_traj:
            step_in_ep += 1

            # ---- Sample action from the active policy ----
            rng, key = jax.random.split(rng)

            if stage == 1:
                a = agent1.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=FLAGS.deterministic,
                    seed=key,
                )
                a = np.asarray(jax.device_get(a))
                a = filt1(a)
            else:
                a = agent2.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=FLAGS.deterministic,
                    seed=key,
                )
                a = np.asarray(jax.device_get(a))
                a = filt2(a)
                # If stage-2 policy was trained with GripperCloseEnv, it outputs 6D.
                # Pad to env's 7D action by adding a neutral gripper command.
                a = _pad_action_to_env(a, env.action_space.shape[0])

            # ---- Step env ----
            next_obs, reward, terminated, truncated, info = env.step(a)
            obs = next_obs
            terminated = False
            truncated = False
            # ---- Stage switching logic ----
            if stage == 1:
                # (A) Gripper sensor-based switch
                if FLAGS.stage1_switch_mode in ("gripper", "either"):
                    # Your controller defines:
                    #   env.unwrapped.gripper_state = [closed_norm, object_detected]
                    closed_norm = float(getattr(env.unwrapped, "gripper_state", [0.0, 0.0])[0])
                    obj_det = float(getattr(env.unwrapped, "gripper_state", [0.0, 0.0])[1])

                    ok = (closed_norm >= FLAGS.stage1_closed_thresh) and (obj_det >= FLAGS.stage1_obj_thresh)
                    gripper_streak = (gripper_streak + 1) if ok else 0
                else:
                    gripper_streak = 0

                # (B) Classifier-based switch
                if FLAGS.stage1_switch_mode in ("clf", "either") and stage1_clf_fn is not None:
                    p1 = _sigmoid_prob(stage1_clf_fn(obs))
                    ok1 = (p1 >= FLAGS.stage1_clf_threshold)
                    clf1_streak = (clf1_streak + 1) if ok1 else 0
                else:
                    clf1_streak = 0

                can_switch = (step_in_ep >= FLAGS.stage1_min_steps)
                hit = False
                if FLAGS.stage1_switch_mode == "gripper":
                    hit = (gripper_streak >= FLAGS.stage1_consecutive)
                elif FLAGS.stage1_switch_mode == "clf":
                    hit = (clf1_streak >= FLAGS.stage1_clf_consecutive)
                else:  # either
                    hit = (gripper_streak >= FLAGS.stage1_consecutive) or (clf1_streak >= FLAGS.stage1_clf_consecutive)

                if can_switch and hit:
                    stage = 2
                    filt2.reset()  # avoid mixing stage-1 filtered action history into stage-2
                    clf2_streak = 0
                    print(f"[SWITCH] Stage-1 success detected at step {step_in_ep}. Switching to stage-2 policy.")
                    # Optional: small pause so the gripper finishes physically closing.
                    time.sleep(0.10)

            else:
                # Stage-2 final success (classifier)
                env.unwrapped.curr_path_length = 0
                # env.unwrapped.max_episode_length = 120
                if FLAGS.stage2_success_mode == "clf" and stage2_clf_fn is not None:
                    p2 = _sigmoid_prob(stage2_clf_fn(obs))
                    ok2 = (p2 >= FLAGS.stage2_clf_threshold)
                    clf2_streak = (clf2_streak + 1) if ok2 else 0
                    if clf2_streak >= FLAGS.stage2_clf_consecutive:
                        dt = time.time() - t0
                        env.unwrapped._send_gripper_command(np.array([-1.0]))
                        time.sleep(0.4)
                        print(f"[SUCCESS] Stage-2 classifier fired at step {step_in_ep} (t={dt:.1f}s).")
                        break

            # ---- Safety/truncation exit ----
            if terminated or truncated:
                print(f"[DONE] Env terminated={terminated} truncated={truncated} at step {step_in_ep}.")
                break

        # End episode summary
        dt = time.time() - t0
        print(f"[EPISODE END] steps={step_in_ep} stage={stage} time={dt:.1f}s")

        # On real hardware you typically need to reset the door/handle between rollouts.
        input("Press Enter when the door/handle is reset to start the next episode...")

    env.close()


if __name__ == "__main__":
    app.run(main)