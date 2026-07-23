#!/usr/bin/env python3
"""Smoke test reset and SpaceMouse intervention for a HIL-SERL experiment."""

import argparse
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from experiments.mappings import CONFIG_MAPPING


_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True


def _state_value(obs, key):
    state = obs.get("state", {})
    if isinstance(state, dict):
        value = state.get(key)
        if isinstance(value, np.ndarray):
            return np.asarray(value).reshape(-1)
        return value
    return np.asarray(state).reshape(-1)


def _print_eef_pose(env, label):
    unwrapped = env.unwrapped
    curr_pos = getattr(unwrapped, "curr_pos", None)
    if curr_pos is None:
        print(f"{label}: curr_pos unavailable")
        return

    curr_pos = np.asarray(curr_pos, dtype=np.float64).reshape(-1)
    if curr_pos.size != 7:
        print(f"{label}: curr_pos={np.array2string(curr_pos, precision=4, suppress_small=True)}")
        return

    xyz = curr_pos[:3]
    quat = curr_pos[3:]
    rotvec = R.from_quat(quat).as_rotvec()
    pose_rotvec = np.concatenate([xyz, rotvec])
    print(f"{label} EEF xyz+quat: {np.array2string(curr_pos, precision=5, suppress_small=True)}")
    print(f"{label} EEF xyz+rotvec: {np.array2string(pose_rotvec, precision=5, suppress_small=True)}")


def main():
    parser = argparse.ArgumentParser(
        description="Reset the robot, then send zero actions while SpaceMouse can intervene."
    )
    parser.add_argument("--exp_name", default="ram_insertion")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print_pose_every", type=int, default=10, help="Print current EEF pose every N steps. Use 0 to disable periodic pose prints.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra sleep after each env.step().")
    parser.add_argument(
        "--fake_env",
        action="store_true",
        help="Construct the fake env only. This will not test the real controller or SpaceMouse wrapper.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if args.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown exp_name={args.exp_name!r}. Available: {sorted(CONFIG_MAPPING)}")

    print(f"Loading experiment: {args.exp_name}")
    config = CONFIG_MAPPING[args.exp_name]()
    env = None

    try:
        env = config.get_environment(fake_env=args.fake_env, save_video=False, classifier=False)

        print("Calling env.reset()...")
        obs, info = env.reset()
        state_value = _state_value(obs, "tcp_pose")
        print(f"Reset done. info={info}")
        if state_value is not None:
            preview = np.asarray(state_value).reshape(-1)[:12]
            print(f"state preview after reset: {np.array2string(preview, precision=4, suppress_small=True)}")
        _print_eef_pose(env, "after reset")

        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        print(
            "Now testing SpaceMouse intervention. Move the SpaceMouse or press its buttons; "
            "intervene=True should appear. Press Ctrl+C to stop."
        )

        for step in range(args.steps):
            if _STOP:
                break

            obs, reward, done, truncated, info = env.step(zero_action)
            intervene_action = info.get("intervene_action")
            intervened = intervene_action is not None
            left = bool(info.get("left", False))
            right = bool(info.get("right", False))

            if intervened:
                action_text = np.array2string(
                    np.asarray(intervene_action).reshape(-1),
                    precision=3,
                    suppress_small=True,
                )
            else:
                action_text = "-"

            print(
                f"step={step:04d} reward={reward:.3f} done={done} truncated={truncated} "
                f"intervene={intervened} left={left} right={right} action={action_text}"
            )

            if args.print_pose_every > 0 and (step % args.print_pose_every == 0 or intervened):
                _print_eef_pose(env, f"step {step:04d}")

            if done or truncated:
                print("Episode ended during smoke test; resetting once more.")
                obs, info = env.reset()
                print(f"Reset done. info={info}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    finally:
        if env is not None:
            print("Closing env...")
            env.close()
        print("Smoke test finished.")


if __name__ == "__main__":
    main()
