#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

rtde_stop_robot() {
    local robot_ip=${UR_ROBOT_IP:-${ROBOT_IP:-}}
    if [[ -z "$robot_ip" ]]; then
        return 0
    fi

    python - "$robot_ip" <<'PY'
import sys

robot_ip = sys.argv[1]
try:
    from rtde_control import RTDEControlInterface

    control = RTDEControlInterface(robot_ip)
    for stop in (
        lambda: control.forceModeStop(),
        lambda: control.servoStop(),
        lambda: control.speedStop(a=2.0),
        lambda: control.stopScript(),
    ):
        try:
            stop()
        except Exception:
            pass
    control.disconnect()
    print(f"[run_actor] RTDE stop sent to {robot_ip}", flush=True)
except Exception as exc:
    print(f"[run_actor] RTDE stop skipped: {exc}", flush=True)
PY
}

cleanup_actor() {
    local status=$?
    trap - EXIT INT TERM HUP
    rtde_stop_robot
    exit "$status"
}
trap cleanup_actor EXIT INT TERM HUP

RUN_ID_FILE=${RUN_ID_FILE:-../../experiments/pick_place_banana/.latest_run_id}
RESUME_REQUESTED=${RESUME_TRAINING:-0}
for arg in "$@"; do
    if [[ "$arg" == "--resume_training" || "$arg" == "--resume_training=true" ]]; then
        RESUME_REQUESTED=1
    fi
done

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    if [[ -n "${RUN_ID:-}" ]]; then
        :
    elif [[ -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(cat "$RUN_ID_FILE")
    else
        RUN_ID=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$(dirname "$RUN_ID_FILE")"
        printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
    fi
    CHECKPOINT_PATH=../../experiments/pick_place_banana/pick_place_banana_${RUN_ID}
fi

RESUME_FLAG=()
ATTACH_FLAG=()
if [[ "$RESUME_REQUESTED" == "1" ]]; then
    if [[ ! -e "$CHECKPOINT_PATH" ]]; then
        echo "Cannot resume: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
        exit 1
    fi
    case " $* " in
        *" --resume_training"*|*" --resume_training=true"*) ;;
        *) RESUME_FLAG+=(--resume_training) ;;
    esac
else
    if [[ -e "$CHECKPOINT_PATH" ]]; then
        ATTACH_FLAG+=(--allow_existing_checkpoint_path)
        echo "[run_actor] attaching to active run directory without restoring checkpoint: $CHECKPOINT_PATH"
    fi
fi

echo "[run_actor] checkpoint_path=$CHECKPOINT_PATH"

python ../../train_rlpd.py "$@" "${RESUME_FLAG[@]}" "${ATTACH_FLAG[@]}" \
    --exp_name=pick_place_banana \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --actor
