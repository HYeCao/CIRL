#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

: "${DEMO_PATH:?Set DEMO_PATH=/abs/path/to/banana_demo.pkl before starting the learner.}"

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
    elif [[ "$RESUME_REQUESTED" == "1" && -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(cat "$RUN_ID_FILE")
    else
        RUN_ID=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$(dirname "$RUN_ID_FILE")"
        printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
    fi
    CHECKPOINT_PATH=../../experiments/pick_place_banana/pick_place_banana_${RUN_ID}
fi

RESUME_FLAG=()
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
        echo "Refusing to auto-resume existing CHECKPOINT_PATH: $CHECKPOINT_PATH" >&2
        echo "Set RESUME_TRAINING=1 or pass --resume_training to continue it." >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
printf '%s\n' "$(basename "$CHECKPOINT_PATH" | sed 's/^pick_place_banana_//')" > "$RUN_ID_FILE"
echo "[run_learner] checkpoint_path=$CHECKPOINT_PATH"

python ../../train_rlpd.py "$@" "${RESUME_FLAG[@]}" \
    --exp_name=pick_place_banana \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --demo_path="${DEMO_PATH}" \
    --learner
