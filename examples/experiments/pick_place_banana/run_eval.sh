#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
EXP_DIR=$SCRIPT_DIR
RUN_ID_FILE=${RUN_ID_FILE:-$EXP_DIR/.latest_run_id}

cd "$EXP_DIR"

if [[ -z "${RUN_ID:-}" ]]; then
    if [[ -f "$RUN_ID_FILE" ]]; then
        RUN_ID=$(cat "$RUN_ID_FILE")
    else
        echo "Set RUN_ID, e.g. export RUN_ID=20260504_test2" >&2
        exit 1
    fi
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$EXP_DIR/pick_place_banana_${RUN_ID}}
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ -z "${EVAL_CHECKPOINT_STEP:-}" ]]; then
    latest=$(find "$CHECKPOINT_PATH" -maxdepth 1 -type d -name 'checkpoint_*' -printf '%f
' | sed 's/^checkpoint_//' | sort -n | tail -1)
    if [[ -z "$latest" ]]; then
        echo "No checkpoint_* directory found in $CHECKPOINT_PATH" >&2
        exit 1
    fi
    EVAL_CHECKPOINT_STEP=$latest
fi

EVAL_N_TRAJS=${EVAL_N_TRAJS:-5}
SAVE_VIDEO=${SAVE_VIDEO:-0}

DEFAULT_VENV=${VENV:-$ROOT_DIR/.venv}
NVIDIA_LD_LIBRARY_PATH=$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cublas/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$DEFAULT_VENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib
export VENV=$DEFAULT_VENV
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$NVIDIA_LD_LIBRARY_PATH
fi

args=(
    ../../train_rlpd.py
    --exp_name=pick_place_banana
    --checkpoint_path="$CHECKPOINT_PATH"
    --actor
    --allow_existing_checkpoint_path
    --eval_checkpoint_step="$EVAL_CHECKPOINT_STEP"
    --eval_n_trajs="$EVAL_N_TRAJS"
    --debug
)

if [[ "$SAVE_VIDEO" == "1" ]]; then
    args+=(--save_video)
fi

echo "[run_eval] RUN_ID=$RUN_ID"
echo "[run_eval] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[run_eval] EVAL_CHECKPOINT_STEP=$EVAL_CHECKPOINT_STEP"
echo "[run_eval] EVAL_N_TRAJS=$EVAL_N_TRAJS"
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv run python "${args[@]}"
