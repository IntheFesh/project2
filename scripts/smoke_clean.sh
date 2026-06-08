#!/bin/bash
# Clean baseline smoke test: run original LIBERO (10 clean spatial tasks) via lerobot-eval
# using PYTHONPATH isolation to get the clean 10-task benchmark instead of LIBERO-plus's 2402.
set -e
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_DOWNLOAD_TIMEOUT=60
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export PYTHONPATH=/root/autodl-tmp/vcr/third_party/LIBERO-orig

echo "=== CLEAN SMOKE TEST: original libero_spatial task 0, 4 episodes ==="
echo "PYTHONPATH=$PYTHONPATH (forces clean 10-task benchmark)"
echo ""

lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.num_steps=10 \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0,1]' \
    --env.is_libero_plus=false \
    --eval.n_episodes=4 \
    --eval.batch_size=1 \
    --output_dir=/tmp/smoke_clean \
    --seed=0 2>&1 | grep -vE "task orders|video_paths|\.mp4" | grep -E "success|pc_success|task_id|Aggreg|running_success|Error|error|Traceback|FileNotFound" | tail -25
