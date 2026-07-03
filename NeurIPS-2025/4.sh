#!/usr/bin/env bash
set -euo pipefail

run_worker() {
  local gpu="$1"
  local step="$2"

  for lr in 3e-4; do
    echo "GPU ${gpu}: RL-v2 prefon, lr=${lr}, step=${step}"
    CUDA_VISIBLE_DEVICES="${gpu}" python /home/nvidia/analog_zx/analog_zx/aihwkit/analog-training/NeurIPS-2025/S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v4" \
      -RPU HfO2 \
      --pref \
      --analog-target fc2-fc3 \
      --freeze-digital \
      --reference-mean 0.05 \
      --reference-std 0.05 \
      --LR "${lr}" \
      --step-size "${step}" \
      --epochs 15 \
      --batch-size 128 \
      --eval-batch-size 256 \
      --seed 123 \
      -CUDA 0
  done
}

run_worker 1 5 &
run_worker 4 10 &
run_worker 6 15 &
wait
