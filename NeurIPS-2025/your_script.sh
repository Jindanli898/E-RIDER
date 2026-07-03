#!/usr/bin/env bash
set -euo pipefail

combos=(
  "0.05 0.05"
  "0.05 0.4"
  "0.05 0.7"
  "0.05 1.0"
  "0.2 0.05"
  "0.2 0.4"
  "0.2 0.7"
  "0.2 1.0"
  "0.3 0.05"
  "0.3 0.4"
  "0.3 0.7"
  "0.3 1.0"
  "0.4 0.05"
  "0.4 0.4"
  "0.4 0.7"
  "0.4 1.0"
)

run_worker() {
  local gpu="$1"
  local offset="$2"

  for ((i=offset; i<${#combos[@]}; i+=3)); do
    read -r mean std <<< "${combos[$i]}"
    echo "GPU ${gpu}: TT-v4 OM, mean=${mean}, std=${std}"

    CUDA_VISIBLE_DEVICES="${gpu}" python /home/nvidia/analog_zx/analog_zx/aihwkit/analog-training/NeurIPS-2025/S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v4" \
      -RPU OM \
      --analog-target fc2-fc3 \
      --freeze-digital \
      --pref \
      --train-per-class 200 \
      --reference-mean "${mean}" \
      --reference-std "${std}" \
      --LR 3e-4 \
      --step-size 5 \
      --epochs 15 \
      --batch-size 64 \
      --eval-batch-size 256 \
      --seed 123 \
      -CUDA 0
  done
}

run_worker 3 0 &
run_worker 4 1 &
run_worker 5 2 &
run_worker 6 3 &
wait
