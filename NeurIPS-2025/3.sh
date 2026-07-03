#!/usr/bin/env bash
set -euo pipefail

run_job() {
  local gpu="$1"
  local setting="$2"
  local pref_mode="$3"
  local mean="$4"
  local std="$5"

  local pref_args=()
  if [[ "$pref_mode" == "on" ]]; then
    pref_args+=(--pref)
  fi

  echo "GPU ${gpu}: setting=${setting}, pref=${pref_mode}, mean=${mean}, std=${std}"

  CUDA_VISIBLE_DEVICES="${gpu}" python /home/nvidia/analog_zx/analog_zx/aihwkit/analog-training/NeurIPS-2025/S4-ImageNet-VGG11.py \
    --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
    -SETTING "${setting}" \
    -RPU HfO2 \
    "${pref_args[@]}" \
    --analog-target fc2-fc3 \
    --freeze-digital \
    --reference-mean "${mean}" \
    --reference-std "${std}" \
    --LR 3e-4 \
    --step-size 5 \
    --epochs 15 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --seed 123 \
    -CUDA 0
}

worker() {
  local gpu="$1"
  shift
  while (($#)); do
    run_job "$gpu" "$1" "$2" "$3" "$4"
    shift 4
  done
}

worker 2 \
  "TT-v2" off 0.05 0.05 \
  "TT-v4" off 0.4 1.0 \
&

worker 3 \
  "TT-v2" off 0.4 1.0 \
  "RL-v2" on 0.05 0.05 \
&

worker 5 \
  "TT-v4" off 0.05 0.05 \
  "RL-v2" on 0.4 1.0 \
&

wait
