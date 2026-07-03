#!/usr/bin/env bash
set -euo pipefail

# Stability-oriented sweep for TT-v4 + pref on.
# We fix the currently best-performing reference point and focus on
# lower LR / earlier decay because many runs peak around epoch 3-4
# and then degrade.

REF_MEAN=0.3
REF_STD=1.0
EPOCHS=8
BATCH_SIZE=128
EVAL_BATCH_SIZE=256
SEED=123

LRS=(1e-4 2e-4 3e-4 5e-4)
GAMMAS=(0.05 0.1)
STEPS=(2 3 4)

combos=()
for lr in "${LRS[@]}"; do
  for gamma in "${GAMMAS[@]}"; do
    for step in "${STEPS[@]}"; do
      combos+=("${lr} ${gamma} ${step}")
    done
  done
done

run_worker() {
  local gpu="$1"
  local offset="$2"

  for ((i=offset; i<${#combos[@]}; i+=3)); do
    read -r lr gamma step <<< "${combos[$i]}"
    echo "GPU ${gpu}: lr=${lr}, lrgamma=${gamma}, step=${step}, rmean=${REF_MEAN}, rstd=${REF_STD}"

    CUDA_VISIBLE_DEVICES="${gpu}" python /home/nvidia/analog_zx/analog_zx/aihwkit/analog-training/NeurIPS-2025/S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v4" \
      -RPU HfO2 \
      --pref \
      --analog-target fc3 \
      --freeze-digital \
      --reference-mean "${REF_MEAN}" \
      --reference-std "${REF_STD}" \
      --LR "${lr}" \
      --step-size "${step}" \
      --gamma "${gamma}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --seed "${SEED}" \
      -CUDA 0
  done
}

run_worker 3 0 &
run_worker 5 1 &
run_worker 7 2 &
wait
