# Install
Since the RL-v2 algorithm has not been integrated into the AIHWKIT now, you need to compile the library by your self. The code is available in the [downstream repo](https://github.com/Zhaoxian-Wu/aihwkit). See the ``TTv3-momentum-H`` branch for RL-v2 implementation and ``feature/customized-response-function`` branch for power/exponential response functions.

# Running
In the bash following commands, replace the `${CUDA_IDX}` variable with specific GPU index, e.g.
```bash
CUDA_IDX=0
```

## Simulation 1: Toy example
**Figure 2.** Comparison of  Comparison of Analog SGD and Tiki-Taka under different parameter $c_{\text{Lin}}$
```bash
python S1-TT-fails.py
```

## Simulation 2: FCN/CNN @ MNIST
**Figure 4.** 
The network archetecture can be fully-connected network (FCN) or convolutional neural network (CNN)

As pointed out in the paper, RL-v1 can be implemented by TT-v1.

Perform simulations on FCN
```bash
python S1-mnist-FCN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7
python S1-mnist-FCN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.7
python S1-mnist-FCN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.7
```
Perform simulations on CNN
```bash
python S2-mnist-CNN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.8
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.8
python S2-mnist-CNN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="TT-v2" --CUDA=${CUDA_IDX} --tau=0.8
python S2-mnist-CNN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="RL-v2" --CUDA=${CUDA_IDX} --tau=0.8
```

## Simulation 3: Resnet/MobileNet @ CIFAR
```bash
# Analog SGD
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet34" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet50" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet34" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet50" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
# RLv1/TTv1
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet34" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet50" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet34" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet50" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
```

```bash
python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV2"  -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV3L" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV3S" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV2"  -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV3L" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV3S" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}

python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV2"  -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV3L" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="MobileNetV3S" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV2"  -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV3L" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="MobileNetV3S" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
```

## Simulation 4: Ablation of $\gamma$
```bash
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.1 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.1 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.2 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.2 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.3 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.3 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.4 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.4 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
```

## Simulation 5: ImageNet Finetuning / Rebuttal Notes

The rebuttal experiments use:

- Script: `S4-ImageNet-VGG11.py`
- Task: ImageNet-1K classification
- Data root: `/data/imagenet_prepared/imagenet1k_hf_train200_valfull`
- Label space: full 1000-way ImageNet-1K
- Typical train subset: `200/class`
- Validation: full ImageNet-1K val

The current script supports:

- `--digital-only`
- `-SETTING "FP SGD" | "Analog SGD" | "TT-v2" | "TT-v4" | "RL-v2"`
- `-RPU HfO2 | OM`
- `--analog-target fc3 | fc2-fc3 | block5-fc3`
- `--pref`
- `--reference-mean <float>`
- `--reference-std <float>`
- `--train-per-class <int>`
- `--val-per-class <int>`
- `--freeze-digital`

Commonly used hyperparameters in the ImageNet experiments:

- `--seed 123`
- `--epochs 15`
- `--batch-size 64`
- `--eval-batch-size 256`
- `--LR 3e-4`
- `--step-size 5`
- `--gamma 0.1`

Reference-offset values that were swept:

- `reference_mean`: `0.05, 0.2, 0.3, 0.4`
- `reference_std`: `0.05, 0.4, 0.7, 1.0`

Additional scheduler / optimization sweeps that were tried:

- `LR`: `3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 3e-3`
- `step-size`: `2, 3, 4, 5, 10, 15`
- `gamma`: `0.05, 0.1, 0.5`

### ImageNet command templates

Digital epoch-0 baseline:

```bash
CUDA_VISIBLE_DEVICES=${CUDA_IDX} python S4-ImageNet-VGG11.py \
  --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
  --digital-only \
  --epochs 0 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --seed 123 \
  -CUDA 0
```

Floating-point analog-wrapper baseline (`fc3`, freeze digital):

```bash
CUDA_VISIBLE_DEVICES=${CUDA_IDX} python S4-ImageNet-VGG11.py \
  --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
  -SETTING "FP SGD" \
  --analog-target fc3 \
  --freeze-digital \
  --LR 3e-4 \
  --step-size 5 \
  --epochs 15 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --seed 123 \
  -CUDA 0
```

Main rebuttal setting (`OM`, `fc2-fc3`, pref on/off comparison):

```bash
CUDA_VISIBLE_DEVICES=${CUDA_IDX} python S4-ImageNet-VGG11.py \
  --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
  -SETTING "TT-v4" \
  -RPU OM \
  --analog-target fc2-fc3 \
  --freeze-digital \
  --train-per-class 200 \
  --reference-mean <mean> \
  --reference-std <std> \
  --LR 3e-4 \
  --step-size 5 \
  --gamma 0.1 \
  --epochs 15 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --seed 123 \
  -CUDA 0
```

Add `--pref` to the command above for pref-on runs.

### 4x4 reference-offset sweep

TT-v4, pref-off, `fc3`, HfO2:

```bash
for mean in 0.05 0.2 0.3 0.4; do
  for std in 0.05 0.4 0.7 1.0; do
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v4" \
      -RPU HfO2 \
      --analog-target fc3 \
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
  done
done
```

TT-v4, pref-on, `fc3`, HfO2:

```bash
for mean in 0.05 0.2 0.3 0.4; do
  for std in 0.05 0.4 0.7 1.0; do
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v4" \
      -RPU HfO2 \
      --pref \
      --analog-target fc3 \
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
  done
done
```

### TT-v2 4-point sweep

```bash
#!/usr/bin/env bash
set -euo pipefail

combos=(
  "0.05 0.05"
  "0.2 0.4"
  "0.3 0.7"
  "0.4 1.0"
)

run_worker() {
  local gpu="$1"
  local offset="$2"

  for ((i=offset; i<${#combos[@]}; i+=3)); do
    read -r mean std <<< "${combos[$i]}"
    CUDA_VISIBLE_DEVICES="${gpu}" python S4-ImageNet-VGG11.py \
      --data-dir /data/imagenet_prepared/imagenet1k_hf_train200_valfull \
      -SETTING "TT-v2" \
      -RPU HfO2 \
      --analog-target fc3 \
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
  done
}

run_worker 2 0 &
run_worker 5 1 &
run_worker 7 2 &
wait
```

### LR / step / gamma stability sweep

Used to diagnose settings where validation accuracy peaks very early and then degrades:

- `TT-v4`
- `pref on`
- `fc3`
- `freeze-digital`
- `reference_mean=0.3`
- `reference_std=1.0`

Swept values:

- `LR`: `1e-4, 2e-4, 3e-4, 5e-4`
- `step-size`: `2, 3, 4`
- `gamma`: `0.05, 0.1`

### Notes

- HfO2 was useful for exposing large deployment gaps, but it was often too aggressive for the rebuttal story.
- OM was later used as a milder analog device configuration.
- The most important practical comparison for rebuttal became:
  - `pref off` baseline
  - `pref on` (our method)
  - with `OM + fc2-fc3`
- Some historical runs were launched from `third_party/`, so their outputs were saved under `third_party/runs/...` instead of this local `runs/...` tree.
