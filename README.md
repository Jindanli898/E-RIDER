# E-RIDER

This is a lightweight public artifact for reproducing the E-RIDER experiments. It is not the official IBM AIHWKIT package. The experiments use a custom AIHWKIT CUDA extension and the modified `rpucuda` source is included for reference.

## Layout

```text
E-RIDER/
  README.md
  aihwkit_zx.yml
  NeurIPS-2025/
  rpucuda/
  prebuilt/
    README.md
```

## Environment

Create the conda environment:

```bash
conda env create -f aihwkit_zx.yml
conda activate aihwkit_zx
```

Install an AIHWKIT Python package that matches the binary extension:

```bash
python -m pip install aihwkit
```

## Prebuilt Extension

This repository does not commit the compiled AIHWKIT extension directly because it is too large for normal GitHub tracking. Download it from the GitHub release asset:

```bash
mkdir -p prebuilt
wget -O prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so \
  https://github.com/Jindanli898/E-RIDER/releases/download/v0.1.0-prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so
```

If `wget` is unavailable, use:

```bash
curl -L -o prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so \
  https://github.com/Jindanli898/E-RIDER/releases/download/v0.1.0-prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so
```

Then replace the installed AIHWKIT simulator extension with the E-RIDER one:

```bash
AIHWKIT_DIR=$(python -c "import aihwkit, pathlib; print(pathlib.Path(aihwkit.__file__).parent)")
cp prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so "$AIHWKIT_DIR/simulator/"
```

Verify that Python is loading the intended extension:

```bash
python -c "import aihwkit; from aihwkit.simulator import rpu_base; print(aihwkit.__file__); print(rpu_base.__file__)"
```

The `rpu_base` path should point to the `aihwkit/simulator/rpu_base.cpython-310-x86_64-linux-gnu.so` file that you just replaced.

## Run

Example MNIST-CNN run:

```bash
cd NeurIPS-2025
CUDA_VISIBLE_DEVICES=0 python S2-mnist-CNN.py --SETTING="RL-v2" --CUDA=0 --tau=0.6
```

Other experiment entry points are in `NeurIPS-2025/`, including `S2-mnist-FCN.py`, `S3-resnet_finetune.py`, and `S4-ImageNet-VGG11.py`.

## Notes

- `rpucuda/` is the modified CUDA/C++ source snapshot used to build the prebuilt extension.
- The prebuilt `.so` is Linux/Python/CUDA/PyTorch-ABI specific. If it does not import on a different machine, rebuild AIHWKIT from the matching modified source.
- Large generated outputs, datasets, credentials, caches, and logs are intentionally excluded from this public artifact.
