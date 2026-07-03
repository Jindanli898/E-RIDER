# Prebuilt AIHWKIT Extension

Put the compiled E-RIDER AIHWKIT simulator extension here locally:

```text
prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so
```

The expected server-side source path after an in-place build is:

```bash
/home/jindan/Desktop/analog_zx/aihwkit/src/aihwkit/simulator/rpu_base.cpython-310-x86_64-linux-gnu.so
```

This binary is larger than GitHub's normal per-file limit, so it should be distributed as a GitHub Release asset. Do not commit it as a regular Git object.

Download the release asset:

```bash
mkdir -p prebuilt
wget -O prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so \
  https://github.com/Jindanli898/E-RIDER/releases/download/v0.1.0-prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so
```

After downloading the binary, copy it into the installed AIHWKIT package:

```bash
AIHWKIT_DIR=$(python -c "import aihwkit, pathlib; print(pathlib.Path(aihwkit.__file__).parent)")
cp prebuilt/rpu_base.cpython-310-x86_64-linux-gnu.so "$AIHWKIT_DIR/simulator/"
```

Then verify:

```bash
python -c "import aihwkit; from aihwkit.simulator import rpu_base; print(rpu_base.__file__)"
```
