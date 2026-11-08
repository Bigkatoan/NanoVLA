#!/bin/bash
# Python environments for NanoVLA (Ubuntu, NVIDIA GPU, python3.10 available).
#   envs/lerobot : teacher (SmolVLA), labels, student training, web app   -- lerobot 0.4.4 is the last release for python < 3.12
#   envs/deploy  : robot / Jetson runtime (ONNX Runtime GPU, OpenCV, lerobot for the SO-101 driver)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.10}
if [ ! -x envs/lerobot/bin/python ]; then
  $PY -m venv envs/lerobot
  envs/lerobot/bin/pip install -q --upgrade pip
  envs/lerobot/bin/pip install -q "lerobot[smolvla]==0.4.4" onnx onnxscript tensorboard
fi
if [ ! -x envs/deploy/bin/python ]; then
  $PY -m venv envs/deploy
  envs/deploy/bin/pip install -q --upgrade pip
  envs/deploy/bin/pip install -q onnxruntime-gpu opencv-python-headless "lerobot==0.4.4"
fi
# SmolVLA base weights + its VLM (public, no token)
envs/lerobot/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("lerobot/smolvla_base", local_dir="ckpt/smolvla_base", token=False)
snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", token=False)
EOF
echo "ok. start the app: envs/lerobot/bin/python -m app.server --port 8092"
