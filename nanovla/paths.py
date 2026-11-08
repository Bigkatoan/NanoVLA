"""Where everything lives. All paths are inside the repository folder (data/, models/, jobs/ are not versioned)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "lerobot")          # LeRobot v3 datasets (recorded, downloaded, merged)
HF = os.path.join(ROOT, "data", "hf")                 # raw downloads from the Hugging Face Hub
CKPT = os.path.join(ROOT, "ckpt")                     # teacher checkpoints (smolvla_base, smolvla_deg, fewshot/...)
MODELS = os.path.join(ROOT, "models")                 # finished students: <name>/student.onnx, meta.json, report.json
JOBS = os.path.join(ROOT, "jobs")                     # one folder per pipeline run: state.json + logs
TB = os.path.join(ROOT, "tb")                         # TensorBoard runs
ENV = os.path.join(ROOT, "envs", "lerobot")           # python 3.10 venv with lerobot[smolvla]==0.4.4
PY = os.path.join(ENV, "bin", "python")
GPU = os.path.join(ROOT, "nanovla", "gpu.sh")          # single-GPU lock: gpu.sh <timeout_s> <cmd...>
TEACHER = os.environ.get("NANOVLA_TEACHER", os.path.join(CKPT, "smolvla_deg", "checkpoints", "020000", "pretrained_model"))
BASE = os.path.join(CKPT, "smolvla_base")

for d in (DATA, MODELS, JOBS, TB):
    os.makedirs(d, exist_ok=True)
