# NanoVLA

Teach a **SO-101** arm a new task from a handful of demonstrations and get a small model that runs on a
**Jetson Orin Nano** with only the wrist camera.

```
collect demos  ──>  pick a dataset  ──>  Distill  ──>  models/<name>/student.onnx  ──>  robot / Jetson
                                          │
                                          ├─ few-shot the SmolVLA teacher (2 cameras) on your demos
                                          ├─ the teacher labels every frame (+ augmented copies)
                                          └─ train the wrist-only student (ResNet18, 13M params, ~1 ms)
```

## Install

```bash
./setup.sh                       # envs/lerobot (lerobot[smolvla]==0.4.4) + envs/deploy (onnxruntime, opencv)
```

A teacher checkpoint is expected at `ckpt/smolvla_deg/checkpoints/020000/pretrained_model` (or train one from the
**Thầy** tab, ~6 h on an RTX 3060).

## Use

```bash
envs/lerobot/bin/python -m app.server --port 8092     # open http://localhost:8092
```

1. **Thu thập**: record demos with the leader arm (wrist + overview camera, joints in degrees). 2–4 demos per instruction.
2. **Dataset**: pick a recorded dataset, or download a public SO-101 one from the Hub.
3. **Distill**: one click. Few-shot teacher → offline check → labels → student → ONNX (~1.5–2 h).
4. **Model**: download `<name>.zip` (student.onnx, meta.json, report.json) and run it on the robot:

```bash
python deploy/run_student.py --onnx models/<name>/student.onnx --meta models/<name>/meta.json \
    --task "<instruction>" --port /dev/ttyACM0 --robot_id so101 --wrist_cam 0
```

Optional: run with `--front_cam <idx> --record data/lerobot/<name>_dagger1`, then **DAgger** on the Model card.
The teacher labels the frames the student actually visited, and the student retrains.

## Results (offline, held-out task the teacher never saw)

`fport/tic-tac-toe-so101`, 9 instructions, 10 test episodes. Error of the 5 arm joints in degrees over the next
0.33 s (10 actions):

| Model | Cameras | Error | Latency |
|---|---|---|---|
| teacher + 9 demos (1 per instruction) | 2 | 2.39° | 42 ms |
| teacher + 35 demos | 2 | 1.83° | 42 ms |
| student distilled from the teacher | wrist | 2.09° | 1.0 ms |
| student trained on the demos only | wrist | 2.22° | 1.0 ms |

These are offline action errors. Success rates on the robot still need to be measured.

## Layout

```
app/        web UI (server.py + static/index.html)
nanovla/    pipeline.py (jobs), teacher.py (SmolVLA eval / labels), student.py, merge.py (dataset prep), hub.py
deploy/     run_student.py (robot loop, ONNX Runtime / TensorRT)
research/   earlier from-scratch models, Isaac Lab simulation benchmark, experiments (not needed for the pipeline)
data/ ckpt/ models/ jobs/ envs/   local, not versioned
```

Joint convention: actuator angles in degrees (`use_degrees=True`), gripper 0–100, the LeRobot SO101Follower default.
