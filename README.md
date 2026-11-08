# NanoVLA

A ~11M-parameter vision-language-action model for the **SO-ARM101** arm, trained from scratch: a small ViT and a
small text encoder are pretrained separately, their embedding spaces are aligned, and an ACT-style action head maps
(front image, wrist image, instruction, joint state) to 1.6 s action chunks. The goal is a few-shot pipeline: adapt to
a new task from tens of demonstrations, measured by closed-loop **success rate**. Target hardware: Jetson Orin Nano.

## Current pipeline: SmolVLA teacher -> few-shot -> distilled wrist-camera student

```
public SO-101 data (degrees) ──> SmolVLA teacher (2 cameras: wrist + overview, ~450M, PC)
your demos (5-20 per task)  ──> few-shot teacher (2,000 steps, action expert only)
teacher labels (demo frames + augmented copies [+ DAgger frames]) ──> student (wrist only, ResNet18, 13M) ──> ONNX ──> Jetson
```

| Step | Command | Notes |
|---|---|---|
| 1. pick public data | `python3 tools/hf_select.py --out data/hf/selected.json` | v3.0, 30 fps, one wrist + one 480x640 overview camera, joints in **degrees** (actuator angle), per-joint means within 25 deg of the consensus (same calibration convention) |
| 2. download + merge | `snapshot_download(...)`, then `tools/hf_unify.py --out data/lerobot/<name> <roots>` | renames cameras to `camera1` (wrist) / `camera2` (overview), fixes stale episode-metadata indices; values are not rescaled |
| 3. teacher | `lerobot-train --policy.path=ckpt/smolvla_base --dataset.root=data/lerobot/so101_deg_v1 --batch_size=32 --steps=20000` | 6 h on an RTX 3060; `tools/log2tb.py` mirrors the log to TensorBoard |
| 4. your task | record with `lerobot-record` (SO101Follower, `use_degrees=True`, wrist + overview camera, 30 fps), then `scripts/fewshot_distill.sh <name> <dataset> "[train eps]" <test eps>` | few-shot teacher -> offline eval -> teacher labels -> student -> ONNX + `.meta.json` |
| 5. robot | `python deploy/run_student.py --onnx ckpt/student/<name>.onnx --meta ckpt/student/<name>.meta.json --task "..." --port /dev/ttyACM0` | 30 Hz control, new chunk every 8 ticks with cross-fade, per-tick joint-step limit; ONNX Runtime picks TensorRT fp16 on the Jetson |
| 6. DAgger | `run_student.py ... --front_cam <idx> --record data/lerobot/<name>_dagger<k>`, then `scripts/dagger_round.sh <name> <k>` | the teacher labels the frames the student visited; the student retrains on all labels |

Results (offline, proxy task `fport/tic-tac-toe-so101` that the teacher never saw: 9 instructions, 10 test episodes;
mean absolute error of the 5 arm joints in degrees, first 10 actions = 0.33 s / full 50-action chunk;
`results/day1/`):

| Model | Cameras | MAE 10 | MAE 50 | Gripper (0-100) | Latency (RTX 3060) |
|---|---|---|---|---|---|
| teacher, zero-shot | 2 | 4.18 | 12.24 | 1.71 | 42 ms |
| teacher + 9 demos (1 per instruction) | 2 | 2.39 | 6.01 | 1.08 | 42 ms |
| teacher + 18 demos | 2 | 1.93 | 4.62 | 0.92 | 42 ms |
| teacher + 35 demos | 2 | **1.83** | 4.28 | 0.93 | 42 ms |
| `smolvla_base` + 35 demos (no teacher pretraining) | 2 | 2.00 | 4.44 | 0.97 | 42 ms |
| student, demos only | wrist | 2.22 | 4.41 | 0.99 | 1.0 ms |
| student, teacher labels | wrist | **2.09** | **4.22** | 0.92 | 1.0 ms |
| student, teacher + demos | wrist | 2.10 | 4.25 | 0.89 | 1.0 ms |

These are offline action errors, not success rates; closed-loop tests (Isaac tabletop, real robot) come next.
Teacher data: `geonmin-kim/SO101-lv4-3color-cube-mat-to-mat…`, `nikodembartnik/so101_pencil_holder_merged`,
`nikodembartnik/so101_pencil_holder_2`, `dpak0724/so101_toh_79ep_2cam`, `aivanni/so101-puzzle-v4` (1,035 episodes, 661k frames).

Environments: `envs/lerobot` (python 3.10, `lerobot[smolvla]==0.4.4`, the last release for python < 3.12) for the
teacher, labels and student; `envs/deploy` (onnxruntime-gpu, opencv) for `deploy/`.

Simulation research line (Isaac Lab tabletop, `isaac_bench/`, `vla/nanovla_*.py`): modular state-based NanoVLA
(grounder / state / grasp / planner + learned IK `tools/nano_ik.py`), event labels, closed-loop evals with
module-oracle ablations (`results/s*/`); `isaac_bench/sim_web.py` serves the simulated arm in a browser
(Cartesian jog, joint sliders, analytic or learned IK).

## Earlier from-scratch model

| Stage | Model | Params | Training data |
|---|---|---|---|
| Vision | ViT-Tiny, 112 px, DINO + inverse dynamics on patch tokens | ~5.5M | 1.29M cleaned frames, 1,169 LeRobot SO-100/101 repos |
| Text | 4-layer transformer, BPE | ~2M | instructions (MLM + SimCSE + attribute-swap hard negatives) |
| Alignment | sigmoid (SigLIP-style) loss with task clusters | - | image/instruction pairs |
| Action head | 1 fusion + 3 decoder layers, d=256, chunk 16 @ 10 Hz, per-robot embedding | 3.4M | 8,260 cleaned episodes |

Data cleaning (glitch repair, leader-follower calibration, camera lag, idle trimming, frozen/blank frames, label audit)
is in `vla/clean_v2.py`, `vla/audit_labels.py`, `vla/task_clusters.py`.

## Results so far (simulation)

MuJoCo SO-101, "pick up the red cube and place it in the blue square", unseen layouts, scripted-expert demos.
Best recipe: pretrained VLA + 50 % joint-state dropout + grasp/release localisation loss + encoder fine-tune (lr 3e-5).

| Demos | Success rate | Notes |
|---|---|---|
| 10 | 12.7 % | 3 seeds x 50 rollouts (`results/mujoco_fewshot_final.json`) |
| 20 | 21.3 % | |
| 50 | 40.7 % | from-scratch baseline 12.7 %; expert ceiling 82.7 % |


Demo-count scaling (`vla/run_scaling.sh`, `results/scaling.json`; 600-demo pool, expert success ~80 %):

| Demos | Pretrained, best recipe (3 seeds, 150 rollouts) | From scratch (1 seed, 50 rollouts) |
|---|---|---|
| 50 | 34.7 % [28, 43] | - |
| 100 | 50.7 % [43, 59] | 20 % |
| 200 | 65.3 % [57, 72] | 24 % |
| 400 | **73.3 %** [66, 80] | 30 % |
| 600 | 68.0 % [60, 75] | 36 % |

Success rate saturates around 70 % (about 90 % of the expert) from ~300-400 demos; pretraining stays worth
2-3x at every data size. Reaching the few-shot target (20-50 demos) needs a better model, not more demos.

Isaac Sim 5.1 / Isaac Lab 2.3: a multi-object tabletop with YCB household objects, randomised table tops and a
node-based trajectory tool for data collection (`isaac_bench/`), plus an evaluation client for the NVIDIA
Sim-to-Real SO-101 workshop task (vials -> rack) to compare against GR00T N1.6.

## Setup

```bash
# model / MuJoCo side: python 3.10 with torch, torchvision, timm, tokenizers, pandas, pyarrow, av, mujoco
# Isaac side: python 3.11 venv with isaacsim[all,extscache]==5.1.0 (pypi.nvidia.com) and Isaac Lab v2.3.2
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab && git -C third_party/IsaacLab checkout v2.3.2
git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git third_party/so101_workshop
git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/sim_assets/SO-ARM100
python isaac_bench/assets/fetch_ycb.py      # YCB objects + physics wrappers
python isaac_bench/make_textures.py         # table-top textures
```

Data, checkpoints, environments and third-party repos are not versioned (see `.gitignore`).

## Collecting data with the trajectory tool

A spec is a list of steps `{"node": id, "gripper": "open" | "close"}`; each step first drives the gripper to that
state, then moves the grasp centre to the node. Nodes are absolute positions or relative to an object
(`{"obj": "$obj", "offset": [0, 0, 0.07], "yaw": "auto"}`), resolved when the step starts, so one template works for
every randomised layout.

```bash
cd isaac_bench
python traj_tool.py --spec specs/place.json --episodes 1 --video out/demo.mp4          # annotated video
python traj_tool.py --spec specs/place.json --episodes 200 --num_envs 4 --out ../data/place   # dataset
```

## Tasks, randomisation and synthetic data

Tasks in the Isaac tabletop (`isaac_bench/tabletop.py`, instruction generated per episode):
`pick`, `pad` ("put X on the pink pad"), `tray`, `bowl` (held out for few-shot tests), `push` ("push X left / away from you"),
`topple` ("knock over X"), `rotate` ("rotate X by 90 degrees"), `rel` ("put X to the left of / behind Y").
Specs: `specs/pick.json`, `specs/place.json` (pad / tray / bowl / rel), `specs/push.json` (push / topple), `specs/rotate.json`.

Domain randomisation (`TABLETOP_DR=1`, default): 11 table textures, per-env lamp (intensity, colour temperature,
position), HDRI dome per run, front-camera pose jitter, object mass / friction, robot joint damping / friction; the
sampled values are stored in each episode's metadata.

Isaac Lab Mimic (`isaac_bench/nanovla_mimic/`): manager-based variants of the tabletop (end-effector pose actions,
subtask signal "grasp") for fixed (task, object, target) triples. Record source demos (`record_src.py`, scripted now,
teleoperation later), then Isaac Lab's `annotate_demos.py --auto` and `generate_dataset.py` (through
`run_isaaclab_script.py`, which registers the tasks) and `hdf5_to_bench.py` to get NanoVLA data. Smoke test: 3 source
demos -> 5 generated (5/6 attempts), see `smoke_tests.sh`.

Cosmos Transfer (`cosmos/`): `traj_tool.py --cosmos_dir` (with `TABLETOP_CAM_EXTRA=1 TABLETOP_CAM_SCALE=4`) exports
RGB + depth + segmentation videos per episode; `make_specs.py` writes Cosmos-Transfer2.5 specs with style prompts,
`run_cosmos.sh` runs them on an 80 GB-class GPU (~65 GB VRAM needed), `import_cosmos.py` turns the outputs into new
episodes with the original actions.

## Layout

```
NanoVLA (~/VLA)
├── vla/                    model + data pipeline + MuJoCo benchmark (python 3.10, ~/VLM/venv)
│   ├── prep_data.py, pack_data.py, fetch_info.py      download / pack LeRobot SO-100/101 data
│   ├── clean_data.py, clean_v2.py, audit_labels.py,   data cleaning, label audit, task clusters,
│   │   task_clusters.py, viz_clean.py, make_videos.py  before/after visualisation
│   ├── models.py, train_vision.py, train_text.py,     encoders (from scratch), alignment, VLA head
│   │   train_align.py, train_vla.py
│   ├── fewshot.py, prep_fewshot.py                     few-shot data prep + fine-tune (real held-out repos, offline L1)
│   ├── sim_env.py, sim_data.py, sim_fewshot.py,       MuJoCo SO-101 env, scripted expert, closed-loop SR benchmark,
│   │   sim_diag.py, sim_phase.py, aggregate_final.py   success/failure diagnostics, result aggregation
│   ├── merge_sim_bench.py                              merge demo sets (sim_bench600)
│   ├── convert_isaac_ds.py, isaac_policy_server.py    LeRobot v3 -> bench format; policy server for Isaac eval
│   ├── run_scaling.sh                                  demo-count scaling study (running)
│   └── runs/                                           earlier experiment launchers (rounds 1-4, final, pretrain)
├── isaac_bench/            Isaac Lab workspace (python 3.11, envs/isaac_env)
│   ├── tabletop.py         table + SO-101 + YCB household objects, random table tops, tasks, success checks
│   ├── tabletop_expert.py  grasp-centre kinematics + DLS IK on the PhysX Jacobian
│   ├── traj_tool.py        node-based trajectory tool (gripper state first, then move) -> data + annotated video
│   ├── specs/              trajectory specs (place.json, pick.json templates; pick_pad_then_tray.json demo)
│   ├── isaac_eval.py       our VLA on the NVIDIA workshop vials->rack task (GR00T comparison)
│   ├── assets/ycb/         YCB meshes + *_phys.usda wrappers (rigid body, mass, collision), objects.json
│   ├── textures/, make_textures.py                     procedural table-top textures
│   ├── out/                videos (traj_demo.mp4, ycb_place.mp4), out/debug/ = probe images and logs
│   ├── dev/                one-off probes (robot links, IK check, scene render, env smoke test)
│   └── legacy/             first block-based scene + expert generator (superseded by YCB scene + traj_tool)
├── data/
│   ├── frames/, packed/, meta/, info/                  pretraining data (LeRobot hub, 1.7M frames)
│   ├── clean_v2/                                       cleaned episodes, calibration, label fixes, report.json
│   ├── bench/, actions/                                held-out real repos for few-shot (offline L1)
│   ├── sim_bench/ (100 demos), sim_bench600/ (600)     MuJoCo expert demos
│   ├── isaac/, isaac_bench/                            NVIDIA workshop teleop dataset (raw / converted)
│   ├── final/, scaling/                                benchmark results (summary.json, per-job .out)
│   └── logs/                                           pretraining + few-shot round logs
├── ckpt/                   vision/, text_v3/, align/, vla/ (v1, used), vla_v2/, sim_* results; _old/ = superseded runs
├── third_party/
│   ├── IsaacLab/           Isaac Lab v2.3.2 source (installed editable into envs/isaac_env)
│   ├── so101_workshop/     NVIDIA Sim-to-Real SO-101 workshop (tasks + SO-101 USD), used via sys.path
│   ├── sim_assets/         TheRobotStudio SO-101 MJCF/URDF
│   └── sim_pkgs/           MuJoCo build for python 3.10
├── envs/isaac_env/         Python 3.11 venv with Isaac Sim 5.1 + Isaac Lab
├── scripts/                fewshot_distill.sh (your task end to end), dagger_round.sh
├── deploy/                 run_student.py: student on the real SO-101 / Jetson (ONNX Runtime, TensorRT fp16)
├── tools/                  hf_select.py, hf_unify.py, to_lerobot.py, smol_offline.py (teacher eval + labels), log2tb.py,
│                           nano_ik.py, eval_*.sh, export_bench.py, dashboards
├── vla/student.py          wrist-camera student (distillation target), vla/smolvla_server.py (teacher in the Isaac eval)
├── viz/                    Manim sources of the architecture / pipeline videos
├── results/                small result summaries that are versioned (JSON)
└── logs/install/           Isaac Sim / Isaac Lab install logs (not versioned)
```

Reports: evaluation report (Claude Doc) and rollout / trajectory-tool pages (Claude artifacts), linked in the chat.
