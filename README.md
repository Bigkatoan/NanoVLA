# NanoVLA

A ~11M-parameter vision-language-action model for the **SO-ARM101** arm, trained from scratch: a small ViT and a
small text encoder are pretrained separately, their embedding spaces are aligned, and an ACT-style action head maps
(front image, wrist image, instruction, joint state) to 1.6 s action chunks. The goal is a few-shot pipeline: adapt to
a new task from tens of demonstrations, measured by closed-loop **success rate**. Target hardware: Jetson Orin Nano.

## Components

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
├── results/                small result summaries that are versioned (JSON)
└── logs/install/           Isaac Sim / Isaac Lab install logs (not versioned)
```

Reports: evaluation report (Claude Doc) and rollout / trajectory-tool pages (Claude artifacts), linked in the chat.
