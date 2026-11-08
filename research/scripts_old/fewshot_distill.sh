#!/bin/bash
# Your task, end to end on the PC: few-shot the SmolVLA teacher on your demos, distil the wrist-only student, export ONNX.
#   scripts/fewshot_distill.sh <name> <dataset_root> <train_episodes> <test_episodes>
#   e.g. scripts/fewshot_distill.sh pick_cup data/lerobot/my_pick_cup "[0,1,2,3,4,5,6,7,8,9,10,11]" 12-14
# <dataset_root>: a LeRobot v3 dataset recorded with lerobot-record (SO-101, use_degrees=True, a wrist + an overview camera).
# Outputs: ckpt/fewshot/<name>/ (teacher), ckpt/student/<name>.{pt,onnx,meta.json}, results/<name>/*.json, TensorBoard runs.
set -e
NAME=$1; DS=$(realpath $2); TRAIN=$3; TEST=$4
V=$(cd $(dirname $0)/.. && pwd); PY=$V/envs/lerobot/bin/python; R=$V/research/results/$NAME; mkdir -p $R; export HF_HUB_OFFLINE=1
TEACH=${TEACH:-$V/ckpt/smolvla_deg/checkpoints/020000/pretrained_model}; STEPS=${STEPS:-2000}
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
G() { $V/research/tools/gpu.sh 30000 "$@"; }
# 1. camera keys -> camera1 (wrist) / camera2 (overview), fix stale episode-metadata indices
$PY - <<PYEOF
import sys, glob, os, re, pandas as pd; sys.path.insert(0, "$V/research/tools"); import hf_unify as H
H.unify("$DS")
for f in glob.glob(os.path.join("$DS", "meta", "episodes", "*", "*.parquet")):
    c, i = (int(x) for x in re.findall(r"chunk-(\d+)/file-(\d+)", f)[0]); df = pd.read_parquet(f)
    if (df["meta/episodes/chunk_index"] != c).any() or (df["meta/episodes/file_index"] != i).any():
        df["meta/episodes/chunk_index"] = c; df["meta/episodes/file_index"] = i; df.to_parquet(f)
PYEOF
log "dataset ready: $DS"
# 2. few-shot the teacher (action expert only, as in SmolVLA's fine-tuning recipe)
( cd $V && G $V/envs/lerobot/bin/lerobot-train --policy.path=$TEACH --dataset.repo_id=local/$NAME --dataset.root=$DS --dataset.episodes="$TRAIN" \
   --batch_size=32 --steps=$STEPS --log_freq=100 --save_freq=$STEPS --output_dir=$V/ckpt/fewshot/$NAME --policy.push_to_hub=false --wandb.enable=false \
   --num_workers=8 --policy.device=cuda --policy.scheduler_warmup_steps=200 --policy.scheduler_decay_steps=$STEPS ) > $R/fewshot.log 2>&1
FS=$(ls -d $V/ckpt/fewshot/$NAME/checkpoints/0*/pretrained_model | tail -1)
G $PY $V/research/tools/smol_offline.py eval --ckpt $FS --root $DS --episodes $TEST --out $R/eval_teacher.json > $R/eval_teacher.log 2>&1; log "teacher: $(tail -1 $R/eval_teacher.log)"
# 3. teacher labels (train episodes + 2 augmented copies; test episodes)
TR=$(echo "$TRAIN" | tr -d '[] ')
G $PY $V/research/tools/smol_offline.py label --ckpt $FS --root $DS --episodes $TR --stride 3 --aug 2 --out $R/labels_train.npz > $R/label_train.log 2>&1
G $PY $V/research/tools/smol_offline.py label --ckpt $FS --root $DS --episodes $TEST --stride 10 --aug 0 --out $R/labels_test.npz > $R/label_test.log 2>&1
log "labels: $(tail -1 $R/label_train.log)"
# 4. student (wrist only), teacher + demo targets; ONNX + meta for deploy/run_student.py
( cd $V/research/vla && G $PY student.py --train $R/labels_train.npz --test $R/labels_test.npz --target both --out $V/ckpt/student/$NAME.pt --tb $V/tb/student_$NAME ) > $R/student.log 2>&1
log "student: $(tail -1 $R/student.log)"
