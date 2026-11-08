#!/bin/bash
# Day 1 of the teacher -> few-shot -> distill schedule (proxy task: fport tic-tac-toe, 45 eps, 9 instructions).
# test episodes: one per instruction + ep 44; few-shot pools: 1 / 2 / all remaining demos per instruction.
export HF_HUB_OFFLINE=1
V=~/VLA; R=$V/results/day1; PY=$V/envs/lerobot/bin/python; DS=$V/data/lerobot/ttt_v1
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
TEST="4,9,14,18,23,28,33,38,43,44"; N9="[0,5,10,15,19,24,29,34,39]"; N18="[0,1,5,6,10,11,15,16,19,20,24,25,29,30,34,35,39,40]"
N35="[0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,19,20,21,22,24,25,26,27,29,30,31,32,34,35,36,37,39,40,41,42]"
TR35="0-3,5-8,10-13,15-17,19-22,24-27,29-32,34-37,39-42"
until [ -d $V/ckpt/smolvla_deg/checkpoints/020000/pretrained_model ] && ! pgrep -f "lerobot-train.*smolvla_deg" >/dev/null; do sleep 60; done
TEACH=$V/ckpt/smolvla_deg/checkpoints/020000/pretrained_model; BASE=$V/ckpt/smolvla_base
log "teacher done: $(grep -oE 'step:20K .*' $V/results/smolvla/train.log | tail -1)"
G() { $V/tools/gpu.sh 20000 "$@"; }
ev() { G $PY $V/tools/smol_offline.py eval --ckpt $1 --root $DS --episodes $TEST --out $R/$2.json > $R/$2.log 2>&1; log "$2: $(tail -1 $R/$2.log)"; }
ev $TEACH eval_teacher_zeroshot
ft() { local name=$1 init=$2 eps=$3
  ( cd $V && HF_HUB_OFFLINE=1 G $V/envs/lerobot/bin/lerobot-train --policy.path=$init --dataset.repo_id=local/ttt --dataset.root=$DS --dataset.episodes="$eps" \
     --batch_size=32 --steps=2000 --log_freq=100 --save_freq=2000 --output_dir=$V/ckpt/fewshot/$name --policy.push_to_hub=false --wandb.enable=false \
     --num_workers=8 --policy.device=cuda --policy.scheduler_warmup_steps=200 --policy.scheduler_decay_steps=2000 ) > $R/ft_$name.log 2>&1
  ev $V/ckpt/fewshot/$name/checkpoints/002000/pretrained_model eval_$name; }
rm -rf $V/ckpt/fewshot
ft teacher_n9 $TEACH "$N9"
ft teacher_n18 $TEACH "$N18"
ft teacher_n35 $TEACH "$N35"
ft base_n35 $BASE "$N35"
FS=$V/ckpt/fewshot/teacher_n35/checkpoints/002000/pretrained_model
G $PY $V/tools/smol_offline.py label --ckpt $FS --root $DS --episodes $TR35 --stride 3 --aug 2 --bs 16 --out $R/labels_train.npz > $R/label_train.log 2>&1; log "labels train: $(tail -1 $R/label_train.log)"
G $PY $V/tools/smol_offline.py label --ckpt $FS --root $DS --episodes $TEST --stride 10 --aug 0 --bs 16 --out $R/labels_test.npz > $R/label_test.log 2>&1; log "labels test: $(tail -1 $R/label_test.log)"
for t in gt teacher both; do
  ( cd $V/vla && G $PY student.py --train $R/labels_train.npz --test $R/labels_test.npz --target $t --out $V/ckpt/student/ttt_$t.pt --tb $V/tb/student_$t ) > $R/student_$t.log 2>&1
  log "student $t: $(tail -1 $R/student_$t.log)"
done
log "DAY1 DONE"
