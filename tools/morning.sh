#!/bin/bash
# Morning pipeline (runs after the overnight data generation): merge -> pretrain VLA-Isaac v1 (wrist only) -> evals.
set -u
V=~/VLA; R=$V/results/final_v1; L=$R/morning.log; mkdir -p $R
log() { echo "[$(date +%T)] $*" >> $L; }
until grep -q "GEN V2 DONE" $V/data/isaac_v2/gen.log 2>/dev/null; do sleep 60; done
log "generation done: $(grep -c 'done' $V/data/isaac_v2/gen.log) chunks"
cd $V/vla && source ~/VLM/venv/bin/activate && export MUJOCO_GL=egl
SRC=$(ls -d $V/data/isaac_v2/*_c* 2>/dev/null | while read d; do [ -f $d/episodes.json ] && echo $d; done)
python merge_sim_bench.py $V/data/isaac_all_v2 $V/data/isaac_pre/place_c0 $V/data/isaac_pre/place_c1 $V/data/isaac_pre/place_c2 $V/data/isaac_pre/place_c3 \
    $V/data/isaac_pre/pick_c0 $V/data/isaac_pre/pick_c1 $V/data/isaac_pre/pick_c2 $V/data/isaac_pre/pick_c3 $SRC >> $L 2>&1
UNS=$(ls -d $V/data/isaac_v2/fs_unseen_* 2>/dev/null | while read d; do [ -f $d/episodes.json ] && echo $d; done)
[ -n "$UNS" ] && python merge_sim_bench.py $V/data/isaac_v2/fs_unseen $UNS >> $L 2>&1
log "merged: $(tail -1 $L)"
$V/tools/gpu.sh 4000 python -u isaac_pretrain.py --bench isaac_all_v2 --cams wrist --aug strong --steps ${STEPS:-20000} --n_val 60 \
    --out $V/ckpt/vla_isaac_wrist_v1/vla.pt > $R/pretrain_v1.log 2>&1
log "pretrain: $(tail -2 $R/pretrain_v1.log | tr '\n' ' ')"
CK=$V/ckpt/vla_isaac_wrist_v1/vla.pt
# A. zero-shot multi-task on seen tasks in the new randomised scene (also measures wrong-object picks)
BENCH=isaac_all_v2 $V/tools/gpu.sh 3000 $V/tools/eval_zero_shot.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_zeroshot_multitask.json 6401
log "A: $(grep RESULT $R/A_zeroshot_multitask.eval.log | cut -c1-200)"
# B. few-shot on the held-out bowl task: VLA-Isaac v1 vs VLA v1 (community pretraining only)
for cfg in "isaac 10" "isaac 50" "v1 50" "isaac 25"; do set -- $cfg
  ck=$CK; [ $1 = v1 ] && ck=$V/ckpt/vla/vla.pt
  $V/tools/gpu.sh 3000 $V/tools/eval_ft.sh $ck isaac_v2/fs_bowl $2 1500 bowl 24 $R/B_bowl_$1_N$2.json 64$((10 + $2))
  log "B $1 N=$2: $(grep RESULT $R/B_bowl_$1_N$2.eval.log | cut -c1-200)"
done
# C. unseen objects (never in pretraining): zero-shot vs fine-tuned on their pick demos
EVAL_X="--objects glue_stick,spool,toy_brick" BENCH=isaac_all_v2 $V/tools/gpu.sh 3000 $V/tools/eval_zero_shot.sh $CK pick 24 $R/C_unseen_zeroshot.json 6481
log "C zero-shot: $(grep RESULT $R/C_unseen_zeroshot.eval.log | cut -c1-200)"
EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_ft.sh $CK isaac_v2/fs_unseen 60 1500 pick 24 $R/C_unseen_ft.json 6482
log "C fine-tuned: $(grep RESULT $R/C_unseen_ft.eval.log | cut -c1-200)"
log "MORNING DONE"
