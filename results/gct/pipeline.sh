#!/bin/bash
# GCT full pipeline (after the overnight stage-0 data): few-shot demo sets -> stage 1 -> G1 -> stage 2 online DAgger
# -> G2 -> stage 3 few-shot adapters -> stage 4 export + latency. Everything logged here and on TensorBoard.
V=~/VLA; R=$V/results/gct; IB=$V/isaac_bench
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
A_TASKS=pick,pad,tray,container,push,topple,rotate,rel
evalA() { ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/$3.eval.log $V/tb/$2 $3 ) & EVAL_X="--diag 1" $V/tools/gpu.sh 3600 $V/tools/eval_c.sh $1 $A_TASKS 48 $R/$3.json $4; log "$3: $(grep RESULT $R/$3.eval.log | cut -c1-230)"; }
evalC() { ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/$3.eval.log $V/tb/$2 $3 ) & EVAL_X="--diag 1 --objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $1 pick 24 $R/$3.json $4; log "$3: $(grep RESULT $R/$3.eval.log | cut -c1-230)"; }
evalPick() { ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/$3.eval.log $V/tb/$2 $3 ) & EVAL_X="--diag 1" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $1 pick 24 $R/$3.json $4; log "$3: $(grep RESULT $R/$3.eval.log | cut -c1-230)"; }
GX="--skip_frames 2 --start_noise 0.3 --start_noise_p 0.5 --dart_mix 0:0.3,0.03:0.4,0.08:0.3 --slow_near 2 --labels 1 --num_envs 12"
until grep -q "GEN RETRY DONE" $V/data/isaac_gct/gen.log 2>/dev/null; do sleep 60; done
touch $V/data/isaac_gct/STOP_TB
log "stage0: $(python3 -c "import json,glob; L=[json.load(open(f)) for f in glob.glob('$V/data/isaac_gct/*_c*/episodes.json')]; print(len(L), 'chunks,', sum(d['kept'] for d in L), 'kept of', sum(d['attempts'] for d in L))")"
# ---- few-shot demo sets with the stage-0 recipe (never used in pre-training)
( cd $IB && source $V/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES
  $V/tools/gpu.sh 3600 python -u traj_tool.py --spec specs/place.json --tasks bowl --episodes 60 --seed 9301 $GX --out $V/data/fs_bowl_gct > $V/data/fs_bowl_gct.log 2>&1
  for o in glue_stick spool toy_brick; do
    $V/tools/gpu.sh 2400 python -u traj_tool.py --spec specs/_pick_$o.json --episodes 30 --seed 9401 $GX --out $V/data/fs_unseen_gct_$o > $V/data/fs_unseen_gct_$o.log 2>&1
  done )
~/VLM/venv/bin/python $V/vla/merge_sim_bench.py $V/data/fs_unseen_gct $V/data/fs_unseen_gct_glue_stick $V/data/fs_unseen_gct_spool $V/data/fs_unseen_gct_toy_brick > $R/merge_unseen.log 2>&1
log "few-shot sets: bowl $(python3 -c "import json; print(json.load(open('$V/data/fs_bowl_gct/episodes.json'))['kept'])" 2>&1 | tail -1), unseen $(tail -1 $R/merge_unseen.log)"
# ---- stage 1
SRC=$(ls -d $V/data/isaac_gct/*_c* | while read d; do [ -f $d/episodes.json ] && echo $d; done | tr '\n' ' ')
~/VLM/venv/bin/python $V/vla/merge_sim_bench.py $V/data/isaac_gct_v1 $V/data/isaac_all_v2 $V/data/dagger_c1/place $SRC > $R/merge.log 2>&1
log "merged: $(tail -1 $R/merge.log)"
CK1=$V/ckpt/nanovla_g/vlag.pt; mkdir -p $(dirname $CK1)
log "stage1 train"
( cd $V/vla && $V/tools/gpu.sh 16000 ~/VLM/venv/bin/python -u nanovla_g.py --run gct_s1 --bench isaac_gct_v1 --steps 20000 --out $CK1 ) > $(dirname $CK1)/train.log 2>&1
log "stage1 trained: $(grep val_rel $(dirname $CK1)/train.log | tail -1)"
evalPick $CK1 gct_s1 G1_pick 7101; evalA $CK1 gct_s1 G1_A 7102; evalC $CK1 gct_s1 G1_C 7103
log "G1 gate: $(grep val_rel $(dirname $CK1)/train.log | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('vision_reliance', d['vision_reliance'], '(>=1.5)', 'loc_err_cm', d['loc_err_cm'], '(<2)')")"
# ---- stage 2 online DAgger
CK2=$V/ckpt/nanovla_g_online/vlag.pt
log "stage2 online DAgger (${S2H:-10} h)"
( cd $IB && source $V/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && \
  TABLETOP_HOLDOUT=glue_stick,spool,toy_brick $V/tools/gpu.sh $(( ${S2H:-10} * 3600 + 3600 )) python -u online_dagger.py --init $CK1 --bench isaac_gct_v1 --num_envs 8 --hours ${S2H:-10} --run gct_s2 --out $CK2 ) > $R/stage2.log 2>&1
log "stage2: $(grep -E '"saved"' $R/stage2.log | tail -1)"
evalPick $CK2 gct_s2 G2_pick 7201; evalA $CK2 gct_s2 G2_A 7202; evalC $CK2 gct_s2 G2_C 7203
# ---- stage 3 few-shot (base = best of stage 1 / 2 on A)
best=$CK2; python3 -c "import json,sys; a=json.load(open('$R/G1_A.json'))['success_rate']; b=json.load(open('$R/G2_A.json'))['success_rate']; sys.exit(0 if b >= a else 1)" || best=$CK1
log "stage3 base: $best"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/B_bowl_zeroshot.eval.log $V/tb/gct_s3 B_bowl_zeroshot ) &
$V/tools/gpu.sh 3000 $V/tools/eval_c.sh $best bowl 24 $R/B_bowl_zeroshot.json 7300; log "B bowl zero-shot: $(grep RESULT $R/B_bowl_zeroshot.eval.log | cut -c1-200)"
for N in 5 10 25 50; do
  AD=$V/ckpt/adapters/bowl_N$N.pt
  ( cd $V/vla && $V/tools/gpu.sh 3600 ~/VLM/venv/bin/python -u fewshot_c.py --base $best --demos fs_bowl_gct --N $N --out $AD ) > $R/fs_bowl_N$N.log 2>&1
  ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/B_bowl_N$N.eval.log $V/tb/gct_s3 B_bowl_N$N ) &
  SRV_X="--adapter $AD" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $best bowl 24 $R/B_bowl_N$N.json 73$((10 + N)); log "B bowl N=$N: $(grep RESULT $R/B_bowl_N$N.eval.log | cut -c1-200)"
done
for N in 10 25; do
  AD=$V/ckpt/adapters/unseen_N$N.pt
  ( cd $V/vla && $V/tools/gpu.sh 3600 ~/VLM/venv/bin/python -u fewshot_c.py --base $best --demos fs_unseen_gct --N $N --out $AD ) > $R/fs_unseen_N$N.log 2>&1
  ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/C_unseen_N$N.eval.log $V/tb/gct_s3 C_unseen_N$N ) &
  SRV_X="--adapter $AD" EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $best pick 24 $R/C_unseen_N$N.json 74$N; log "C unseen N=$N: $(grep RESULT $R/C_unseen_N$N.eval.log | cut -c1-200)"
done
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/A_forget_bowlN25.eval.log $V/tb/gct_s3 A_forget_bowlN25 ) &
SRV_X="--adapter $V/ckpt/adapters/bowl_N25.pt" $V/tools/gpu.sh 3600 $V/tools/eval_c.sh $best $A_TASKS 24 $R/A_forget_bowlN25.json 7350; log "A with bowl adapter (forgetting): $(grep RESULT $R/A_forget_bowlN25.eval.log | cut -c1-200)"
# ---- stage 4 export + latency
( cd $V/vla && $V/tools/gpu.sh 1800 $V/envs/deploy/bin/python -u $V/tools/export_bench.py --ckpt $best --out $V/deploy ) > $R/stage4.log 2>&1
log "stage4: $(tail -3 $R/stage4.log | tr '\n' ' ')"
wait; log "PIPELINE DONE"
