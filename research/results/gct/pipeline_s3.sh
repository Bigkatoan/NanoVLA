#!/bin/bash
# continuation after stopping stage 2 (online DAgger degraded): record the partial online checkpoint, then stage 3 + 4
V=~/VLA; R=$V/research/results/gct; IB=$V/research/isaac_bench
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
A_TASKS=pick,pad,tray,container,push,topple,rotate,rel
CK1=$V/ckpt/nanovla_g/vlag.pt; CK2=$V/ckpt/nanovla_g_online/vlag.pt
log "stage2 STOPPED after 4.9 h (1146 episodes): success with expert help fell 0.59 -> 0.34, interventions/episode 5.0 -> 6.6"
( cd $V/research/vla && ~/VLM/venv/bin/python $V/research/tools/tb_eval.py $R/G2_pick_partial.eval.log $V/tb/gct_s2 G2_pick_partial ) &
EVAL_X="--diag 1" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $CK2 pick 24 $R/G2_pick_partial.json 7211; log "G2_pick_partial: $(grep RESULT $R/G2_pick_partial.eval.log | cut -c1-200)"
best=$CK1; python3 -c "import json,sys; a=json.load(open('$R/G1_pick.json'))['success_rate']; b=json.load(open('$R/G2_pick_partial.json'))['success_rate']; sys.exit(0 if b > a else 1)" && best=$CK2
log "stage3 base: $best"
( cd $V/research/vla && ~/VLM/venv/bin/python $V/research/tools/tb_eval.py $R/B_bowl_zeroshot.eval.log $V/tb/gct_s3 B_bowl_zeroshot ) &
$V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $best bowl 24 $R/B_bowl_zeroshot.json 7300; log "B bowl zero-shot: $(grep RESULT $R/B_bowl_zeroshot.eval.log | cut -c1-200)"
for N in 5 10 25 50; do
  AD=$V/ckpt/adapters/bowl_N$N.pt
  ( cd $V/research/vla && $V/research/tools/gpu.sh 3600 ~/VLM/venv/bin/python -u fewshot_c.py --base $best --demos fs_bowl_gct --N $N --out $AD ) > $R/fs_bowl_N$N.log 2>&1
  ( cd $V/research/vla && ~/VLM/venv/bin/python $V/research/tools/tb_eval.py $R/B_bowl_N$N.eval.log $V/tb/gct_s3 B_bowl_N$N ) &
  SRV_X="--adapter $AD" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $best bowl 24 $R/B_bowl_N$N.json 73$((10 + N)); log "B bowl N=$N: $(grep RESULT $R/B_bowl_N$N.eval.log | cut -c1-200)"
done
for N in 10 25; do
  AD=$V/ckpt/adapters/unseen_N$N.pt
  ( cd $V/research/vla && $V/research/tools/gpu.sh 3600 ~/VLM/venv/bin/python -u fewshot_c.py --base $best --demos fs_unseen_gct --N $N --out $AD ) > $R/fs_unseen_N$N.log 2>&1
  ( cd $V/research/vla && ~/VLM/venv/bin/python $V/research/tools/tb_eval.py $R/C_unseen_N$N.eval.log $V/tb/gct_s3 C_unseen_N$N ) &
  SRV_X="--adapter $AD" EVAL_X="--objects glue_stick,spool,toy_brick" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $best pick 24 $R/C_unseen_N$N.json 74$N; log "C unseen N=$N: $(grep RESULT $R/C_unseen_N$N.eval.log | cut -c1-200)"
done
( cd $V/research/vla && ~/VLM/venv/bin/python $V/research/tools/tb_eval.py $R/A_forget_bowlN25.eval.log $V/tb/gct_s3 A_forget_bowlN25 ) &
SRV_X="--adapter $V/ckpt/adapters/bowl_N25.pt" $V/research/tools/gpu.sh 3600 $V/research/tools/eval_c.sh $best $A_TASKS 24 $R/A_forget_bowlN25.json 7350; log "A with bowl adapter (forgetting): $(grep RESULT $R/A_forget_bowlN25.eval.log | cut -c1-200)"
# ---- stage 4 export + latency
( cd $V/research/vla && $V/research/tools/gpu.sh 1800 $V/envs/deploy/bin/python -u $V/research/tools/export_bench.py --ckpt $best --out $V/deploy ) > $R/stage4.log 2>&1
log "stage4: $(tail -3 $R/stage4.log | tr '\n' ' ')"
wait; log "PIPELINE DONE"
