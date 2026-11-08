#!/bin/bash
# DAgger on NanoVLA-C: policy drives short segments in sim, expert recovers; aggregate; fine-tune; closed-loop eval. 2 rounds.
V=~/VLA; R=$V/results/closed; CUR=$V/ckpt/nanovla_c/vlac.pt; BASE=isaac_all_v2
log() { echo "$(date +%H:%M) $*" | tee -a $R/pipeline.log; }
export SERVER=vlac_server.py
log "DAgger smoke test (2 envs, 2 episodes)"
NE=2 $V/tools/gpu.sh 900 $V/tools/dagger_gen.sh $CUR pick pick 2 $V/data/dagger_c_smoke 6900 9
if [ ! -f $V/data/dagger_c_smoke/episodes.json ]; then log "SMOKE FAILED: $(grep -E 'Error|Traceback' -A3 $V/data/dagger_c_smoke.log | head -6 | tr '\n' ' ')"; exit 1; fi
log "smoke ok: $(python3 -c "import json; d=json.load(open('$V/data/dagger_c_smoke/episodes.json')); print(d['kept'], 'kept of', d['attempts'], [e.get('policy_frac') for e in d['episodes']])")"
for RD in 1 2; do
  D=$V/data/dagger_c$RD; mkdir -p $D; SRC=""; TBR=$V/tb/closed_dagger$RD; rm -f $D/STOP
  ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_gen.py $TBR "$D/*.log" $D/STOP ) &
  for J in "place pad,tray,container,rel" "pick pick" "push push,topple" "rotate rotate"; do set -- $J
    log "DAgger r$RD gen $1"
    $V/tools/gpu.sh 5400 $V/tools/dagger_gen.sh $CUR $1 $2 ${EPJ:-60} $D/$1 69$RD$((${#1} % 10)) $((7000 + 100 * RD + ${#1}))
    log "  $1: $(python3 -c "import json; d=json.load(open('$D/$1/episodes.json')); print(d['kept'], 'kept of', d['attempts'])" 2>&1 | tail -1)"
    [ -f $D/$1/episodes.json ] && SRC="$SRC $D/$1"
  done
  touch $D/STOP
  ~/VLM/venv/bin/python $V/vla/merge_sim_bench.py $V/data/isaac_dagger_c$RD $V/data/$BASE $SRC > $D/merge.log 2>&1
  log "merged r$RD: $(tail -1 $D/merge.log)"
  CK=$V/ckpt/nanovla_c_dagger$RD/vlac.pt; mkdir -p $(dirname $CK)
  log "train closed_dagger$RD"
  ( cd $V/vla && $V/tools/gpu.sh 9000 ~/VLM/venv/bin/python -u nanovla_c.py --run closed_dagger$RD --bench isaac_dagger_c$RD --init $CUR --steps 10000 \
      --lr 1e-4 --enc_lr 1e-5 --own_ramp 0.01 --out $CK ) > $(dirname $CK)/train.log 2>&1
  log "trained r$RD: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
  ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/A_dagger$RD.eval.log $TBR A_multitask ) &
  $V/tools/gpu.sh 3600 $V/tools/eval_c.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_dagger$RD.json 690$RD
  log "A r$RD: $(grep RESULT $R/A_dagger$RD.eval.log | cut -c1-220)"
  ( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/C_dagger$RD.eval.log $TBR C_unseen ) &
  EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $CK pick 24 $R/C_dagger$RD.json 691$RD
  log "C r$RD: $(grep RESULT $R/C_dagger$RD.eval.log | cut -c1-220)"
  CUR=$CK; BASE=isaac_dagger_c$RD
done
wait; log "DAGGER DONE"
