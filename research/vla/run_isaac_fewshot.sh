#!/bin/bash
# Few-shot test on the held-out bowl task in the Isaac tabletop:
# start point (VLA v1 | VLA-Isaac | scratch) x N demos x seeds -> fine-tune + serve (py3.10) -> closed-loop eval (Isaac).
VLA=~/VLA; OUT=$VLA/research/results/isaac_fewshot; mkdir -p $OUT; LOG=$OUT/run.log
SEEDS=${SEEDS:-"0 1"}; NS=${NS:-"10 20 50"}; STARTS=${STARTS:-"isaac v1 scratch"}; EPIS=${EPIS:-50}
# CAMS=wrist: wrist-camera-only policy + client (start 'isaac' = ckpt/vla_isaac_wrist), AUG=strong: heavy image augmentation
CAMS=${CAMS:-both}; AUG=${AUG:-base}; SRV_X=""; EVAL_X=""; ISAAC_CK=$VLA/ckpt/vla_isaac/vla.pt
if [ "$CAMS" = wrist ]; then SRV_X="--cams wrist"; EVAL_X="--cams wrist"; ISAAC_CK=${ISAAC_CK_WRIST:-$VLA/ckpt/vla_isaac_wrist/vla.pt}; OUT=$VLA/research/results/isaac_fewshot_wrist; mkdir -p $OUT; LOG=$OUT/run.log; fi
[ "$AUG" = strong ] && SRV_X="$SRV_X --aug strong"
port=6100
for seed in $SEEDS; do for N in $NS; do for start in $STARTS; do
  tag=${start}_N${N}_s${seed}; [ -f $OUT/$tag.json ] && continue
  case $start in
    isaac) cfg=fixenc; ck=$ISAAC_CK;;
    v1) cfg=fixenc; ck=$VLA/ckpt/vla/vla.pt;;
    scratch) cfg=scratch; ck=$VLA/ckpt/vla/vla.pt;;
  esac
  port=$((port + 1))
  ( cd $VLA/research/vla && source ~/VLM/venv/bin/activate && export MUJOCO_GL=egl && \
    exec python -u isaac_policy_server.py --bench isaac_pre/bowl_fewshot --cfg $cfg --vla $ck --N $N --seed $seed --steps 1500 --port $port $SRV_X ) > $OUT/$tag.server.log 2>&1 &
  spid=$!
  until grep -q '"ready"' $OUT/$tag.server.log || ! kill -0 $spid 2>/dev/null; do sleep 5; done
  if kill -0 $spid 2>/dev/null; then
    ( cd $VLA/research/isaac_bench && source $VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && \
      python -u tabletop_eval.py --tasks bowl --episodes $EPIS --num_envs 4 --seed $((20000 + seed)) --port $port --out $OUT/$tag.json $EVAL_X ) > $OUT/$tag.eval.log 2>&1
  fi
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  echo "[$(date)] $tag: $(grep RESULT $OUT/$tag.eval.log | cut -c1-160)" >> $LOG
done; done; done
echo "[$(date)] FEWSHOT DONE" >> $LOG
