#!/bin/bash
# Closed-loop eval of a SmolVLA checkpoint: smolvla_server + tabletop_eval (wrist, 240x320).  eval_smol.sh <ckpt> <tasks> <eps> <out.json> <port> [extra]
CK=$1; TASKS=$2; EP=$3; OUT=$4; PORT=$5; shift 5
mkdir -p $(dirname $OUT)
( cd ~/VLA && HF_HUB_OFFLINE=1 exec ~/VLA/envs/lerobot/bin/python -u vla/smolvla_server.py --ckpt $CK --port $PORT --cams both --log ${OUT%.json}.lat.jsonl ) > ${OUT%.json}.server.log 2>&1 &
SP=$!
until grep -q '"ready"' ${OUT%.json}.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/research/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES TABLETOP_RES=240x320 TABLETOP_FRONT=1 && \
  python -u tabletop_eval.py --cams both --tasks $TASKS --episodes $EP --num_envs 4 --port $PORT --diag 1 --out $OUT --video ${OUT%.json}.mp4 "$@" ) > ${OUT%.json}.eval.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep RESULT ${OUT%.json}.eval.log | cut -c1-300
