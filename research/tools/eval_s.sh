#!/bin/bash
# Closed-loop eval of NanoVLA-S: vlas_server + tabletop_eval_s (IK servo).  eval_s.sh <ckpt> <tasks> <episodes> <out.json> <port> [extra eval args...]
CK=$1; TASKS=$2; EP=$3; OUT=$4; PORT=$5; shift 5
mkdir -p $(dirname $OUT)
( cd ~/VLA/research/vla && exec ~/VLM/venv/bin/python -u vlas_server.py --ckpt $CK --port $PORT --log ${OUT%.json}.policy.jsonl ${SRV_X:-} ) > ${OUT%.json}.server.log 2>&1 &
SP=$!
until grep -q '"ready"' ${OUT%.json}.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/research/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && \
  python -u tabletop_eval_s.py --tasks $TASKS --episodes $EP --num_envs 4 --seed 30000 --port $PORT --out $OUT --video ${OUT%.json}.mp4 "$@" ) > ${OUT%.json}.eval.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep RESULT ${OUT%.json}.eval.log | cut -c1-300
