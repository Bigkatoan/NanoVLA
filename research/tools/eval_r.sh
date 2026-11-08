#!/bin/bash
# Closed-loop eval of a NanoVLA-R checkpoint (recurrent server) in the Isaac tabletop, same client/seed as eval_zero_shot.sh.
#   eval_r.sh <ckpt> <tasks> <episodes> <out.json> <port> [extra tabletop env vars...]
CK=$1; TASKS=$2; EP=$3; OUT=$4; PORT=$5; shift 5
mkdir -p $(dirname $OUT)
( cd ~/VLA/research/vla && exec ~/VLM/venv/bin/python -u vlar_server.py --ckpt $CK --port $PORT ${SRV_X:-} ) > ${OUT%.json}.server.log 2>&1 &
SP=$!
until grep -q '"ready"' ${OUT%.json}.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/research/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && export "$@" && \
  python -u tabletop_eval.py --cams wrist --tasks $TASKS --episodes $EP --num_envs 4 --seed 30000 --exec ${EXEC:-1} --port $PORT --out $OUT --video ${OUT%.json}.mp4 --video_eps 3 ${EVAL_X:-} ) > ${OUT%.json}.eval.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep RESULT ${OUT%.json}.eval.log | cut -c1-300
