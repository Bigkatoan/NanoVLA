#!/bin/bash
# Zero-shot closed-loop eval of a pretrained NanoVLA checkpoint (no fine-tuning) in the Isaac tabletop.
#   eval_zero_shot.sh <ckpt> <tasks> <episodes> <out.json> <port> [extra tabletop env vars...]
CK=$1; TASKS=$2; EP=$3; OUT=$4; PORT=$5; shift 5
mkdir -p $(dirname $OUT)
( cd ~/VLA/vla && source ~/VLM/venv/bin/activate && export MUJOCO_GL=egl && \
  exec python -u isaac_policy_server.py --bench ${BENCH:-isaac_pre_all} --cams wrist --cfg fixenc --vla $CK --N 200 --steps 0 --port $PORT ) > ${OUT%.json}.server.log 2>&1 &
SP=$!
until grep -q '"ready"' ${OUT%.json}.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && export "$@" && \
  python -u tabletop_eval.py --cams wrist --tasks $TASKS --episodes $EP --num_envs 4 --seed 30000 --port $PORT --out $OUT --video ${OUT%.json}.mp4 --video_eps 3 ${EVAL_X:-} ) > ${OUT%.json}.eval.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep RESULT ${OUT%.json}.eval.log | cut -c1-300
