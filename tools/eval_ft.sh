#!/bin/bash
# Fine-tune (isaac_policy_server) then closed-loop eval in the Isaac tabletop, wrist camera only.
#   eval_ft.sh <ckpt> <bench> <N> <steps> <tasks> <episodes> <out.json> <port> [tabletop env vars...]
CK=$1; BENCH=$2; N=$3; STEPS=$4; TASKS=$5; EP=$6; OUT=$7; PORT=$8; shift 8
mkdir -p $(dirname $OUT)
( cd ~/VLA/vla && source ~/VLM/venv/bin/activate && export MUJOCO_GL=egl && \
  exec python -u isaac_policy_server.py --bench $BENCH --cams wrist --cfg fixenc --vla $CK --N $N --steps $STEPS --port $PORT ${SRV_X:-} ) > ${OUT%.json}.server.log 2>&1 &
SP=$!
until grep -q '"ready"' ${OUT%.json}.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && [ $# -gt 0 ] && export "$@"; \
  python -u tabletop_eval.py --cams wrist --tasks $TASKS --episodes $EP --num_envs 4 --seed ${EVAL_SEED:-30000} --port $PORT --out $OUT --video ${OUT%.json}.mp4 --video_eps 2 ) > ${OUT%.json}.eval.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep RESULT ${OUT%.json}.eval.log | cut -c1-260
