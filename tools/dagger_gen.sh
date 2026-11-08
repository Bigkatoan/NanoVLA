#!/bin/bash
# One DAgger data chunk (run under tools/gpu.sh): NanoVLA-M policy server + traj_tool in DAgger mode.
#   dagger_gen.sh <ckpt> <spec> <tasks> <episodes> <out_dir> <port> <seed>
CK=$1; SPEC=$2; TASKS=$3; EP=$4; OUT=$5; PORT=$6; SEED=$7
( cd ~/VLA/vla && exec ~/VLM/venv/bin/python -u ${SERVER:-vlam_server.py} --ckpt $CK --port $PORT ) > $OUT.server.log 2>&1 &
SP=$!
until grep -q '"ready"' $OUT.server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
( cd ~/VLA/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && \
  TABLETOP_HOLDOUT=glue_stick,spool,toy_brick python -u traj_tool.py --spec specs/$SPEC.json --tasks $TASKS --episodes $EP --num_envs ${NE:-12} \
     --seed $SEED --dagger_port $PORT --max_steps 1800 --out $OUT ) > $OUT.log 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
