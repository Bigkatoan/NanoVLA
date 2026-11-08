#!/bin/bash
# Multi-task Isaac demos for NanoVLA pretraining (bowl task held out for the few-shot test), then bowl demos for few-shot.
cd ~/VLA/research/isaac_bench; source ~/VLA/envs/isaac_env/bin/activate; export OMNI_KIT_ACCEPT_EULA=YES
OUT=~/VLA/data/isaac_pre; LOG=$OUT/gen.log; mkdir -p $OUT
NE=${NUM_ENVS:-12}
for c in 0 1 2 3; do
  for job in "place pad,tray 200" "pick pick 80"; do
    set -- $job; d=$OUT/$1_c$c
    [ -f $d/episodes.json ] && continue
    echo "[$(date)] start $1 chunk $c" >> $LOG
    python -u traj_tool.py --spec specs/$1.json --tasks $2 --episodes $3 --num_envs $NE --seed $((100 + 10 * c)) --out $d > $d.log 2>&1
    echo "[$(date)] done $1 chunk $c: $(python3 -c "import json; d=json.load(open('$d/episodes.json')); print(d['kept'], 'kept of', d['attempts'])" 2>&1)" >> $LOG
  done
done
d=$OUT/bowl_fewshot
[ -f $d/episodes.json ] || python -u traj_tool.py --spec specs/place.json --tasks bowl --episodes 60 --num_envs $NE --seed 900 --out $d > $d.log 2>&1
echo "[$(date)] GEN DONE" >> $LOG
