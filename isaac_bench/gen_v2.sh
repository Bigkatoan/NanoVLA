#!/bin/bash
# Overnight NanoVLA data v2: wrist camera only, random robot base, full domain randomisation, 56-object library,
# room background; tasks pick / pad / tray / container / rel / push / topple / rotate. Held out: the bowl task and
# the objects in HOLD (used only for the few-shot / generalisation sets generated at the end).
cd ~/VLA/isaac_bench; source ~/VLA/envs/isaac_env/bin/activate; export OMNI_KIT_ACCEPT_EULA=YES
OUT=~/VLA/data/isaac_v2; LOG=$OUT/gen.log; mkdir -p $OUT
STOP=${STOP:-"05:45"}; NE=${NUM_ENVS:-12}; EP=${EP_PER_CHUNK:-80}
HOLD=glue_stick,spool,toy_brick
JOBS=("place pad,tray,container,rel" "pick pick" "push push,topple" "rotate rotate")
c=0
while [ "$(date +%H:%M)" \< "$STOP" ] || [ "$(date +%H)" -ge 12 ]; do
  set -- ${JOBS[$((c % 4))]}; d=$OUT/$1_c$c
  if [ ! -f $d/episodes.json ]; then
    echo "[$(date +%T)] start $1 ($2) chunk $c" >> $LOG
    TABLETOP_HOLDOUT=$HOLD ~/VLA/tools/gpu.sh 3600 python -u traj_tool.py --spec specs/$1.json --tasks $2 --episodes $EP --num_envs $NE \
        --seed $((1000 + 37 * c)) --out $d > $d.log 2>&1
    echo "[$(date +%T)] done $1 chunk $c: $(python3 -c "import json; d=json.load(open('$d/episodes.json')); print(d['kept'], 'kept of', d['attempts'])" 2>&1 | tail -1)" >> $LOG
  fi
  c=$((c + 1))
done
# few-shot / generalisation sets (never used for pretraining)
~/VLA/tools/gpu.sh 3600 python -u traj_tool.py --spec specs/place.json --tasks bowl --episodes 80 --num_envs $NE --seed 9001 --out $OUT/fs_bowl > $OUT/fs_bowl.log 2>&1
echo "[$(date +%T)] done fs_bowl" >> $LOG
for o in glue_stick spool toy_brick; do       # unseen objects: pick demos for the generalisation few-shot test
  python3 -c "import json; s=json.load(open('specs/pick.json')); s['force_obj']='$o'; json.dump(s, open('specs/_pick_$o.json','w'))"
  ~/VLA/tools/gpu.sh 2400 python -u traj_tool.py --spec specs/_pick_$o.json --episodes 25 --num_envs $NE --seed 9100 --out $OUT/fs_unseen_$o > $OUT/fs_unseen_$o.log 2>&1
  echo "[$(date +%T)] done fs_unseen_$o" >> $LOG
done
echo "[$(date +%T)] GEN V2 DONE" >> $LOG
