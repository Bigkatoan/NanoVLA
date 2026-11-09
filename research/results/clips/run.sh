#!/bin/bash
# Isaac Sim clips for the release video: scripted expert on 4 task families (front + wrist camera, 640x480 each),
# and the distilled student sim_pick driving the arm from its wrist camera (12 episodes, env 0 recorded).
V=~/VLA; R=$V/research/results/clips; cd $V/research/isaac_bench
source $V/envs/isaac_env/bin/activate; export OMNI_KIT_ACCEPT_EULA=YES TABLETOP_FRONT=1 TABLETOP_RES=480x640 TABLETOP_CAM_SCALE=1 TABLETOP_HOLDOUT=glue_stick,spool,toy_brick
for t in "pick pick" "place pad,tray,container" "push push,topple" "rotate rotate"; do set -- $t
  $V/nanovla/gpu.sh 3600 python -u traj_tool.py --spec specs/$1.json --tasks $2 --episodes 3 --num_envs 2 --seed 4242 --video $R/expert_$1.mp4 --video_eps 3 --markers 0 > $R/expert_$1.log 2>&1
  echo "expert $1: $(grep -c '"ep"' $R/expert_$1.log) eps"
done
( cd $V && HF_HUB_OFFLINE=1 PYTHONPATH=$V exec $V/envs/lerobot/bin/python -m nanovla.sim_server --model $V/models/sim_pick --port 6141 ) > $R/student_server.log 2>&1 &
SP=$!; until grep -q '"ready"' $R/student_server.log || ! kill -0 $SP 2>/dev/null; do sleep 3; done
$V/nanovla/gpu.sh 3600 python -u tabletop_eval.py --cams both --tasks pick --episodes 12 --num_envs 1 --port 6141 --diag 1 --seed 777 --out $R/student_pick.json --video $R/student_pick.mp4 --video_eps 12 > $R/student_pick.log 2>&1
kill $SP; grep RESULT $R/student_pick.log | cut -c1-200; echo CLIPS DONE
