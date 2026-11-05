#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log
until grep -q "ROUND2B DONE" $LOG; do sleep 30; done
echo "[$(date)] round-3 start (N=10, A_v1: temporal ensembling / longer fine-tune)" >> $LOG
run() { python sim_fewshot.py --Ns ${4:-10} --conds A_v1 --rollouts 30 --steps $2 --exec_steps $1 --videos 1 --tag r3_e$1_s$2_N${4:-10} > ~/VLA/data/r3_e$1_s$2_N${4:-10}.out 2>&1
        echo "[$(date)] r3 exec=$1 steps=$2 N=${4:-10}: $(grep '"cond"' ~/VLA/data/r3_e$1_s$2_N${4:-10}.out | cut -c1-170)" >> $LOG; }
run 0 1500 &
run 4 4000 &
wait
run 0 4000 &
wait
echo "[$(date)] ROUND3 DONE" >> $LOG
