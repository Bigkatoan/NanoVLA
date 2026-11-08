#!/bin/bash
cd ~/VLA/research/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log
echo "[$(date)] round-2 ablation start (N=10, A_v1enc, exec2)" >> $LOG
for cfg in "0.0 0.0" "0.5 0.0" "0.0 1.0" "0.5 1.0"; do
  set -- $cfg
  python sim_fewshot.py --Ns 10 --conds A_v1enc --rollouts 30 --steps 1500 --exec_steps 2 --state_drop $1 --aux_w $2 --videos 0 --tag r2_sd$1_aux$2 > ~/VLA/data/r2_sd$1_aux$2.out 2>&1
  echo "[$(date)] r2 sd=$1 aux=$2: $(grep '"cond"' ~/VLA/data/r2_sd$1_aux$2.out | cut -c1-160)" >> $LOG
done
echo "[$(date)] ROUND2 DONE" >> $LOG
