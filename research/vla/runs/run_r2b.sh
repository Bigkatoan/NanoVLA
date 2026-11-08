#!/bin/bash
cd ~/VLA/research/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log
echo "[$(date)] round-2b ablation start (N=10, A_v1 frozen, exec4)" >> $LOG
run() { python sim_fewshot.py --Ns 10 --conds A_v1 --rollouts 30 --steps 1500 --exec_steps $1 --state_drop $2 --aux_w $3 --videos 1 --tag r2b_e$1_sd$2_aux$3 > ~/VLA/data/r2b_e$1_sd$2_aux$3.out 2>&1
        echo "[$(date)] r2b exec=$1 sd=$2 aux=$3: $(grep '"cond"' ~/VLA/data/r2b_e$1_sd$2_aux$3.out | cut -c1-170)" >> $LOG; }
run 4 0.5 0.0 &
run 4 0.0 1.0 &
wait
run 4 0.5 1.0 &
run 8 0.0 0.0 &
wait
echo "[$(date)] ROUND2B DONE" >> $LOG
