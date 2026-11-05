#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log
until grep -q "SIM SR R1 DONE" $LOG; do sleep 30; done
echo "[$(date)] round-4 start (N=50, A_v1 + state dropout + localisation loss)" >> $LOG
python sim_fewshot.py --Ns 50 --conds A_v1 --rollouts 50 --steps 1500 --exec_steps 4 --state_drop 0.5 --aux_w 1.0 --videos 1 --tag r4_sd_aux_N50 > ~/VLA/data/r4_sd_aux_N50.out 2>&1
echo "[$(date)] r4 N=50 sd+aux: $(grep '"cond"' ~/VLA/data/r4_sd_aux_N50.out | cut -c1-170)" >> $LOG
echo "[$(date)] ROUND4 DONE" >> $LOG
