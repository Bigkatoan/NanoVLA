#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log
echo "[$(date)] waiting for VLA v2" >> $LOG
until grep -q "VLA v2 done" ~/VLA/data/vla.log 2>/dev/null; do sleep 60; done
echo "[$(date)] start sim success-rate benchmark (round 1)" >> $LOG
python sim_fewshot.py --Ns 10,20,50 --conds A_v2,A_v1,B,C --rollouts 50 --steps 1500 --tag sim_sr_r1 > ~/VLA/data/sim_sr_r1.out 2>&1
echo "[$(date)] SIM SR R1 DONE" >> $LOG
grep '"cond"' ~/VLA/data/sim_sr_r1.out >> $LOG
