#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
LOG=~/VLA/data/sim_sr.log; OUT=~/VLA/data/final; mkdir -p $OUT
until grep -q "ROUND3 DONE" $LOG; do sleep 30; done
echo "[$(date)] FINAL evaluation start" >> $LOG
jobs_file=$OUT/jobs.txt; : > $jobs_file
for seed in 0 1 2; do for N in 10 20 50; do
  echo "A_v1 0.0 0.0 $N $seed base" >> $jobs_file
  echo "A_v1 0.5 1.0 $N $seed fix" >> $jobs_file
  echo "A_v1enc 0.5 1.0 $N $seed fixenc" >> $jobs_file
  echo "C 0.0 0.0 $N $seed scratch" >> $jobs_file
done; done
cat $jobs_file | xargs -P 3 -L 1 bash -c 'python sim_fewshot.py --Ns $3 --conds $0 --rollouts 50 --steps 1500 --exec_steps 4 --state_drop $1 --aux_w $2 --seed $4 --videos 0 --tag final_$5_N$3_s$4 > '"$OUT"'/$5_N$3_s$4.out 2>&1; echo "[$(date)] final $5 N=$3 seed=$4: $(grep "\"cond\"" '"$OUT"'/$5_N$3_s$4.out | cut -c1-150)" >> '"$LOG"
echo "[$(date)] FINAL DONE" >> $LOG
