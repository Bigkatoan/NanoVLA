#!/bin/bash
# Demo-count scaling study: best recipe (fixenc) at N=50..600, 3 seeds; scratch baseline 1 seed.
cd ~/VLA/vla; source ~/VLM/venv/bin/activate; export MUJOCO_GL=egl
OUT=~/VLA/data/scaling; LOG=$OUT/run.log; mkdir -p $OUT; jobs=$OUT/jobs.txt; : > $jobs
steps() { case $1 in 50) echo 1500;; 100) echo 2500;; 200) echo 4000;; 400) echo 6000;; 600) echo 8000;; esac; }
for seed in 0 1 2; do for N in 600 400 200 100 50; do echo "A_v1enc 0.5 1.0 $N $seed fixenc $(steps $N)" >> $jobs; done; done
for N in 600 400 200 100; do echo "C 0.0 0.0 $N 0 scratch $(steps $N)" >> $jobs; done
echo "[$(date)] start $(wc -l < $jobs) jobs" >> $LOG
cat $jobs | xargs -P 3 -L 1 bash -c 'python sim_fewshot.py --bench sim_bench600 --Ns $3 --conds $0 --rollouts 50 --steps $6 --exec_steps 4 --state_drop $1 --aux_w $2 --seed $4 --videos 0 --tag scale_$5_N$3_s$4 > '"$OUT"'/$5_N$3_s$4.out 2>&1; echo "[$(date)] $5 N=$3 seed=$4: $(grep "\"cond\"" '"$OUT"'/$5_N$3_s$4.out | cut -c1-160)" >> '"$LOG"
echo "[$(date)] SCALING DONE" >> $LOG
