#!/bin/bash
# Planner-only lr sweep (is the 1-3 cm offset an optimisation problem?) -> mod4 = mod3 + best Planner -> gated evals
V=~/VLA; R=$V/research/results/s4; CK=$V/ckpt/nanovla_mod/mod4.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until grep -q "GATE2 DONE" $V/research/results/s3/pipeline.log; do sleep 15; done
log "planner sweep"
( cd $V/research/vla && $V/research/tools/gpu.sh 5400 ~/VLM/venv/bin/python -u train_planner.py --init $V/ckpt/nanovla_mod/mod3.pt --out $CK --lrs 3e-4,1e-3,3e-3 --steps 20000 --run planner_sweep ) > $R/sweep.log 2>&1
log "sweep: $(grep -E 'init|saved' $R/sweep.log | tr '\n' ' ' | cut -c1-400)"
[ -f $CK ] || { log "no mod4"; exit; }
G=$(grep -o 'gate2[^:]*: RESULT {[^}]*"success_rate": [0-9.]*' $V/research/results/s3/pipeline.log >/dev/null; echo 0.01)
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
SRV_X="--oracle grounder --gate $G" ev pick_gate_oracle_grounder pick 24 7652 --oracle_spec $SP --oracle_modules grounder
SRV_X="--gate $G" ev pick_gate pick 24 7651
log "S4 DONE"
