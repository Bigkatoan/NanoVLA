#!/bin/bash
# event gate: StateNet may step k -> k+1 only after the arm reached step k's target (1 cm)
V=~/VLA; R=$V/research/results/s3; CK=$V/ckpt/nanovla_mod/mod3.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until [ -f $R/pick.json ] && ! pgrep -f "port 7621" >/dev/null; do sleep 15; done; log "pick: $(grep RESULT $R/pick.eval.log | cut -c1-200)"
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
SRV_X="--oracle grounder --gate 0.01" ev pick_gate_oracle_grounder pick 24 7632 --oracle_spec $SP --oracle_modules grounder
SRV_X="--gate 0.01" ev pick_gate pick 24 7631
log "GATE DONE"
