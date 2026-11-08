#!/bin/bash
V=~/VLA; R=$V/research/results/s3; CK=$V/ckpt/nanovla_mod/mod3.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until grep -q "GATE DONE" $R/pipeline.log; do sleep 15; done
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
SRV_X="--oracle grounder --gate 0.02" ev pick_gate2_oracle_grounder pick 24 7642 --oracle_spec $SP --oracle_modules grounder
SRV_X="--gate 0.02" ev pick_gate2 pick 24 7641
log "GATE2 DONE"
