#!/bin/bash
# mod2: event labels (state = what to do from where the arm is, no timer phases) + the arm holds while the jaw ramps.
V=~/VLA; R=$V/research/results/s2; CK=$V/ckpt/nanovla_mod/mod2.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
while pgrep -f "results/s/pipeline.sh" >/dev/null; do sleep 30; done
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"
log "train mod2"
( cd $V/research/vla && $V/research/tools/gpu.sh 6000 ~/VLM/venv/bin/python -u nanovla_mod.py --run mod2 --bench isaac_s_v1 --steps 6000 --init_ckpt $V/ckpt/nanovla_mod/mod.pt --out $CK ) > $R/train.log 2>&1
log "trained: $(grep '"val"' $R/train.log | tail -1)"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
ev pick pick 24 7611
SRV_X="--oracle grounder" ev pick_oracle_grounder pick 24 7612 --oracle_spec $SP --oracle_modules grounder
ev A pick,pad,tray,container,push,topple,rotate,rel 48 7613
log "MOD2 DONE"
