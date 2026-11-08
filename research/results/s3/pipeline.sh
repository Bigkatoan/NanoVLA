#!/bin/bash
# mod3: Planner trained with translation augmentation (follow the object, not the joints) + half the batch on the Grounder's estimates.
V=~/VLA; R=$V/research/results/s3; CK=$V/ckpt/nanovla_mod/mod3.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"
log "train mod3"
( cd $V/research/vla && $V/research/tools/gpu.sh 6000 ~/VLM/venv/bin/python -u nanovla_mod.py --run mod3 --bench isaac_s_v1 --steps 4000 --init_ckpt $V/ckpt/nanovla_mod/mod2.pt --plan_shift 0.05 --plan_real 0.5 --out $CK ) > $R/train.log 2>&1
log "trained: $(grep '"val"' $R/train.log | tail -1)"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
SRV_X="--oracle grounder" ev pick_oracle_grounder pick 24 7622 --oracle_spec $SP --oracle_modules grounder
ev pick pick 24 7621
ev A pick,pad,tray,container,push,topple,rotate,rel 48 7623
log "MOD3 DONE"
