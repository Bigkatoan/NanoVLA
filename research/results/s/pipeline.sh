#!/bin/bash
# NanoVLA-Mod: wait for the state-labelled data, merge with the GCT data, train the split models, closed-loop evals and
# module-oracle ablations (each module replaced by the true answer) to find the bottleneck.
V=~/VLA; R=$V/research/results/s; CK=$V/ckpt/nanovla_mod/mod.pt; SP=$V/research/isaac_bench/specs/pick.json
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until grep -q "GEN S DONE" $V/data/isaac_s/gen.log 2>/dev/null; do sleep 60; done
touch $V/data/isaac_s/STOP_TB
SRC=$(ls -d $V/data/isaac_gct/*_c* $V/data/isaac_s/*_c* | while read d; do [ -f $d/episodes.json ] && echo $d; done | tr '\n' ' ')
~/VLM/venv/bin/python $V/research/vla/merge_sim_bench.py $V/data/isaac_s_v1 $SRC > $R/merge.log 2>&1; log "merged: $(tail -1 $R/merge.log)"
mkdir -p $(dirname $CK); log "train mod1"
( cd $V/research/vla && $V/research/tools/gpu.sh 12000 ~/VLM/venv/bin/python -u nanovla_mod.py --run mod1 --bench isaac_s_v1 --steps 15000 --out $CK ) > $(dirname $CK)/train.log 2>&1
log "trained: $(grep '"val"' $(dirname $CK)/train.log | tail -1)"
NIK="--nano_ik $V/ckpt/nano_ik/nano_ik.pt"            # all-learned pipeline: NanoIK replaces the analytic IK (oracle: 21/24 vs 19/24)
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/research/tools/gpu.sh 4000 $V/research/tools/eval_s.sh $CK $tasks $eps $R/$name.json $port $NIK "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-200)"; }
ev pick pick 24 7601
ev pick_jitter pick 24 7602 --query_every 6 --jitter 3
SRV_X="--oracle grounder" ev pick_oracle_grounder pick 24 7605 --oracle_spec $SP --oracle_modules grounder
SRV_X="--oracle state" ev pick_oracle_state pick 24 7606 --oracle_spec $SP --oracle_modules state
SRV_X="--oracle grasp" ev pick_oracle_grasp pick 24 7607 --oracle_spec $SP --oracle_modules grasp
SRV_X="--oracle grounder,state,grasp" ev pick_oracle_all pick 24 7608 --oracle_spec $SP --oracle_modules grounder,state,grasp
NIK="" ev pick_analytic_ik pick 24 7609
ev A pick,pad,tray,container,push,topple,rotate,rel 48 7603
ev C pick 24 7604 --objects glue_stick,spool,toy_brick
log "MOD DONE"
