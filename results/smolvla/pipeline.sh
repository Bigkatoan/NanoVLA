#!/bin/bash
# SmolVLA teacher (wrist only): wait for the LeRobot conversion, fine-tune smolvla_base, closed-loop evals.
V=~/VLA; R=$V/results/smolvla; OUT=$V/ckpt/smolvla_sim
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until grep -q '"done"' $R/convert.log; do sleep 30; done
log "converted: $(grep '"done"' $R/convert.log)"
( cd $V/tools && setsid nohup ~/VLM/venv/bin/python log2tb.py $R/train.log $V/tb/smolvla_sim > $R/log2tb.out 2>&1 < /dev/null & )
log "train smolvla_sim 20k"
( cd $V && HF_HUB_OFFLINE=1 $V/tools/gpu.sh 30000 ~/VLA/envs/lerobot/bin/lerobot-train --policy.path=ckpt/smolvla_base --dataset.repo_id=local/nanovla_sim \
   --dataset.root=data/lerobot/sim_v1 --batch_size=32 --steps=20000 --log_freq=100 --save_freq=5000 --output_dir=$OUT \
   --policy.push_to_hub=false --wandb.enable=false --num_workers=6 --policy.device=cuda ) > $R/train.log 2>&1
CK=$OUT/checkpoints/last/pretrained_model; [ -d $CK ] || CK=$(ls -d $OUT/checkpoints/0*/pretrained_model | tail -1)
log "trained: $CK"
ev() { local name=$1 tasks=$2 eps=$3 port=$4; shift 4; $V/tools/gpu.sh 6000 $V/tools/eval_smol.sh $CK $tasks $eps $R/$name.json $port "$@"; log "$name: $(grep RESULT $R/$name.eval.log | cut -c1-220)"; }
ev pick pick 24 6121
ev A pick,pad,tray,container,push,topple,rotate,rel 48 6122
ev C pick 24 6123 --objects glue_stick,spool,toy_brick
log "SMOLVLA DONE"
