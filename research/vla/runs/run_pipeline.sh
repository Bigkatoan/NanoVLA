#!/bin/bash
# Wait for prep_data.py to finish, then pack, train text encoder, train vision encoder.
set -e
cd ~/VLA/research/vla
source ~/VLM/venv/bin/activate
LOG=~/VLA/data/pipeline.log
echo "[$(date)] waiting for prep" >> $LOG
while pgrep -f "scratch_vla/prep_data.py" > /dev/null; do sleep 60; done
echo "[$(date)] prep done: $(grep -c ok ~/VLA/data/prep_log.txt) ok / $(grep -c ERR ~/VLA/data/prep_log.txt) err" >> $LOG
python pack_data.py >> $LOG 2>&1
echo "[$(date)] packed; training text" >> $LOG
python train_text.py --steps 30000 > ~/VLA/data/train_text.out 2>&1
echo "[$(date)] text done; training vision" >> $LOG
rm -f ~/VLA/ckpt/vision/log.jsonl ~/VLA/ckpt/vision/last.pt
python train_vision.py --steps 40000 --workers 10 > ~/VLA/data/train_vision.out 2>&1
echo "[$(date)] vision done" >> $LOG
