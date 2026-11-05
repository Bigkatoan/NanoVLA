#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/vla.log
echo "[$(date)] waiting for align" >> $LOG
while pgrep -f "train_align.py" > /dev/null; do sleep 60; done
echo "[$(date)] start VLA pretrain" >> $LOG
python train_vla.py --steps 30000 > ~/VLA/data/train_vla.out 2>&1
echo "[$(date)] VLA done: $(tail -1 ~/VLA/data/train_vla.out)" >> $LOG
