#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/vla.log
until grep -q "VLA done" $LOG 2>/dev/null; do sleep 60; done
echo "[$(date)] start VLA v2 (emb_drop 0.3, 12k steps)" >> $LOG
python train_vla.py --steps 12000 --emb_drop 0.3 --out ~/VLA/ckpt/vla_v2 --workers 6 > ~/VLA/data/train_vla_v2.out 2>&1
echo "[$(date)] VLA v2 done: $(tail -1 ~/VLA/data/train_vla_v2.out)" >> $LOG
