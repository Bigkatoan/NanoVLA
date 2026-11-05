#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/align.log
echo "[$(date)] waiting for text_v3 + vision" >> $LOG
until [ -f ~/VLA/ckpt/text_v3/text_encoder.pt ] && [ -f ~/VLA/ckpt/vision/vision_encoder.pt ]; do sleep 60; done
sleep 30
echo "[$(date)] start align" >> $LOG
python train_align.py --steps 15000 > ~/VLA/data/train_align.out 2>&1
echo "[$(date)] align done: $(tail -2 ~/VLA/data/train_align.out | head -1 | cut -c1-300)" >> $LOG
