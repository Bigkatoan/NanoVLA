#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/fewshot.log
echo "[$(date)] waiting for VLA" >> $LOG
until grep -q "VLA done" ~/VLA/data/vla.log 2>/dev/null; do sleep 120; done
echo "[$(date)] start few-shot benchmark" >> $LOG
python fewshot.py --repos 8 --Ns 10,50 --conds A,B,C --steps 1000 > ~/VLA/data/fewshot.out 2>&1
echo "[$(date)] FEWSHOT DONE" >> $LOG
tail -8 ~/VLA/data/fewshot.out >> $LOG
