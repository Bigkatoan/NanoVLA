#!/bin/bash
cd ~/VLA/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/clean_v2.log
echo "[$(date)] waiting for action download" >> $LOG
while pgrep -f "prep_fewshot.py --mode actions" > /dev/null; do sleep 60; done
echo "[$(date)] actions: $(ls ~/VLA/data/actions | wc -l) repos; label audit" >> $LOG
python audit_labels.py >> $LOG 2>&1
echo "[$(date)] clean v2" >> $LOG
python clean_v2.py >> $LOG 2>&1
echo "[$(date)] videos" >> $LOG
CLEAN_DIR=clean_v2 python make_videos.py >> $LOG 2>&1
echo "[$(date)] ALL DONE" >> $LOG
