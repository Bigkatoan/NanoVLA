#!/bin/bash
cd ~/VLA/research/vla; source ~/VLM/venv/bin/activate
LOG=~/VLA/data/clean_v2.log
echo "[$(date)] clean v2b (no cam_uncorrelated, repo-level lag)" >> $LOG
python clean_v2.py >> $LOG 2>&1 && echo "[$(date)] videos v2b" >> $LOG && CLEAN_DIR=clean_v2 python make_videos.py >> $LOG 2>&1
echo "[$(date)] ALL DONE v2b" >> $LOG
