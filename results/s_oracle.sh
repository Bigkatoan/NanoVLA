#!/bin/bash
cd ~/VLA/isaac_bench && source ~/VLA/envs/isaac_env/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p ~/VLA/results/s
python -u tabletop_eval_s.py --tasks pick --episodes 12 --oracle_spec specs/pick.json --out ~/VLA/results/s/oracle_pick.json > ~/VLA/results/s/oracle_pick.log 2>&1
python -u tabletop_eval_s.py --tasks pick --episodes 12 --oracle_spec specs/pick.json --query_every 6 --jitter 3 --out ~/VLA/results/s/oracle_pick_jitter.json > ~/VLA/results/s/oracle_pick_jitter.log 2>&1
