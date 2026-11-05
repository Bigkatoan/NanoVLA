#!/bin/bash
cd ~/VLA/isaac_bench; source ~/VLA/envs/isaac_env/bin/activate; export OMNI_KIT_ACCEPT_EULA=YES
O=out/smoke; mkdir -p $O/mimic; LOG=$O/log.txt; : > $LOG
run() { echo "[$(date +%T)] $1" >> $LOG; local n=$2; shift 2; timeout -s KILL 2400 "$@" > $O/$n.log 2>&1; echo "  exit $?" >> $LOG; grep -h '^{' $O/$n.log | cut -c1-200 >> $LOG; }
run "topple" topple python -u traj_tool.py --spec specs/push.json --tasks topple --episodes 3 --video_eps 3 --video $O/topple.mp4 --seed 7
run "rel" rel python -u traj_tool.py --spec specs/place.json --tasks rel --episodes 4 --video_eps 3 --video $O/rel.mp4 --seed 7
export PYTHONPATH=$PWD:$HOME/VLA/third_party/so101_workshop/source:$PYTHONPATH
IL=~/VLA/third_party/IsaacLab/scripts/imitation_learning/isaaclab_mimic; W=nanovla_mimic/run_isaaclab_script.py
T=NanoVLA-SO101-tray-marker-Mimic-v0
run "mimic annotate" annotate python -u $W $IL/annotate_demos.py --task $T --auto --input_file $O/mimic/src.hdf5 --output_file $O/mimic/annot.hdf5 --headless --enable_cameras
run "mimic generate" generate python -u $W $IL/generate_dataset.py --task $T --input_file $O/mimic/annot.hdf5 --output_file $O/mimic/gen.hdf5 --generation_num_trials 5 --num_envs 1 --headless --enable_cameras
run "mimic convert" convert python -u nanovla_mimic/hdf5_to_bench.py --input $O/mimic/gen.hdf5 --text "put the marker in the tray" --out $O/mimic/bench
echo "[$(date +%T)] TESTS2 DONE" >> $LOG
