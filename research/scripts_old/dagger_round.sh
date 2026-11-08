#!/bin/bash
# One DAgger round: the teacher labels the frames the STUDENT visited on the robot, the student is retrained on all labels.
#   1) on the robot:  python deploy/run_student.py ... --front_cam <idx> --record data/lerobot/<name>_dagger<k>
#   2) on the PC:     scripts/dagger_round.sh <name> <k>
set -e
NAME=$1; K=$2; V=$(cd $(dirname $0)/.. && pwd); PY=$V/envs/lerobot/bin/python; R=$V/research/results/$NAME; export HF_HUB_OFFLINE=1
FS=$(ls -d $V/ckpt/fewshot/$NAME/checkpoints/0*/pretrained_model | tail -1); DG=$V/data/lerobot/${NAME}_dagger$K
N=$($PY -c "import json; print(json.load(open('$DG/meta/info.json'))['total_episodes'] - 1)")
$V/research/tools/gpu.sh 30000 $PY $V/research/tools/smol_offline.py label --ckpt $FS --root $DG --episodes 0-$N --stride 3 --aug 1 --out $R/labels_dagger$K.npz > $R/label_dagger$K.log 2>&1
ALL=$(ls $R/labels_train.npz $R/labels_dagger*.npz | tr '\n' ',' | sed 's/,$//')
( cd $V/research/vla && $V/research/tools/gpu.sh 30000 $PY student.py --train $ALL --test $R/labels_test.npz --target teacher --out $V/ckpt/student/$NAME.pt --tb $V/tb/student_${NAME}_dagger$K ) > $R/student_dagger$K.log 2>&1
tail -1 $R/student_dagger$K.log
