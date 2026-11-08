#!/bin/bash
# Run on a machine with an 80 GB-class GPU (H100 / A100-80G; Cosmos-Transfer2.5-2B needs ~65 GB VRAM).
# Usage: bash run_cosmos.sh <specs_dir>     (copy data/cosmos_export/... and cosmos/specs/... to the machine first;
#        spec paths are absolute - rewrite them with sed if the directory layout differs)
set -e
SPECS=${1:?specs dir}
if [ ! -d cosmos-transfer2.5 ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
  cd cosmos-transfer2.5 && pip install -e . && cd ..      # follow the repo's setup docs for the exact environment
  # model weights are downloaded from Hugging Face on first use (accept the NVIDIA model license; huggingface-cli login)
fi
cd cosmos-transfer2.5
for s in "$SPECS"/*.json; do
  out=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$s")
  [ -d "$out" ] && continue
  python examples/inference.py -i "$s" -o "$out"
done
