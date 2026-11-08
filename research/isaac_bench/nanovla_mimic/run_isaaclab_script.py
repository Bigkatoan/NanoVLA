"""Run an Isaac Lab script (e.g. Mimic's annotate_demos.py / generate_dataset.py) with the NanoVLA tasks registered.

    python nanovla_mimic/run_isaaclab_script.py $ISAACLAB/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task ... [args]
"""
import os, runpy, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
import nanovla_mimic  # noqa: F401,E402  (gym registration only; no Isaac imports)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
