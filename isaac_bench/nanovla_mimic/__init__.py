"""Isaac Lab Mimic for the NanoVLA SO-101 tabletop.

Registers manager-based (Mimic-capable) variants of the tabletop for fixed (task, object, target) triples, e.g.
    NanoVLA-SO101-tray-marker-Mimic-v0       "put the marker in the tray"
    NanoVLA-SO101-pad-soup_can-pink_pad-Mimic-v0
Layouts are re-sampled at every reset with the tabletop sampler (positions, distractors, table texture). Actions are
absolute end-effector poses (differential IK on the 5 arm joints, env frame -> robot root frame inside the Mimic env)
plus the jaw position, which is what Isaac Lab Mimic transforms between object frames.

Usage (Isaac Lab scripts, PYTHONPATH must contain isaac_bench and the workshop source):
    python nanovla_mimic/record_src.py --task NanoVLA-SO101-tray-marker-Mimic-v0 --demos 10 --out datasets/tray_marker_src.hdf5
    python $ISAACLAB/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task nanovla_mimic:NanoVLA-SO101-tray-marker-Mimic-v0 \
        --auto --input_file datasets/tray_marker_src.hdf5 --output_file datasets/tray_marker_annot.hdf5 --headless --enable_cameras
    python $ISAACLAB/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task nanovla_mimic:NanoVLA-SO101-tray-marker-Mimic-v0 \
        --input_file datasets/tray_marker_annot.hdf5 --output_file datasets/tray_marker_gen.hdf5 --generation_num_trials 200 --num_envs 4 --headless --enable_cameras
    python nanovla_mimic/hdf5_to_bench.py --input datasets/tray_marker_gen.hdf5 --out ~/VLA/data/mimic/tray_marker
"""
import gymnasium as gym

VARIANTS = [
    ("tray", "marker", "tray"), ("tray", "soup_can", "tray"), ("tray", "foam_brick", "tray"),
    ("pad", "marker", "pink_pad"), ("pad", "soup_can", "pink_pad"), ("pad", "gelatin_box", "white_pad"),
    ("pick", "marker", None), ("pick", "pudding_box", None),
    ("bowl", "marker", "bowl"), ("bowl", "soup_can", "bowl"), ("bowl", "meat_can", "bowl"),
]


def task_id(task, obj, target):
    return f"NanoVLA-SO101-{task}-{obj}" + (f"-{target}" if target and task == "pad" else "") + "-Mimic-v0"


def _register():
    """registration only needs gymnasium: config classes are built lazily by nanovla_mimic.env.__getattr__ once the app runs"""
    for task, obj, target in VARIANTS:
        tid = task_id(task, obj, target)
        if tid in gym.registry:
            continue
        gym.register(id=tid, entry_point="nanovla_mimic.env:SO101TabletopMimicEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"nanovla_mimic.env:Cfg__{task}__{obj}__{target or 'none'}"})


_register()
