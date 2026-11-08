"""Follow a lerobot-train log and mirror its metrics into TensorBoard (lerobot itself only logs to wandb / stdout).
Parses lines like '... step:2K smpl:64K ep:1K epch:0.3 loss:0.081 grdn:0.52 lr:1.0e-04 updt_s:0.4 data_s:0.01'.

    python tools/log2tb.py results/smolvla/train.log ~/VLA/tb/smolvla_sim
"""
import os, re, sys, time
from torch.utils.tensorboard import SummaryWriter

UNIT = {"K": 1e3, "M": 1e6, "B": 1e9}


def num(s):
    s = s.strip()
    return float(s[:-1]) * UNIT[s[-1]] if s and s[-1] in UNIT else float(s)


def main():
    log, out = sys.argv[1], sys.argv[2]; tb = SummaryWriter(out); pos = 0; last = -1
    pat = re.compile(r"step:(\S+)\s+smpl:\S+\s+ep:\S+\s+epch:(\S+)\s+loss:(\S+)\s+grdn:(\S+)\s+lr:(\S+)\s+updt_s:(\S+)\s+data_s:(\S+)")
    while True:
        if os.path.exists(log):
            with open(log, errors="ignore") as f:
                f.seek(pos); txt = f.read(); pos = f.tell()
            for m in pat.finditer(txt):
                try:
                    step = int(num(m.group(1)))
                    if step == last:
                        continue
                    last = step
                    for k, g in (("epoch", 2), ("loss", 3), ("grad_norm", 4), ("lr", 5), ("update_s", 6), ("data_s", 7)):
                        tb.add_scalar(f"train/{k}", num(m.group(g)), step)
                except ValueError:
                    pass
            tb.flush()
            if "End of training" in txt:
                break
        time.sleep(20)


if __name__ == "__main__":
    main()
