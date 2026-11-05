"""Aggregate the demo-count scaling study (data/scaling/*.out) into results/scaling.json:
pooled success rate per (config, N) with Wilson 95% CI, per-seed rates."""
import glob, json, math, os, re
import pandas as pd

ROOT = os.path.expanduser("~/VLA")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "data/scaling/*.out")):
        m = re.match(r"(\w+)_N(\d+)_s(\d+)\.out", os.path.basename(f))
        for line in open(f):
            if line.startswith("{") and '"cond"' in line:
                r = json.loads(line)
                rows.append(dict(cfg=m.group(1), N=int(m.group(2)), seed=int(m.group(3)), sr=r["success_rate"], n=r["n"], k=round(r["success_rate"] * r["n"])))
    D = pd.DataFrame(rows)
    out = []
    for (cfg, N), g in D.groupby(["cfg", "N"]):
        k, n = int(g.k.sum()), int(g.n.sum()); lo, hi = wilson(k, n)
        out.append(dict(cfg=cfg, N=int(N), seeds=len(g), successes=k, rollouts=n, sr=round(k / n, 4), ci95=[round(lo, 4), round(hi, 4)],
                        per_seed={int(s): v for s, v in zip(g.seed, g.sr)}))
    out.sort(key=lambda r: (r["cfg"], r["N"]))
    res = dict(task="MuJoCo SO-101 pick red cube -> blue square, unseen layouts, 50 rollouts per seed",
               demos="scripted expert, data/sim_bench600 (600 demos, expert success ~80%)",
               configs={"fixenc": "pretrained VLA + state dropout 0.5 + localisation loss + encoder fine-tune",
                        "scratch": "same architecture trained from scratch on the N demos"}, table=out)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results/scaling.json"), "w"), indent=1)
    for r in out:
        print(f"{r['cfg']:8s} N={r['N']:4d} seeds={r['seeds']} SR={r['sr']:.3f} CI=[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]")


if __name__ == "__main__":
    main()
