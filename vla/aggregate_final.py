"""Aggregate the final closed-loop evaluation (data/final/*.out): pooled success rate per (config, N) with Wilson CI,
per-seed mean/std, fraction of the expert ceiling, and two-proportion z-tests for the key comparisons."""
import glob, json, math, os, re
import numpy as np, pandas as pd

ROOT = os.path.expanduser("~/VLA/data")
NAMES = {"base": "A_v1: pretrained VLA (frozen encoders)", "fix": "A_v1 + state dropout + localisation loss",
         "fixenc": "A_v1 + state dropout + localisation loss + encoder fine-tune", "scratch": "C: everything from scratch"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) + 1e-12
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def load():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "final", "*.out")):
        m = re.match(r"(\w+)_N(\d+)_s(\d+)\.out", os.path.basename(f))
        if not m:
            continue
        for line in open(f):
            if line.startswith("{") and '"cond"' in line:
                r = json.loads(line)
                rows.append(dict(cfg=m.group(1), N=int(m.group(2)), seed=int(m.group(3)), sr=r["success_rate"], n=r["n"], k=round(r["success_rate"] * r["n"])))
    return pd.DataFrame(rows)


def main():
    D = load()
    ceil = json.loads(open(os.path.join(ROOT, "final", "expert_ceiling.out")).read().strip().splitlines()[-1])["mean"]
    out = []
    for (cfg, N), g in D.groupby(["cfg", "N"]):
        k, n = int(g.k.sum()), int(g.n.sum()); lo, hi = wilson(k, n)
        out.append(dict(cfg=cfg, config=NAMES.get(cfg, cfg), N=N, seeds=len(g), successes=k, rollouts=n, sr=k / n, ci_lo=lo, ci_hi=hi,
                        seed_mean=g.sr.mean(), seed_std=g.sr.std(ddof=0), frac_of_expert=(k / n) / ceil))
    T = pd.DataFrame(out).sort_values(["N", "cfg"])
    tests = []
    for N in sorted(T.N.unique()):
        t = T[T.N == N].set_index("cfg")
        for a, b in [("fix", "base"), ("fixenc", "base"), ("base", "scratch"), ("fix", "scratch")]:
            if a in t.index and b in t.index:
                z, p = ztest(t.loc[a, "successes"], t.loc[a, "rollouts"], t.loc[b, "successes"], t.loc[b, "rollouts"])
                tests.append(dict(N=N, a=a, b=b, diff=t.loc[a, "sr"] - t.loc[b, "sr"], z=z, p=p))
    res = dict(expert_ceiling=ceil, table=T.to_dict("records"), tests=tests, complete=len(D) == 36, n_jobs=len(D))
    json.dump(res, open(os.path.join(ROOT, "final", "summary.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(T[["N", "cfg", "successes", "rollouts", "sr", "ci_lo", "ci_hi", "seed_mean", "seed_std", "frac_of_expert"]].round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print("expert ceiling", round(ceil, 3), "| jobs", len(D), "/ 36")


if __name__ == "__main__":
    main()
