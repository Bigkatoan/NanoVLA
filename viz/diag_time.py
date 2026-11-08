import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/VLA/vla"))
import nanovla2 as v2, nanovla_r as R, nanovla_c as C
torch.set_num_threads(4)
ck = torch.load(os.path.expanduser("~/VLA/ckpt/nanovla_c/vlac.pt"), map_location="cpu", weights_only=False)
P = v2.prepare("isaac_all_v2", 100, 0); P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
R.T_MAX = 10000
m, tok = C.build("cpu", ck["st"], init=""); m.load_state_dict(ck["model"]); m.eval()
eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:100]
ds = R.SeqData(P, eps, False, "own")
bins = {}; ph = {}; tk = {}; lens = []
with torch.no_grad():
    for i in range(len(eps)):
        b = R.collate([ds[i]]); lens.append(int(b["mask"].sum()))
        img = R.gpu_images(b["img"], v2.SIZE["own"], False)
        plans, *_ = m.run_seq(img, b["state"], tok(b["label"]))
        tg = b["tgt"][0, :, :C.HM]; va = b["valid"][0, :, :C.HM].unsqueeze(-1).float(); p = plans[0]
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        err = ((p - tg).abs() * va).sum((1, 2)).numpy(); hold = ((z - tg).abs() * va).sum((1, 2)).numpy()
        task = P["task"][eps[i]].split()[0]
        for k in range(len(err)):
            for key, d in ((min(k // 50, 5), bins), (int(b["phase"][0, k]), ph), (task, tk)):
                r = d.setdefault(key, [0.0, 0.0]); r[0] += err[k]; r[1] += hold[k]
out = dict(by_time={f"{k*5}-{k*5+5}s" if k < 5 else "25s+": round(float(v[0] / v[1]), 3) for k, v in sorted(bins.items())},
           by_phase={["approach", "holding", "released"][k]: round(float(v[0] / v[1]), 3) for k, v in sorted(ph.items())},
           by_first_word={k: round(float(v[0] / v[1]), 3) for k, v in sorted(tk.items())}, ep_len_s=[round(float(np.percentile(lens, q)) / 10, 1) for q in (10, 50, 90)])
json.dump(out, open(os.path.expanduser("~/VLA/viz/diag_time.json"), "w"), indent=1); print(json.dumps(out, indent=1))
