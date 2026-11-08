"""Group each repo's episode labels into task clusters (paraphrases of the same task -> same cluster)
using the from-scratch text encoder v3; average-linkage, cosine threshold."""
import os, re, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from models import TextEncoder
from train_text import Enc, _attr_of
ROOT = os.path.expanduser("~/VLA/data"); CK = os.path.expanduser("~/VLA/ckpt/text_v3")
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
ck = torch.load(os.path.join(CK, "text_encoder.pt")); m = TextEncoder(ck["cfg"]["vocab"], max_len=ck["cfg"]["max_len"]).cuda().eval(); m.load_state_dict(ck["model"])
enc = Enc(Tokenizer.from_file(os.path.join(CK, "tokenizer.json")), 64)
E = pd.read_parquet(os.path.join(ROOT, "clean_v2/episodes.parquet")); E = E[(E["drop"].fillna("") == "") & E.label.notna()].copy()
SYN = {"mug": "cup", "glass": "cup", "block": "cube", "box": "bin", "container": "bin", "grey": "gray", "bucket": "bin", "basket": "bin"}
def attrs(t):
    d = {}
    for w in re.findall(r"[a-z0-9]+", str(t).lower()):
        k = _attr_of.get(w)
        if k in ("color", "object", "place", "side"):
            d.setdefault(k, set()).add(SYN.get(w, w))
    return d
def conflict(a, b):
    return any(k in a and k in b and not (a[k] & b[k]) for k in ("color", "object", "place", "side"))
rows = []
for rid, g in E.groupby("repo_id"):
    labs = g.label.tolist()
    with torch.no_grad():
        Z = F.normalize(m(enc(labs).cuda())[1], dim=-1)
    S = (Z @ Z.T).cpu().numpy()
    A = [attrs(l) for l in labs]
    cl = list(range(len(labs)))
    while True:  # average-linkage agglomeration
        ids = sorted(set(cl)); best = (-1, None)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = [k for k, c in enumerate(cl) if c == ids[i]]; b = [k for k, c in enumerate(cl) if c == ids[j]]
                s = S[np.ix_(a, b)].mean()
                if any(conflict(A[x], A[y]) for x in a for y in b):
                    continue  # cannot-link: different colour/object/target/side -> different task
                if s > best[0]: best = (s, (ids[i], ids[j]))
        if best[1] is None or best[0] < TH: break
        cl = [best[1][0] if c == best[1][1] else c for c in cl]
    rel = {c: k for k, c in enumerate(sorted(set(cl)))}
    for (ep, lab), c in zip(zip(g.ep, labs), cl):
        mem = [k for k, cc in enumerate(cl) if cc == c]
        medoid = labs[mem[int(np.argmax(S[np.ix_(mem, mem)].mean(1)))]]
        rows.append(dict(repo_id=int(rid), ep=int(ep), task=f"{rid}:{rel[c]}", task_text=medoid, n_tasks_repo=len(rel)))
D = pd.DataFrame(rows); D.to_parquet(os.path.join(ROOT, "clean_v2/task_clusters.parquet"))
r = D.drop_duplicates("repo_id")
print(f"threshold {TH}: repos {len(r)}, tasks/repo p50 {r.n_tasks_repo.median()} mean {r.n_tasks_repo.mean():.2f}, single-task repos {(r.n_tasks_repo==1).mean():.2f}, total tasks {D.task.nunique()}")
pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 200)
ex = D[D.repo_id == r[r.n_tasks_repo >= 3].repo_id.iloc[3]].merge(E[["repo_id", "ep", "label"]], on=["repo_id", "ep"])
print(ex[["task", "label"]].sort_values("task").head(10).to_string(index=False))
