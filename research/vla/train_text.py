"""Train a from-scratch BPE tokenizer + tiny text encoder on robot instructions.

Objectives:  MLM (15%)  +  SimCSE-style InfoNCE where positives are
  (a) a second stochastic augmentation of the same instruction,
  (b) (annotated instruction, original repo task) of the same episode when available.
Held-out evaluation is split by repo.
"""
import argparse, glob, json, os, random, re, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from models import TextEncoder, MLMHead, n_params

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.expanduser("~/VLA/ckpt/text")
SPECIAL = ["[PAD]", "[UNK]", "[MASK]"]

SYN = [
    ["grasp", "grab", "pick up", "take", "seize"],
    ["place", "put", "set", "drop", "deposit"],
    ["move", "bring", "carry", "transfer"],
    ["into", "inside", "in"],
    ["bin", "container", "box"],
    ["cube", "block"],
    ["lift", "raise"],
    ["release", "let go of", "drop"],
    ["onto", "on", "on top of"],
]
FILLERS = ["the robotic arm", "robotic arm", "securely", "firmly", "carefully", "the arm", "gently"]
_syn_map = {w: g for g in SYN for w in g}

# attribute classes used to build hard negatives (swap one attribute -> different command)
ATTR = {
    "color": ["red", "blue", "green", "yellow", "white", "orange", "black", "purple", "gray", "pink", "brown", "beige", "silver", "grey"],
    "object": ["cube", "ball", "cup", "duck", "lego", "pen", "marker", "bottle", "pawn", "tape", "lid", "banana", "pineapple", "toy", "mug",
               "candle", "carrot", "lemon", "sponge", "doll", "egg", "spoon", "scissors", "pencil", "fork", "towel", "ring", "strawberry",
               "eraser", "dice", "battery", "screwdriver", "brush", "cylinder", "triangle", "knight", "rook"],
    "place": ["bin", "box", "bowl", "tray", "plate", "basket", "bucket", "pot", "cabinet", "drawer", "mat", "jar", "holder", "crate", "bag", "container"],
    "side": ["left", "right", "center", "middle", "top", "bottom", "front", "back", "leftmost", "rightmost", "corner"],
    "number": [str(i) for i in range(1, 10)] + ["one", "two", "three", "first", "second", "third"],
    "verb": ["open", "close", "push", "pull", "stack", "fold", "pour", "flip", "rotate", "press"],
}
_attr_of = {w: k for k, ws in ATTR.items() for w in ws}


def make_negative(t, rng):
    """Change exactly one attribute (or swap two colours) so the command means something else. None if impossible."""
    toks = re.findall(r"\w+|\W+", t)
    idx = [i for i, w in enumerate(toks) if w.lower() in _attr_of]
    if not idx:
        return None
    cols = [i for i in idx if _attr_of[toks[i].lower()] == "color"]
    if len(cols) >= 2 and toks[cols[0]].lower() != toks[cols[1]].lower() and rng.random() < 0.3:
        i, j = cols[0], cols[1]
        toks[i], toks[j] = toks[j], toks[i]  # role swap: "red cube -> blue bin" vs "blue cube -> red bin"
        return "".join(toks)
    i = rng.choice(idx)
    w = toks[i].lower()
    alt = [x for x in ATTR[_attr_of[w]] if x != w and not (w in ("gray", "grey") and x in ("gray", "grey"))]
    toks[i] = rng.choice(alt)
    return "".join(toks)


def clean(t):
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def augment(t, rng):
    s = " " + t.lower() + " "
    for f in FILLERS:
        if rng.random() < 0.5:
            s = s.replace(" " + f + " ", " ")
    words = s.split()
    out = []
    i = 0
    while i < len(words):
        w = words[i].strip(".,;")
        if w in _syn_map and rng.random() < 0.4:
            out.append(rng.choice(_syn_map[w]))
        elif rng.random() < 0.08 and len(words) > 4:
            pass  # word dropout
        else:
            out.append(words[i])
        i += 1
    s = " ".join(out)
    if rng.random() < 0.5:
        s = s.replace(",", "").replace(".", "")
    return s if s.strip() else t


def load_corpus():
    a = pd.read_parquet(os.path.join(ROOT, "meta/annotations_all.parquet"))
    a["task"] = a.task.map(clean)
    pairs = []  # (repo, annotated, original)
    for f in glob.glob(os.path.join(ROOT, "frames/*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["task", "task_orig"]).drop_duplicates()
        except Exception:
            continue
        repo = os.path.basename(f)[:-8].replace("__", "/", 1)
        for t, o in d.itertuples(index=False):
            if o and clean(o).lower() != clean(t).lower():
                pairs.append((repo, clean(t), clean(o)))
    return a, pd.DataFrame(pairs, columns=["repo", "task", "orig"])


def load_corpus_clean():
    """Cleaned corpus: annotations without degenerate labels / benchmark repos, colour-fixed labels from clean_v2,
    (annotation, original) pairs only when the original task is neither a template nor contradicts the annotation."""
    a = pd.read_parquet(os.path.join(ROOT, "meta/annotations_all.parquet"))
    bench = set(json.load(open(os.path.join(ROOT, "bench/repos.json"))))
    a = a[~a.repo.isin(bench)]
    a = a[~a.task.str.contains("FROZEN FRAMES", case=False, na=True)]
    a["task"] = a.task.map(clean)
    a = a[a.task.str.split().str.len() > 2]
    E = pd.read_parquet(os.path.join(ROOT, "clean_v2/episodes.parquet"))
    fixed = E[E.get("label_src", pd.Series(dtype=str)) == "annotation+color_fixed"]
    fx = {(r, int(e)): clean(l) for r, e, l in zip(fixed.repo, fixed.ep, fixed.label)}
    a["task"] = [fx.get((r, int(e)), t) for r, e, t in zip(a.repo, a.ep, a.task)]
    dropped = set(zip(E[E["drop"].fillna("") != ""].repo, E[E["drop"].fillna("") != ""].ep.astype(int)))
    a = a[[(r, int(e)) not in dropped for r, e in zip(a.repo, a.ep)]]
    LA = pd.read_parquet(os.path.join(ROOT, "clean/label_audit.parquet"))
    LA = LA[~LA["flags"].str.contains("orig_template|orig_conflict|degenerate") & ~LA.repo.isin(bench)]
    LA = LA[[(r, int(e)) not in dropped for r, e in zip(LA.repo, LA.ep)]]
    pairs = [(r, fx.get((r, int(e)), clean(t)), clean(o)) for r, e, t, o in zip(LA.repo, LA.ep, LA.task, LA.orig)
             if o and clean(o).lower() != clean(t).lower()]
    print(f"clean corpus: {len(a)} episodes, {a.task.nunique()} unique, {len(fx)} colour-fixed, {len(pairs)} trusted ann/orig pairs")
    return a[["repo", "ep", "task"]], pd.DataFrame(pairs, columns=["repo", "task", "orig"]).drop_duplicates()


def train_tokenizer(texts, vocab):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
    tok.decoder = decoders.BPEDecoder()
    tr = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2, special_tokens=SPECIAL)
    tok.train_from_iterator(texts, tr)
    return tok


class Enc:
    def __init__(self, tok, max_len):
        self.tok, self.L = tok, max_len

    def __call__(self, texts):
        ids = torch.zeros(len(texts), self.L, dtype=torch.long)
        for i, e in enumerate(self.tok.encode_batch(texts)):
            x = e.ids[: self.L]
            ids[i, : len(x)] = torch.tensor(x)
        return ids


def mask_tokens(ids, vocab, p=0.15, mask_id=2):
    labels = ids.clone()
    real = ids != 0
    m = (torch.rand(ids.shape, device=ids.device) < p) & real
    labels[~m] = -100
    x = ids.clone()
    r = torch.rand(ids.shape, device=ids.device)
    x[m & (r < 0.8)] = mask_id
    rnd = m & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(3, vocab, (int(rnd.sum()),), device=ids.device)
    return x, labels


def info_nce(a, b, keys, tau):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / tau
    same = torch.tensor([[ka == kb for kb in keys] for ka in keys], device=a.device)
    eye = torch.eye(len(keys), dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(same & ~eye, -1e4)  # remove false negatives (same canonical text)
    y = torch.arange(len(keys), device=a.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)), (logits.argmax(1) == y).float().mean()


def info_nce_hn(a, b, n, n_valid, keys, tau):
    """InfoNCE a->b with extra hard negatives n (one per anchor, only valid ones used as columns)."""
    a, b, n = F.normalize(a, dim=-1), F.normalize(b, dim=-1), F.normalize(n, dim=-1)
    same = torch.tensor([[ka == kb for kb in keys] for ka in keys], device=a.device)
    eye = torch.eye(len(keys), dtype=torch.bool, device=a.device)
    lb = (a @ b.T / tau).masked_fill(same & ~eye, -1e4)
    ln = (a @ n.T / tau).masked_fill(~n_valid.unsqueeze(0), -1e4)
    y = torch.arange(len(keys), device=a.device)
    loss = F.cross_entropy(torch.cat([lb, ln], 1), y) * 0.5 + F.cross_entropy(lb.T, y) * 0.5
    return loss, (lb.argmax(1) == y).float().mean()


@torch.no_grad()
def attr_margin(model, enc, texts, dev, rng, n=1024):
    """mean cos(anchor, paraphrase-aug) - mean cos(anchor, attribute-swapped). Higher = more attribute-sensitive."""
    model.eval()
    T, P, N = [], [], []
    for t in rng.sample(texts, min(len(texts), 4 * n)):
        ng = make_negative(t, rng)
        if ng is None or ng.lower() == t.lower():
            continue
        T.append(t); P.append(augment(t, rng)); N.append(ng)
        if len(T) >= n:
            break
    e = lambda x: F.normalize(model(enc(x).to(dev))[1], dim=-1)
    a, p, q = e(T), e(P), e(N)
    sp, sn = (a * p).sum(1), (a * q).sum(1)
    model.train()
    return dict(sim_para=round(sp.mean().item(), 3), sim_swap=round(sn.mean().item(), 3), margin=round((sp - sn).mean().item(), 3),
                para_gt_swap=round((sp > sn).float().mean().item(), 3))


@torch.no_grad()
def evaluate(model, enc, texts, pairs, dev, rng, tau):
    model.eval()
    res = attr_margin(model, enc, texts, dev, rng)
    model.eval()
    t = rng.sample(texts, min(512, len(texts)))
    a = model(enc(t).to(dev))[1]
    b = model(enc([augment(x, rng) for x in t]).to(dev))[1]
    _, acc = info_nce(a, b, t, tau)
    res["aug_r1@512"] = acc.item()
    if len(pairs):
        p = pairs.sample(min(512, len(pairs)), random_state=0)
        a = model(enc(p.task.tolist()).to(dev))[1]
        b = model(enc(p.orig.tolist()).to(dev))[1]
        _, acc = info_nce(a, b, p.task.tolist(), tau)
        res["ann2orig_r1"] = acc.item()
        res["n_pairs"] = len(p)
    model.train()
    return res


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--hardneg", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)
    ann, pairs = load_corpus_clean() if a.clean else load_corpus()
    repos = sorted(ann.repo.unique())
    rng.shuffle(repos)
    val_repos = set(repos[: len(repos) // 10])
    tr_texts = sorted(set(ann[~ann.repo.isin(val_repos)].task))
    va_texts = sorted(set(ann[ann.repo.isin(val_repos)].task) - set(tr_texts))
    tr_pairs = pairs[~pairs.repo.isin(val_repos)]
    va_pairs = pairs[pairs.repo.isin(val_repos)]
    print(f"train texts {len(tr_texts)} val texts {len(va_texts)} pairs tr {len(tr_pairs)} va {len(va_pairs)}")

    corpus = tr_texts + tr_pairs.orig.tolist() + [augment(t, rng) for t in tr_texts for _ in range(3)]
    tok = train_tokenizer(corpus, a.vocab)
    tok.save(os.path.join(OUT, "tokenizer.json"))
    V = tok.get_vocab_size()
    enc = Enc(tok, a.max_len)
    lens = [len(e.ids) for e in tok.encode_batch(tr_texts)]
    print("vocab", V, "tokens/instr p50 %d p95 %d max %d" % (np.median(lens), np.percentile(lens, 95), max(lens)))

    dev = "cuda"
    model = TextEncoder(V, max_len=a.max_len).to(dev)
    head = MLMHead(model.d, V, model.tok).to(dev)
    print("text encoder params %.2fM" % n_params(model))
    params = list(model.parameters()) + [p for n, p in head.named_parameters() if not n.startswith("emb")]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    pair_list = list(tr_pairs[["task", "orig"]].itertuples(index=False))
    log = open(os.path.join(OUT, "log.jsonl"), "w")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        n_pair = min(len(pair_list), a.bs // 4)
        P = rng.sample(pair_list, n_pair) if n_pair else []
        T = rng.sample(tr_texts, a.bs - n_pair)
        anchors = [p[0] for p in P] + T
        views = [p[1] if rng.random() < 0.7 else augment(p[0], rng) for p in P] + [augment(t, rng) for t in T]
        ids_a, ids_b = enc(anchors).to(dev), enc(views).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xm, lab = mask_tokens(ids_a, V)
            h, _, _ = model(xm)
            mlm = F.cross_entropy(head(h).flatten(0, 1).float(), lab.flatten(), ignore_index=-100)
            _, pa, _ = model(ids_a)
            _, pb, _ = model(ids_b)
            if a.hardneg:
                negs = [make_negative(t, rng) for t in anchors]
                nv = torch.tensor([x is not None for x in negs], device=dev)
                _, pn, _ = model(enc([x if x is not None else t for x, t in zip(negs, anchors)]).to(dev))
                cl, acc = info_nce_hn(pa.float(), pb.float(), pn.float(), nv, anchors, a.tau)
            else:
                cl, acc = info_nce(pa.float(), pb.float(), anchors, a.tau)
            loss = mlm + cl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            r = dict(step=step, mlm=round(mlm.item(), 3), cl=round(cl.item(), 3), tr_r1=round(acc.item(), 3), t=round(time.time() - t0))
            if step % 2500 == 0 or step == a.steps:
                r.update({k: round(v, 3) if isinstance(v, float) else v for k, v in evaluate(model, enc, va_texts, va_pairs, dev, rng, a.tau).items()})
            print(r, flush=True)
            log.write(json.dumps(r) + "\n"); log.flush()
    torch.save(dict(model=model.state_dict(), cfg=dict(vocab=V, max_len=a.max_len), args=vars(a)), os.path.join(OUT, "text_encoder.pt"))
    # nearest-neighbour sanity check on held-out instructions
    model.eval()
    with torch.no_grad():
        q = va_texts[:8]
        bank = va_texts
        E = F.normalize(torch.cat([model(enc(bank[i:i + 512]).to(dev))[1] for i in range(0, len(bank), 512)]), dim=-1)
        Q = F.normalize(model(enc(q).to(dev))[1], dim=-1)
        nn_ = (Q @ E.T).topk(3, dim=1).indices
        for i, t in enumerate(q):
            print("Q:", t)
            for j in nn_[i][1:]:
                print("    ->", bank[j])


if __name__ == "__main__":
    main()
