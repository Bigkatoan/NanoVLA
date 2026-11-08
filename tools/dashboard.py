"""Live dashboard for the NanoVLA-M fixes (scheduled sampling + DAgger): reads the real logs, refreshes every 5 s.
    python3 tools/dashboard.py --port 8765        # then open http://localhost:8765
"""
import argparse, glob, json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

V = os.path.expanduser("~/VLA")
RUNS = {"M-L1 (gốc)": "ckpt/nanovla_m_l1/train.log", "M + scheduled sampling": "ckpt/nanovla_m_ss/train.log",
        "M + SS + vel + jitter": "ckpt/nanovla_m_vel/train.log", "M + DAgger r1": "ckpt/nanovla_m_dagger1/train.log", "M + DAgger r2": "ckpt/nanovla_m_dagger2/train.log"}
EVALS = {"v1": "results/final_v1/A_zeroshot_multitask.eval.log", "M-L1": "results/nanovla_m/A_l1.eval.log",
         "M+SS": "results/fix/A_ss.eval.log", "M+SS+vel+jitter": "results/fix/A_vel.eval.log", "M+SS+DAgger r1": "results/fix/A_dagger1.eval.log", "M+SS+DAgger r2": "results/fix/A_dagger2.eval.log",
         "C: M-L1": "results/nanovla_m/C_l1.eval.log", "C: M+SS+DAgger r1": "results/fix/C_dagger1.eval.log", "C: M+SS+DAgger r2": "results/fix/C_dagger2.eval.log"}


def jl(path):
    out = []
    try:
        for l in open(os.path.join(V, path)):
            l = l.strip()
            if l.startswith("{"):
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return out


def data():
    runs = {}
    for name, p in RUNS.items():
        L = jl(p)
        if not L:
            continue
        tr = [(r["step"], r["act"]) for r in L if "act" in r]
        va = [(r["step"], r["val_rel"]) for r in L if "val_rel" in r]
        ss = [(r["step"], r.get("ss_p", 0)) for r in L if "act" in r]
        last = tr[-1][0] if tr else 0
        tot = next((r.get("steps") for r in L if False), None)
        runs[name] = dict(train=tr[::2], val=va, ss=ss[::4], last=last, done=any("saved" in r for r in L), sps=next((r["s_per_step"] for r in reversed(L) if "s_per_step" in r), None))
    evals = {}
    for name, p in EVALS.items():
        L = [r for r in jl(p) if "success" in r]
        if name == "v1":
            L = [dict(r, success=r["success"] and not (r["task"] == "rel" and r["t"] < 1.0)) for r in L]   # 2 spurious rel successes = failures
        if not L:
            continue
        by = {}
        for r in L:
            b = by.setdefault(r["task"], [0, 0]); b[0] += r["success"]; b[1] += 1
        evals[name] = dict(n=len(L), succ=sum(r["success"] for r in L), by=by)
    gen = []
    for f in sorted(glob.glob(os.path.join(V, "data/dagger_r*/*.log"))):
        L = [r for r in jl(os.path.relpath(f, V)) if "kept" in r]
        if L:
            gen.append(dict(chunk=os.path.relpath(f, os.path.join(V, "data")), attempts=len(L), kept=sum(r["kept"] for r in L),
                            policy_frac=round(sum(r.get("policy_frac", 0) for r in L) / len(L), 3), expert_ok=round(sum(r["success_check"] for r in L) / len(L), 3)))
    stage = []
    try:
        stage = open(os.path.join(V, "results/fix/pipeline.log")).read().strip().splitlines()[-12:]
    except FileNotFoundError:
        pass
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        gpu = dict(util=int(g[0]), mem=int(g[1]), tot=int(g[2]))
    except Exception:
        gpu = None
    return dict(t=time.strftime("%H:%M:%S"), runs=runs, evals=evals, gen=gen, stage=stage, gpu=gpu)


PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NanoVLA Training</title>
<style>
:root{--bg:#f5f6f8;--fg:#15171c;--mut:#5d6472;--card:#fff;--line:#dde0e6;--a:#e8590c;--b:#1c7ed6;--c:#2f9e44;--d:#ae3ec9;--e:#868e96}
@media (prefers-color-scheme:dark){:root{--bg:#0d0f13;--fg:#e7e9ee;--mut:#9aa1b0;--card:#161920;--line:#2a2f39;--a:#ff922b;--b:#4dabf7;--c:#69db7c;--d:#da77f2;--e:#adb5bd;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:20px max(16px,env(safe-area-inset-left,0px))}
main{max-width:1150px;margin:0 auto;display:grid;gap:14px}
h1{font-size:1.25rem;margin:0}.mut{color:var(--mut)}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.card h2{font-size:.95rem;margin:0 0 8px}
svg{width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:4px 6px;border-top:1px solid var(--line);text-align:left}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--c)}
pre{margin:0;font-size:.8rem;white-space:pre-wrap;color:var(--mut)}
.lg span{display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:.82rem}.lg b{width:14px;height:3px;display:inline-block}
</style></head><body><main>
<h1>NanoVLA-M: sửa lệch phân phối (scheduled sampling + DAgger)</h1>
<div class="mut" id="hdr"></div>
<div class="grid">
<div class="card"><h2>Loss train (L1 hành động, chuẩn hoá)</h2><div id="train"></div><div class="lg" id="lg1"></div></div>
<div class="card"><h2>Lỗi val offline (÷ mức đứng yên, thấp = tốt)</h2><div id="val"></div><div class="lg" id="lg2"></div></div>
</div>
<div class="grid">
<div class="card"><h2>Eval vòng kín (bài A: 48 episode · C: 24 vật chưa thấy)</h2><div id="evals"></div></div>
<div class="card"><h2>Sinh dữ liệu DAgger (policy lái từng đoạn, chuyên gia sửa)</h2><div id="gen"></div></div>
</div>
<div class="card"><h2>Pipeline</h2><pre id="stage"></pre></div>
</main><script>
const COL=["var(--e)","var(--a)","var(--c)","var(--b)","var(--d)"];
function chart(series,key,ymaxFix){const W=520,H=230,P=34;let xs=[],ys=[];
 series.forEach(s=>s[key].forEach(p=>{xs.push(p[0]);ys.push(p[1])}));if(!xs.length)return'<div class="mut">chưa có dữ liệu</div>';
 const xm=Math.max(...xs),ym=ymaxFix||Math.max(...ys)*1.05;const X=x=>P+(W-P-8)*x/xm,Y=y=>H-22-(H-30)*Math.min(y,ym)/ym;
 let g=`<svg viewBox="0 0 ${W} ${H}"><line x1="${P}" y1="${H-22}" x2="${W-8}" y2="${H-22}" stroke="var(--line)"/>`;
 for(let k=0;k<=4;k++){const y=ym*k/4;g+=`<text x="2" y="${Y(y)+4}" font-size="10" fill="var(--mut)">${y.toFixed(2)}</text><line x1="${P}" x2="${W-8}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--line)" stroke-dasharray="2 4"/>`}
 g+=`<text x="${W-8}" y="${H-6}" font-size="10" text-anchor="end" fill="var(--mut)">bước ${xm}</text>`;
 series.forEach(s=>{const pts=s[key];if(!pts.length)return;g+=`<polyline fill="none" stroke="${s.col}" stroke-width="2" points="${pts.map(p=>X(p[0])+','+Y(p[1])).join(' ')}"/>`;
  if(key=='val')pts.forEach(p=>g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="${s.col}"/>`)});return g+'</svg>'}
async function tick(){let d;try{d=await (await fetch('data.json')).json()}catch(e){return}
 const names=Object.keys(d.runs),ser=names.map((n,i)=>({...d.runs[n],col:COL[i%COL.length],name:n}));
 document.getElementById('train').innerHTML=chart(ser,'train');document.getElementById('val').innerHTML=chart(ser,'val');
 const lg=ser.map(s=>`<span><b style="background:${s.col}"></b>${s.name} · ${s.last} bước${s.done?' ✓':''}</span>`).join('');
 document.getElementById('lg1').innerHTML=lg;document.getElementById('lg2').innerHTML=lg;
 let t='<table><tr><th>Model</th><th>Tiến độ</th><th>Thành công</th><th>Theo tác vụ</th></tr>';
 for(const [n,e] of Object.entries(d.evals)){const tot=n.startsWith('C:')?24:48;
  t+=`<tr><td>${n}</td><td><div class="bar"><i style="width:${100*e.n/tot}%"></i></div>${e.n}/${tot}</td><td><b>${e.succ}</b>/${e.n}</td><td class="mut">${Object.entries(e.by).map(([k,v])=>k+' '+v[0]+'/'+v[1]).join(' · ')}</td></tr>`}
 document.getElementById('evals').innerHTML=t+'</table>';
 let g='<table><tr><th>Chunk</th><th>Episode giữ</th><th>Thử</th><th>Policy lái</th><th>Chuyên gia cứu được</th></tr>';
 d.gen.forEach(c=>g+=`<tr><td>${c.chunk}</td><td>${c.kept}</td><td>${c.attempts}</td><td>${Math.round(100*c.policy_frac)}%</td><td>${Math.round(100*c.expert_ok)}%</td></tr>`);
 document.getElementById('gen').innerHTML=d.gen.length?g+'</table>':'<div class="mut">chưa bắt đầu</div>';
 document.getElementById('stage').textContent=d.stage.join('\n');
 document.getElementById('hdr').textContent=`cập nhật ${d.t}`+(d.gpu?` · GPU ${d.gpu.util}% · ${d.gpu.mem}/${d.gpu.tot} MiB`:'')}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data.json"):
            b = json.dumps(data()).encode(); ct = "application/json"
        else:
            b = PAGE.encode(); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); a = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
