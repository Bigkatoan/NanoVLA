"""Job runner behind the web app. A job is a list of stages run one after another as subprocesses; its state lives in
jobs/<id>/state.json (status, stage, progress, metrics) and every stage writes jobs/<id>/<stage>.log.

Job kinds
  collect  record demonstrations on the real SO-101 with lerobot-record (leader arm, wrist + overview camera)
  distill  dataset -> few-shot teacher -> teacher eval -> teacher labels -> wrist-only student -> models/<name>/
  dagger   frames the student visited on the robot (deploy/run_student.py --record) -> teacher labels -> student retrained
  teacher  public SO-101 datasets (degrees) -> download -> merge -> SmolVLA teacher (hours)
  import   one public LeRobot dataset from the Hub -> data/lerobot/<owner>__<name>, prepared
  run      a finished student on the real SO-101 (deploy/run_student.py), optionally recording frames for DAgger
  sim_collect  scripted demos in the Isaac Sim tabletop (wrist + front camera) -> LeRobot dataset in degrees
  sim_eval     closed-loop success rate of a student in the Isaac Sim tabletop

    python -m nanovla.pipeline new distill '{"dataset": "data/lerobot/my_task", "name": "my_task"}'   # create + start
    python -m nanovla.pipeline run jobs/<id>                                                          # (used by new)
"""
import json, os, re, shutil, signal, subprocess, sys, time, uuid
from nanovla import paths as P

STAGES = {
    "collect": ["record"],
    "distill": ["prepare", "fewshot", "eval_teacher", "label_train", "label_test", "student", "package"],
    "dagger": ["prepare", "label_dagger", "student", "package"],
    "teacher": ["select", "download", "merge", "train"],
    "import": ["download", "prepare_import"],
    "run": ["robot"],
    "sim_collect": ["generate", "convert"],
    "sim_eval": ["isaac"],
}
SIM_SPECS = {"pick": ("pick", "pick"), "place": ("place", "pad,tray,container"), "push": ("push", "push"), "rotate": ("rotate", "rotate")}


def jdir(jid):
    return os.path.join(P.JOBS, jid)


def load(jid):
    return json.load(open(os.path.join(jdir(jid), "state.json")))


def save(st):
    p = os.path.join(jdir(st["id"]), "state.json"); tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1); os.replace(tmp, p)


def new(kind, params):
    if kind not in STAGES:
        raise ValueError(f"unknown job kind {kind}")
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    os.makedirs(jdir(jid))
    st = dict(id=jid, kind=kind, params=params, status="queued", created=time.time(), stage=None, progress=0.0, pid=None,
              stages=[dict(name=s, status="pending") for s in STAGES[kind]], result={}, error=None)
    save(st)
    p = subprocess.Popen([sys.executable, "-m", "nanovla.pipeline", "run", jid], cwd=P.ROOT, start_new_session=True,
                         stdout=open(os.path.join(jdir(jid), "runner.log"), "w"), stderr=subprocess.STDOUT)
    st["pid"] = p.pid; save(st)
    return st


def stop(jid):
    st = load(jid)
    if st["status"] in ("running", "queued") and st.get("pid"):
        try:
            os.killpg(st["pid"], signal.SIGTERM); time.sleep(2); os.killpg(st["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    st = load(jid); st["status"] = "stopped"; save(st)
    return st


def list_jobs():
    out = []
    for j in sorted(os.listdir(P.JOBS), reverse=True):
        try:
            out.append(load(j))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ running
PROG = [(re.compile(r"step:(\d+(?:\.\d+)?)(K?)\s"), "lerobot"), (re.compile(r'"labelled": (\d+)'), "label"), (re.compile(r'"step": (\d+)'), "student"),
        (re.compile(r"(\d+)/(\d+) \["), "tqdm")]


def sh(st, stage, cmd, gpu=True, total=None, env=None):
    """run one stage command, follow its log for progress; raise on failure"""
    log = os.path.join(jdir(st["id"]), f"{stage}.log")
    full = ([P.GPU, "86400"] if gpu else []) + cmd
    e = dict(os.environ); e.setdefault("HF_HUB_OFFLINE", "1"); e["PYTHONPATH"] = P.ROOT; e.update(env or {})
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(full) + "\n"); fh.flush()
        p = subprocess.Popen(full, cwd=P.ROOT, stdout=fh, stderr=subprocess.STDOUT, env=e)
        while p.poll() is None:
            time.sleep(5)
            if total:
                try:
                    txt = open(log, errors="ignore").read()[-20000:]; best = None
                    for rx, kind in PROG:
                        m = rx.findall(txt)
                        if m:
                            v = m[-1]
                            if kind == "lerobot":
                                best = float(v[0]) * (1000 if v[1] == "K" else 1)
                            elif kind == "tqdm":
                                best = float(v[0]) if int(v[1]) == total else best
                            else:
                                best = float(v)
                            if best is not None:
                                break
                    if best is not None:
                        st["progress"] = round(min(1.0, best / total), 3); save(st)
                except Exception:
                    pass
    if p.returncode != 0:
        tail = open(log, errors="ignore").read()[-1500:]
        raise RuntimeError(f"stage {stage} failed (exit {p.returncode}):\n{tail}")
    return open(log, errors="ignore").read()


def last_json(txt):
    for line in reversed(txt.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


def eps_str(eps):
    return ",".join(str(e) for e in eps)


def run(jid):
    st = load(jid); st["status"] = "running"; st["started"] = time.time(); save(st)
    pr = st["params"]; J = jdir(jid); PYM = [P.PY, "-m"]
    try:
        for s in st["stages"]:
            s["status"] = "running"; s["started"] = time.time(); st["stage"] = s["name"]; st["progress"] = 0.0; save(st)
            name = s["name"]; res = {}
            # ---------------- collect
            if st["kind"] == "collect" and name == "record":
                root = os.path.join(P.DATA, pr["name"])
                cams = "{" + ", ".join(f"{k}: {{type: opencv, index_or_path: {v}, width: 640, height: 480, fps: 30}}" for k, v in (("wrist", pr["wrist_cam"]), ("front", pr["front_cam"]))) + "}"
                cmd = [os.path.join(P.ENV, "bin", "lerobot-record"), "--robot.type=so101_follower", f"--robot.port={pr['robot_port']}", f"--robot.id={pr.get('robot_id', 'so101')}",
                       f"--robot.cameras={cams}", "--teleop.type=so101_leader", f"--teleop.port={pr['leader_port']}", f"--teleop.id={pr.get('leader_id', 'leader')}",
                       f"--dataset.repo_id=local/{pr['name']}", f"--dataset.root={root}", f"--dataset.single_task={pr['task']}",
                       f"--dataset.num_episodes={pr.get('episodes', 10)}", f"--dataset.episode_time_s={pr.get('episode_s', 30)}",
                       f"--dataset.reset_time_s={pr.get('reset_s', 10)}", "--dataset.push_to_hub=false", "--display_data=false", "--play_sounds=false"]
                if os.path.exists(root) and pr.get("resume", True):
                    cmd.append("--resume=true")
                sh(st, name, cmd, gpu=False); res = dict(dataset=root)
            # ---------------- Isaac Sim
            elif name == "generate":
                spec, tasks = SIM_SPECS[pr.get("task", "pick")]; raws = []
                env = {"TABLETOP_FRONT": "1", "TABLETOP_RES": "240x320", "TABLETOP_CAM_SCALE": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "TABLETOP_HOLDOUT": "glue_stick,spool,toy_brick"}
                n = int(pr.get("chunks", 2))
                for c in range(n):
                    out = os.path.join(J, f"raw_c{c}"); raws.append(out)
                    sh(st, name, [os.path.join(P.ROOT, "envs", "isaac_env", "bin", "python"), "-u", os.path.join(P.ROOT, "research", "isaac_bench", "traj_tool.py"),
                       "--spec", os.path.join(P.ROOT, "research", "isaac_bench", "specs", f"{spec}.json"), "--tasks", tasks, "--episodes", str(pr.get("episodes_per_chunk", 75)),
                       "--seed", str(int(pr.get("seed", 70000)) + 37 * c), "--skip_frames", "2", "--start_noise", "0.3", "--start_noise_p", "0.5",
                       "--dart_mix", "0:0.3,0.03:0.4,0.08:0.3", "--slow_near", "2", "--num_envs", "12", "--out", out], env=env)
                    st["progress"] = (c + 1) / n; save(st)
                pr["raws"] = raws; res = dict(chunks=len(raws))
            elif name == "convert":
                out = os.path.join(P.DATA, pr["name"])
                res = last_json(sh(st, name, [P.PY, os.path.join(P.ROOT, "research", "tools", "to_lerobot.py"), "--out", out, "--cams", "wrist,front", "--hw", "240x320",
                                              "--scale", "57.29578", "--repo_id", f"local/{pr['name']}", "--src"] + pr["raws"], gpu=False))
                from nanovla import merge as M
                M.prepare(out); res = {k: v for k, v in M.describe(out).items() if k != "episodes_by_task"}
                for r in pr["raws"]:
                    shutil.rmtree(r, ignore_errors=True)
            elif name == "isaac":
                md = os.path.join(P.MODELS, pr["model"]); port = str(6200 + int(time.time()) % 500); out = os.path.join(J, "sim_eval.json")
                srv = subprocess.Popen([P.PY, "-m", "nanovla.sim_server", "--model", md, "--port", port], cwd=P.ROOT, stdout=open(os.path.join(J, "sim_server.log"), "w"),
                                       stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=P.ROOT))
                try:
                    for _ in range(120):
                        if '"ready"' in open(os.path.join(J, "sim_server.log")).read():
                            break
                        if srv.poll() is not None:
                            raise RuntimeError("student server failed:\n" + open(os.path.join(J, "sim_server.log")).read()[-1500:])
                        time.sleep(2)
                    env = {"TABLETOP_RES": "240x320", "OMNI_KIT_ACCEPT_EULA": "YES"}
                    cmd = [os.path.join(P.ROOT, "envs", "isaac_env", "bin", "python"), "-u", os.path.join(P.ROOT, "research", "isaac_bench", "tabletop_eval.py"),
                           "--cams", "wrist", "--tasks", pr.get("tasks", "pick"), "--episodes", str(pr.get("episodes", 24)), "--num_envs", "4", "--port", port,
                           "--diag", "1", "--out", out, "--video", os.path.join(J, "sim_eval.mp4")]
                    if pr.get("objects"):
                        cmd += ["--objects", pr["objects"]]
                    sh(st, name, cmd, env=env)
                finally:
                    srv.kill()
                r = json.load(open(out)); res = dict(success_rate=r["success_rate"], episodes=r["episodes"], tasks=r["tasks"], by=r.get("by", {}))
                rp = os.path.join(md, "report.json"); rep = json.load(open(rp)); rep.setdefault("sim", []).append(dict(res, when=time.strftime("%Y-%m-%d %H:%M"), job=jid))
                json.dump(rep, open(rp, "w"), indent=1)
            # ---------------- run a student on the robot
            elif st["kind"] == "run" and name == "robot":
                md = os.path.join(P.MODELS, pr["model"])
                cmd = [os.path.join(P.ROOT, "envs", "deploy", "bin", "python"), os.path.join(P.ROOT, "deploy", "run_student.py"), "--onnx", os.path.join(md, "student.onnx"),
                       "--meta", os.path.join(md, "meta.json"), "--task", pr["task"], "--port", pr["robot_port"], "--robot_id", pr.get("robot_id", "so101"),
                       "--wrist_cam", str(pr.get("wrist_cam", 0)), "--seconds", str(pr.get("seconds", 30))]
                if pr.get("dry_run"):
                    cmd.append("--dry_run")
                if pr.get("record"):
                    rec = os.path.join(P.DATA, f"{pr['model']}_dagger{int(time.time())}"); pr["recorded"] = rec
                    cmd += ["--front_cam", str(pr.get("front_cam", 2)), "--record", rec]
                res = last_json(sh(st, name, cmd, gpu=False)); res["recorded"] = pr.get("recorded")
            # ---------------- distill / dagger
            elif name == "prepare":
                from nanovla import merge as M
                root = os.path.abspath(pr["dataset"]); M.prepare(root); d = M.describe(root)
                if st["kind"] == "distill":
                    tr, te = M.default_split(d)
                    cap = int(pr.get("max_train_per_task", 0))             # large datasets: keep at most N demos per instruction
                    if cap:
                        keep = set(e for v in d["episodes_by_task"].values() for e in [x for x in v if x not in te][:cap]); tr = [e for e in tr if e in keep]
                    pr.setdefault("train", tr); pr.setdefault("test", te)
                    if not pr["test"]:
                        raise ValueError("need at least 2 episodes of some task to hold one out for testing")
                res = {k: v for k, v in d.items() if k != "episodes_by_task"}; res.update(train=len(pr.get("train", [])), test=len(pr.get("test", [])))
            elif name == "fewshot":
                steps = int(pr.get("fewshot_steps", 2000)); out = os.path.join(P.CKPT, "fewshot", pr["name"])
                if os.path.exists(out):
                    shutil.rmtree(out)
                cmd = [os.path.join(P.ENV, "bin", "lerobot-train"), f"--policy.path={pr.get('teacher', P.TEACHER)}", f"--dataset.repo_id=local/{pr['name']}",
                       f"--dataset.root={os.path.abspath(pr['dataset'])}", f"--dataset.episodes=[{eps_str(pr['train'])}]", "--batch_size=32", f"--steps={steps}",
                       "--log_freq=50", f"--save_freq={steps}", f"--output_dir={out}", "--policy.push_to_hub=false", "--wandb.enable=false", "--num_workers=8",
                       "--policy.device=cuda", "--policy.scheduler_warmup_steps=200", f"--policy.scheduler_decay_steps={steps}"]
                sh(st, name, cmd, total=steps)
                pr["fewshot_ckpt"] = sorted(__import__("glob").glob(os.path.join(out, "checkpoints", "0*", "pretrained_model")))[-1]; res = dict(ckpt=pr["fewshot_ckpt"])
            elif name == "eval_teacher":
                res = last_json(sh(st, name, PYM + ["nanovla.teacher", "eval", "--ckpt", pr["fewshot_ckpt"], "--root", os.path.abspath(pr["dataset"]),
                                                   "--episodes", eps_str(pr["test"]), "--out", os.path.join(J, "eval_teacher.json")]))
            elif name in ("label_train", "label_test", "label_dagger"):
                ck = pr.get("fewshot_ckpt") or pr["teacher_ckpt"]
                eps = {"label_train": pr.get("train"), "label_test": pr.get("test")}.get(name)
                if name == "label_dagger":
                    from nanovla import merge as M
                    eps = list(range(M.describe(os.path.abspath(pr["dataset"]))["episodes"]))
                stride, aug = (3, int(pr.get("aug", 2))) if name != "label_test" else (10, 0)
                if name == "label_dagger":
                    stride, aug = 3, 1
                out = os.path.join(J, f"{name}.npz")
                res = last_json(sh(st, name, PYM + ["nanovla.teacher", "label", "--ckpt", ck, "--root", os.path.abspath(pr["dataset"]), "--episodes", eps_str(eps),
                                                    "--stride", str(stride), "--aug", str(aug), "--bs", "16", "--out", out]))
                pr[name] = out
            elif name == "student":
                out = os.path.join(J, "student.pt")
                if st["kind"] == "distill":
                    train, test, target = pr["label_train"], pr["label_test"], "both"
                else:                                              # dagger: all earlier labels of that model + the new ones
                    rep = json.load(open(os.path.join(P.MODELS, pr["model"], "report.json")))
                    train = ",".join(rep["labels"] + [pr["label_dagger"]]); test, target = rep["label_test"], "teacher"
                    pr["labels_all"] = train.split(","); pr["label_test"] = test
                res = last_json(sh(st, name, PYM + ["nanovla.student", "--train", train, "--test", test, "--target", target, "--out", out,
                                                    "--steps", str(pr.get("student_steps", 12000)), "--tb", os.path.join(P.TB, f"student_{pr['name']}")],
                                   total=int(pr.get("student_steps", 12000))))
            elif name == "package":
                name_ = pr["name"] if st["kind"] == "distill" else pr["model"]; md = os.path.join(P.MODELS, name_); os.makedirs(md, exist_ok=True)
                for src, dst in (("student.onnx", "student.onnx"), ("student.meta.json", "meta.json"), ("student.pt", "student.pt")):
                    shutil.copy(os.path.join(J, src), os.path.join(md, dst))
                prev = {}
                if os.path.exists(os.path.join(md, "report.json")):
                    prev = json.load(open(os.path.join(md, "report.json")))
                ev_t = {}
                if os.path.exists(os.path.join(J, "eval_teacher.json")):
                    ev_t = json.load(open(os.path.join(J, "eval_teacher.json")))
                elif prev:
                    ev_t = prev.get("teacher", {})
                stu = json.load(open(os.path.join(J, "student.json")))
                try:                                                      # chunk spacing = the dataset's frame rate (sim data is 10 fps, real 30)
                    from nanovla import merge as M
                    mj = json.load(open(os.path.join(md, "meta.json"))); mj["fps"] = M.describe(os.path.abspath(pr["dataset"]))["fps"]; json.dump(mj, open(os.path.join(md, "meta.json"), "w"), indent=1)
                except Exception:
                    pass
                rep = dict(name=name_, created=time.strftime("%Y-%m-%d %H:%M"), dataset=os.path.abspath(pr["dataset"]) if st["kind"] == "distill" else prev.get("dataset"),
                           tasks=json.load(open(os.path.join(md, "meta.json")))["tasks"], teacher_ckpt=pr.get("fewshot_ckpt") or prev.get("teacher_ckpt"),
                           teacher=ev_t, student=stu, labels=pr.get("labels_all") or [pr["label_train"]], label_test=pr["label_test"],
                           train_episodes=pr.get("train") or prev.get("train_episodes"), test_episodes=pr.get("test") or prev.get("test_episodes"),
                           dagger_rounds=prev.get("dagger_rounds", 0) + (1 if st["kind"] == "dagger" else 0), job=jid)
                json.dump(rep, open(os.path.join(md, "report.json"), "w"), indent=1)
                shutil.make_archive(os.path.join(P.MODELS, name_), "zip", md)
                res = dict(model=md, zip=os.path.join(P.MODELS, name_ + ".zip")); st["result"] = rep
            # ---------------- teacher
            elif name == "select":
                out = os.path.join(J, "selected.json")
                if pr.get("repo_ids"):
                    json.dump(dict(selected=[dict(id=r) for r in pr["repo_ids"]]), open(out, "w"))
                else:
                    sh(st, name, PYM + ["nanovla.hub", "select", "--out", out], gpu=False)
                pr["repo_ids"] = [r["id"] for r in json.load(open(out))["selected"]][: int(pr.get("max_datasets", 5))]; res = dict(repo_ids=pr["repo_ids"])
            elif name == "prepare_import":
                from nanovla import merge as M
                src = pr["roots"][0]; dst = os.path.join(P.DATA, os.path.basename(src))
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst); M.prepare(dst); d = M.describe(dst); res = {k: v for k, v in d.items() if k != "episodes_by_task"}
            elif name == "download":
                sh(st, name, PYM + ["nanovla.hub", "download"] + pr["repo_ids"], gpu=False, env={"HF_HUB_OFFLINE": "0"})
                pr["roots"] = [os.path.join(P.HF, r.replace("/", "__")) for r in pr["repo_ids"]]
            elif name == "merge":
                out = os.path.join(P.DATA, pr.get("name", "teacher_mix"))
                res = last_json(sh(st, name, PYM + ["nanovla.merge", "--out", out] + pr["roots"], gpu=False)); pr["teacher_data"] = out
            elif name == "train":
                steps = int(pr.get("steps", 20000)); out = os.path.join(P.CKPT, pr.get("name", "teacher"))
                cmd = [os.path.join(P.ENV, "bin", "lerobot-train"), f"--policy.path={P.BASE}", "--dataset.repo_id=local/teacher", f"--dataset.root={pr['teacher_data']}",
                       "--batch_size=32", f"--steps={steps}", "--log_freq=100", "--save_freq=5000", f"--output_dir={out}", "--policy.push_to_hub=false",
                       "--wandb.enable=false", "--num_workers=8", "--policy.device=cuda"]
                sh(st, name, cmd, total=steps); res = dict(teacher=os.path.join(out, "checkpoints", "last", "pretrained_model"))
            s["status"] = "done"; s["ended"] = time.time(); s["result"] = res; st["params"] = pr; save(st)
        st["status"] = "done"; st["progress"] = 1.0; st["ended"] = time.time(); save(st)
    except Exception as e:
        import traceback
        st = st if isinstance(st, dict) else load(jid)
        for s in st["stages"]:
            if s["status"] == "running":
                s["status"] = "failed"
        st["status"] = "failed"; st["error"] = f"{e}"[-3000:]; st["ended"] = time.time(); save(st)
        traceback.print_exc()


def main():
    if sys.argv[1] == "new":
        print(json.dumps(new(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}), indent=1))
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "stop":
        print(json.dumps(stop(sys.argv[2])))
    elif sys.argv[1] == "list":
        for j in list_jobs():
            print(j["id"], j["kind"], j["status"], j.get("stage"), j.get("progress"))


if __name__ == "__main__":
    main()
