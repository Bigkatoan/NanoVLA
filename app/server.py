"""NanoVLA Studio: the web UI of the pipeline (collect -> choose dataset -> distill -> model).

    envs/lerobot/bin/python -m app.server --port 8092          # then open http://localhost:8092
"""
import argparse, asyncio, glob, json, os, subprocess, time
from aiohttp import web
from nanovla import paths as P, pipeline as PL, merge as M, preview as V

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_desc_cache = {}


def describe_cached(root):
    key = (root, os.path.getmtime(os.path.join(root, "meta", "info.json")))
    if key not in _desc_cache:
        try:
            d = M.describe(root)
        except Exception as e:
            d = dict(name=os.path.basename(root), root=root, error=str(e))
        _desc_cache[key] = d
    return _desc_cache[key]


def datasets():
    out = []
    for info in sorted(glob.glob(os.path.join(P.DATA, "*", "meta", "info.json"))):
        d = dict(describe_cached(os.path.dirname(os.path.dirname(info)))); d.pop("episodes_by_task", None); d["source"] = V.source(d["root"])
        d["root"] = os.path.relpath(d["root"], P.ROOT); out.append(d)
    return out


def models():
    out = []
    for rep in sorted(glob.glob(os.path.join(P.MODELS, "*", "report.json"))):
        try:
            r = json.load(open(rep)); r["has_zip"] = os.path.exists(os.path.join(P.MODELS, r["name"] + ".zip")); out.append(r)
        except Exception:
            pass
    return sorted(out, key=lambda r: r.get("created", ""), reverse=True)


def gpu_busy():
    lock = os.path.join(P.ROOT, ".gpu.lock")
    if not os.path.exists(lock):
        return False
    return subprocess.run(["flock", "-n", lock, "true"]).returncode != 0


def devices():
    return dict(serial=sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")), cameras=sorted(glob.glob("/dev/video*")))


def tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n)); return f.read().decode(errors="ignore")
    except FileNotFoundError:
        return ""


async def index(_):
    return web.FileResponse(os.path.join(STATIC, "index.html"))


async def api_status(_):
    return web.json_response(dict(teacher=os.path.exists(P.TEACHER), teacher_path=os.path.relpath(P.TEACHER, P.ROOT), gpu_busy=gpu_busy(), devices=devices(),
                                  running=[j["id"] for j in PL.list_jobs() if j["status"] in ("running", "queued")]))


async def api_datasets(_):
    return web.json_response(datasets())


async def api_dataset(req):
    root = os.path.join(P.DATA, req.match_info["name"])
    if not os.path.exists(os.path.join(root, "meta", "info.json")):
        raise web.HTTPNotFound()
    d = dict(describe_cached(root)); tr, te = M.default_split(d) if d.get("episodes_by_task") else ([], [])
    d["default_train"], d["default_test"] = tr, te; d["root"] = os.path.relpath(root, P.ROOT)
    return web.json_response(d)


def _root(req):
    root = os.path.join(P.DATA, os.path.basename(req.match_info["name"]))
    if not os.path.exists(os.path.join(root, "meta", "info.json")):
        raise web.HTTPNotFound()
    return root


async def api_ds_summary(req):
    return web.json_response(await asyncio.get_event_loop().run_in_executor(None, V.summary, _root(req)))


async def api_ds_episodes(req):
    q = req.query; return web.json_response(await asyncio.get_event_loop().run_in_executor(None, V.episode_list, _root(req), int(q.get("offset", 0)), int(q.get("limit", 60))))


async def api_ds_frame(req):
    q = req.query
    try:
        b = await asyncio.get_event_loop().run_in_executor(None, V.frame, _root(req), int(q["ep"]), q.get("cam", "camera2"), float(q.get("f", 0.5)), int(q.get("w", 320)))
    except (KeyError, ValueError, IndexError):
        raise web.HTTPNotFound()
    return web.Response(body=b, content_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


async def api_ds_traj(req):
    return web.json_response(await asyncio.get_event_loop().run_in_executor(None, V.trajectory, _root(req), int(req.query["ep"])))


async def api_models(_):
    return web.json_response(models())


async def api_jobs(_):
    return web.json_response([{k: v for k, v in j.items() if k != "params"} | dict(name=j["params"].get("name") or j["params"].get("model") or "") for j in PL.list_jobs()[:40]])


async def api_job(req):
    jid = req.match_info["id"]
    try:
        st = PL.load(jid)
    except FileNotFoundError:
        raise web.HTTPNotFound()
    st["log"] = tail(os.path.join(PL.jdir(jid), f"{st.get('stage')}.log")) if st.get("stage") else ""
    return web.json_response(st)


async def api_new_job(req):
    body = await req.json(); kind = body.get("kind"); params = body.get("params", {})
    if kind in ("distill", "dagger") and not os.path.exists(P.TEACHER) and not params.get("teacher"):
        return web.json_response(dict(error="Chưa có thầy (teacher). Hãy train thầy trước ở tab Thầy."), status=400)
    if kind == "distill":
        params.setdefault("name", os.path.basename(params.get("dataset", "model").rstrip("/")))
        params["dataset"] = os.path.join(P.ROOT, params["dataset"]) if not os.path.isabs(params["dataset"]) else params["dataset"]
    if kind == "dagger":
        params["dataset"] = os.path.join(P.ROOT, params["dataset"]) if not os.path.isabs(params["dataset"]) else params["dataset"]
        rep = json.load(open(os.path.join(P.MODELS, params["model"], "report.json"))); params["teacher_ckpt"] = rep["teacher_ckpt"]; params["name"] = params["model"]
    try:
        st = await asyncio.get_event_loop().run_in_executor(None, PL.new, kind, params)
    except Exception as e:
        return web.json_response(dict(error=str(e)), status=400)
    return web.json_response(st)


async def api_retry(req):
    st = PL.load(req.match_info["id"]); pr = {k: v for k, v in st["params"].items() if k in ("dataset", "name", "model", "task", "episodes", "episode_s", "reset_s", "robot_port",
        "robot_id", "leader_port", "leader_id", "wrist_cam", "front_cam", "fewshot_steps", "student_steps", "aug", "repo_ids", "max_datasets", "steps", "seconds", "record", "dry_run", "teacher_ckpt")}
    return web.json_response(await asyncio.get_event_loop().run_in_executor(None, PL.new, st["kind"], pr))


async def api_stop(req):
    return web.json_response(PL.stop(req.match_info["id"]))


async def model_zip(req):
    p = os.path.join(P.MODELS, req.match_info["name"] + ".zip")
    if not os.path.exists(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Content-Disposition": f'attachment; filename="{req.match_info["name"]}.zip"'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8092); ap.add_argument("--host", default="127.0.0.1"); a = ap.parse_args()
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/status", api_status), web.get("/api/datasets", api_datasets), web.get("/api/datasets/{name}", api_dataset), web.get("/api/datasets/{name}/summary", api_ds_summary),
                    web.get("/api/datasets/{name}/episodes", api_ds_episodes), web.get("/api/datasets/{name}/frame", api_ds_frame), web.get("/api/datasets/{name}/trajectory", api_ds_traj),
                    web.get("/api/models", api_models), web.get("/api/jobs", api_jobs), web.get("/api/jobs/{id}", api_job), web.post("/api/jobs", api_new_job),
                    web.post("/api/jobs/{id}/stop", api_stop), web.post("/api/jobs/{id}/retry", api_retry), web.get("/models/{name}.zip", model_zip), web.static("/static", STATIC)])
    print(json.dumps(dict(event="ready", url=f"http://{a.host}:{a.port}")), flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
