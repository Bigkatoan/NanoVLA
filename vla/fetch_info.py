import json, os, time, concurrent.futures as cf
from huggingface_hub import hf_hub_download
rl=[r.split(":",1)[1] for r in json.load(open("data/meta/molmoact2/repo_list.json"))]
def get(r):
    out=f"data/info/{r.replace('/','__')}.json"
    if os.path.exists(out): return r,"cached"
    for k in range(6):
        try:
            p=hf_hub_download(r,"meta/info.json",repo_type="dataset")
            os.system(f"cp '{p}' '{out}'"); return r,"ok"
        except Exception as e:
            s=str(e)
            if "429" in s: time.sleep(30*(k+1)); continue
            return r,"ERR "+s[:80]
    return r,"ratelimited"
with cf.ThreadPoolExecutor(8) as ex:
    res=list(ex.map(get,rl))
import collections; print(collections.Counter(x[1][:12] for x in res))
json.dump(res,open("data/info/_status.json","w"))
