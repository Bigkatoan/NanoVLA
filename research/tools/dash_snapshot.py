"""Static snapshot of the live dashboard (data inlined) for publishing as an artifact."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard as D
d = D.data()
page = D.PAGE.replace("let d;try{d=await (await fetch('data.json')).json()}catch(e){return}", "const d=" + json.dumps(d) + ";")
page = page.replace("tick();setInterval(tick,5000);", "tick();")
page = page.replace("<title>NanoVLA Training</title>", "<title>NanoVLA Training</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">")
page = page.split("<body>", 1)[1].replace("</body></html>", "")
head = D.PAGE.split("<style>", 1)[1].split("</style>", 1)[0]
open(sys.argv[1], "w").write("<title>NanoVLA Training</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n<style>" + head.replace(
    "@media (prefers-color-scheme:dark){:root{", "@media (prefers-color-scheme:dark){:root:not([data-theme=\"light\"]){") + 
    ":root[data-theme=\"dark\"]{--bg:#0d0f13;--fg:#e7e9ee;--mut:#9aa1b0;--card:#161920;--line:#2a2f39;--a:#ff922b;--b:#4dabf7;--c:#69db7c;--d:#da77f2;--e:#adb5bd;color-scheme:dark}</style>\n" + page)
print("snapshot", d["t"])
