import json
import subprocess
import time
from pathlib import Path

from app import Clients

NODE = ("require('https').get(process.argv[1], r => { console.log(r.statusCode); r.resume(); })"
        ".on('error', e => { console.error(e.message); process.exit(1); })")
FETCH = str(Path(__file__).with_name("Fetch.java"))

url = Clients().settings.url
results = {}
for name, argv in (("node", ["node", "-e", NODE, url]), ("java", ["java", FETCH, url]),
                   ("copied java", ["/opt/copied-jdk/bin/java", FETCH, url]),
                   ("curl", ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", url])):
    done = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    said = (done.stdout if done.returncode == 0 else done.stderr).strip().splitlines()
    results[name] = said[-1] if said else f"exited {done.returncode}"
Path.home().joinpath(".tls.json").write_text(json.dumps(results))
while True:
    time.sleep(3600)
