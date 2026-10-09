set -euo pipefail
mkdir -p /tmp/agentenv /tmp/poc-artifacts /tmp/poc-logs
cd /tmp/agentenv
tar -xzf /tmp/source.tar.gz
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-buildx
build_one() {
  docker build -f "$1" -t "$3" "$2" > "/tmp/poc-logs/$3-build.log" 2>&1
  docker save "$3" | gzip -1 > "/tmp/poc-artifacts/$3.tar.gz"
  docker image inspect "$3" --format '{{.Id}}' > "/tmp/poc-artifacts/$3.image-id"
  echo "BUILT $3"
}
build_one src/agent_env/env/gateway/Dockerfile src/agent_env/env poc-gateway
build_one src/agent_env/env/envs/service_db/Dockerfile src/agent_env/env/envs/service_db poc-db
build_one src/agent_env/env/envs/service_db/Dockerfile.db-web src/agent_env/env/envs/service_db poc-db-web
build_one src/agent_env/env/envs/service_db/Dockerfile.db-mcp src/agent_env/env/envs/service_db poc-db-mcp
mkdir -p /tmp/items-context /tmp/agent-context
cp -R packages/agentenv-protocol/src/agentenv_protocol /tmp/items-context/agentenv_protocol
cp tst/data/agentenv_mcp/Dockerfile tst/data/agentenv_mcp/server.py tst/data/agentenv_mcp/seed.json /tmp/items-context/
build_one /tmp/items-context/Dockerfile /tmp/items-context poc-items
cp -R packages/agentenv-protocol /tmp/agent-context/agentenv-protocol
cp tst/data/a2a_agent/Dockerfile /tmp/agent-context/Dockerfile
cp /tmp/fixture_agent.py /tmp/agent-context/agent.py
python3 - <<'PYDOCKER'
from pathlib import Path
p=Path('/tmp/agent-context/Dockerfile')
s=p.read_text()
s=s.replace('USER agent', 'RUN pip install --no-cache-dir mcp==1.28.1\nRUN python -c "from mcp.client.streamable_http import streamable_http_client; import agent"\nUSER agent')
p.write_text(s)
PYDOCKER
build_one /tmp/agent-context/Dockerfile /tmp/agent-context poc-agent
python3 - <<'PY'
from pathlib import Path
import json,hashlib
p=Path('/tmp/poc-artifacts')
manifest={f.stem.removesuffix('.tar'):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'image_id':f.with_name(f.stem.removesuffix('.tar')+'.image-id').read_text().strip()} for f in p.glob('*.tar.gz')}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2))
assert len(manifest)==6
print('ALL SIX IMAGE BUILDS PASSED')
PY
