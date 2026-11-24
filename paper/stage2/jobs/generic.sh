set -euo pipefail
: "${SRC_REPO:?set SRC_REPO=<your-hf-user>/schemasmith-src}"
nvidia-smi --query-gpu=name,memory.total --format=csv
pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /tmp/constraints.txt || true
pip install -q --root-user-action=ignore huggingface_hub
# SRC_REPO: a Hub model repo you own that holds your own wheel at aws/schemasmith-0.2.0-py3-none-any.whl
# (build it with `python -m pip wheel --no-deps -w dist .` from the repo root). nightmoose/schemasmith-src is private.
python -c "import os; from huggingface_hub import hf_hub_download as d; d(os.environ['SRC_REPO'],'aws/schemasmith-0.2.0-py3-none-any.whl',local_dir='/tmp/j')"
pip install -q --root-user-action=ignore -c /tmp/constraints.txt /tmp/j/aws/schemasmith-0.2.0-py3-none-any.whl curatorkit==0.1.23 auditkit==1.1.6 aligntune==0.1.32 2>&1 | tail -3
pip list 2>/dev/null | grep -iE '^(torch|transformers|peft) ' || true
echo "$RET_PY_B64" | base64 -d > /tmp/run.py
python /tmp/run.py
tar czf /tmp/out.tgz -C /tmp/out . && base64 -w 8000 /tmp/out.tgz | awk '{print "B64CHUNK " NR " " $0}'
echo "B64END $(sha256sum /tmp/out.tgz | cut -c1-64)"
echo JOB_DONE
