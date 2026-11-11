#!/bin/bash
# Runs inside a SageMaker training job (AWS PyTorch DLC). Fetched from the
# private source repo by the bootstrap in the job's environment.
# JOB_KIND=train: TAG, SEED, NO_ADV=1 optional. Writes adapters to /opt/ml/model.
# JOB_KIND=eval: scores base, v1 SFT/DPO, and every adapter channel under
# /opt/ml/input/data on the same GPU. Results go to /opt/ml/model and, as
# base64 chunks, to the job log so the box can read them without S3.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "from huggingface_hub import hf_hub_download as d; d('nightmoose/schemasmith-src', 'aws/schemasmith-0.2.0-py3-none-any.whl', local_dir='/tmp/j')"
pip install -q --root-user-action=ignore torch==2.13.0 /tmp/j/aws/schemasmith-0.2.0-py3-none-any.whl curatorkit==0.1.23 auditkit==1.1.6 aligntune==0.1.32 2>&1 | tail -3
pip uninstall -y -q torchaudio torchvision 2>/dev/null || true  # DLC builds do not match torch 2.13
# Pip torch 2.13 bundles cuDNN 9.20; the DLC image has 9.10 on LD_LIBRARY_PATH.
# The v1data ablation (lexsi-v1data-s1-1110-1639) finished SFT, then failed at
# DPO trainer init with that mismatch. Prefer torch's libs, and fall back to
# torch/nvidia paths only if a mixed path still fails the preflight.
_torch_libs() { python3 -c "import glob,os,site,torch; ps=[os.path.join(os.path.dirname(torch.__file__),'lib')]; ps+=[p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp,'nvidia','*','lib'))]; print(':'.join(dict.fromkeys(p for p in ps if os.path.isdir(p))))"; }
export LD_LIBRARY_PATH="$(_torch_libs)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cudnn', torch.backends.cudnn.version())"; then
  echo "cuDNN still mismatched; retrying with torch/nvidia libs only"
  export LD_LIBRARY_PATH="$(_torch_libs)"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cudnn', torch.backends.cudnn.version())"
fi
pip list 2>/dev/null | grep -iE '^(torch|transformers|trl|peft|bitsandbytes|aligntune) '
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
W=/tmp/w
python3 -m schemasmith.aws.run prep --work $W
emit() {  # emit NAME DIR: tar.gz + base64 to the log in 8000-char lines
  tar czf /tmp/$1.tgz -C "$2" . && base64 -w 8000 /tmp/$1.tgz | awk -v n="$1" '{print "B64CHUNK " n " " NR " " $0}'
  echo "B64END $1 $(sha256sum /tmp/$1.tgz | cut -c1-64)"
}
if [ "$JOB_KIND" = train ]; then
  extra=""; [ "${NO_ADV:-0}" = 1 ] && extra="--no-adv"
  python3 -m schemasmith.aws.run train --work $W --tag "$TAG" --seed "$SEED" $extra
  cp -r $W/$TAG/sft-adapter $W/$TAG/dpo-adapter $W/$TAG/train_log.json /opt/ml/model/
  cat $W/$TAG/train_log.json
else
  python3 -m schemasmith.aws.run eval --work $W --tag v1 --base \
    --sft nightmoose/schemasmith-tiny-aya-sft --dpo nightmoose/schemasmith-tiny-aya-dpo
  for ch in /opt/ml/input/data/*; do
    # SageMaker may add a *-manifest channel when the S3 uri is a single object.
    [ -f "$ch/model.tar.gz" ] || { echo "skip $(basename $ch) (no model.tar.gz)"; continue; }
    tag=$(basename $ch); mkdir -p $W/$tag
    tar xzf $ch/model.tar.gz -C $W/$tag
    python3 -m schemasmith.aws.run eval --work $W --tag $tag
  done
  mkdir -p /tmp/out
  for d in $W/*/eval; do t=$(basename $(dirname $d)); mkdir -p /tmp/out/$t; cp -r $d /tmp/out/$t/; done
  for f in $W/*/train_log.json; do [ -f "$f" ] && cp $f /tmp/out/$(basename $(dirname $f))/; done
  cp -r /tmp/out/* /opt/ml/model/
  emit evals /tmp/out
fi
echo JOB_DONE
