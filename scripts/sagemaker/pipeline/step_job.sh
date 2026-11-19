#!/bin/bash
# Runs inside a SageMaker Pipelines training step (AWS PyTorch training DLC).
# Code comes from the `code` channel (S3), not the Hub source repo.
#   JOB_KIND=train : TAG, SEED -> /opt/ml/model/{sft-adapter,dpo-adapter,train_log.json}
#   JOB_KIND=eval  : scores the adapter in the `adapter` channel (SFT and SFT+DPO)
#                    on held-out 256, hostile v1 (26), hostile v2 (30) -> /opt/ml/model/eval/
# The gated base model needs an HF read token; the execution role reads it from
# Secrets Manager (AmazonSageMaker-lexsi-hf). It is never echoed.
set -euo pipefail
C=/opt/ml/input/data/code
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_TOKEN="$(python3 -c "import boto3,json;print(json.loads(boto3.client('secretsmanager',region_name='us-east-1').get_secret_value(SecretId='AmazonSageMaker-lexsi-hf')['SecretString'])['HF_TOKEN'])")"
pip install -q --root-user-action=ignore torch==2.13.0 $C/schemasmith-0.2.0-py3-none-any.whl curatorkit==0.1.23 auditkit==1.1.6 aligntune==0.1.32 2>&1 | tail -3
pip uninstall -y -q torchaudio torchvision 2>/dev/null || true
# Same cuDNN fix as scripts/sagemaker/job.sh: prefer pip torch's bundled libs.
_torch_libs() { python3 -c "import glob,os,site,torch; ps=[os.path.join(os.path.dirname(torch.__file__),'lib')]; ps+=[p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp,'nvidia','*','lib'))]; print(':'.join(dict.fromkeys(p for p in ps if os.path.isdir(p))))"; }
export LD_LIBRARY_PATH="$(_torch_libs)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cudnn', torch.backends.cudnn.version())"
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
W=/tmp/w
python3 -m schemasmith.aws.run prep --work $W
if [ "$JOB_KIND" = train ]; then
  python3 -m schemasmith.aws.run train --work $W --tag "$TAG" --seed "$SEED"
  cp -r $W/$TAG/sft-adapter $W/$TAG/dpo-adapter $W/$TAG/train_log.json /opt/ml/model/
  cat $W/$TAG/train_log.json
else
  f=$(find /opt/ml/input/data/adapter -name model.tar.gz | head -1)
  [ -n "$f" ] || { echo "no model.tar.gz in adapter channel"; exit 1; }
  mkdir -p $W/cand && tar xzf "$f" -C $W/cand
  python3 -m schemasmith.aws.run eval --work $W --tag cand
  mkdir -p /opt/ml/model/eval && cp -r $W/cand/eval/. /opt/ml/model/eval/
  [ -f $W/cand/train_log.json ] && cp $W/cand/train_log.json /opt/ml/model/
fi
echo JOB_DONE
