#!/usr/bin/env bash
# Recompute every Stage 2 statistic from saved predictions. No GPU, no model download.
# Run from this directory after `pip install -e "../..[hf]"` (plus matplotlib for figures).
set -euo pipefail
cd "$(dirname "$0")"
LOGS=${LOGS:-logs}  # Stage 2 logs from the public dataset nightmoose/schemasmith-eval-logs (stage2/)
EVALS=evals    # v1/v2 SageMaker predictions, already in this repo under results/aws/sagemaker/
OUT=out
mkdir -p "$OUT"

if [ ! -d "$LOGS/stage2" ]; then
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("nightmoose/schemasmith-eval-logs", repo_type="dataset",
                  allow_patterns=["stage2/*"], local_dir="$LOGS")
PY
fi

SM=../../results/aws/sagemaker
rm -rf "$EVALS"; mkdir -p "$EVALS"
ln -s "../$SM/eval-v2/v1" "$EVALS/v1"
ln -s "../$SM/eval-v2/v2-s2" "$EVALS/v2-s2"
ln -s "../$SM/eval-v2/v2-s3" "$EVALS/v2-s3"
ln -s "../$SM/eval-v2-s1" "$EVALS/s1"

S=$LOGS/stage2
python stage2_stats.py --evals "$EVALS" --out "$OUT/stats.json"
python ablation_stats.py --evals "$EVALS" --abl "$S/ablation" \
  --ret "$S/retention/v1/retention.json" --out "$OUT/ablation_retention.json"
python mcnemar_stats.py --runs v1="$EVALS/v1/eval" v2-s1="$EVALS/s1/v2-s1/eval" \
  v2-s2="$EVALS/v2-s2/eval" v2-s3="$EVALS/v2-s3/eval" \
  v1data-s1="$S/ablation/v1data-s1/eval" v1data-s2="$S/ablation/v1data-s2/eval" \
  v1data-s3="$S/ablation/v1data-s3/eval" --out "$OUT/mcnemar.json"
python round2_stats.py --ret-v1 "$S/retention/v1/retention.json" \
  --ret-v2 "$S/retention/v2-s1/retention.json" "$S/retention/v2-s3/retention.json" \
  --safety "$S/xstest/outputs.json" --out "$OUT/round2.json"
if python -c "import matplotlib" 2>/dev/null; then
  python make_figures.py --stats "$OUT/stats.json" --ablation "$OUT/ablation_retention.json" --out "$OUT/figures"
fi
for f in stats ablation_retention mcnemar round2; do
  if cmp -s "$OUT/$f.json" "$S/stats/$f.json"; then echo "$f.json: identical to the published file"
  else echo "$f.json: DIFFERS from $S/stats/$f.json"; fi
done
