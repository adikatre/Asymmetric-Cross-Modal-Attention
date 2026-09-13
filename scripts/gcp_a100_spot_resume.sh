#!/usr/bin/env bash
# Resume a failed VQA campaign from its retained run root.  Run on the VM.
set -Eeuo pipefail

RUN_ROOT="${1:?run root required}"
BUCKET="${2:?bucket required}"
RUN_ID="${3:?run id required}"
BUDGET_HOURS="${4:?continuation budget required}"
DATA_ROOT="/mnt/vqa-data/data"
REPO_DIR="$RUN_ROOT/repo"
VENV="$RUN_ROOT/venv"
OUTPUT_ROOT="$RUN_ROOT/output"
CONTINUATION_DIR="$RUN_ROOT/repair"
PREFIX="gs://${BUCKET}/runs/${RUN_ID}"
NOTEBOOK="$REPO_DIR/notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb"
CONTINUATION_NOTEBOOK="$CONTINUATION_DIR/train_evaluate_visualize_colab_unfrozen_5_gcp_s7_budget${BUDGET_HOURS}.ipynb"
LOG="$CONTINUATION_DIR/resume.log"

mkdir -p "$CONTINUATION_DIR"
exec > >(tee -a "$LOG") 2>&1

sync_artifacts() {
  gcloud storage rsync --recursive "$OUTPUT_ROOT" "$PREFIX/output" || true
  gcloud storage rsync --recursive "$CONTINUATION_DIR" "$PREFIX/continuation" || true
}
trap sync_artifacts EXIT TERM INT

"$VENV/bin/python" - "$NOTEBOOK" "$CONTINUATION_NOTEBOOK" "$BUDGET_HOURS" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
budget = float(sys.argv[3])
notebook = json.loads(source.read_text())
replacements = 0
for cell in notebook["cells"]:
    lines = cell.get("source", [])
    updated = []
    for line in lines:
        if line.strip().startswith("GPU_BUDGET_HOURS"):
            updated.append(f"GPU_BUDGET_HOURS    = {budget}\n")
            replacements += 1
        else:
            updated.append(line)
    cell["source"] = updated
if replacements != 1:
    raise RuntimeError(f"expected one GPU_BUDGET_HOURS setting, found {replacements}")
notebook.setdefault("metadata", {}).setdefault("vqa_continuation", {})
notebook["metadata"]["vqa_continuation"].update({
    "reason": "resume after validated Test2015 repair and complete seed 123",
    "gpu_budget_hours": budget,
    "source_notebook_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
})
destination.write_text(json.dumps(notebook, indent=1))
PY

"$VENV/bin/python" - "$CONTINUATION_DIR/continuation_manifest.json" "$BUDGET_HOURS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "repair Test2015 staging and complete previously budget-blocked seed 123",
    "gpu_budget_hours": float(sys.argv[2]),
}, indent=2))
PY

export VQA_PROJECT_ROOT="$REPO_DIR"
export VQA_DATA_DIR="$DATA_ROOT"
export VQA_OUTPUT_DIR="$OUTPUT_ROOT"
export HF_HOME="/mnt/vqa-data/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

(
  while true; do
    sleep 300
    sync_artifacts
  done
) &
SYNC_LOOP_PID=$!
trap 'kill "$SYNC_LOOP_PID" 2>/dev/null || true; sync_artifacts' EXIT TERM INT

set +e
"$VENV/bin/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output "$OUTPUT_ROOT/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.continuation.executed.ipynb" \
  "$CONTINUATION_NOTEBOOK" 2>&1 | tee "$CONTINUATION_DIR/notebook.log"
NOTEBOOK_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$NOTEBOOK_STATUS" > "$CONTINUATION_DIR/exit_status.txt"
sync_artifacts
(( NOTEBOOK_STATUS == 0 )) || exit "$NOTEBOOK_STATUS"

"$VENV/bin/python" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
predictions = sorted((output / "predictions").glob("*_test_predictions.json"))
if len(predictions) != 2:
    raise RuntimeError(f"expected two Test2015 prediction JSONs, found {len(predictions)}")
for path in predictions:
    records = json.loads(path.read_text())
    if len(records) != 447_793:
        raise RuntimeError(f"invalid prediction count in {path}: {len(records)}")
reports = sorted((output / "metrics").glob("*_final_training_report.json"))
if not reports:
    raise RuntimeError("final training report missing")
report = json.loads(reports[-1].read_text())
if report.get("seeds_completed") != [7, 42, 123]:
    raise RuntimeError(f"incomplete campaign: {report.get('seeds_completed')}")
PY

printf '0\n' > "$OUTPUT_ROOT/exit_status.txt"
date -u +%FT%TZ > "$OUTPUT_ROOT/finished_utc.txt"
gcloud storage cp /dev/null "$PREFIX/SUCCESS"
sync_artifacts
