#!/usr/bin/env bash
# GCE startup script for the VQA A100 Spot run.  It deliberately never formats
# the persistent data disk and writes all mutable run state beneath RUN_ROOT.
set -Eeuo pipefail

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
PROJECT_ID="${PROJECT_ID:-$(metadata PROJECT_ID)}"
BUCKET="${BUCKET:-$(metadata BUCKET)}"
RUN_ID="${RUN_ID:-$(metadata RUN_ID)}"
GIT_COMMIT="${GIT_COMMIT:-$(metadata GIT_COMMIT)}"
NOTEBOOK_SHA256="${NOTEBOOK_SHA256:-$(metadata NOTEBOOK_SHA256)}"
REPO_URL="${REPO_URL:-$(metadata REPO_URL)}"
DATA_DISK="${DATA_DISK:-$(metadata DATA_DISK)}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/vqa-data}"
RUN_PREFIX="gs://${BUCKET}/runs/${RUN_ID}"
LOG_FILE="/var/log/vqa-${RUN_ID}.log"

exec > >(tee -a "$LOG_FILE" | logger -t vqa-startup -s 2>/dev/console) 2>&1
echo "$(date -Is) starting ${RUN_ID}"

sync_artifacts() {
  if [[ -d "${RUN_ROOT:-}/output" ]]; then
    gcloud storage rsync --recursive "$RUN_ROOT/output" "$RUN_PREFIX/output" || true
  fi
  [[ -f "$LOG_FILE" ]] && gcloud storage cp "$LOG_FILE" "$RUN_PREFIX/logs/startup.log" || true
}
trap sync_artifacts EXIT TERM INT

DEVICE="/dev/disk/by-id/google-${DATA_DISK}"
for _ in $(seq 1 10); do [[ -e "$DEVICE" ]] && break; sleep 2; done
# The first manually attached incarnation of this disk used Compute's default
# device name.  New workers set device-name explicitly; retain this exact,
# one-disk fallback solely for the current instance.
if [[ ! -e "$DEVICE" && -e /dev/disk/by-id/google-persistent-disk-1 ]]; then
  DEVICE="/dev/disk/by-id/google-persistent-disk-1"
  echo "Using legacy persistent-disk-1 attachment for $DATA_DISK"
fi
[[ -e "$DEVICE" ]] || { echo "Persistent disk $DATA_DISK did not appear"; exit 20; }
mkdir -p "$MOUNT_POINT"
FS_TYPE="$(blkid -o value -s TYPE "$DEVICE" || true)"
[[ -n "$FS_TYPE" ]] || { echo "Refusing to format unrecognized disk $DEVICE"; exit 21; }

# Inspect safely before allowing any writes.  The disk is known to contain
# prior VQA state and must never be initialized by this run.
mount -o ro "$DEVICE" "$MOUNT_POINT"
echo "Persistent-disk inventory (read-only):"
df -h "$MOUNT_POINT"
find "$MOUNT_POINT" -maxdepth 2 -mindepth 1 -printf '%y %s %p\n' | sort | head -200 || true
umount "$MOUNT_POINT"
mount "$DEVICE" "$MOUNT_POINT"

RUN_ROOT="$MOUNT_POINT/work/runs/$RUN_ID"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/output" "$MOUNT_POINT/cache/huggingface"
AVAILABLE_GB="$(df -BG --output=avail "$MOUNT_POINT" | tail -1 | tr -dc '0-9')"
if (( AVAILABLE_GB < 100 )); then
  echo "Only ${AVAILABLE_GB} GB free on the persistent disk; refusing unknown-file cleanup."
  exit 22
fi

# Locate an existing validated cache/data layout before downloading anything.
H5_EXISTING="$(find "$MOUNT_POINT" -maxdepth 4 -type f -name vqa_images_224.h5 -print -quit || true)"
if [[ -n "$H5_EXISTING" ]]; then
  DATA_ROOT="$(dirname "$H5_EXISTING")"
else
  DATA_ROOT="$MOUNT_POINT/data"
fi
mkdir -p "$DATA_ROOT/questions" "$DATA_ROOT/answers" "$DATA_ROOT/images" "$RUN_ROOT/staging"

stage_object() {
  local object="$1" destination="$2"
  [[ -f "$destination" ]] || gcloud storage cp "gs://${BUCKET}/${object}" "$destination"
}

if [[ ! -f "$DATA_ROOT/vqa_images_224.h5" ]]; then
  stage_object vqa_images_224.h5 "$DATA_ROOT/vqa_images_224.h5"
fi
if [[ ! -f "$DATA_ROOT/questions/v2_OpenEnded_mscoco_train2014_questions.json" ]]; then
  stage_object questions.tar.gz "$RUN_ROOT/staging/questions.tar.gz"
  tar -xzf "$RUN_ROOT/staging/questions.tar.gz" -C "$DATA_ROOT"
fi
if [[ ! -f "$DATA_ROOT/answers/v2_mscoco_train2014_annotations.json" ]]; then
  stage_object answers.tar.gz "$RUN_ROOT/staging/answers.tar.gz"
  tar -xzf "$RUN_ROOT/staging/answers.tar.gz" -C "$DATA_ROOT"
fi
if [[ ! -d "$DATA_ROOT/images/test2015" ]]; then
  stage_object test2015.zip "$RUN_ROOT/staging/test2015.zip"
  python3 -m zipfile -e "$RUN_ROOT/staging/test2015.zip" "$DATA_ROOT/images"
fi
if [[ ! -f "$DATA_ROOT/questions/v2_OpenEnded_mscoco_test2015_questions.json" ]]; then
  stage_object v2_Questions_Test_mscoco.zip "$RUN_ROOT/staging/test_questions.zip"
  python3 -m zipfile -e "$RUN_ROOT/staging/test_questions.zip" "$RUN_ROOT/staging/test_questions"
  TEST_QUESTIONS="$(find "$RUN_ROOT/staging/test_questions" -type f -name v2_OpenEnded_mscoco_test2015_questions.json -print -quit)"
  [[ -n "$TEST_QUESTIONS" ]] || { echo "Test question JSON absent from uploaded archive"; exit 23; }
  cp "$TEST_QUESTIONS" "$DATA_ROOT/questions/"
fi

python3 - "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
needed = [
    root / "vqa_images_224.h5",
    root / "questions/v2_OpenEnded_mscoco_train2014_questions.json",
    root / "questions/v2_OpenEnded_mscoco_val2014_questions.json",
    root / "questions/v2_OpenEnded_mscoco_test2015_questions.json",
    root / "answers/v2_mscoco_train2014_annotations.json",
    root / "answers/v2_mscoco_val2014_annotations.json",
]
missing = [str(p) for p in needed if not p.is_file()]
if missing:
    raise SystemExit(f"Missing required inputs: {missing}")
with (root / "vqa_images_224.h5").open("rb") as f:
    assert f.read(8) == b"\x89HDF\r\n\x1a\n", "invalid HDF5 signature"
with (root / "questions/v2_OpenEnded_mscoco_test2015_questions.json").open() as f:
    assert len(json.load(f)["questions"]) == 447_793
image_count = sum(1 for _ in (root / "images/test2015").glob("*.jpg"))
assert image_count == 81_434, image_count
print(f"Validated data root {root}; test images={image_count}")
PY

# Archives are reproducible from GCS and no longer needed after the checks.
rm -f "$RUN_ROOT/staging/questions.tar.gz" "$RUN_ROOT/staging/answers.tar.gz" \
      "$RUN_ROOT/staging/test2015.zip" "$RUN_ROOT/staging/test_questions.zip"

REPO_DIR="$RUN_ROOT/repo"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --quiet origin "$GIT_COMMIT"
git -C "$REPO_DIR" checkout --detach --quiet "$GIT_COMMIT"
actual_sha="$(sha256sum "$REPO_DIR/notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb" | awk '{print $1}')"
[[ "$actual_sha" == "$NOTEBOOK_SHA256" ]] || { echo "Notebook SHA mismatch"; exit 24; }

VENV="$RUN_ROOT/venv"
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  if ! dpkg-query -W -f='${Status}' python3-venv 2>/dev/null | grep -q 'install ok installed'; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
  fi
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REPO_DIR/requirements.txt" nbconvert

export VQA_PROJECT_ROOT="$REPO_DIR"
export VQA_DATA_DIR="$DATA_ROOT"
export VQA_OUTPUT_DIR="$RUN_ROOT/output"
export HF_HOME="$MOUNT_POINT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$VQA_OUTPUT_DIR"

sync_artifacts &
SYNC_PID=$!
(
  while kill -0 "$$" 2>/dev/null; do sleep 300; sync_artifacts; done
) &
LOOP_PID=$!
trap 'kill "$LOOP_PID" "$SYNC_PID" 2>/dev/null || true; sync_artifacts' EXIT TERM INT

"$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
set +e
"$VENV/bin/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output "$RUN_ROOT/output/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.executed.ipynb" \
  "$REPO_DIR/notebooks/train_evaluate_visualize_colab_unfrozen_5_gcp_s7.ipynb" \
  2>&1 | tee "$RUN_ROOT/output/notebook.log"
NOTEBOOK_STATUS=${PIPESTATUS[0]}
set -e
sync_artifacts
if (( NOTEBOOK_STATUS != 0 )); then
  echo "$NOTEBOOK_STATUS" > "$RUN_ROOT/output/exit_status.txt"
  gcloud storage cp "$RUN_ROOT/output/exit_status.txt" "$RUN_PREFIX/exit_status.txt"
  exit "$NOTEBOOK_STATUS"
fi

for p in "$RUN_ROOT/output/metrics" "$RUN_ROOT/output/figures" "$RUN_ROOT/output/predictions" "$RUN_ROOT/output/checkpoints"; do
  [[ -d "$p" ]] || { echo "Expected output missing: $p"; exit 25; }
done
find "$RUN_ROOT/output/predictions" -name '*test_predictions.json' -type f | grep -q . || {
  echo "No Test2015 prediction files were produced"; exit 26;
}
printf '0\n' > "$RUN_ROOT/output/exit_status.txt"
date -u +%FT%TZ > "$RUN_ROOT/output/finished_utc.txt"
sync_artifacts
gcloud storage cp "$RUN_ROOT/output/exit_status.txt" "$RUN_PREFIX/exit_status.txt"
gcloud storage cp "$RUN_ROOT/output/finished_utc.txt" "$RUN_PREFIX/finished_utc.txt"
gcloud storage cp /dev/null "$RUN_PREFIX/SUCCESS"
echo "$(date -Is) completed ${RUN_ID}"
