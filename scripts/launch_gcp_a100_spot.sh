#!/usr/bin/env bash
# Creates the single Spot A100 worker.  Re-run with the same RUN_ID after a
# preemption; the persistent data disk and output prefix are retained.
set -Eeuo pipefail

PROJECT_ID="training-sdsu"
ZONE="us-central1-a"
BUCKET="visualqa_data"
DATA_DISK="vqa-s7-a100-data"
INSTANCE="vqa-a100-spot"
GIT_COMMIT="18a531626832cf6cb3582718ac09cacd430e80cc"
NOTEBOOK_SHA256="5d6752dfffa23dd8a47e3eaebaf29bcfe755270cacc3dfdbdca981aaaa2fc006"
REPO_URL="https://github.com/adikatre/Asymmetric-Cross-Modal-Attention"
RUN_ID="${RUN_ID:-unfrozen-gcp-s7-${GIT_COMMIT:0:7}-$(date -u +%Y%m%dT%H%M%SZ)}"

if gcloud compute instances describe "$INSTANCE" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo "Instance $INSTANCE already exists; refusing to create a second worker."
  exit 1
fi

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=50GB --boot-disk-type=pd-balanced \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --disk=name="$DATA_DISK",device-name="$DATA_DISK",mode=rw,boot=no,auto-delete=no \
  --scopes=cloud-platform \
  --metadata=PROJECT_ID="$PROJECT_ID",BUCKET="$BUCKET",RUN_ID="$RUN_ID",GIT_COMMIT="$GIT_COMMIT",NOTEBOOK_SHA256="$NOTEBOOK_SHA256",REPO_URL="$REPO_URL",DATA_DISK="$DATA_DISK" \
  --metadata-from-file=startup-script=scripts/gcp_a100_spot_startup.sh

printf 'Instance: %s\nRun ID: %s\nPrefix: gs://%s/runs/%s\n' "$INSTANCE" "$RUN_ID" "$BUCKET" "$RUN_ID"
