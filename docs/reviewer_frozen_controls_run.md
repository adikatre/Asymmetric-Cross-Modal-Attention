# Frozen VQA shared-control experiment

This is the first experiment requested on November 17, 2026. It addresses
Reviewer 1 item 1 and Reviewer 2 minor item e. Results are pending; the launch
controller's operational state is under `runs/colab/` (ignored by Git).

The reviewer notes that **"block applications and aggregation also differ"**
and requests **"a conventional single-block shared baseline and an alternative
parameter-matched control."** The experiment compares four architectures:

| Architecture | Fusion parameters | Block calls | Directional aggregation |
| --- | ---: | ---: | --- |
| Conventional shared | 3,153,408 | 2 | One block shared across directions |
| Sequential shared | 6,306,816 | 4 | Two shared blocks composed in each direction |
| Existing asymmetric | 6,306,816 | 2 | One independent block per direction |
| Existing parallel shared | 6,306,816 | 4 | Average two shared-block outputs per direction |

For the sequential control, each direction computes
`B1(B0(query, original_other), original_other)`. The opposite modality supplies
the original keys/values at both depths. This adds query depth and changes
aggregation; it does not isolate directional specialization by itself.

The canonical frozen notebook supplies the unchanged CLIP ViT-B/32 and
RoBERTa-base encoders, 224px preprocessing, 20-token question limit, top-3,000
answer vocabulary, soft BCE targets, AdamW, AMP, batch size 2,048, twelve-epoch
horizon, paired seeds 7/42/123, and seed/epoch data order. Checkpoint selection
uses the same fixed 10,000-example validation subset, with full validation
after selection. Four architectures times three seeds means twelve runs in
one frozen-control experiment. Fine-tuning and other reviewer experiments
are outside this launch.

## Data recovery and current locations

The original executed notebook used:

```text
/mnt/vqa-data/data/vqa_images_224.h5
/mnt/vqa-data/data/vqa_frozen_features_dcfce3a81574.h5
```

That mount is absent here. Searches found neither cache in connected Drive.
The historical GCS location `gs://prashant-vqa/data/` could not be inspected
with the saved GCP identity, which reports a deleted account. Colab's separate
OAuth identity works.

Official VQA/COCO archives are being downloaded to the workspace's `data/zip/`.
Question and answer JSON files must match the original experiment's SHA-256
hashes. The local CPU build creates:

```text
/home/adikatre/Asymmetric-Cross-Modal-Attention/data/vqa_images_224.h5
```

It stores 123,287 images in the original sorted train-then-validation order.
JPEGs are read directly from archives to avoid extracting another image tree.
The CPU Pillow resize/crop implementation passed pixel equality against
torchvision across odd/even/extreme image shapes and JPEG round trips.

The local image cache is complete and verified. The controller can stage it
and the metadata in `/content/acma/data/` on one fresh L4. On the current retry,
measured WSL upload throughput was about 4 MB/s, so the VM instead rebuilds
its image cache from the same hash-checked public archives with the verified
Pillow preprocessing path. The complete WSL copy remains preserved. The notebook reconstructs the frozen
feature HDF5 there. Actual cache bytes, encoder state hashes/revisions,
software versions and source fingerprints are bound into run provenance.
Both existing architectures are rerun on the rebuilt cache to support a
comparison within the same environment.

The VM reconstruction was verified against the WSL copy: both completed HDF5
files contain 18,567,190,536 bytes with SHA-256
`19f9d4b3dc7a8d4f745e8296c5b2c3ae658a612816274211833eec8f37734cc9`.
They match byte for byte. The receipt is `image_cache_equivalence.json` in the
current attempt directory.

The first full-data preflight exposed slow HDF5 fancy selections and projected
58.45 hours. A matched 2,048-example read benchmark produced identical full
batch hashes with ordinary row reads, which were 2.5--10 times faster. The
reader now caches the four dataset handles per worker and uses ordinary row
reads. A separate on-VM test verifies exact shuffled batch values and dtypes
against the canonical implementation. Redundant raw target tensors are released
after retaining question/image IDs. Batch size, data order, model, optimizer,
seeds, horizon and selection stay the same. No epoch of the slow-reader attempt
completed; all arms restart under a newly fingerprinted reader configuration.
Current records are under
`runs/colab/acma-r1-shared-frozen-l4-20261117-rowio/` on the same L4.

## Hardware verification and credit policy

An initial A100 was terminated before training. The L4 verification runtime
was also terminated after testing, while local preprocessing runs. At that
point Colab reported zero active assignments and zero units/hour.

All four heads passed real forward/backward/optimizer, validation and
checkpoint-reload checks on NVIDIA L4 with the intended AMP batch of 2,048.
Both two-block shared controls peaked at 7.78 GiB allocated and 8.43 GiB
reserved. Frozen encoder extraction passed at batch 512, peaking at 1.83 GiB
allocated and 2.17 GiB reserved. These are memory preflights on synthetic
inputs, not accuracy results or steady-state latency measurements.

The tested L4 rate was 1.94 compute units/hour. It has about 22.0 GiB visible
VRAM, 53 GiB system RAM, and 236 GB free disk. Torch 2.14.0, torchvision 0.29.0
and Transformers 5.17.0 are pinned. No purchase is authorized or attempted.

The first unattended launch failed before training: the installed Colab CLI
lost its initial Jupyter connection and leaked a local WebSocket thread. The
controller eventually timed out and terminated that L4. The retry uses SSH
for short setup commands, bounds the runtime probe to two minutes, kills the
local transport process group on timeout, and requires pipeline telemetry
within two minutes of training submission. It permits one retry only after
confirming no remote job has started. Current operational records are under
`runs/colab/acma-r1-shared-frozen-l4-20261117-attempt2/`.

A fresh GPU is allocated only after local preparation completes. The
controller verifies the actual device, checks staging hashes, and launches a
fail-fast remote pipeline. Failures retain local data and diagnostic records;
owned idle/failed runtimes are released. The monitor polls file telemetry
every minute, mirrors immutable checkpoint chunks with hashes, and releases
the owned VM after verified completion. Network failures alone do not prove
the training process has stopped.

An independent worker also attempts to stream the completed frozen feature
cache back to local `data/`. Its status and final size/hash receipt live under
the campaign's `cache-mirror/` directory. A pending or failed transfer is not a
durable cache; checkpoint export proceeds independently.

The user's five-hour Codex allowance is separate from Colab units. At less
than 5% remaining, preserve useful running computation and terminate an idle
VM. Local preprocessing requires no Colab runtime. The independent controller
and monitor do not require continued model sampling, but depend on WSL and
the Windows host remaining awake and online.

## Files and verification

- `tools/colab/prepare_vqa.py`: resumable public downloads, original metadata
  hashes, CPU HDF5 construction and atomic preparation status.
- `tools/colab/l4_preflight.py`: unchanged-batch GPU memory/optimizer/checkpoint
  and real frozen-encoder inference checks.
- `tools/colab/reviewer_controls.py`: reuses canonical notebook code, adds the
  two shared controls, profiles latency/memory/operator coverage, and publishes
  immutable checkpoints and result artifacts.
- `tools/colab/pipeline.py` and `bootstrap.py`: verified source staging,
  synthetic checkpoint/resume check, data check and frozen experiment.
- `tools/colab/launch.py`: local preparation through owned-runtime launch.
- `tools/colab/monitor.py`: independent file polling, verified artifact export
  and owned-runtime cleanup.
- `tools/colab/mirror_feature_cache.py`: optional independent full-cache copy
  with a size/hash receipt, without delaying checkpoint monitoring.

The FLOP report explicitly lists operator coverage. `torch.profiler` does not
provide a complete end-to-end FLOP count; its supported-operator total must
not be described as one. Historical 40 GB A100 cost measurements remain
separate from the new L4 measurements.
