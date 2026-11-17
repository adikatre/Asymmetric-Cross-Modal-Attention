#!/usr/bin/env python3
"""Run the frozen VQA R1-1 controls while reusing the canonical notebook.

The notebook remains the authority for preprocessing, targets, optimization,
paired data order, validation selection, and epoch-boundary checkpoints. This
entrypoint adds only the two requested controls and operational telemetry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback

NOTEBOOK = "notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb"
CONTROL_TYPES = ("symmetric_single", "symmetric_sequential")
DATA_CELLS = (1, 3, 4, 6, 8, 10, 11, 12, 14, 16, 18, 21, 23)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Notebook source drift: expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def cell_source(notebook, index):
    cell = notebook["cells"][index]
    if cell["cell_type"] != "code":
        raise ValueError(f"Expected code cell {index}")
    return "".join(cell["source"])


def adapted_source(notebook, index):
    source = cell_source(notebook, index)
    if index == 6:
        source = replace_once(source, 'EXPERIMENT_VERSION = "directional_parallel_v1"',
                              'EXPERIMENT_VERSION = "reviewer_shared_controls_v1"')
        # No scientific hyperparameter changes. Operational budget is controlled
        # by the session controller instead of dropping the last seed pair.
    elif index == 12:
        source = replace_once(source, '"fusion_flow": "parallel",',
                              '"fusion_flow": "variant_specific_shared_control",')
        source = replace_once(source, '"feature_cache_sha256": FEATURE_CACHE_SHA256,',
                              '"feature_cache_sha256": FEATURE_CACHE_SHA256,\n'
                              '    "cache_reader": "row_cached_handles_v1",\n'
                              '    "source_fingerprints": SOURCE_FINGERPRINTS,\n'
                              '    "models": list(REQUESTED_VARIANTS),')
        source = replace_once(source,
                              'with open(METRICS_DIR / f"{RUN_TAG}_config.json", "w") as f:\n'
                              '    json.dump(EXPERIMENT_CONFIG, f, indent=2)',
                              '# Final immutable config is written after hashing actual cache bytes.')
        source = replace_once(source,
                              'with open(CHECKPOINT_DIR / f"{RUN_TAG}_answer_vocab.json", "w") as f:\n'
                              '    json.dump(answer_to_idx, f, indent=2)',
                              '# Versioned vocabulary is saved after final cache/config binding.')
    elif index == 21:
        source = replace_once(source, '                    idx += bs',
                              '                    idx += bs\n'
                              '                    report_progress(phase="feature_extraction", '
                              'split=split_name, feature_samples=idx, feature_total=n)')
        source = replace_once(source, '    assert _txt_enc.roberta.config.hidden_size == TEXT_HIDDEN_DIM',
                              '    assert _txt_enc.roberta.config.hidden_size == TEXT_HIDDEN_DIM\n'
                              '    record_encoder_provenance(_img_enc.clip, _txt_enc.roberta)')
    elif index == 23:
        # Preserve row values and order while avoiding HDF5's expensive large
        # point selections. DataLoader retains its original batching/workers.
        tree = ast.parse(source)
        methods = [node for cls in ast.walk(tree)
                   if isinstance(cls, ast.ClassDef) and cls.name == 'PrecomputedVQADataset'
                   for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == '__getitems__']
        if len(methods) != 1:
            raise ValueError('Notebook cache reader source drift')
        lines = source.splitlines(keepends=True)
        del lines[methods[0].lineno-1:methods[0].end_lineno]
        source = ''.join(lines)
        source = replace_once(source, '                self._h5f = h5py.File(self.h5_path, "r")',
                              '                self._h5f = h5py.File(self.h5_path, "r")\n'
                              '                self._fields = tuple(self._h5f[f"{self.split}/{name}"] for name in '
                              '("image_features", "text_features", "attention_mask", "answer_target"))')
        for number, name in enumerate(('image_features', 'text_features', 'attention_mask', 'answer_target')):
            source = replace_once(source, f'self._h5f[f"{{self.split}}/{name}"][idx]',
                                  f'self._fields[{number}][idx]')
        source = replace_once(source,
                              '    train_ds.samples, val_ds.samples = raw_train_ds.samples, raw_val_ds.samples',
                              '    train_ds.samples = [{key: row[key] for key in ("question_id", "image_id")} '
                              'for row in raw_train_ds.samples]\n'
                              '    val_ds.samples = [{key: row[key] for key in ("question_id", "image_id")} '
                              'for row in raw_val_ds.samples]\n'
                              '    del raw_train_ds, raw_val_ds')
        source += '\n# Release redundant raw targets after the cached loaders replace raw loaders.\ngc.collect()\n'
    elif index == 25:
        source = replace_once(source, '        total_loss += loss.item() * answers.size(0)\n        soft_score_sum',
                              '        report_progress(phase="training", step=step, '
                              'steps=len(loader), loss=float(loss.item()), '
                              'optimizer_steps=optimizer_steps, '
                              'learning_rate=float(optimizer.param_groups[0]["lr"]))\n'
                              '        total_loss += loss.item() * answers.size(0)\n        soft_score_sum')
        source = replace_once(source, '        total += answers.size(0)\n\n    metrics = {',
                              '        total += answers.size(0)\n'
                              '        report_progress(phase="validation", '
                              'validation_samples=total, loss=float(loss.item()))\n\n    metrics = {')
    elif index == 26:
        source = replace_once(source, '"attention_calls": 4 if model_type == "symmetric_matched" else 2,',
                              '"attention_calls": attention_calls(model_type),')
        source = replace_once(source, '    for epoch in range(start_epoch, EPOCHS + 1):',
                              '    for epoch in range(start_epoch, EPOCHS + 1):\n'
                              '        report_progress(phase="epoch_start", epoch=epoch, '
                              'seed=seed, model_type=model_type, run_name=run_name)')
        source = replace_once(source, '        save_json(history, history_path)',
                              '        save_json(history, history_path)\n'
                              '        report_progress(phase="checkpoint_ready", '
                              'epoch=epoch, checkpoint=str(latest_path), best_checkpoint=str(best_path), '
                              'best_epoch=best_epoch, best_val_acc=best_val_acc)')
    elif index == 27:
        source = replace_once(source, '"attention_calls": 4 if variant == "symmetric_matched" else 2,',
                              '"attention_calls": attention_calls(variant),')
        source = source.split('\nif not RUN_PREFLIGHT_BENCHMARK:')[0]
    ast.parse(source)
    return source


CONTROL_SOURCE = r'''
class SequentialSharedCrossModalFusion(nn.Module):
    """Two shared blocks composed per direction, with original opposite keys.

    For each direction B1(B0(query, original_other), original_other).
    Both directions use the same ordered blocks; image/text updates do not
    feed into each other. This introduces query depth instead of averaging.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.shared_experts = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout),
            CrossAttentionBlock(embed_dim, num_heads, dropout)])

    def _apply_blocks(self, query, key_value, key_padding_mask=None, return_attention=False):
        weights = None
        for block in self.shared_experts:
            query, weights = block(query, key_value, key_padding_mask, return_attention)
        return query, weights

    def forward(self, image_features, text_features, text_padding_mask=None, return_attention=False):
        image, i2t = self._apply_blocks(image_features, text_features, text_padding_mask, return_attention)
        text, t2i = self._apply_blocks(text_features, image_features, None, return_attention)
        return image, text, i2t, t2i


class SequentialSharedVQAModel(LegacySymmetricVQAModel):
    def __init__(self, num_answers, embed_dim=512, num_heads=8, dropout=0.3):
        super().__init__(num_answers, embed_dim, num_heads, dropout)
        self.fusion = SequentialSharedCrossModalFusion(embed_dim, num_heads, dropout)


def attention_calls(variant):
    return 4 if variant in ("symmetric_matched", "symmetric_sequential") else 2


def make_model(model_type, vision_profile_name=ACTIVE_VISION_PROFILE):
    if not FREEZE_ENCODERS:
        raise ValueError("This first reviewer experiment is frozen encoders only")
    classes = {"asymmetric": AsymmetricVQAModel,
               "symmetric_matched": ParameterMatchedSymmetricVQAModel,
               "symmetric_single": LegacySymmetricVQAModel,
               "symmetric_sequential": SequentialSharedVQAModel}
    return classes[model_type](NUM_ANSWERS, EMBED_DIM, NUM_HEADS, DROPOUT)


def fusion_blocks(model, model_type):
    if model_type == "asymmetric":
        return [model.fusion.cross_attn_img_to_txt, model.fusion.cross_attn_txt_to_img]
    if model_type == "symmetric_single":
        return [model.fusion.shared_cross_attn]
    return list(model.fusion.shared_experts)


def module_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def make_paired_model(model_type, seed, check_initialization=True):
    set_seed(seed)
    model = make_model(model_type)
    blocks = fusion_blocks(model, model_type)
    # Exact common-module order and seed offsets from notebook cell 26.
    common = {"classifier": model.classifier,
              "image_projection": model.image_projection,
              "text_projection": model.text_projection}
    with torch.random.fork_rng(devices=[]):
        for offset, module in enumerate(common.values(), start=100):
            torch.random.default_generator.manual_seed(seed + offset)
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        for offset, block in enumerate(blocks, start=200):
            torch.random.default_generator.manual_seed(seed + offset)
            initial = CrossAttentionBlock(EMBED_DIM, NUM_HEADS, DROPOUT)
            block.load_state_dict(initial.state_dict())
    with torch.no_grad():
        model.classifier[-1].bias.copy_(torch.logit(answer_priors))
    modules = {**common, **{f"block_{i}": b for i, b in enumerate(blocks)}}
    model.initialization_hashes = {name: module_hash(m) for name, m in modules.items()}
    expected = 3_153_408 if model_type == "symmetric_single" else EXPECTED_FUSION_PARAMS
    assert count_parameters(model.fusion) == expected
    expected_head = 6_004_664 if model_type == "symmetric_single" else EXPECTED_FROZEN_HEAD_PARAMS
    assert count_parameters(model) == expected_head
    if check_initialization:
        path = METRICS_DIR / f"{RUN_TAG}_s{seed}_{model_type}_initialization.json"
        if path.exists():
            if json.loads(path.read_text()) != model.initialization_hashes:
                raise ValueError("Initialization changed from this run's saved provenance")
        else:
            path.write_text(json.dumps(model.initialization_hashes, indent=2))
        reference = REFERENCE_INITIALIZATIONS.get(str(seed), {})
        for name, digest in model.initialization_hashes.items():
            if name in reference and reference[name] != digest:
                raise ValueError(f"Initialization differs from original parallel run: {seed}/{name}")
    return model.to(device)


def audit_reviewer_controls():
    audit_active_architectures()
    for variant in CONTROL_TYPES:
        model = make_paired_model(variant, SEED)
        fusion = model.fusion
        fusion.eval()
        x, y = torch.randn(2, 3, EMBED_DIM, device=device), torch.randn(2, 3, EMBED_DIM, device=device)
        with torch.no_grad():
            xy = fusion(x, y)
            yx = fusion(y, x)
        torch.testing.assert_close(xy[0], yx[1])
        torch.testing.assert_close(xy[1], yx[0])
        fusion.train()
        optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-4)
        output = fusion(x, y)
        loss = output[0].square().mean() + output[1].square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
        optimizer.step()
        del optimizer, model, fusion, output, loss
    gc.collect()
    torch.cuda.empty_cache()


def run_controls():
    """All three paired seeds are required; no accuracy-dependent stopping."""
    complete = []
    for seed_index, seed in enumerate(SEEDS):
        variants = ACTIVE_MODEL_TYPES if seed_index % 2 == 0 else ACTIVE_MODEL_TYPES[::-1]
        pair = {}
        for variant in variants:
            report_progress(phase="run_start", seed=seed, model_type=variant)
            pair[variant] = finish_run(variant, seed)
            if pair[variant]["epochs_completed"] != EPOCHS:
                raise RuntimeError(f"Incomplete horizon for {seed}/{variant}")
        complete.append({"seed": seed, "models": pair})
        save_json(complete, METRICS_DIR / f"{RUN_TAG}_campaign.json")
    save_json({"status": "complete", "paired_seeds": list(SEEDS),
               "models": list(ACTIVE_MODEL_TYPES), "config_sha256": CONFIG_SHA256,
               "runs": complete, "scope": "frozen encoders only",
               "comparison_note": "single shared control changes capacity; sequential control changes depth/aggregation"},
              METRICS_DIR / f"{RUN_TAG}_final_training_report.json")
    return complete


def profile_controls():
    """Warm synchronized cached-head timing; profiler FLOPs have explicit coverage."""
    batch = next(iter(val_loader))
    inputs = tuple(x.to(device, non_blocking=True) for x in batch[:3])
    target = batch[3].to(device, non_blocking=True)
    rows = {}
    for variant in ACTIVE_MODEL_TYPES:
        report_progress(phase="profiling", model_type=variant)
        model = make_paired_model(variant, SEED)
        optimizer, _, _ = build_optimizer_and_scheduler(model)
        scaler = GradScaler() if USE_AMP else None
        criterion = nn.BCEWithLogitsLoss()

        def step(training):
            model.train(training)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), (autocast() if USE_AMP else nullcontext()):
                logits, _ = model(*inputs)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite profiling loss")
            if training:
                if scaler is None:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()

        row = {"batch_size": int(inputs[0].shape[0]), "precision": "autocast_fp16" if USE_AMP else "fp32",
               "warmup_steps": 5, "measured_steps": 10,
               "total_params": count_parameters(model), "fusion_params": count_parameters(model.fusion),
               "attention_calls": attention_calls(variant)}
        for mode, training in (("forward", False), ("training_step", True)):
            for _ in range(5):
                step(training)
            cuda_sync()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(10):
                cuda_sync()
                start = time.monotonic()
                step(training)
                cuda_sync()
                samples.append(time.monotonic() - start)
            row[mode] = {"seconds_per_step": samples, "median_seconds": float(np.median(samples)),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        # Profile a small fixed batch for tractable operator records; scale is
        # reported, not silently extrapolated to the throughput batch.
        small = tuple(x[:4] for x in inputs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_flops=True) as prof:
            with autocast() if USE_AMP else nullcontext():
                logits, _ = model(*small)
                loss = criterion(logits, target[:4])
            loss.backward()
            cuda_sync()
        events = prof.key_averages()
        row["flop_profile"] = {
            "batch_size": int(small[0].shape[0]), "scope": "cached_head_forward_and_backward",
            "reported_flops": sum(int(event.flops or 0) for event in events),
            "coverage": "torch.profiler estimates FLOPs only for supported operators; this is an incomplete count",
            "operators": [{"operator": event.key, "calls": event.count, "flops": int(event.flops or 0)}
                          for event in events],
            "encoder_cost": "excluded: encoders run once during separately recorded feature extraction"}
        rows[variant] = row
        del model, optimizer, scaler, logits, loss, prof, events
        gc.collect()
        torch.cuda.empty_cache()
    report = {"runtime": runtime_metadata(), "models": rows, "config_sha256": CONFIG_SHA256}
    save_json(report, METRICS_DIR / f"{RUN_TAG}_profile.json")
    return report


def synthetic_resume_test():
    """Separate smoke configuration: compare interrupted vs uninterrupted epochs."""
    global train_ds, val_ds, train_loader, val_loader, val_subset_loader
    global answer_priors, EXPERIMENT_CONFIG, CONFIG_SHA256, RUN_TAG, EPOCHS
    global BATCH_SIZE, _loader_kwargs, NUM_WORKERS, make_epoch_train_loader
    set_seed(SEED)
    n = 4
    answer_priors = torch.full((NUM_ANSWERS,), 0.1)
    tensors = (torch.randn(n, IMAGE_SEQ_LEN, IMAGE_HIDDEN_DIM, dtype=torch.float16),
               torch.randn(n, MAX_QUESTION_LEN, TEXT_HIDDEN_DIM, dtype=torch.float16),
               torch.ones(n, MAX_QUESTION_LEN, dtype=torch.int64),
               torch.zeros(n, NUM_ANSWERS))
    tensors[-1][:, 0] = 1
    train_ds = val_ds = torch.utils.data.TensorDataset(*tensors)
    BATCH_SIZE, NUM_WORKERS, _loader_kwargs = 2, 0, {"num_workers": 0}
    train_loader = val_loader = val_subset_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    def make_epoch_train_loader(seed, epoch):
        generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(epoch))
        return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=generator, **_loader_kwargs)
    EXPERIMENT_CONFIG = {"experiment_version": "reviewer_controls_synthetic_resume_test",
                         "source_fingerprints": SOURCE_FINGERPRINTS,
                         "resume_granularity": "epoch_boundary"}
    CONFIG_SHA256 = hashlib.sha256(json.dumps(EXPERIMENT_CONFIG, sort_keys=True).encode()).hexdigest()
    RUN_TAG = "synthetic_" + CONFIG_SHA256[:12]
    results = {}
    for variant in CONTROL_TYPES:
        EPOCHS = 2
        model, history = run_training(variant, run_name=RUN_TAG + "_continuous_" + variant)
        continuous = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        del model
        EPOCHS = 1
        model, _ = run_training(variant, run_name=RUN_TAG + "_resumed_" + variant)
        del model
        EPOCHS = 2
        model, resumed = run_training(variant, run_name=RUN_TAG + "_resumed_" + variant)
        maximum_error = 0.0
        for name, tensor in model.state_dict().items():
            expected = continuous[name]
            actual = tensor.detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            maximum_error = max(maximum_error, float((actual - expected).abs().max()))
        assert len(history) == len(resumed) == 2
        results[variant] = {"epochs": 2, "max_parameter_abs_error": maximum_error,
                            "resumed_checkpoint": str(CHECKPOINT_DIR / f"{RUN_TAG}_resumed_{variant}_latest.pt")}
        del model, continuous
        gc.collect()
        torch.cuda.empty_cache()
    save_json(results, METRICS_DIR / "synthetic_resume_test.json")
    return results
'''


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


class Telemetry:
    def __init__(self, output):
        self.path = output / "status.json"
        self.lock = threading.Lock()
        self.state = {"status": "running", "phase": "initializing", "started_at": time.time(),
                      "progress_at": time.time(), "step": 0}
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def resources(self):
        disk = shutil.disk_usage(self.path.parent)
        result = {"disk_free_bytes": disk.free, "disk_total_bytes": disk.total}
        try:
            values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            result["ram_available_bytes"] = int(values["MemAvailable"].split()[0]) * 1024
            result["ram_total_bytes"] = int(values["MemTotal"].split()[0]) * 1024
        except (OSError, KeyError):
            pass
        return result

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["progress_at"] = time.time()
            self.write()

    def write(self):
        atomic_json(self.path, {**self.state, "heartbeat_at": time.time(), **self.resources()})

    def heartbeat(self):
        while not self.stopped.wait(20):
            with self.lock:
                self.write()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)


class ArtifactPublisher:
    """Publish immutable chunks before advertising checkpoint candidates."""
    CHUNK_BYTES = 32 * 1024 * 1024

    def __init__(self, output):
        self.output = output
        self.path = output / "artifact_manifest.json"
        self.artifacts = {}
        self.complete = False
        if self.path.exists():
            self.artifacts = {item["path"]: item for item in json.loads(self.path.read_text())["artifacts"]}

    def checkpoint(self, source, run_name, epoch, kind):
        path = Path("snapshots") / run_name / f"epoch_{int(epoch):04d}_{kind}.pt"
        if str(path) in self.artifacts:
            return
        destination = self.output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        total, digest, chunks = 0, hashlib.sha256(), []
        with Path(source).open("rb") as stream:
            number = 0
            while data := stream.read(self.CHUNK_BYTES):
                chunk_path = Path(str(path) + f".part{number:04d}")
                temporary = self.output / (str(chunk_path) + ".partial")
                temporary.write_bytes(data)
                temporary.replace(self.output / chunk_path)
                chunk_hash = hashlib.sha256(data).hexdigest()
                chunks.append({"path": str(chunk_path), "size_bytes": len(data), "sha256": chunk_hash})
                digest.update(data)
                total += len(data)
                number += 1
        self.artifacts[str(path)] = {"path": str(path), "source_path": str(Path(source).relative_to(self.output)),
                                     "size_bytes": total, "sha256": digest.hexdigest(), "chunks": chunks}
        self.flush()

    def compact_results(self):
        files = [source for folder in ("metrics", "predictions")
                 for source in sorted((self.output / folder).glob("*"))
                 if source.suffix in (".json", ".npz")]
        files.extend(source for name in ("source_manifest.json", "cache_provenance.json", "encoder_provenance.json")
                     if (source := self.output / name).exists())
        files.extend(sorted((self.output / "checkpoints").glob("*answer_vocab.json")))
        for source in files:
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            path = Path("snapshots") / "results" / digest[:16] / source.relative_to(self.output)
            destination = self.output / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(data)
            temporary.replace(destination)
            self.artifacts[str(path)] = {"path": str(path), "size_bytes": len(data),
                                        "sha256": digest}
        self.flush()

    def flush(self):
        atomic_json(self.path, {"format": "colab_artifact_chunks_v1", "updated_at": time.time(),
                                "complete": self.complete,
                                "artifacts": list(self.artifacts.values())})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path(NOTEBOOK))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(CONTROL_TYPES),
                        choices=(*CONTROL_TYPES, "asymmetric", "symmetric_matched"))
    parser.add_argument("--self-test", action="store_true", help="Synthetic train/checkpoint/resume test")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-source", action="store_true", help="Compile source without importing GPU dependencies")
    args = parser.parse_args(argv)
    notebook_bytes = args.notebook.read_bytes()
    notebook = json.loads(notebook_bytes)
    sources = {i: adapted_source(notebook, i) for i in (*DATA_CELLS, 25, 26, 27)}
    for i, source in sources.items():
        compile(source, f"{args.notebook}:cell-{i}", "exec")
    compile(CONTROL_SOURCE, "reviewer-controls-extension", "exec")
    if args.validate_source:
        print("Canonical notebook cells and reviewer extension compile successfully")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = Telemetry(args.output_dir)
    publisher = ArtifactPublisher(args.output_dir)

    def report_progress(**values):
        if values.get("phase") == "checkpoint_ready":
            telemetry.update(phase="publishing_checkpoint", epoch=values["epoch"])
            run_name = Path(values["checkpoint"]).name.removesuffix("_latest.pt")
            publisher.checkpoint(values["checkpoint"], run_name, values["epoch"], "latest")
            publisher.checkpoint(values["best_checkpoint"], run_name, values["best_epoch"], "best")
            publisher.compact_results()
        telemetry.update(**values)

    def record_encoder_provenance(image_encoder, text_encoder):
        telemetry.update(phase="encoder_content_hashing")
        records = {}
        for name, model in (("image", image_encoder), ("text", text_encoder)):
            digest = hashlib.sha256()
            for key, tensor in model.state_dict().items():
                digest.update(key.encode())
                digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
            records[name] = {"resolved_revision": getattr(model.config, "_commit_hash", None),
                             "name_or_path": model.config._name_or_path,
                             "state_dict_sha256": digest.hexdigest(), "config": model.config.to_dict()}
        atomic_json(args.output_dir / "encoder_provenance.json", records)

    def record_cache_provenance(namespace):
        cache = Path(namespace["FEATURES_H5"])
        telemetry.update(phase="feature_cache_content_hashing", feature_cache=str(cache))
        total, digest = 0, hashlib.sha256()
        with cache.open("rb") as stream:
            while data := stream.read(32 * 1024 * 1024):
                digest.update(data)
                total += len(data)
                if total % (1024 * 1024 * 1024) == 0:
                    telemetry.update(phase="feature_cache_content_hashing", bytes_hashed=total,
                                     bytes_total=cache.stat().st_size)
        provenance = {
            "path": str(cache), "size_bytes": total, "sha256": digest.hexdigest(),
            "feature_cache_config_sha256": namespace["FEATURE_CACHE_SHA256"],
            "feature_cache_config": namespace["FEATURE_CACHE_CONFIG"],
            "runtime": {"gpu": namespace["torch"].cuda.get_device_name(),
                        "torch_version": str(namespace["torch"].__version__),
                        "transformers_version": str(namespace["transformers"].__version__)}}
        atomic_json(args.output_dir / "cache_provenance.json", provenance)
        config = namespace["EXPERIMENT_CONFIG"]
        config["feature_cache_content_sha256"] = provenance["sha256"]
        config["encoder_provenance"] = {
            name: {key: value for key, value in record.items() if key != "config"}
            for name, record in json.loads((args.output_dir / "encoder_provenance.json").read_text()).items()}
        config["runtime"] = provenance["runtime"]
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        namespace["CONFIG_SHA256"] = config_sha
        namespace["RUN_TAG"] = f"clipB32_224_frozen_top{namespace['NUM_ANSWERS']}_reviewer_shared_controls_v1_{config_sha[:12]}"
        atomic_json(namespace["METRICS_DIR"] / f"{namespace['RUN_TAG']}_config.json", config)
        atomic_json(namespace["CHECKPOINT_DIR"] / f"{namespace['RUN_TAG']}_answer_vocab.json",
                    namespace["answer_to_idx"])
    fingerprints = {"notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
                    "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(args.output_dir / "source_manifest.json", fingerprints)
    references = {}
    if args.reference_metrics:
        for path in args.reference_metrics.glob("*_initialization.json"):
            seed = path.stem.split("_s")[-1].split("_")[0]
            if seed in {"7", "42", "123"}:
                value = json.loads(path.read_text())
                if seed in references and references[seed] != value:
                    raise ValueError(f"Conflicting reference initialization records for seed {seed}")
                references[seed] = value
    os.environ["VQA_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["VQA_OUTPUT_DIR"] = str(args.output_dir.resolve())
    namespace = {"__name__": "reviewer_control_run", "SOURCE_FINGERPRINTS": fingerprints,
                 "CONTROL_TYPES": CONTROL_TYPES, "REQUESTED_VARIANTS": tuple(args.variants),
                 "REFERENCE_INITIALIZATIONS": references,
                 "report_progress": report_progress,
                 "record_encoder_provenance": record_encoder_provenance}
    try:
        initial_cells = (1, 3, 4, 6, 14, 16, 18) if args.self_test else DATA_CELLS
        for index in initial_cells:
            telemetry.update(phase=f"notebook_cell_{index}", notebook_cell=index)
            exec(compile(sources[index], f"{args.notebook}:cell-{index}", "exec"), namespace)
            if index == 4:
                if namespace["device"].type != "cuda":
                    raise RuntimeError("The reviewer experiment requires a verified GPU runtime")
                torch = namespace["torch"]
                telemetry.update(gpu=torch.cuda.get_device_name(),
                                 gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
            if index == 21:
                record_cache_provenance(namespace)
        if args.prepare_only:
            telemetry.update(status="prepared", phase="feature_cache_ready",
                             feature_cache=str(namespace["FEATURES_H5"]),
                             feature_cache_sha256=namespace["FEATURE_CACHE_SHA256"])
            return 0
        for index in (25, 26, 27):
            exec(compile(sources[index], f"{args.notebook}:cell-{index}", "exec"), namespace)
        exec(compile(CONTROL_SOURCE, "reviewer-controls-extension", "exec"), namespace)
        namespace["ACTIVE_MODEL_TYPES"] = tuple(args.variants)
        if args.self_test:
            namespace["REFERENCE_INITIALIZATIONS"] = {}
            telemetry.update(phase="synthetic_train_checkpoint_resume_test")
            namespace["synthetic_resume_test"]()
            publisher.compact_results()
            telemetry.update(status="self_test_complete", phase="self_test_complete")
            return 0
        telemetry.update(phase="architecture_audit")
        namespace["audit_reviewer_controls"]()
        telemetry.update(phase="preflight")
        namespace["preflight_report"] = namespace["run_preflight"]()
        namespace["profile_controls"]()
        if args.preflight_only:
            telemetry.update(status="preflight_complete", phase="preflight_complete")
            return 0
        namespace["run_controls"]()
        publisher.compact_results()
        publisher.complete = True
        publisher.flush()
        telemetry.update(status="complete", phase="complete")
        return 0
    except BaseException as error:
        telemetry.update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
