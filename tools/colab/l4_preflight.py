#!/usr/bin/env python3
"""Test the unchanged frozen VQA workload on L4 before preparing the dataset.

Uses synthetic cached states at the study batch (2048) and real pretrained
CLIP/RoBERTa inference at the extraction batch (512). No batch/precision fallback.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb"))
    parser.add_argument("--controls", type=Path, default=Path(__file__).with_name("reviewer_controls.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "l4_preflight_status.json"
    result_path = args.output_dir / "l4_preflight.json"
    started = time.time()
    report = {"status": "running", "started_at": started,
              "protocol": {"cached_training_batch": 2048, "feature_extraction_batch": 512,
                           "precision": "fp16_autocast_with_grad_scaler", "image_tokens": 50,
                           "text_tokens": 20, "image_size": 224, "cached_hidden_dim": 768,
                           "embedding_dim": 512, "num_answers": 3000,
                           "grad_accum_steps": 1, "batch_or_precision_fallback": False},
              "source_fingerprints": {"notebook_sha256": hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
                                      "controls_sha256": hashlib.sha256(args.controls.read_bytes()).hexdigest(),
                                      "preflight_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "heads": {}, "feature_extraction": {"status": "pending"}}

    def progress(phase, **values):
        atomic_json(status_path, {"status": "running", "phase": phase,
                                  "updated_at": time.time(), **values})

    try:
        progress("loading_canonical_definitions")
        spec = importlib.util.spec_from_file_location("reviewer_controls", args.controls)
        controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controls)
        notebook = json.loads(args.notebook.read_text())
        os.environ["VQA_OUTPUT_DIR"] = str(args.output_dir.resolve())
        os.environ["VQA_DATA_DIR"] = str((args.output_dir / "unused_data").resolve())
        namespace = {"__name__": "l4_preflight_runtime", "CONTROL_TYPES": controls.CONTROL_TYPES,
                     "REFERENCE_INITIALIZATIONS": {}, "report_progress": lambda **values: None}
        for index in (1, 3, 4, 6, 14, 16, 18):
            exec(compile(controls.cell_source(notebook, index), f"notebook-cell-{index}", "exec"), namespace)
        exec(compile(controls.CONTROL_SOURCE, "reviewer-controls-extension", "exec"), namespace)
        torch, nn, np = namespace["torch"], namespace["nn"], namespace["np"]
        if not torch.cuda.is_available():
            raise RuntimeError("Preflight requires the requested L4 CUDA runtime")
        device = namespace["device"]
        gpu_name = torch.cuda.get_device_name()
        if "L4" not in gpu_name:
            raise RuntimeError(f"Expected an L4, got {gpu_name}")
        namespace["answer_priors"] = torch.full((3000,), 0.1)
        props = torch.cuda.get_device_properties(0)
        report["runtime"] = {"gpu": gpu_name, "gpu_total_bytes": props.total_memory,
                             "torch_version": str(torch.__version__),
                             "transformers_version": str(namespace["transformers"].__version__),
                             "cuda_version": torch.version.cuda,
                             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}

        def memory():
            free, total = torch.cuda.mem_get_info()
            return {"free_bytes": free, "total_bytes": total,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

        report["initial_memory"] = memory()
        variants = ("symmetric_single", "symmetric_sequential", "asymmetric", "symmetric_matched")
        for variant in variants:
            progress("cached_head_preflight", variant=variant)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            namespace["set_seed"](7)
            model = namespace["make_paired_model"](variant, 7, check_initialization=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=namespace["LEARNING_RATE"],
                                          weight_decay=namespace["WEIGHT_DECAY"], fused=True)
            scaler = namespace["GradScaler"]()
            criterion = nn.BCEWithLogitsLoss()
            image = torch.randn(2048, 50, 768, dtype=torch.float16, device=device)
            text = torch.randn(2048, 20, 768, dtype=torch.float16, device=device)
            mask = torch.ones(2048, 20, dtype=torch.int64, device=device)
            target = torch.zeros(2048, 3000, dtype=torch.float32, device=device)
            target[:, 0] = 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            before = model.classifier[-1].weight.detach().clone()
            torch.cuda.synchronize()
            step_started = time.monotonic()
            with namespace["autocast"]():
                logits, _ = model(image, text, mask)
                loss = criterion(logits, target)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite forward output in {variant}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), namespace["MAX_GRAD_NORM"])
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite gradients in {variant}")
            missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
            if missing:
                raise RuntimeError(f"Unused parameters in {variant}: {missing}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            step_seconds = time.monotonic() - step_started
            if scaler.get_scale() < old_scale or not optimizer.state:
                raise RuntimeError(f"Optimizer update was skipped for {variant}")
            parameter_delta = float((model.classifier[-1].weight.detach() - before).abs().max())
            if parameter_delta <= 0:
                raise RuntimeError(f"No classifier parameter update for {variant}")
            row = {"status": "passed", "batch_size": 2048, "loss": float(loss.item()),
                   "step_seconds": step_seconds, "gradient_norm_before_clip": float(gradient_norm),
                   "parameter_max_abs_delta": parameter_delta,
                   "fusion_params": namespace["count_parameters"](model.fusion),
                   "total_params": namespace["count_parameters"](model),
                   "training_memory": memory()}
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.inference_mode(), namespace["autocast"]():
                validation_logits, _ = model(image, text, mask)
                validation_loss = criterion(validation_logits, target)
            torch.cuda.synchronize()
            if not torch.isfinite(validation_loss) or not torch.isfinite(validation_logits).all():
                raise FloatingPointError(f"Nonfinite validation output in {variant}")
            row["validation_loss"] = float(validation_loss.item())
            checkpoint = args.output_dir / f"l4_preflight_{variant}.pt"
            temporary = checkpoint.with_suffix(".pt.partial")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(), "rng_state": {
                            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                        "protocol": report["protocol"], "runtime": report["runtime"],
                        "variant": variant, "scope": "synthetic_preflight_only"}, temporary)
            temporary.replace(checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            expected = model.state_dict()
            assert set(loaded["model_state_dict"]) == set(expected)
            assert all(loaded["model_state_dict"][name].shape == tensor.shape for name, tensor in expected.items())
            assert loaded["optimizer_state_dict"]["state"] and loaded["scaler_state_dict"]
            row["checkpoint"] = {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                                 "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 "reloaded_on_cpu": True}
            report["heads"][variant] = row
            atomic_json(result_path, report)
            del expected, loaded, model, optimizer, scaler, criterion, image, text, mask, target, logits
            del validation_logits, validation_loss, loss, before
            gc.collect()
            torch.cuda.empty_cache()

        progress("loading_frozen_encoders")
        report["feature_extraction"] = {"status": "loading_weights"}
        atomic_json(result_path, report)
        # Actual encoder instances and autocast path from canonical cell 21.
        image_encoder = namespace["ImageEncoder"](512, freeze=True).to(device)
        text_encoder = namespace["TextEncoder"](512, freeze=True).to(device)
        image_encoder.eval()
        text_encoder.eval()
        namespace["set_seed"](7)
        images = torch.randn(512, 3, 224, 224, dtype=torch.float32, device=device)
        input_ids = torch.full((512, 20), 42, dtype=torch.int64, device=device)
        input_ids[:, 0] = 0
        input_ids[:, -1] = 2
        attention_mask = torch.ones(512, 20, dtype=torch.int64, device=device)
        torch.cuda.reset_peak_memory_stats()
        progress("feature_extraction_preflight", batch_size=512)
        torch.cuda.synchronize()
        feature_started = time.monotonic()
        with torch.inference_mode(), namespace["autocast"]():
            image_features = image_encoder.clip(pixel_values=images).last_hidden_state
            text_features = text_encoder.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        torch.cuda.synchronize()
        assert tuple(image_features.shape) == (512, 50, 768)
        assert tuple(text_features.shape) == (512, 20, 768)
        assert torch.isfinite(image_features).all() and torch.isfinite(text_features).all()
        report["feature_extraction"] = {
            "status": "passed", "batch_size": 512,
            "seconds": time.monotonic() - feature_started,
            "image_feature_shape": list(image_features.shape),
            "text_feature_shape": list(text_features.shape), "memory": memory(),
            "image_revision": getattr(image_encoder.clip.config, "_commit_hash", None),
            "text_revision": getattr(text_encoder.roberta.config, "_commit_hash", None)}
        report["status"] = "passed"
        report["finished_at"] = time.time()
        atomic_json(result_path, report)
        atomic_json(status_path, {"status": "complete", "phase": "l4_preflight_passed",
                                  "updated_at": time.time(), "report": str(result_path)})
        print("L4_PREFLIGHT_PASSED", json.dumps({"report": str(result_path), "gpu": gpu_name}))
        return 0
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["finished_at"] = time.time()
        atomic_json(result_path, report)
        atomic_json(status_path, {"status": "failed", "phase": "failed", "updated_at": time.time(),
                                  "error": repr(error), "report": str(result_path)})
        print("L4_PREFLIGHT_FAILED", repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
