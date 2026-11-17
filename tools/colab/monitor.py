#!/usr/bin/env python3
"""Poll one owned Colab experiment and mirror hash-verified artifacts.

This process never queues kernel execution, restarts training, or allocates a VM.
Run it outside the filesystem sandbox so the signed-in Colab CLI can use its
normal authentication and history files. Local files survive a Colab VM loss;
the monitor still depends on the local computer remaining awake and online.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
import traceback


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in path.parts):
        raise ValueError(f"Artifact path must be relative: {value!r}")
    return path


def checked_spec(value):
    result = dict(value)
    result["path"] = str(relative_path(result["path"]))
    result["size_bytes"] = int(result["size_bytes"])
    if result["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", result["sha256"]):
        raise ValueError(f"Invalid artifact identity: {result['path']}")
    return result


class Monitor:
    def __init__(self, args):
        self.args = args
        self.root = args.local_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        state_path = self.root / "monitor_state.json"
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if self.state.get("session", args.session) != args.session:
            raise ValueError("The local monitor directory belongs to another Colab session")
        self.state.update(session=args.session, pid=os.getpid(), poll_seconds=args.interval,
                          monitor_started_at=time.time(), monitor_status="starting")
        self.receipts_path = self.root / "artifact_receipts.json"
        self.receipts = json.loads(self.receipts_path.read_text()) if self.receipts_path.exists() else {}
        self.lock = (self.root / "monitor.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        self.save()

    def save(self):
        self.state["local_heartbeat_at"] = time.time()
        atomic_json(self.root / "monitor_state.json", self.state)

    def event(self, kind, **values):
        payload = {"timestamp": time.time(), "utc": datetime.now(timezone.utc).isoformat(),
                   "kind": kind, "session": self.args.session, **values}
        with (self.root / "monitor_events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)

    def cli(self, *arguments, timeout=90):
        result = subprocess.run([self.args.colab, *arguments], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"colab {arguments[0]} failed: {result.stdout[-1800:]}")
        return result.stdout

    def download(self, remote, local, timeout=90):
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(local.name + ".download")
        self.cli("download", "-s", self.args.session, remote, str(temporary), timeout=timeout)
        if not temporary.exists():
            raise RuntimeError(f"CLI reported success without downloaded file: {remote}")
        temporary.replace(local)

    def get_json(self, remote, name):
        destination = self.root / "telemetry" / name
        self.download(remote, destination)
        value = json.loads(destination.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object from {remote}")
        return value

    def remote_path(self, relative):
        return str(PurePosixPath(self.args.remote_output) / relative_path(relative))

    def materialize_piece(self, spec):
        spec = checked_spec(spec)
        destination = self.root / "artifacts" / spec["path"]
        if (destination.exists() and destination.stat().st_size == spec["size_bytes"]
                and sha256(destination) == spec["sha256"]):
            return destination
        self.download(self.remote_path(spec["path"]), destination, timeout=self.args.transfer_timeout)
        if destination.stat().st_size != spec["size_bytes"] or sha256(destination) != spec["sha256"]:
            destination.rename(destination.with_name(destination.name + f".corrupt-{int(time.time())}"))
            raise ValueError(f"Downloaded artifact failed size/hash verification: {spec['path']}")
        return destination

    def mirror(self, manifest):
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("artifact_manifest.json must contain an artifacts list")
        verified = []
        for entry in artifacts:
            spec = checked_spec(entry)
            destination = self.root / "artifacts" / spec["path"]
            receipt_key = spec["path"] + ":" + spec["sha256"]
            receipt = self.receipts.get(receipt_key)
            if (receipt and destination.exists() and destination.stat().st_size == spec["size_bytes"]
                    and destination.stat().st_mtime_ns == receipt.get("mtime_ns")):
                verified.append(spec["path"])
                continue
            if spec.get("chunks"):
                chunks = [checked_spec(chunk) for chunk in spec["chunks"]]
                if sum(chunk["size_bytes"] for chunk in chunks) != spec["size_bytes"]:
                    raise ValueError(f"Chunk lengths do not match artifact: {spec['path']}")
                pieces = [self.materialize_piece(chunk) for chunk in chunks]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".assembly")
                with temporary.open("wb") as output:
                    for piece in pieces:
                        with piece.open("rb") as stream:
                            while data := stream.read(1024 * 1024):
                                output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != spec["size_bytes"] or sha256(temporary) != spec["sha256"]:
                    raise ValueError(f"Assembled artifact failed verification: {spec['path']}")
                temporary.replace(destination)
            else:
                if spec["size_bytes"] > self.args.max_direct_mib * 1024**2:
                    raise ValueError(f"Artifact requires bounded transport chunks: {spec['path']}")
                destination = self.materialize_piece(spec)
            self.receipts[receipt_key] = {"path": spec["path"], "sha256": spec["sha256"],
                                          "size_bytes": spec["size_bytes"], "verified_at": time.time(),
                                          "mtime_ns": destination.stat().st_mtime_ns,
                                          "local_path": str(destination)}
            atomic_json(self.receipts_path, self.receipts)
            self.event("artifact_verified", path=spec["path"], size_bytes=spec["size_bytes"],
                       sha256=spec["sha256"])
            verified.append(spec["path"])
        self.state["verified_artifact_count"] = len(self.receipts)
        return verified

    def export_log(self):
        try:
            self.cli("log", "-s", self.args.session, "-o", str(self.root / "execution.ipynb"), timeout=120)
            self.state["execution_log_exported_at"] = time.time()
        except Exception as error:
            self.event("log_export_error", error=str(error))

    def stop_owned_session(self, reason):
        if self.args.no_stop:
            self.event("session_retained", reason=reason)
            return
        self.cli("stop", "-s", self.args.session, timeout=90)
        self.state["session_stopped_at"] = time.time()
        self.event("session_stopped", reason=reason)

    def assess(self, current, now):
        age = now - float(current.get("progress_at", current.get("updated_at", now)))
        heartbeat_age = now - float(current.get("heartbeat_at", current.get("updated_at", now)))
        phase = str(current.get("phase", "unknown"))
        threshold = self.args.stall_minutes * 60
        if any(word in phase.lower() for word in ("feature", "cache", "notebook_cell", "download", "prepar")):
            threshold = max(threshold, 3600)
        health = "healthy" if age < threshold and heartbeat_age < max(300, threshold) else "stale_progress"
        self.state.update(phase=phase, progress_age_seconds=age, heartbeat_age_seconds=heartbeat_age,
                          health=health, remote_status=current.get("status"))
        loss = current.get("loss")
        if loss is not None and (not isinstance(loss, (int, float)) or not float("-inf") < loss < float("inf")):
            self.state["health"] = "nonfinite_loss"
        if health != "healthy" and now - self.state.get("last_anomaly_event_at", 0) >= 300:
            self.event("progress_review_needed", phase=phase, age_seconds=age, heartbeat_age=heartbeat_age)
            self.state["last_anomaly_event_at"] = now

    def poll(self):
        now = time.time()
        current = None
        staging = None
        pipeline = None
        errors = {}
        if self.args.pipeline_status_path:
            try:
                pipeline = self.get_json(self.args.pipeline_status_path, "pipeline_status.json")
                self.state["pipeline_last_received_at"] = now
                self.state["pipeline"] = pipeline
            except Exception as error:
                errors["pipeline"] = str(error)
        try:
            current = self.get_json(self.args.status_path, "training_status.json")
            self.state["training_last_received_at"] = now
            self.state["training"] = current
            if (pipeline and pipeline.get("started_at") and current.get("started_at")
                    and current["started_at"] < pipeline["started_at"]):
                # A repaired pipeline can leave an earlier attempt's telemetry
                # on disk until the new training process reaches initialization.
                self.state["ignored_old_training_status"] = True
                current = None
            else:
                self.state.pop("ignored_old_training_status", None)
        except Exception as error:
            errors["training"] = str(error)
        if self.args.data_status_path:
            try:
                staging = self.get_json(self.args.data_status_path, "data_status.json")
                self.state["data_last_received_at"] = now
                self.state["data"] = staging
            except Exception as error:
                errors["data"] = str(error)
        if current is None and staging is None and pipeline is None:
            self.state["consecutive_missed_polls"] = self.state.get("consecutive_missed_polls", 0) + 1
            self.state.update(monitor_status="poll_failed", poll_errors=errors)
            self.event("poll_failed", consecutive_missed=self.state["consecutive_missed_polls"], errors=errors)
            self.save()
            return False
        self.state.update(consecutive_missed_polls=0, monitor_status="monitoring", last_received_at=now,
                          poll_errors=errors)
        self.assess(current or pipeline or staging, now)
        manifest = None
        verified = []
        try:
            manifest = self.get_json(self.args.manifest_path, "artifact_manifest.json")
            verified = self.mirror(manifest)
            self.state.pop("artifact_error", None)
        except Exception as error:
            # A manifest is not expected during download/cache preparation.
            self.state["artifact_error"] = str(error)
            if current and current.get("status") == "complete":
                self.event("completion_verification_pending", error=str(error))
        active = current or pipeline or staging
        self.event("health", phase=self.state.get("phase"), health=self.state.get("health"),
                   seed=active.get("seed"), model_type=active.get("model_type"),
                   epoch=active.get("epoch"), step=active.get("step"),
                   status=active.get("status"), verified_artifacts=len(verified))
        # A pipeline record describes the overall attempt. A stale data failure
        # from an earlier repaired stage cannot override a live pipeline.
        failure_candidates = (pipeline, current) if pipeline else (current, staging)
        failure = next((value for value in failure_candidates
                        if value and value.get("status") in ("failed", "error")), None)
        if failure:
            self.state.update(monitor_status="failed", failure=failure)
            self.export_log()
            if not self.args.keep_on_failure:
                self.stop_owned_session("explicit experiment failure; diagnostics exported")
            self.save()
            return True
        if current and current.get("status") == "complete":
            # Terminal telemetry alone is insufficient: require the final report
            # and a full checkpoint in the verified immutable artifact manifest.
            final_report = any("final_training_report" in path and path.endswith(".json") for path in verified)
            checkpoint = any(path.endswith((".pt", ".pth", ".ckpt")) for path in verified)
            if (manifest and manifest.get("complete") is True and final_report and checkpoint
                    and len(verified) == len(manifest.get("artifacts", []))):
                self.state.update(monitor_status="complete", completed_at=now)
                self.export_log()
                self.stop_owned_session("all final artifacts verified outside the VM")
                self.save()
                return True
            self.state["monitor_status"] = "verifying_completion"
        self.save()
        return False

    def run(self):
        self.event("monitor_started", pid=os.getpid(), poll_seconds=self.args.interval)
        while not self.stop_requested:
            began = time.monotonic()
            try:
                done = self.poll()
            except Exception as error:
                self.state.update(monitor_status="monitor_error", error=str(error))
                self.event("monitor_error", error=str(error), traceback=traceback.format_exc())
                self.save()
                done = False
            if done or self.args.once:
                return
            remaining = self.args.interval - (time.monotonic() - began)
            while remaining > 0 and not self.stop_requested:
                time.sleep(min(remaining, 1))
                remaining = self.args.interval - (time.monotonic() - began)
        self.state["monitor_status"] = "stopped_locally"
        self.save()
        self.event("monitor_stopped_locally", remote_session_retained=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--remote-output", default="/content/acma/output")
    parser.add_argument("--status-path", default="/content/acma/output/status.json")
    parser.add_argument("--pipeline-status-path", default="/content/acma/pipeline_status.json")
    parser.add_argument("--data-status-path", default="/content/acma/data_status.json")
    parser.add_argument("--manifest-path", default="/content/acma/output/artifact_manifest.json")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stall-minutes", type=int, default=20)
    parser.add_argument("--transfer-timeout", type=int, default=300)
    parser.add_argument("--max-direct-mib", type=int, default=64)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-stop", action="store_true", help="Keep VM after completion/failure (diagnostic mode)")
    parser.add_argument("--keep-on-failure", action="store_true", help="Retain failed VM for investigation")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be positive")
    monitor = Monitor(args)
    def request_stop(*_):
        monitor.stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    monitor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
