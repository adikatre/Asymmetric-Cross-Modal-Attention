#!/usr/bin/env python3
"""Prepare data locally, allocate one L4, stage verified inputs, and monitor.

No GPU is allocated during local downloads/preprocessing. A restart reconciles
the one recorded session and never submits duplicate training or allocations.
The independent watchdog only releases the owned VM after positive evidence
that no campaign process remains; missed network polls alone never stop a VM.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_command(pid):
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, ValueError):
        return ""


RECEIVER = r'''
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]); expected_size=int(sys.argv[2]); expected_hash=sys.argv[3]
target.parent.mkdir(parents=True,exist_ok=True)
temporary=target.with_name(target.name+'.uploading')
digest=hashlib.sha256(); total=0
with temporary.open('wb') as output:
    while True:
        block=sys.stdin.buffer.read(8*1024*1024)
        if not block: break
        output.write(block); digest.update(block); total+=len(block)
    output.flush(); os.fsync(output.fileno())
if total!=expected_size or digest.hexdigest()!=expected_hash:
    raise RuntimeError('ACMA_TRANSFER_HASH_MISMATCH: '+str(total)+' '+digest.hexdigest())
temporary.replace(target)
print('ACMA_TRANSFER_OK '+json.dumps({'path':str(target),'size_bytes':total,'sha256':digest.hexdigest()}),flush=True)
'''

PROCESS_PROBE = r'''
import json,os,pathlib,time
root=pathlib.Path('/content/acma'); statuses={}
for name in ('pipeline_status.json','output/status.json'):
    try: statuses[name]=json.loads((root/name).read_text())
    except (OSError,ValueError): pass
processes=[]
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    if int(path.parent.name)==os.getpid(): continue
    try: command=path.read_bytes().replace(bytes([0]),b' ').decode()
    except (OSError,UnicodeError): continue
    if 'ACMA_WATCHDOG_PROBE' in command: continue
    if (any('/tools/colab/'+name in command for name in ('pipeline.py','reviewer_controls.py','setup_runtime.py'))
            or 'ACMA_TRANSFER_HASH_MISMATCH' in command
            or ('pip install' in command and 'torch==2.14.0' in command)
            or 'unittest discover -s tools/colab' in command):
        processes.append({'pid':int(path.parent.name),'command':command[:300]})
print('ACMA_WATCHDOG_PROBE '+json.dumps({'time':time.time(),'statuses':statuses,'processes':processes}),flush=True)
'''


class Launcher:
    def __init__(self, args):
        self.args = args
        self.repo = args.repo.resolve()
        self.root = args.campaign_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "launch_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if self.state.get("session", args.session) != args.session:
            raise ValueError("Campaign directory is bound to a different session")
        self.lock = (self.root / "launch.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state.update(session=args.session, pid=os.getpid(),
                          started_at=self.state.get("started_at", time.time()))
        self.lock_state = threading.Lock()
        self.closed = threading.Event()
        self.heartbeat = threading.Thread(target=self.beat, daemon=True)
        self.save()
        self.heartbeat.start()

    def save(self, **fields):
        self.state.update(fields, heartbeat_at=time.time())
        atomic_json(self.state_path, self.state)

    def beat(self):
        while not self.closed.wait(20):
            with self.lock_state:
                self.save()

    def update(self, **fields):
        with self.lock_state:
            self.save(**fields)
        print(json.dumps({"time": time.time(), **fields}), flush=True)

    def run(self, argv, *, name, timeout=600, stdin=None):
        log_path = self.root / (name + ".log")
        self.update(phase=name, active_command=list(map(str, argv)))
        with log_path.open("ab") as log:
            process = subprocess.Popen(list(map(str, argv)), cwd=self.repo,
                                       stdin=stdin or subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # CLI handshake failures can leak a non-daemon WebSocket
                # thread. Terminate the complete local transport process group.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if returncode:
            raise RuntimeError(f"{name} exited {returncode}; see {log_path}")
        return log_path.read_text(errors="replace")

    def cli(self, *argv, name, timeout=600, stdin=None):
        return self.run([self.args.colab, *argv], name=name, timeout=timeout, stdin=stdin)

    def fetch_json(self, remote, name):
        destination = self.root / name
        self.cli("download", "-s", self.args.session, remote, str(destination),
                 name="fetch_" + name.replace(".", "_"), timeout=120)
        return json.loads(destination.read_text())

    def wait_download(self):
        pid = self.args.wait_download_pid
        if not pid:
            return
        while command := process_command(pid):
            if "prepare_vqa.py" not in command or "--download-only" not in command:
                raise RuntimeError(f"PID {pid} no longer identifies the expected download-only job")
            self.update(phase="waiting_for_existing_download", download_pid=pid)
            self.closed.wait(30)
        self.update(phase="existing_download_finished", download_pid=pid)

    def prepare_local(self):
        self.wait_download()
        python = self.args.data_python or self.repo / ".venv-colab-data/bin/python"
        if not Path(python).is_file():
            raise FileNotFoundError(f"Prepared local data environment missing: {python}")
        self.run([python, self.repo / "tools/colab/prepare_vqa.py", "--data-dir", self.repo / "data",
                  "--status", self.root / "local_data_status.json", "--image-backend", "pillow"],
                 name="local_data_preparation", timeout=self.args.local_timeout)
        verification = self.root / "verify_local_data.py"
        verification.write_text('''import hashlib,json,sys,h5py
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import prepare_vqa as p
data=Path(sys.argv[2]);manifest=json.loads((data/'source_manifest.json').read_text())
for name,expected in p.EXPECTED_JSON.items():
 folder='answers' if 'annotations' in name else 'questions'
 actual=p.sha256(data/folder/name)
 if actual!=expected: raise ValueError('Original metadata mismatch: '+name)
cache=data/'vqa_images_224.h5'
if not p.complete_cache(cache): raise ValueError('Incomplete or incompatible 224px cache')
with h5py.File(cache,'r') as f:
 if f['images'].shape!=(123287,224,224,3): raise ValueError('Wrong image count/shape')
actual=p.sha256(cache);receipt=manifest['image_cache']
if receipt.get('sha256')!=actual or receipt.get('count')!=123287: raise ValueError('Image content receipt mismatch')
Path(sys.argv[3]).write_text(json.dumps({'status':'complete','images':123287,'size_bytes':cache.stat().st_size,'sha256':actual,'metadata':p.EXPECTED_JSON},indent=2))
print('ACMA_LOCAL_DATA_VERIFIED',flush=True)
''')
        self.run([python, verification, self.repo / "tools/colab", self.repo / "data",
                  self.root / "local_data_verification.json"], name="local_data_verification", timeout=1800)
        self.update(local_data_verified_at=time.time())

    def bundle(self):
        files = sorted((self.repo / "tools/colab").glob("*.py"))
        files.append(self.repo / "notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb")
        manifest = {str(path.relative_to(self.repo)): sha256(path) for path in files}
        manifest_path = self.root / "source_manifest.json"
        atomic_json(manifest_path, manifest)
        bundle = self.root / "source_bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(self.repo)))
            archive.add(manifest_path, arcname="source_manifest.json")
        metadata_bundle = self.root / "metadata_bundle.tar.gz"
        with tarfile.open(metadata_bundle, "w:gz") as archive:
            for path in sorted((self.repo / "data/zip").glob("v2_*.zip")):
                archive.add(path, arcname="zip/" + path.name)
            archive.add(self.repo / "data/source_manifest.json", arcname="source_manifest.json")
        return bundle, metadata_bundle

    def allocate(self):
        prior_request = self.state.get("allocation_requested_at")
        sessions = self.cli("sessions", name="sessions_before_allocation", timeout=120)
        exists = self.args.session in sessions
        if exists and not prior_request:
            raise RuntimeError("Named session already exists without an ownership record; refusing to reuse or stop it")
        if prior_request and not exists:
            raise RuntimeError("The previously requested runtime is absent; automatic reallocation is disabled")
        if not exists:
            self.update(allocation_requested_at=time.time(), phase="allocating_l4")
            self.start_watchdog()
            try:
                self.cli("new", "-s", self.args.session, "--gpu", "L4", name="allocate_l4", timeout=600)
            except (subprocess.TimeoutExpired, RuntimeError):
                # A timed-out client can still have allocated a VM. Reconcile
                # exactly once and never issue a second allocation request.
                sessions = self.cli("sessions", name="reconcile_allocation_timeout", timeout=120)
                if self.args.session not in sessions:
                    raise RuntimeError("Allocation failed with no confirmed owned runtime; no retry was submitted")
        self.update(owned_session=True, allocated_at=self.state.get("allocated_at", time.time()))

    def start_watchdog(self):
        previous = self.state.get("watchdog_pid")
        if previous and "launch.py" in process_command(previous) and "--watchdog" in process_command(previous):
            return
        log = (self.root / "watchdog.log").open("ab")
        argv = [sys.executable, str(Path(__file__).resolve()), "--repo", str(self.repo),
                "--campaign-dir", str(self.root), "--session", self.args.session,
                "--colab", self.args.colab, "--watchdog"]
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        log.close()
        self.update(watchdog_pid=process.pid)

    def execute_marker(self, source, name, marker, timeout=120):
        # Short setup scripts need no notebook state. SSH avoids the installed
        # CLI's initial Jupyter handshake cleanup bug and propagates exceptions.
        self.cli("ssh", "-s", self.args.session, "--", "python3", "-c", source.read_text(),
                 name=name, timeout=timeout)
        if marker not in (self.root / (name + ".log")).read_text(errors="replace"):
            raise RuntimeError(f"{name} lacks explicit completion marker {marker}")

    def setup(self):
        self.execute_marker(self.repo / "tools/colab/runtime_probe.py", "runtime_probe", "ACMA_RUNTIME_PROBE_OK")
        probe = self.fetch_json("/content/acma/runtime_probe.json", "runtime_probe.json")
        if "L4" not in probe["gpu"]:
            raise RuntimeError(f"Expected L4; allocated {probe['gpu']}")
        if int(probe["disk"]["free"]) < 115_000_000_000:
            raise RuntimeError("Insufficient remote disk for image/feature caches and checkpoints")
        self.execute_marker(self.repo / "tools/colab/setup_runtime.py", "setup_runtime", "ACMA_SETUP_OK", timeout=600)
        versions = self.fetch_json("/content/acma/setup_versions.json", "setup_versions.json")
        for name, expected in (("torch", "2.14.0"), ("torchvision", "0.29.0"), ("transformers", "5.17.0")):
            if versions[name].split("+")[0] != expected:
                raise RuntimeError(f"Pinned environment mismatch: {name}={versions[name]}")
        self.update(environment_verified_at=time.time(), gpu=probe["gpu"], packages=versions)

    def stream(self, source, remote, name):
        expected = {"size_bytes": source.stat().st_size, "sha256": sha256(source)}
        with source.open("rb") as stream:
            text = self.cli("ssh", "-s", self.args.session, "--", "python3", "-c", RECEIVER,
                            remote, str(expected["size_bytes"]), expected["sha256"], name=name,
                            timeout=self.args.transfer_timeout, stdin=stream)
        matches = re.findall(r"ACMA_TRANSFER_OK (\{[^\n]+\})", text)
        if not matches:
            raise RuntimeError(f"{name} lacks remote transfer receipt")
        receipt = json.loads(matches[-1])
        if (receipt["size_bytes"], receipt["sha256"]) != (expected["size_bytes"], expected["sha256"]):
            raise ValueError(f"{name} remote transfer hash mismatch")
        atomic_json(self.root / (name + "_receipt.json"), {**receipt, "local_path": str(source),
                                                         "verified_at": time.time()})

    def stage(self, bundle, metadata):
        self.stream(bundle, "/content/acma/source_bundle.tar.gz", "stage_source")
        self.stream(metadata, "/content/acma/metadata_bundle.tar.gz", "stage_metadata")
        unpack = self.root / "unpack_metadata.py"
        unpack.write_text('''import hashlib,json,pathlib,tarfile,zipfile
root=pathlib.Path('/content/acma');data=root/'data';data.mkdir(exist_ok=True)
with tarfile.open(root/'metadata_bundle.tar.gz') as archive: archive.extractall(data,filter='data')
manifest=json.loads((data/'source_manifest.json').read_text())
for archive_path in sorted((data/'zip').glob('v2_*.zip')):
 with zipfile.ZipFile(archive_path) as archive:
  for member in archive.namelist():
   if not member.endswith('.json') or pathlib.Path(member).name!=member: raise ValueError('Unsafe metadata member')
   folder=data/('answers' if 'annotations' in member else 'questions');folder.mkdir(exist_ok=True)
   target=folder/member;target.write_bytes(archive.read(member))
   actual=hashlib.sha256(target.read_bytes()).hexdigest()
   if actual!=manifest['files'][member]['sha256']:raise ValueError('Metadata checksum mismatch')
print('ACMA_METADATA_STAGED_OK',flush=True)
''')
        self.execute_marker(unpack, "unpack_metadata", "ACMA_METADATA_STAGED_OK")
        if self.args.stage_images_mode == "rebuild":
            rebuild = self.root / "rebuild_remote_images.py"
            rebuild.write_text('''import hashlib,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path('/content/acma');source=root/'source';source.mkdir(exist_ok=True)
with tarfile.open(root/'source_bundle.tar.gz') as archive:archive.extractall(source,filter='data')
manifest=json.loads((source/'source_manifest.json').read_text())
for name,expected in manifest.items():
 if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source hash mismatch')
subprocess.run([sys.executable,str(source/'tools/colab/prepare_vqa.py'),'--data-dir',str(root/'data'),'--status',str(root/'data_status.json'),'--image-backend','pillow'],check=True)
print('ACMA_REMOTE_IMAGES_REBUILT_OK',flush=True)
''')
            self.execute_marker(rebuild, "rebuild_remote_images", "ACMA_REMOTE_IMAGES_REBUILT_OK", timeout=3600)
        else:
            self.stream(self.repo / "data/vqa_images_224.h5", "/content/acma/data/vqa_images_224.h5", "stage_images")
        self.update(inputs_staged_at=time.time())

    def start_training(self):
        if self.state.get("training_submitted_at"):
            self.update(phase="reconciling_previous_training_submission")
            return
        argv = [self.args.colab, "exec", "-s", self.args.session, "--timeout", "0",
                "--env", "ACMA_MODE=campaign", "--env", "ACMA_SESSION=" + self.args.session,
                "-f", str(self.repo / "tools/colab/bootstrap.py")]
        for attempt in (1, 2):
            # Record intent before spawning. No retry follows an uncertain
            # remote state; require positive evidence that no job is running.
            self.update(training_submitted_at=time.time(), phase="submitting_training",
                        training_submission_attempt=attempt)
            with (self.root / "bootstrap.log").open("ab") as log:
                process = subprocess.Popen(argv, cwd=self.repo, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            self.update(training_client_pid=process.pid, phase="awaiting_pipeline_start")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    receipt = self.fetch_json("/content/acma/pipeline_status.json", "pipeline_start_receipt.json")
                    self.update(phase="training_submitted", pipeline_start_receipt=receipt)
                    return
                except Exception:
                    time.sleep(10)
            if self.productive():
                self.update(phase="training_submitted", pipeline_start_note="Remote job confirmed through process probe")
                return
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            self.update(training_submitted_at=None, phase="confirmed_failed_kernel_handshake")
        raise RuntimeError("Two kernel connection attempts failed before any remote job started")

    def start_cache_mirror(self):
        previous = self.state.get("cache_mirror_pid")
        replacement = self.root / "cache-mirror/restarted_worker.json"
        if replacement.exists():
            candidate = json.loads(replacement.read_text()).get("pid")
            if candidate and "mirror_feature_cache.py" in process_command(candidate):
                previous = candidate
        if previous and "mirror_feature_cache.py" in process_command(previous):
            self.update(cache_mirror_pid=previous)
            return
        script = self.repo / "tools/colab/mirror_feature_cache.py"
        if not script.exists():
            raise FileNotFoundError(f"Feature-cache mirror worker is missing: {script}")
        argv = [sys.executable, str(script), "--session", self.args.session,
                "--local-data-dir", str(self.repo / "data"),
                "--state-dir", str(self.root / "cache-mirror")]
        log = (self.root / "cache_mirror.log").open("ab")
        process = subprocess.Popen(argv, cwd=self.repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
        log.close()
        self.update(cache_mirror_pid=process.pid,
                    cache_mirror_note="Independent worker; durability requires its verified receipt")

    def monitor(self):
        self.update(phase="monitoring")
        argv = [sys.executable, str(self.repo / "tools/colab/monitor.py"), "--session", self.args.session,
                "--local-dir", str(self.root / "monitor")]
        self.run(argv, name="monitor", timeout=self.args.monitor_timeout)
        status = json.loads((self.root / "monitor/monitor_state.json").read_text())
        self.update(monitor_result=status.get("monitor_status"),
                    session_stopped_at=status.get("session_stopped_at"))
        if status.get("monitor_status") != "complete":
            raise RuntimeError("Remote experiment did not complete; see monitor diagnostics")
        self.run([sys.executable, self.repo / "tools/colab/summarize_controls.py",
                  "--campaign-dir", self.root], name="summarize_results", timeout=120)
        self.update(phase="complete", status="complete", completed_at=time.time())

    def productive(self):
        text = self.cli("ssh", "-s", self.args.session, "--", "python3", "-c", PROCESS_PROBE,
                        name="reconcile_productivity", timeout=120)
        matches = re.findall(r"ACMA_WATCHDOG_PROBE (\{[^\n]+\})", text)
        if not matches:
            raise RuntimeError("No trustworthy productivity probe receipt")
        value = json.loads(matches[-1])
        recent = any(record.get("status") == "running" and
                     time.time() - record.get("heartbeat_at", 0) < 180
                     for record in value["statuses"].values())
        return bool(value["processes"] or recent)

    def cleanup_failure(self):
        if not self.state.get("owned_session") or self.state.get("session_stopped_at"):
            return
        try:
            self.cli("log", "-s", self.args.session, "-o", str(self.root / "failure_execution.ipynb"),
                     name="failure_log_export", timeout=120)
        except Exception:
            pass
        try:
            if self.productive():
                self.update(remote_retained_reason="Campaign process still confirmed active")
                return
            self.cli("stop", "-s", self.args.session, name="failure_stop_owned", timeout=120)
            self.update(session_stopped_at=time.time())
        except Exception as error:
            self.update(cleanup_uncertain=str(error), remote_retained_reason="No confirmed idle state")

    def main(self):
        try:
            if self.state.get("training_submitted_at"):
                self.allocate()  # Reconciliation only; no new request permitted.
                self.start_watchdog()
                self.start_cache_mirror()
                self.monitor()
                return
            self.prepare_local()
            bundle, metadata = self.bundle()
            self.allocate()
            self.start_watchdog()
            self.setup()
            self.stage(bundle, metadata)
            self.start_training()
            self.start_cache_mirror()
            self.monitor()
        except BaseException as error:
            self.update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc())
            self.cleanup_failure()
            raise
        finally:
            self.closed.set()
            self.heartbeat.join(timeout=2)


def watchdog(args):
    """Independent idle cleanup; a network failure is never proof of idleness."""
    root = args.campaign_dir.resolve()
    while True:
        try:
            state = json.loads((root / "launch_state.json").read_text())
            if state.get("session_stopped_at"):
                return
            if not (state.get("owned_session") or state.get("allocation_requested_at")):
                time.sleep(60)
                continue
            now = time.time()
            launcher_stale = now - state.get("heartbeat_at", now) >= 600
            # A live local heartbeat only proves the controller process exists.
            # After submission, independently inspect the remote job even if
            # monitor polling and the local heartbeat thread remain alive.
            post_submission_check = bool(state.get("training_submitted_at") and
                                         now - state["training_submitted_at"] >= 600)
            if not launcher_stale and not post_submission_check:
                time.sleep(60)
                continue
            sessions = subprocess.run([args.colab, "sessions"], stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=120)
            if sessions.returncode or args.session not in sessions.stdout:
                atomic_json(root / "watchdog_status.json", {"status": "uncertain_or_vm_absent",
                                                            "checked_at": time.time()})
                time.sleep(60)
                continue
            result = subprocess.run([args.colab, "ssh", "-s", args.session, "--", "python3", "-c", PROCESS_PROBE],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
            matches = re.findall(r"ACMA_WATCHDOG_PROBE (\{[^\n]+\})", result.stdout)
            if result.returncode or not matches:
                raise RuntimeError("Productivity probe unavailable; retaining VM")
            probe = json.loads(matches[-1])
            recent = any(value.get("status") == "running" and time.time() - value.get("heartbeat_at", 0) < 180
                         for value in probe["statuses"].values())
            if probe["processes"] or recent:
                atomic_json(root / "watchdog_status.json", {"status": "productive_remote_process_confirmed",
                                                            "checked_at": time.time(), "probe": probe})
                time.sleep(60)
                continue
            result = subprocess.run([args.colab, "stop", "-s", args.session], stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=120)
            atomic_json(root / "watchdog_status.json", {"status": "stopped_confirmed_idle_owned_vm" if not result.returncode else "stop_failed",
                                                        "checked_at": time.time(), "probe": probe,
                                                        "returncode": result.returncode})
            if not result.returncode:
                return
        except Exception as error:
            atomic_json(root / "watchdog_status.json", {"status": "uncertain_retained", "error": repr(error),
                                                        "checked_at": time.time()})
        time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--session", default="acma-r1-shared-frozen-l4-20261117-run")
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--wait-download-pid", type=int)
    parser.add_argument("--data-python", type=Path)
    parser.add_argument("--local-timeout", type=int, default=24 * 3600)
    parser.add_argument("--transfer-timeout", type=int, default=4 * 3600)
    parser.add_argument("--stage-images-mode", choices=("upload", "rebuild"), default="upload")
    parser.add_argument("--monitor-timeout", type=int, default=30 * 3600)
    parser.add_argument("--watchdog", action="store_true")
    args = parser.parse_args(argv)
    if args.watchdog:
        watchdog(args)
    else:
        Launcher(args).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
