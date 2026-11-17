"""No-allocation checks for durable launch ordering, transport, and idle cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import launch


class EndWatch(BaseException):
    pass


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(campaign_dir=self.root, session="owned-test-session", colab="colab")

    def tearDown(self):
        self.temporary.cleanup()

    def write_watchdog_state(self):
        launch.atomic_json(self.root / "launch_state.json", {
            "session": self.args.session, "owned_session": True,
            "phase": "monitor", "heartbeat_at": time.time(),
            "training_submitted_at": time.time() - 601})

    def probe_result(self, processes=None, statuses=None):
        return subprocess.CompletedProcess([], 0, stdout="ACMA_WATCHDOG_PROBE " + json.dumps({
            "time": time.time(), "processes": processes or [], "statuses": statuses or {}}), stderr="")

    def test_fresh_launcher_heartbeat_does_not_hide_idle_remote(self):
        self.write_watchdog_state()
        replies = [subprocess.CompletedProcess([], 0, stdout="[owned-test-session] endpoint", stderr=""),
                   self.probe_result(), subprocess.CompletedProcess([], 0, stdout="stopped", stderr="")]
        with patch.object(launch.subprocess, "run", side_effect=replies) as run:
            launch.watchdog(self.args)
        self.assertEqual([row.args[0][1] for row in run.call_args_list], ["sessions", "ssh", "stop"])
        self.assertEqual(json.loads((self.root / "watchdog_status.json").read_text())["status"],
                         "stopped_confirmed_idle_owned_vm")

    def test_active_upload_or_training_process_is_retained(self):
        self.write_watchdog_state()
        for command in ("python upload ACMA_TRANSFER_HASH_MISMATCH", "python /content/acma/source/tools/colab/reviewer_controls.py"):
            replies = [subprocess.CompletedProcess([], 0, stdout="[owned-test-session] endpoint", stderr=""),
                       self.probe_result(processes=[{"pid": 123, "command": command}])]
            with patch.object(launch.subprocess, "run", side_effect=replies) as run, \
                 patch.object(launch.time, "sleep", side_effect=EndWatch):
                with self.assertRaises(EndWatch):
                    launch.watchdog(self.args)
            self.assertEqual([row.args[0][1] for row in run.call_args_list], ["sessions", "ssh"])

    def test_missed_network_poll_never_stops_vm(self):
        self.write_watchdog_state()
        with patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="network")) as run, \
             patch.object(launch.time, "sleep", side_effect=EndWatch):
            with self.assertRaises(EndWatch):
                launch.watchdog(self.args)
        self.assertEqual(run.call_count, 1)

    def test_binary_receiver_checks_full_size_and_hash_before_rename(self):
        payload = bytes(range(256)) * 257
        target = self.root / "received.bin"
        command = [sys.executable, "-c", launch.RECEIVER, str(target), str(len(payload)),
                   hashlib.sha256(payload).hexdigest()]
        result = subprocess.run(command, input=payload, capture_output=True, check=True)
        self.assertIn(b"ACMA_TRANSFER_OK", result.stdout)
        self.assertEqual(target.read_bytes(), payload)
        target.unlink()
        result = subprocess.run(command, input=payload[:-1], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(target.exists())

    def test_allocation_timeout_reconciles_without_second_new(self):
        controller = launch.Launcher.__new__(launch.Launcher)
        controller.args = self.args
        controller.state = {}
        controller.update = lambda **fields: controller.state.update(fields)
        controller.start_watchdog = lambda: None
        commands = []
        def cli(*args, **kwargs):
            commands.append(args)
            if args[0] == "new":
                raise subprocess.TimeoutExpired(args, 600)
            return "[owned-test-session] endpoint" if len(commands) > 1 else "No active sessions"
        controller.cli = cli
        controller.allocate()
        self.assertEqual(sum(command[0] == "new" for command in commands), 1)
        self.assertTrue(controller.state["owned_session"])

    def test_existing_unowned_session_is_never_reused_or_stopped(self):
        controller = launch.Launcher.__new__(launch.Launcher)
        controller.args = self.args
        controller.state = {}
        controller.cli = lambda *args, **kw: "[owned-test-session] endpoint"
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            controller.allocate()

    def test_failure_cleanup_preserves_confirmed_running_or_uncertain_vm(self):
        for probe in (True, RuntimeError("network failed")):
            controller = launch.Launcher.__new__(launch.Launcher)
            controller.args, controller.root = self.args, self.root
            controller.state = {"owned_session": True}
            controller.update = lambda **fields: controller.state.update(fields)
            commands = []
            controller.cli = lambda *args, **kw: commands.append(args)
            def productive():
                if isinstance(probe, Exception):
                    raise probe
                return probe
            controller.productive = productive
            controller.cleanup_failure()
            self.assertFalse(any(command[0] == "stop" for command in commands))

    def submission_controller(self):
        controller = launch.Launcher.__new__(launch.Launcher)
        controller.args = self.args
        controller.repo, controller.root = self.root, self.root
        controller.state = {}
        controller.update = lambda **fields: controller.state.update(fields)
        return controller

    def test_setup_uses_bounded_ssh_with_required_marker(self):
        controller = self.submission_controller()
        source = self.root / 'probe.py'
        source.write_text("print('PROBE_OK')")
        calls = []
        def cli(*args, **kwargs):
            calls.append((args, kwargs))
            (self.root / 'probe.log').write_text('PROBE_OK')
        controller.cli = cli
        controller.execute_marker(source, 'probe', 'PROBE_OK')
        self.assertEqual(calls[0][0][0], 'ssh')
        self.assertEqual(calls[0][1]['timeout'], 120)

    def test_no_duplicate_submission_when_remote_process_is_confirmed(self):
        controller = self.submission_controller()
        controller.productive = lambda: True
        process = MagicMock(pid=12345)
        with patch.object(launch.subprocess, 'Popen', return_value=process) as start, \
             patch.object(launch.time, 'monotonic', side_effect=[0, 121]):
            controller.start_training()
        self.assertEqual(start.call_count, 1)
        self.assertEqual(controller.state['phase'], 'training_submitted')

    def test_kernel_handshake_retry_is_bounded_after_confirmed_no_work(self):
        controller = self.submission_controller()
        controller.productive = lambda: False
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with patch.object(launch.subprocess, 'Popen', return_value=process) as start, \
             patch.object(launch.time, 'monotonic', side_effect=[0, 121, 122, 243]), \
             patch.object(launch.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'Two kernel connection attempts'):
                controller.start_training()
        self.assertEqual(start.call_count, 2)
        self.assertEqual(kill.call_count, 2)
        self.assertIsNone(controller.state['training_submitted_at'])


if __name__ == "__main__":
    unittest.main()
