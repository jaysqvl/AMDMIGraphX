"""Check profile and publication guards without requiring a Docker daemon."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "owned"))
from gfx90c_code import MACHINES
from runtime import PROFILES, Runtime, memory_bytes, validate_profile


class Profiles(unittest.TestCase):
    def test_each_profile_matches_a_pinned_recipe(self):
        for name in PROFILES:
            profile = validate_profile(name)
            lock = json.loads(
                (HERE / profile["recipe"] / "source-lock.json").read_text()
            )
            self.assertEqual(profile["rocm"], lock["runtime"]["rocm"])
            self.assertTrue(set(profile["gpu_targets"]) <= MACHINES.keys())
            self.assertEqual(profile["status"], "experimental")

    def test_legacy_cannot_accidentally_enable_unsupported_backends(self):
        bad = copy.deepcopy(PROFILES["rocm-10.1-gfx90c"])
        bad["hipblaslt"] = True
        with (
            patch.dict(PROFILES, broken=bad),
            self.assertRaisesRegex(ValueError, "disables CK"),
        ):
            validate_profile("broken")

    def test_empty_duplicate_and_shell_targets_are_rejected(self):
        for targets in ([], ["gfx900", "gfx900"], ["gfx90c;touch /tmp/bad"]):
            bad = copy.deepcopy(PROFILES["rocm-10.1-gfx90c"])
            bad["gpu_targets"] = targets
            with patch.dict(PROFILES, broken=bad), self.assertRaises(ValueError):
                validate_profile("broken")

    def test_plan_does_not_contact_docker(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "runtime.py"),
                "plan",
                "--profile",
                "rocm-10.1-vega",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout)["gpu_targets"], ["gfx900", "gfx906", "gfx90c"]
        )

    def test_explicit_memory_units(self):
        self.assertEqual(memory_bytes("12g"), 12 * 1024**3)
        for invalid in ("0", "12gb", "-1", "unlimited"):
            with self.assertRaises(ValueError):
                memory_bytes(invalid)


class Publication(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime.__new__(Runtime)
        self.runtime.journal = {"runtime_recipe_sha256": "expected", "stages": {}}

    def record(self, state, recipe="expected"):
        self.runtime.journal["stages"]["compiler"] = {
            "state": state,
            "runtime_recipe_sha256": recipe,
        }

    def test_detached_running_zero_exit_and_oom_are_failures(self):
        for state in (
            {},
            "running",
            {"Status": "running", "ExitCode": 0},
            {"Status": "exited", "ExitCode": 1},
            {"Status": "exited", "ExitCode": 0, "OOMKilled": True},
        ):
            self.record(state)
            with self.assertRaises(RuntimeError):
                self.runtime.require_passed(["compiler"])

    def test_different_recipe_is_not_publishable(self):
        self.record({"Status": "exited", "ExitCode": 0}, recipe="old")
        with self.assertRaisesRegex(RuntimeError, "different recipe"):
            self.runtime.require_passed(["compiler"])

    def test_success_requires_matching_recipe(self):
        self.record({"Status": "exited", "ExitCode": 0})
        self.runtime.require_passed(["compiler"])


if __name__ == "__main__":
    unittest.main()
