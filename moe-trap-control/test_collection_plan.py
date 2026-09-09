"""Check that coverage extensions preserve frozen identities and screening counts."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from analyze_collection_experiment import completion_counts
from collection_protocol import HERE, load_plan
from collection_storage import atomic_json


class FrozenPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / "design/experiment_long_batch1_plan_20260908.json").read_text())
        with (HERE / "design/collection_manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        cls.screen = next(row for row in rows if row["suite"] == "libero_10" and row["benchmark"] == "pro"
                          and row["screen"] == "1")
        cls.extension = next(row for row in rows if row["variant_id"] == cls.screen["variant_id"]
                             and row["init_index"] == cls.screen["init_index"] and row["policy_seed_index"] == "1")

    def validate(self, task, partition=None, pending=False):
        plan = dict(self.plan, tasks=[dict(task)])
        if partition is not None:
            plan["sample_partition"] = partition
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            atomic_json(path, plan)
            return load_plan(path, "long", pending_only=pending)

    def test_extension_requires_explicit_partition(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            self.validate(self.extension)
        accepted = self.validate(self.extension, "coverage_extension")
        self.assertEqual(accepted["tasks"][0]["noise_seed"], self.extension["noise_seed"])

    def test_partition_cannot_reclassify_screen(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            self.validate(self.screen, "coverage_extension")
        with self.assertRaisesRegex(ValueError, "partition"):
            self.validate(self.extension, "unknown")

    def test_original_seed_is_immutable(self):
        task = dict(self.extension, noise_seed=str(int(self.extension["noise_seed"]) + 1))
        with self.assertRaisesRegex(ValueError, "noise_seed"):
            self.validate(task, "coverage_extension")

    def test_completed_screen_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "already audited"):
            self.validate(self.plan["tasks"][0], pending=True)

    def test_extension_does_not_reduce_screen_remaining(self):
        counts = completion_counts([self.screen["main_id"], self.extension["main_id"]])
        self.assertEqual(counts["completed_count"], 2)
        self.assertEqual(counts["completed_screen_count"], 1)
        self.assertEqual(counts["completed_extension_count"], 1)
        self.assertEqual(counts["screen_planned_remaining"], 3799)
        self.assertEqual(counts["coverage_planned_remaining"], 14028)


if __name__ == "__main__":
    unittest.main()
