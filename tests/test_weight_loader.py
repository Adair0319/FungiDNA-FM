import os
import tempfile
import unittest
from pathlib import Path

import yaml

from fungidna.invoke.handlers.base import WeightLoader, WeightsNotFoundError


class TestWeightLoader(unittest.TestCase):
    def setUp(self):
        # Point the weights config at a nonexistent temp file by default so
        # tests never read the user's real ~/.config/fungidna/weights.yaml.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmpdir.name) / "weights.yaml"
        os.environ["FUNGIDNA_WEIGHTS_CONFIG"] = str(self.config_path)

    def tearDown(self):
        os.environ.pop("FUNGIDNA_TAXONOMY_WEIGHTS", None)
        os.environ.pop("FUNGIDNA_WEIGHTS_CONFIG", None)
        self.tmpdir.cleanup()

    def write_config(self, mapping):
        self.config_path.write_text(yaml.safe_dump(mapping))

    def test_registry_wins(self):
        wl = WeightLoader({"taxonomy": "/tmp/tax.pt"})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/tax.pt"))

    def test_env_fallback(self):
        os.environ["FUNGIDNA_TAXONOMY_WEIGHTS"] = "/tmp/env.pt"
        wl = WeightLoader({})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/env.pt"))

    def test_missing_raises(self):
        wl = WeightLoader({})
        with self.assertRaises(WeightsNotFoundError) as cm:
            wl.resolve("splice_site")
        self.assertEqual(cm.exception.task, "splice_site")

    def test_yaml_fallback(self):
        self.write_config({"splice_site": "/tmp/yaml.pt"})
        wl = WeightLoader({})
        self.assertEqual(wl.resolve("splice_site"), Path("/tmp/yaml.pt"))

    def test_registry_beats_env_beats_yaml(self):
        os.environ["FUNGIDNA_TAXONOMY_WEIGHTS"] = "/tmp/env.pt"
        self.write_config({"taxonomy": "/tmp/yaml.pt"})
        wl = WeightLoader({"taxonomy": "/tmp/registry.pt"})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/registry.pt"))

    def test_missing_yaml_file_raises(self):
        # FUNGIDNA_WEIGHTS_CONFIG points at a file that does not exist.
        wl = WeightLoader({})
        with self.assertRaises(WeightsNotFoundError):
            wl.resolve("cds")

    def test_malformed_yaml_raises_with_message(self):
        self.config_path.write_text("taxonomy: [1, 2\n")
        wl = WeightLoader({})
        with self.assertRaises(WeightsNotFoundError) as cm:
            wl.resolve("cds")
        self.assertIn("unreadable", str(cm.exception))
        self.assertEqual(cm.exception.task, "cds")


if __name__ == "__main__":
    unittest.main()
