from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from jepa_pu.config import load_config
from jepa_pu.data import extract_answer
from jepa_pu.llada import make_candidates


class ConfigTests(unittest.TestCase):
    def test_environment_expansion_and_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "paths:\n  model: ${TEST_MODEL}\n  output_dir: ${MISSING_TEST_ROOT:-artifacts}/out\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_MODEL": "/models/llada"}, clear=False):
                loaded = load_config(config)
            self.assertEqual(loaded["paths"]["model"], "/models/llada")
            self.assertEqual(loaded["paths"]["output_dir"], "artifacts/out")

    def test_missing_required_environment_variable_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("value: ${DEFINITELY_UNSET_JEPA_TEST_VAR}\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DEFINITELY_UNSET_JEPA_TEST_VAR", None)
                with self.assertRaisesRegex(ValueError, "DEFINITELY_UNSET_JEPA_TEST_VAR"):
                    load_config(config)


class DataTests(unittest.TestCase):
    def test_extract_answer_prefers_gsm8k_marker(self) -> None:
        self.assertEqual(extract_answer("Work: 12 + 3 = 15. #### $1,234.50"), "1234.5")

    def test_extract_answer_handles_fraction(self) -> None:
        self.assertEqual(extract_answer("Final answer: 3/4"), "0.75")


class CandidateTests(unittest.TestCase):
    def test_candidates_are_confidence_prefixes(self) -> None:
        sampled = torch.tensor([[10, 11, 12, 13]])
        confidence = torch.tensor([[0.2, 0.9, 0.5, 0.7]])
        margin = torch.zeros_like(confidence)
        active = torch.ones_like(sampled, dtype=torch.bool)
        candidates = make_candidates(sampled, confidence, margin, active, [1, 3])
        self.assertEqual(candidates[0].positions.tolist(), [1])
        self.assertEqual(candidates[1].positions.tolist(), [1, 3, 2])


if __name__ == "__main__":
    unittest.main()
