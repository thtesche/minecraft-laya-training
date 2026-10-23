"""Tests for the CLI helpers (`mlaya.cli`). Stdlib unittest, no extra deps."""
from __future__ import annotations

import unittest

from mlaya.cli import _train_base
from mlaya.config import Settings


class TrainBaseTest(unittest.TestCase):
    """The fine-tune starts from the checkpoint the loop serves, not the bare Hub alias."""

    def test_a_re_pointed_name_resolves_to_its_checkpoint(self) -> None:
        settings = Settings()
        settings.laya.extra_models = {"english": "/ckpt/minecraft-20261014-104114"}
        self.assertEqual(_train_base(settings), "/ckpt/minecraft-20261014-104114")

    def test_without_a_registration_the_name_stands(self) -> None:
        self.assertEqual(_train_base(Settings()), Settings().train.base_model)

    def test_an_explicit_base_wins(self) -> None:
        settings = Settings()
        settings.laya.extra_models = {"english": "/ckpt/a"}
        self.assertEqual(_train_base(settings, "/ckpt/b"), "/ckpt/b")


if __name__ == "__main__":
    unittest.main()
