"""Smoke tests for the command-line interface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from darcy.cli import main


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"].count(".") == 2


def test_solve_generated_field(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "--grid", "48", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "converged     True" in out


def test_solve_from_file(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    field = tmp_path / "a.npy"
    np.save(field, np.full((32, 32), 4.0))
    assert main(["solve", "--input", str(field)]) == 0
    assert "32 x 32" in capsys.readouterr().out


def test_solve_writes_output(tmp_path) -> None:
    out = tmp_path / "sol.npz"
    assert main(["solve", "--grid", "32", "--output", str(out)]) == 0
    data = np.load(out)
    assert data["a"].shape == (32, 32) and data["u"].shape == (32, 32)


def test_dataset_generation(tmp_path) -> None:
    out = tmp_path / "train.npz"
    assert main(["dataset", "--n-samples", "8", "--grid", "32", "--output", str(out)]) == 0
    data = np.load(out)
    assert data["a"].shape == (8, 32, 32)
    assert data["u"].dtype == np.float32


def test_invalid_input_returns_error_code(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    field = tmp_path / "bad.npy"
    np.save(field, np.full((8, 8), -1.0))  # non-positive permeability
    assert main(["solve", "--input", str(field)]) == 2
    assert "error" in capsys.readouterr().err
