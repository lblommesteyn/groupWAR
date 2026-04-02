from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_cli_optimizes_example_nhl_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "nhl_selected.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groupwar.cli",
            "optimize",
            "--league",
            "nhl",
            "--players",
            str(ROOT / "examples" / "nhl_players.csv"),
            "--score-column",
            "war",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )

    assert output_path.exists()
    assert "league=nhl" in completed.stdout
