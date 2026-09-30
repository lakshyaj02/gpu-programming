import os
from pathlib import Path
import subprocess

import pytest


def _benchmark_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        os.environ.get("ATTENTION_BENCHMARK"),
        root / "build" / "week08_attention" / "week08_attention_benchmark",
        root / "build" / "week08_attention_benchmark",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    pytest.skip("Build week08_attention_benchmark or set ATTENTION_BENCHMARK")


@pytest.mark.parametrize(
    ("m", "n", "d", "dv", "causal"),
    [
        (8, 8, 4, 4, False),
        (8, 8, 4, 4, True),
        (7, 11, 5, 9, False),
        (7, 11, 5, 9, True),
        (33, 35, 17, 19, False),
    ],
)
def test_attention_against_cpu_reference(m, n, d, dv, causal):
    result = subprocess.run(
        [str(_benchmark_path()), str(m), str(n), str(d), str(dv), str(int(causal))],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "correctness=PASS" in result.stdout
