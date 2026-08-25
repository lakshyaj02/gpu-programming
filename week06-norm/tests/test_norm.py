from pathlib import Path


def test_norm_sources_are_present():
    project_root = Path(__file__).resolve().parents[1]
    expected_sources = {
        "rmsnorm_naive.cu",
        "rmsnorm_warp.cu",
        "rmsnorm_half.cu",
        "rmsnorm_vectorized.cu",
        "layernorm.cu",
        "fused_residual_rmsnorm.cu",
        "main.cu",
    }

    assert {path.name for path in (project_root / "src").glob("*.cu")} == expected_sources
