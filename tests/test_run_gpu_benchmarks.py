from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_gpu_benchmarks.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_gpu_progress_prefix_handles_iteration_separator(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
    write_executable(fake_bin / "make", "#!/usr/bin/env bash\nprintf 'benchmark output\\n'\n")

    log_dir = tmp_path / "logs"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ITERATIONS": "1",
        "LOG_DIR": str(log_dir),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[gpu0 run 1/1] benchmark output" in (
        log_dir / "gpu0_console.log"
    ).read_text(encoding="utf-8")
    assert "[gpu1 run 1/1] benchmark output" in (
        log_dir / "gpu1_console.log"
    ).read_text(encoding="utf-8")
