from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR


def create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_run_paths() -> dict[str, Path]:
    run_dir = create_run_dir()
    timestamp = run_dir.name.removeprefix("run_")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "log_file": run_dir / f"run_log_{timestamp}.txt",
        "summary_file": run_dir / f"run_summary_{timestamp}.txt",
    }


def append_summary(summary_file: Path, line: str) -> None:
    with open(summary_file, "a", encoding="utf8") as handle:
        handle.write(line + "\n")


def log_block(log_file: Path, fn, *args, **kwargs):
    with open(log_file, "a", encoding="utf8") as handle:
        with redirect_stdout(handle):
            return fn(*args, **kwargs)
