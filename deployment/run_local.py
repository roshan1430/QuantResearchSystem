import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "deployment" / "logs"


def start_process(name: str, command: list[str], cwd: Path) -> subprocess.Popen:
    print(f"[start] {name}: {' '.join(command)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / f"{name}.out.log", "w", encoding="utf-8")
    err = open(LOG_DIR / f"{name}.err.log", "w", encoding="utf-8")
    return subprocess.Popen(command, cwd=str(cwd), env=os.environ.copy(), stdout=out, stderr=err)


def main() -> int:
    processes = [
        start_process(
            "ml-engine",
            [sys.executable, "-m", "uvicorn", "engine.main:app", "--app-dir", "ml-engine", "--host", "127.0.0.1", "--port", "8001"],
            ROOT,
        ),
        start_process(
            "backend",
            [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000"],
            ROOT,
        ),
        start_process("frontend", ["npm.cmd", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"], ROOT / "frontend"),
    ]

    print("[info] frontend: http://127.0.0.1:5173")
    print("[info] backend docs: http://127.0.0.1:8000/docs")
    print("[info] ml engine: http://127.0.0.1:8001/models")
    print(f"[info] service logs: {LOG_DIR}")

    # Give services a brief startup window; fail early with clear logs.
    time.sleep(4.0)
    for process, name in zip(processes, ["ml-engine", "backend", "frontend"]):
        if process.poll() is not None:
            print(f"[error] {name} exited early with code {process.returncode}")
            return process.returncode or 1

    try:
        while True:
            for process, name in zip(processes, ["ml-engine", "backend", "frontend"]):
                if process.poll() is not None:
                    print(f"[error] {name} exited with code {process.returncode}")
                    return process.returncode or 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[stop] shutting down local platform")
        for process in processes:
            process.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
