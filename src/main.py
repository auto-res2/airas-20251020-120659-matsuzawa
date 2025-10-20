"""src/main.py – orchestration wrapper that launches src.train as subprocess."""
from __future__ import annotations

import subprocess
import sys
from typing import List

def main() -> None:
    # Pass all command line arguments directly to train.py
    args: List[str] = sys.argv[1:]
    
    cmd: List[str] = [sys.executable, "-u", "-m", "src.train", *args]
    
    print("[main] Launching subprocess:\n  ", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
