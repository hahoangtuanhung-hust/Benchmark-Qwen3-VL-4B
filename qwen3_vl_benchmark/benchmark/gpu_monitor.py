"""GPU sampling uses the existing shared NVML/nvidia-smi monitor."""
from runpy import run_path
from pathlib import Path

if __name__ == "__main__":
    run_path(str(Path(__file__).parents[2] / "scripts" / "gpu_monitor.py"), run_name="__main__")

