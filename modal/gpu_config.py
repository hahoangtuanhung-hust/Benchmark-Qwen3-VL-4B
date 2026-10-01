from __future__ import annotations

# This benchmark deliberately pins one GPU family so results from different
# architectures cannot accidentally be merged into the same report.
GPU_KEY = "l4"
MODAL_GPU = "L4"
EXPECTED_SM = "89"
EXPECTED_ARCH = "Ada"
EXPECTED_GPU_LABEL = "NVIDIA L4"
