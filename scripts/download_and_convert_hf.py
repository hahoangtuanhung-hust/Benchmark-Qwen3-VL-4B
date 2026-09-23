import yaml
import os
import subprocess
import sys
import logging
import argparse
import shutil
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def is_valid_mmproj(mmproj_path: str, fp16_path: str) -> bool:
    """Check that a cached file is a vision projector, not a text model.

    Older converter versions can reject ``--mmproj``.  The previous fallback
    converted the language model into the requested mmproj filename, which
    leaves a plausible-looking but unusable cache entry behind.  The
    projector metadata is written near the GGUF header, so this check is
    inexpensive and does not need to scan the multi-gigabyte file.
    """
    try:
        mmproj_size = os.path.getsize(mmproj_path)
        fp16_size = os.path.getsize(fp16_path)
        if mmproj_size <= 0 or abs(mmproj_size - fp16_size) < 1024 * 1024:
            return False

        with open(mmproj_path, "rb") as handle:
            header = handle.read(4 * 1024 * 1024).lower()
        # Qwen3-VL uses the generic CLIP projector key for its vision encoder.
        return header.startswith(b"gguf") and b"clip.projector_type" in header
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Only download/convert this specific model label")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, "configs", "models.yaml")

    # Detect llama.cpp directory (prefer LLAMA_CPP_DIR env, then local)
    llamacpp_dir = os.environ.get("LLAMA_CPP_DIR", os.path.join(project_dir, "llama.cpp"))
    convert_script = os.path.join(llamacpp_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llamacpp_dir, "build", "bin", "llama-quantize")

    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if not os.path.exists(convert_script):
        logging.error(f"convert_hf_to_gguf.py not found at {convert_script}. Did you build llama.cpp?")
        sys.exit(1)

    if not os.path.exists(quantize_bin):
        logging.error(f"llama-quantize not found at {quantize_bin}. Did you build llama.cpp?")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = []
    base_cfg = config.get("base_model", {})
    hf_repo = base_cfg.get("hf_repo", "")
    hf_cache_dir = os.path.join(project_dir, base_cfg.get("hf_cache_dir", "models/base_hf_cache"))

    # =========================================================================
    # Step 1: Download the base HF model (only once, cached in Volume)
    # =========================================================================
    if not os.path.exists(hf_cache_dir) or not os.listdir(hf_cache_dir):
        logging.info(f"Downloading base model: {hf_repo} ...")
        try:
            snapshot_download(repo_id=hf_repo, local_dir=hf_cache_dir)
            logging.info(f"Base model downloaded to {hf_cache_dir}")
        except Exception as e:
            logging.error(f"Failed to download base model {hf_repo}: {e}")
            sys.exit(1)
    else:
        logging.info(f"Base model already cached at {hf_cache_dir}, skipping download.")

    # =========================================================================
    # Step 2: Convert base model to F16 GGUF (the "golden master")
    # =========================================================================
    fp16_cfg = None
    for p in config.get("requested_precisions", []):
        if p.get("quantize_type") == "F16":
            fp16_cfg = p
            break

    if fp16_cfg is None:
        logging.error("No F16 entry found in requested_precisions. Cannot proceed.")
        sys.exit(1)

    fp16_dir = os.path.join(project_dir, fp16_cfg["directory"])
    fp16_path = os.path.join(fp16_dir, fp16_cfg["filename"])
    os.makedirs(fp16_dir, exist_ok=True)

    if not os.path.exists(fp16_path):
        logging.info(f"Converting base model to GGUF F16: {fp16_path}")
        convert_cmd = [
            sys.executable, convert_script, hf_cache_dir,
            "--outfile", fp16_path,
            "--outtype", "f16"
        ]
        try:
            subprocess.run(convert_cmd, check=True)
            logging.info(f"[OK] F16 GGUF created: {fp16_path}")
            results.append({"label": "FP16", "target": fp16_cfg["filename"], "success": True})
        except subprocess.CalledProcessError:
            logging.error(f"Failed to convert base model to F16 GGUF")
            results.append({"label": "FP16", "target": fp16_cfg["filename"], "success": False, "error": "convert_hf_to_gguf failed"})
            sys.exit(1)
    else:
        logging.info(f"F16 GGUF already exists: {fp16_path}")
        results.append({"label": "FP16", "target": fp16_cfg["filename"], "success": True})

    # =========================================================================
    # Step 3: Convert base model to mmproj GGUF (vision encoder)
    # =========================================================================
    mmproj_cfg = config.get("mmproj", {})
    mmproj_dir = os.path.join(project_dir, mmproj_cfg.get("directory", "models/mmproj"))
    mmproj_path = os.path.join(mmproj_dir, mmproj_cfg.get("filename", "mmproj.gguf"))
    os.makedirs(mmproj_dir, exist_ok=True)

    if os.path.exists(mmproj_path) and not is_valid_mmproj(mmproj_path, fp16_path):
        logging.warning(
            f"Cached mmproj is invalid (likely a text-model conversion): {mmproj_path}. "
            "Removing it so it can be rebuilt."
        )
        os.remove(mmproj_path)

    if not os.path.exists(mmproj_path):
        logging.info(f"Converting mmproj (vision encoder) to GGUF: {mmproj_path}")
        convert_cmd = [
            sys.executable, convert_script, hf_cache_dir,
            "--outfile", mmproj_path,
            "--outtype", "f16",
            "--mmproj",
        ]
        try:
            subprocess.run(convert_cmd, check=True)
            if not is_valid_mmproj(mmproj_path, fp16_path):
                raise RuntimeError("converter produced a file without projector metadata")
            logging.info(f"[OK] mmproj GGUF created: {mmproj_path}")
            results.append({"label": "mmproj", "target": mmproj_cfg["filename"], "success": True})
        except subprocess.CalledProcessError:
            logging.error(
                "mmproj conversion failed. The installed convert_hf_to_gguf.py "
                "must support --mmproj; refusing to create a text-model fallback."
            )
            if os.path.exists(mmproj_path):
                os.remove(mmproj_path)
            results.append({"label": "mmproj", "target": mmproj_cfg["filename"], "success": False, "error": "mmproj conversion failed"})
            sys.exit(1)
        except RuntimeError as exc:
            logging.error(f"mmproj conversion produced an invalid artifact: {exc}")
            if os.path.exists(mmproj_path):
                os.remove(mmproj_path)
            results.append({"label": "mmproj", "target": mmproj_cfg["filename"], "success": False, "error": str(exc)})
            sys.exit(1)
    else:
        logging.info(f"mmproj GGUF already exists: {mmproj_path}")
        results.append({"label": "mmproj", "target": mmproj_cfg["filename"], "success": True})

    # =========================================================================
    # Step 4: Use llama-quantize to create all other quantized variants
    # =========================================================================
    for model_cfg in config.get("requested_precisions", []):
        quant_type = model_cfg.get("quantize_type", "")
        label = model_cfg.get("label", "")

        # Skip F16 — already done above
        if quant_type == "F16":
            continue

        # If a single model is requested, filter
        if args.model:
            if args.model.lower() not in label.lower() and args.model.lower() not in quant_type.lower():
                continue

        target_dir = os.path.join(project_dir, model_cfg["directory"])
        target_path = os.path.join(target_dir, model_cfg["filename"])
        os.makedirs(target_dir, exist_ok=True)

        if os.path.exists(target_path):
            logging.info(f"[{label}] Already exists: {target_path}")
            results.append({"label": label, "target": model_cfg["filename"], "success": True})
            continue

        logging.info(f"[{label}] Quantizing F16 -> {quant_type}: {target_path}")
        quantize_cmd = [quantize_bin, fp16_path, target_path, quant_type]

        try:
            subprocess.run(quantize_cmd, check=True)
            logging.info(f"[OK] {label} GGUF created: {target_path}")
            results.append({"label": label, "target": model_cfg["filename"], "success": True})
        except subprocess.CalledProcessError:
            logging.error(f"[FAILED] Quantization failed for {label} ({quant_type})")
            results.append({"label": label, "target": model_cfg["filename"], "success": False, "error": f"llama-quantize {quant_type} failed"})

    # =========================================================================
    # Summary
    # =========================================================================
    logging.info("--- Conversion Summary ---")
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        logging.info(f"[{status}] {r['label']} -> {r['target']}")
        if not r["success"]:
            logging.info(f"      Reason: {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
