#!/usr/bin/env python3
"""
Wait for llama-server to become healthy.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §31 (orchestration)
"""

import argparse
import json
import sys
import time

import requests


def wait_for_server(host: str, port: int, timeout: int, interval: float = 2.0) -> bool:
    """Poll /health endpoint until server is ready or timeout."""
    url = f"http://{host}:{port}/health"
    start = time.time()
    attempt = 0

    print(f"[INFO] Waiting for server at {url} (timeout={timeout}s)...")

    while time.time() - start < timeout:
        attempt += 1
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    status = data.get("status", "unknown")
                except (json.JSONDecodeError, ValueError):
                    status = "ok"

                if status in ("ok", "no slot available"):
                    elapsed = time.time() - start
                    print(f"[OK] Server healthy after {elapsed:.1f}s (attempt {attempt})")
                    return True
                else:
                    print(f"  [{attempt}] Status: {status}, waiting...")
            else:
                print(f"  [{attempt}] HTTP {resp.status_code}, waiting...")
        except requests.ConnectionError:
            print(f"  [{attempt}] Connection refused, waiting...")
        except requests.Timeout:
            print(f"  [{attempt}] Timeout, waiting...")
        except Exception as e:
            print(f"  [{attempt}] Error: {e}, waiting...")

        time.sleep(interval)

    elapsed = time.time() - start
    print(f"[FAIL] Server not ready after {elapsed:.1f}s ({attempt} attempts)")
    return False


def main():
    parser = argparse.ArgumentParser(description="Wait for llama-server health check")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds")
    args = parser.parse_args()

    ok = wait_for_server(args.host, args.port, args.timeout, args.interval)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
