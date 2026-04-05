#!/usr/bin/env python3
"""
MIND EASE — Latency Benchmark
Measures per-layer processing time across N iterations.
"""
import argparse
import json
import struct
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_test_mode(binary: Path):
    """Run the engine self-test and report."""
    result = subprocess.run([str(binary), "--test"], capture_output=True, text=True)
    print(result.stderr)
    return result.returncode == 0

def generate_test_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic sine-wave audio as PCM16 bytes."""
    import struct, math
    n_samples = int(duration_s * sample_rate)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # 440Hz tone at moderate amplitude
        val = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * t))
        samples.append(val)
    return struct.pack(f"<{n_samples}h", *samples)

def send_ipc(proc, data: str):
    encoded = data.encode("utf-8")
    header = struct.pack("<I", len(encoded))
    proc.stdin.write(header + encoded)
    proc.stdin.flush()

def send_ipc_bytes(proc, data: bytes):
    header = struct.pack("<I", len(data))
    proc.stdin.write(header + data)
    proc.stdin.flush()

def recv_ipc(proc) -> dict:
    header = proc.stdout.read(4)
    if len(header) < 4:
        return {}
    length = struct.unpack("<I", header)[0]
    data = proc.stdout.read(length)
    return json.loads(data.decode("utf-8"))

def main():
    parser = argparse.ArgumentParser(description="MIND EASE latency benchmark")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--safety-test", action="store_true")
    args = parser.parse_args()

    binary = ROOT / "engine" / "build" / "mindease_engine"

    if not binary.exists():
        print(f"Engine binary not found at {binary}")
        print("Run scripts/build_engine.sh first, or test without binary (mock mode)")
        sys.exit(0)

    print(f"Running {args.iterations} iterations against {binary}")
    print("=" * 60)

    audio_data = generate_test_audio(2.0)
    results = {"stt": [], "emotion": [], "llm": [], "total": []}

    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    # Wait for ready
    msg = recv_ipc(proc)
    if msg.get("status") != "ready":
        print("Engine not ready")
        proc.kill()
        sys.exit(1)
    print("Engine ready.")

    for i in range(args.iterations):
        t_start = time.perf_counter()

        # Send audio signal then audio data
        send_ipc(proc, json.dumps({"type": "audio"}))
        send_ipc_bytes(proc, audio_data)

        # Collect results
        iteration_results = {}
        timeout = time.time() + 5.0
        while time.time() < timeout:
            msg = recv_ipc(proc)
            if not msg:
                break
            mtype = msg.get("type", "")
            if mtype == "transcript":
                iteration_results["stt"] = msg.get("duration_ms", 0)
            elif mtype == "emotion":
                iteration_results["emotion"] = msg.get("duration_ms", 0)
            elif mtype == "response":
                iteration_results["llm"] = msg.get("duration_ms", 0)
                break
            elif mtype == "safety_alert":
                iteration_results["llm"] = 0  # bypassed
                break

        t_end = time.perf_counter()
        total_ms = int((t_end - t_start) * 1000)

        for k, v in iteration_results.items():
            if k in results:
                results[k].append(v)
        results["total"].append(total_ms)

        print(f"  [{i+1:2d}] STT={iteration_results.get('stt',0)}ms "
              f"Emotion={iteration_results.get('emotion',0)}ms "
              f"LLM={iteration_results.get('llm',0)}ms "
              f"Total={total_ms}ms {'✓' if total_ms < 600 else '✗'}")

    proc.kill()

    print()
    print("=" * 60)
    print("Results (avg / p95):")
    for layer, vals in results.items():
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        p95 = sorted(vals)[int(len(vals) * 0.95)]
        target = {"stt": 180, "emotion": 80, "llm": 250, "total": 600}.get(layer, 999)
        status = "✓" if avg <= target else "✗"
        print(f"  {layer:<10} avg={avg:.0f}ms  p95={p95}ms  target={target}ms  {status}")

if __name__ == "__main__":
    main()
