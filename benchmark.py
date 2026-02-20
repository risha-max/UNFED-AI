"""
UNFED AI Benchmark — measures inference latency for before/after optimization comparison.

Connects to a running pipeline (registry + nodes), runs a fixed prompt N times,
and reports time-to-first-token (TTFT), per-token decode latency, and throughput.

Usage:
    python benchmark.py                    # default: 10 runs, 50 tokens
    python benchmark.py --runs 20          # more runs for better stats
    python benchmark.py --max-tokens 100   # longer generation
    python benchmark.py --registry host:50050 --model Qwen/Qwen2.5-0.5B
"""

import argparse
import statistics
import sys
import time

from client.client import UnfedClient


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_single(client: UnfedClient, prompt: str, max_tokens: int) -> dict:
    """Run a single generation and return timing metrics."""
    token_times = []
    gen_start = time.perf_counter()
    ttft = None

    for token_text in client.generate(prompt, max_new_tokens=max_tokens):
        now = time.perf_counter()
        if ttft is None:
            ttft = now - gen_start
        token_times.append(now)

    total_time = time.perf_counter() - gen_start
    num_tokens = len(token_times)

    decode_latencies = []
    for i in range(1, len(token_times)):
        decode_latencies.append(token_times[i] - token_times[i - 1])

    return {
        "ttft": ttft or 0.0,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "tok_per_sec": num_tokens / total_time if total_time > 0 else 0.0,
        "decode_latencies": decode_latencies,
        "mean_decode": statistics.mean(decode_latencies) if decode_latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="UNFED AI Inference Benchmark")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of benchmark runs (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Tokens to generate per run (default: 50)")
    parser.add_argument("--prompt", type=str,
                        default="Explain the theory of relativity in simple terms.",
                        help="Fixed prompt for benchmarking")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs to discard (default: 1)")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID")
    args = parser.parse_args()

    total_runs = args.warmup + args.runs

    print(f"UNFED AI Benchmark")
    print(f"  Prompt: {args.prompt[:60]}{'...' if len(args.prompt) > 60 else ''}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs: {args.runs} (+ {args.warmup} warmup)")
    print()

    client = UnfedClient(
        registry_address=args.registry,
        model_id=args.model,
    )

    all_results = []
    for i in range(total_runs):
        is_warmup = i < args.warmup
        label = f"warmup {i + 1}" if is_warmup else f"run {i + 1 - args.warmup}/{args.runs}"
        sys.stdout.write(f"  {label}...")
        sys.stdout.flush()

        result = run_single(client, args.prompt, args.max_tokens)
        tag = " (discarded)" if is_warmup else ""
        print(f" {result['num_tokens']} tokens, {result['tok_per_sec']:.1f} tok/s, "
              f"TTFT={result['ttft'] * 1000:.0f}ms{tag}")

        if not is_warmup:
            all_results.append(result)

    client.close()

    if not all_results:
        print("\nNo results to report.")
        return

    ttfts = [r["ttft"] * 1000 for r in all_results]
    tok_per_secs = [r["tok_per_sec"] for r in all_results]
    all_decode = []
    for r in all_results:
        all_decode.extend([d * 1000 for d in r["decode_latencies"]])

    print()
    print("=" * 60)
    print(f"  TTFT:       mean={statistics.mean(ttfts):.0f}ms  "
          f"min={min(ttfts):.0f}ms  max={max(ttfts):.0f}ms  "
          f"p95={percentile(ttfts, 95):.0f}ms")
    if all_decode:
        print(f"  Decode:     mean={statistics.mean(all_decode):.1f}ms/tok  "
              f"min={min(all_decode):.1f}ms  max={max(all_decode):.1f}ms  "
              f"p95={percentile(all_decode, 95):.1f}ms")
    print(f"  Throughput: {statistics.mean(tok_per_secs):.1f} tok/s  "
          f"min={min(tok_per_secs):.1f}  max={max(tok_per_secs):.1f}  "
          f"p95={percentile(tok_per_secs, 95):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
