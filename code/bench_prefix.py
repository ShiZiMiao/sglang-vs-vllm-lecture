#!/usr/bin/env python3
"""
Minimal prefix-sharing benchmark for OpenAI-compatible LLM endpoints.

It sends N streaming chat requests and reports client-side TTFT / E2E
percentiles plus an approximate TPOT. Two modes:

  shared : every request uses the same long prefix, plus a unique suffix
  unique : every request uses its own random prefix (no prefix reuse)

Example:
  python bench_prefix.py --base-url http://localhost:8000/v1 --model qwen \
      --mode shared --shared-words 1500 --suffix-words 200 \
      --num-requests 64 --concurrency 16 --max-tokens 64

Requires: pip install openai
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Please install the OpenAI client: pip install openai", file=sys.stderr)
    raise SystemExit(1)

WORDS = (
    "system model token cache memory batch request prompt decode prefill "
    "attention layer tensor kernel latency throughput schedule block prefix "
    "suffix stream server client network compute storage vector matrix router "
    "worker queue budget sample context window grammar schema output input "
    "engine runtime radix paged chunk graph parallel pipeline cluster gpu cpu "
    "bandwidth latency policy eviction hash tree node leaf branch split merge "
    "agent tool retrieval document conversation history session tenant adapter "
    "quantization precision float integer benchmark workload arrival service "
    "deadline priority fairness isolation replica shard transfer connector "
    "prefill decode speculative draft verify accept reject sample temperature "
    "topology interconnect fabric memory pool slot index table mapping offset "
    "throughput percentile median average tail head cold warm steady burst"
).split()


@dataclass
class Record:
    index: int
    ttft_ms: Optional[float]
    e2e_ms: Optional[float]
    tpot_ms: Optional[float]
    chunks: int
    error: Optional[str] = None


def make_text(n_words: int, rng: random.Random) -> str:
    if n_words <= 0:
        return ""
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def one_request(client: OpenAI, model: str, prompt: str, max_tokens: int,
                temperature: float, timeout: float, index: int) -> Record:
    start = time.perf_counter()
    ttft = None
    last = None
    chunks = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            timeout=timeout,
        )
        for chunk in stream:
            now = time.perf_counter()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if ttft is None:
                    ttft = now - start
                chunks += 1
                last = now
        e2e = (last if last is not None else time.perf_counter()) - start
        tpot = None
        if ttft is not None and chunks > 1:
            # Approximate: assumes one content chunk per output token.
            tpot = (e2e - ttft) / (chunks - 1)
        return Record(index=index, ttft_ms=ttft * 1000 if ttft else None,
                      e2e_ms=e2e * 1000,
                      tpot_ms=tpot * 1000 if tpot is not None else None,
                      chunks=chunks)
    except Exception as exc:  # noqa: BLE001 - report per-request failures
        return Record(index=index, ttft_ms=None, e2e_ms=None, tpot_ms=None,
                      chunks=0, error=f"{type(exc).__name__}: {exc}")


def build_prompts(args: argparse.Namespace) -> List[str]:
    """Build prompts before timing so client-side generation is not measured."""
    rng = random.Random(args.seed)
    shared_prefix = make_text(args.shared_words, rng)
    prompts: List[str] = []
    for i in range(args.num_requests):
        suffix = make_text(args.suffix_words, rng)
        if args.mode == "shared":
            prefix = shared_prefix
        elif args.mode == "unique":
            prefix = make_text(args.shared_words, rng)
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        prompts.append(f"{prefix}\n\nRequest {i}. {suffix}\n\nAnswer:")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["shared", "unique"], default="shared")
    parser.add_argument("--shared-words", type=int, default=1500,
                        help="length of the shared/unique prefix in words")
    parser.add_argument("--suffix-words", type=int, default=200,
                        help="length of each request's unique suffix in words")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=4,
                        help="sequential warmup requests; 0 disables")
    parser.add_argument("--request-rate", type=float, default=0.0,
                        help="requests/second; 0 means submit as fast as possible")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None,
                        help="optional path to write raw records as JSON")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.num_requests < 1:
        parser.error("--num-requests must be >= 1")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    max_retries=0, timeout=args.timeout)
    prompts = build_prompts(args)

    # Warmup: compile kernels, capture CUDA graphs, and (in shared mode)
    # populate the prefix cache. Excluded from statistics.
    # In unique mode, use prompts that are NOT part of the measured set, so
    # warmup does not accidentally create cache hits for measured requests.
    warmup_rng = random.Random(args.seed + 9999)
    for i in range(args.warmup):
        print(f"[warmup] {i + 1}/{args.warmup}", flush=True)
        if args.mode == "shared":
            warmup_prompt = prompts[i % len(prompts)]
        else:
            warmup_prompt = (
                f"{make_text(args.shared_words, warmup_rng)}\n\n"
                f"Warmup {i}. {make_text(args.suffix_words, warmup_rng)}\n\nAnswer:"
            )
        one_request(client, args.model, warmup_prompt,
                    args.max_tokens, args.temperature, args.timeout, -1)

    records: List[Record] = []
    t0 = time.perf_counter()
    next_submit = t0 - (1.0 / args.request_rate if args.request_rate > 0 else 0.0)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i, prompt in enumerate(prompts):
            if args.request_rate > 0:
                next_submit += 1.0 / args.request_rate
                delay = next_submit - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            futures.append(pool.submit(one_request, client, args.model, prompt,
                                        args.max_tokens, args.temperature,
                                        args.timeout, i))
        for fut in as_completed(futures):
            records.append(fut.result())
    wall = time.perf_counter() - t0

    records.sort(key=lambda r: r.index)
    ok = [r for r in records if r.error is None and r.ttft_ms is not None]
    failed = [r for r in records if r.error is not None]

    print()
    print(f"mode={args.mode} requests={len(records)} ok={len(ok)} failed={len(failed)} "
          f"concurrency={args.concurrency} wall={wall:.2f}s")
    if ok:
        ttfts = [r.ttft_ms for r in ok]  # type: ignore[misc]
        e2es = [r.e2e_ms for r in ok if r.e2e_ms is not None]  # type: ignore[misc]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]  # type: ignore[misc]
        total_chunks = sum(r.chunks for r in ok)
        print(f"TTFT ms : p50={percentile(ttfts, 50):8.1f}  "
              f"p90={percentile(ttfts, 90):8.1f}  "
              f"p99={percentile(ttfts, 99):8.1f}  "
              f"mean={statistics.fmean(ttfts):8.1f}")
        print(f"E2E  ms : p50={percentile(e2es, 50):8.1f}  "
              f"p90={percentile(e2es, 90):8.1f}  "
              f"p99={percentile(e2es, 99):8.1f}")
        if tpots:
            print(f"TPOT ms : p50={percentile(tpots, 50):8.2f}  "
                  f"p90={percentile(tpots, 90):8.2f}  "
                  f"p99={percentile(tpots, 99):8.2f}  (approx)")
        print(f"output chunks/s (approx tokens/s): {total_chunks / wall:.2f}")
    if failed:
        print(f"\nfirst failures ({len(failed)} total):")
        for r in failed[:5]:
            print(f"  request {r.index}: {r.error}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "wall_seconds": wall,
                    "records": [asdict(r) for r in records],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nraw records written to {args.json_out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
