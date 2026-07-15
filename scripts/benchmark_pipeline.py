#!/usr/bin/env python3
"""
Single entrypoint for benchmarks + plots.
Runs scaling benchmark and then generates plots.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    cp = subprocess.run(cmd, check=False, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run benchmark and plots in one command.")
    p.add_argument(
        "--include-cuda",
        action="store_true",
        help="Include a fixed single-GPU comparison (not CUDA scaling).",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--t", type=int, default=120)
    p.add_argument("--weak-n", type=int, default=100)
    p.add_argument("--weak-k", type=int, default=8)
    p.add_argument("--weak-t", type=int, default=120)
    p.add_argument("--weak-ants-per-worker", type=int, default=64)
    p.add_argument("--omp-threads", default="1,2,4,8,16,32")
    p.add_argument("--mpi-ranks", default="1,2,4")
    p.add_argument("--mpi-threads-per-rank", type=int, default=32)
    p.add_argument("--sync-every", type=int, default=1)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    bench_cmd = [
        sys.executable,
        "scripts/benchmark_scaling.py",
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--timeout-s",
        str(args.timeout_s),
        "--output-dir",
        args.output_dir,
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--m",
        str(args.m),
        "--t",
        str(args.t),
        "--weak-n",
        str(args.weak_n),
        "--weak-k",
        str(args.weak_k),
        "--weak-t",
        str(args.weak_t),
        "--weak-ants-per-worker",
        str(args.weak_ants_per_worker),
        "--omp-threads",
        args.omp_threads,
        "--mpi-ranks",
        args.mpi_ranks,
        "--mpi-threads-per-rank",
        str(args.mpi_threads_per_rank),
        "--sync-every",
        str(args.sync_every),
    ]
    if args.include_cuda:
        bench_cmd.append("--include-cuda")
    if args.skip_build:
        bench_cmd.append("--skip-build")

    run(bench_cmd)
    run(
        [
            sys.executable,
            "scripts/plot_scaling_results.py",
            "--summary",
            f"{args.output_dir}/scaling_summary.csv",
            "--raw",
            f"{args.output_dir}/scaling_results_raw.csv",
            "--out-dir",
            f"{args.output_dir}/plots",
        ]
    )
    print("Pipeline completed: benchmark + plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
