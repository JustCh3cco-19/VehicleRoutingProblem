#!/usr/bin/env python3
"""
Benchmark strong/weak scaling for the VRP ACO implementations.

This script executes sequential, OpenMP, and MPI+OpenMP binaries,
collects elapsed times, computes speedup/efficiency, and writes:
  - reports/scaling_results_raw.csv
  - reports/scaling_summary.csv
  - reports/scaling_report.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ELAPSED_RE = re.compile(r"elapsed:\s*([0-9]*\.?[0-9]+)")
COST_RE = re.compile(r"best cost:\s*([0-9]*\.?[0-9]+)")
MAX_NODES = 4
MAX_CPUS_PER_TASK = 32
MAX_TOTAL_CORES = 128
MAX_TIMEOUT_S = 300


def parse_int_list(text: str, name: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid integer in {name}: '{token}'") from exc
        if value <= 0:
            raise ValueError(f"invalid value in {name}: {value} (must be > 0)")
        values.append(value)
    if not values:
        raise ValueError(f"{name} is empty")
    return sorted(set(values))


def run_make(targets: list[str]) -> None:
    cmd = ["make"] + targets
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"build failed for {' '.join(targets)}")


def parse_metric(regex: re.Pattern[str], output: str) -> float | None:
    match = regex.search(output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def run_case(cmd: list[str], timeout_s: int, env: dict[str, str] | None) -> dict[str, Any]:
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=timeout_s,
    )
    wall = time.perf_counter() - start
    output = f"{completed.stdout}\n{completed.stderr}"

    elapsed = parse_metric(ELAPSED_RE, output)
    if elapsed is None:
        elapsed = wall

    best_cost = parse_metric(COST_RE, output)
    status = "ok" if completed.returncode == 0 else "error"

    return {
        "status": status,
        "return_code": completed.returncode,
        "elapsed_s": elapsed,
        "best_cost": best_cost if best_cost is not None else "",
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in raw_rows:
        if row["warmup"] != 0 or row["status"] != "ok":
            continue
        key = (
            row["experiment"],
            row["backend"],
            row["ranks"],
            row["threads_per_rank"],
            row["scale"],
            row["n"],
            row["K"],
            row["m"],
            row["T"],
        )
        grouped.setdefault(key, []).append(float(row["elapsed_s"]))

    summary_rows: list[dict[str, Any]] = []
    for key, times in grouped.items():
        (
            experiment,
            backend,
            ranks,
            threads_per_rank,
            scale,
            n,
            k_val,
            m,
            t_val,
        ) = key
        mean_s = statistics.mean(times)
        std_s = statistics.stdev(times) if len(times) > 1 else 0.0
        summary_rows.append(
            {
                "experiment": experiment,
                "backend": backend,
                "ranks": ranks,
                "threads_per_rank": threads_per_rank,
                "scale": scale,
                "n": n,
                "K": k_val,
                "m": m,
                "T": t_val,
                "samples": len(times),
                "mean_s": mean_s,
                "std_s": std_s,
                "min_s": min(times),
                "max_s": max(times),
                "speedup": "",
                "efficiency": "",
            }
        )

    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_experiment.setdefault(str(row["experiment"]), []).append(row)

    for experiment_rows in by_experiment.values():
        baseline = min(experiment_rows, key=lambda r: int(r["scale"]))
        baseline_scale = float(baseline["scale"])
        baseline_time = float(baseline["mean_s"])
        for row in experiment_rows:
            scale = float(row["scale"])
            if baseline_time <= 0.0 or baseline_scale <= 0.0:
                row["speedup"] = ""
                row["efficiency"] = ""
                continue
            speedup = baseline_time / float(row["mean_s"])
            normalized_scale = scale / baseline_scale
            efficiency = speedup / normalized_scale if normalized_scale > 0.0 else 0.0
            row["speedup"] = speedup
            row["efficiency"] = efficiency

    summary_rows.sort(key=lambda r: (r["experiment"], int(r["scale"])))
    return summary_rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report(
    report_path: Path,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pyver = platform.python_version()
    host = platform.node() or "unknown-host"
    system = f"{platform.system()} {platform.release()}"

    lines: list[str] = []
    lines.append("# Scaling Report")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Host: {host}")
    lines.append(f"- System: {system}")
    lines.append(f"- Python: {pyver}")
    lines.append("")
    lines.append("## Benchmark Configuration")
    lines.append("")
    lines.append(f"- Repeats (measured): {args.repeats}")
    lines.append(f"- Warmups (discarded): {args.warmups}")
    lines.append(f"- Strong scaling workload: n={args.n}, K={args.k}, m={args.m}, T={args.t}")
    lines.append(
        f"- Weak scaling base ants/worker: {args.weak_ants_per_worker}, "
        f"n={args.weak_n}, K={args.weak_k}, T={args.weak_t}"
    )
    lines.append(f"- OpenMP threads tested: {args.omp_threads}")
    lines.append(f"- MPI ranks tested: {args.mpi_ranks}")
    lines.append(f"- MPI threads per rank: {args.mpi_threads_per_rank}")
    lines.append(f"- MPI sync_every: {args.sync_every}")
    lines.append(
        f"- CUDA fixed single-GPU comparison included (not scaling): "
        f"{'yes' if args.include_cuda else 'no'}"
    )
    lines.append("")

    lines.append("## Aggregated Results")
    lines.append("")
    if not summary_rows:
        lines.append("No successful measurements were collected.")
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in summary_rows:
            grouped.setdefault(str(row["experiment"]), []).append(row)

        for experiment, rows in grouped.items():
            lines.append(f"### {experiment}")
            lines.append("")
            table_rows: list[list[str]] = []
            for row in rows:
                table_rows.append(
                    [
                        str(row["scale"]),
                        str(row["ranks"]),
                        str(row["threads_per_rank"]),
                        str(row["m"]),
                        format_float(float(row["mean_s"]), 6),
                        format_float(float(row["std_s"]), 6),
                        format_float(float(row["speedup"]), 3)
                        if row["speedup"] != ""
                        else "-",
                        format_float(float(row["efficiency"]), 3)
                        if row["efficiency"] != ""
                        else "-",
                    ]
                )
            lines.append(
                markdown_table(
                    [
                        "Scale",
                        "Ranks",
                        "Threads/Rank",
                        "m",
                        "Mean Time (s)",
                        "Std Dev (s)",
                        "Speedup",
                        "Efficiency",
                    ],
                    table_rows,
                )
            )
            lines.append("")

    failed_rows = [
        row
        for row in raw_rows
        if row["warmup"] == 0 and row["status"] != "ok"
    ]
    lines.append("## Failures")
    lines.append("")
    if not failed_rows:
        lines.append("No benchmark failures.")
    else:
        fail_table: list[list[str]] = []
        for row in failed_rows:
            err = str(row["stderr"]).replace("\n", " ").replace("\\n", " ").strip()
            err = re.sub(r"\s+", " ", err)
            if len(err) > 120:
                err = err[:117] + "..."
            fail_table.append(
                [
                    str(row["experiment"]),
                    str(row["backend"]),
                    f"{row['ranks']}x{row['threads_per_rank']}",
                    str(row["status"]),
                    err if err else f"return_code={row['return_code']}",
                ]
            )
        lines.append(
            markdown_table(
                ["Experiment", "Backend", "Scale", "Status", "Reason"],
                fail_table,
            )
        )
    lines.append("")
    lines.append(
        "Raw samples are available in `scaling_results_raw.csv`; aggregated values "
        "are in `scaling_summary.csv`."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scaling benchmarks for VRP ACO.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs per configuration.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per configuration.")
    parser.add_argument("--timeout-s", type=int, default=300, help="Timeout per run (seconds).")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated reports.")

    parser.add_argument("--n", type=int, default=100, help="Strong-scaling customers.")
    parser.add_argument("--k", type=int, default=8, help="Strong-scaling vehicles.")
    parser.add_argument("--m", type=int, default=256, help="Strong-scaling ants.")
    parser.add_argument("--t", type=int, default=120, help="Strong-scaling iterations.")

    parser.add_argument("--weak-n", type=int, default=100, help="Weak-scaling customers.")
    parser.add_argument("--weak-k", type=int, default=8, help="Weak-scaling vehicles.")
    parser.add_argument("--weak-t", type=int, default=120, help="Weak-scaling iterations.")
    parser.add_argument(
        "--weak-ants-per-worker",
        type=int,
        default=64,
        help="Weak-scaling ants per worker (rank*thread).",
    )

    parser.add_argument("--omp-threads", default="1,2,4,8,16,32", help="Comma-separated OpenMP thread counts.")
    parser.add_argument("--mpi-ranks", default="1,2,4", help="Comma-separated MPI ranks.")
    parser.add_argument("--mpi-threads-per-rank", type=int, default=32, help="Fixed OpenMP threads per MPI rank.")
    parser.add_argument("--sync-every", type=int, default=1, help="MPI pheromone sync period.")

    parser.add_argument("--skip-build", action="store_true", help="Skip make targets before benchmarking.")
    parser.add_argument(
        "--include-cuda",
        action="store_true",
        help="Include one fixed single-GPU comparison; CUDA is excluded from scaling curves.",
    )

    args = parser.parse_args()

    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be > 0 and warmups must be >= 0")
    if args.timeout_s <= 0 or args.timeout_s > MAX_TIMEOUT_S:
        raise ValueError(f"timeout-s must be in the range 1..{MAX_TIMEOUT_S}")
    if args.n <= 0 or args.k <= 0 or args.m <= 0 or args.t <= 0:
        raise ValueError("strong scaling parameters must be > 0")
    if args.weak_n <= 0 or args.weak_k <= 0 or args.weak_t <= 0:
        raise ValueError("weak scaling parameters must be > 0")
    if args.weak_ants_per_worker <= 0:
        raise ValueError("weak_ants_per_worker must be > 0")
    if args.mpi_threads_per_rank <= 0:
        raise ValueError("mpi_threads_per_rank must be > 0")
    if args.sync_every <= 0:
        raise ValueError("sync_every must be > 0")

    omp_threads = parse_int_list(args.omp_threads, "omp_threads")
    mpi_ranks = parse_int_list(args.mpi_ranks, "mpi_ranks")
    if max(omp_threads) > MAX_CPUS_PER_TASK:
        raise ValueError(f"OpenMP threads cannot exceed {MAX_CPUS_PER_TASK}")
    if max(mpi_ranks) > MAX_NODES:
        raise ValueError(f"MPI ranks/nodes cannot exceed {MAX_NODES}")
    if args.mpi_threads_per_rank > MAX_CPUS_PER_TASK:
        raise ValueError(f"MPI threads per rank cannot exceed {MAX_CPUS_PER_TASK}")
    if max(mpi_ranks) * args.mpi_threads_per_rank > MAX_TOTAL_CORES:
        raise ValueError(f"total MPI+OpenMP cores cannot exceed {MAX_TOTAL_CORES}")
    args.omp_threads = ",".join(str(v) for v in omp_threads)
    args.mpi_ranks = ",".join(str(v) for v in mpi_ranks)

    if not args.skip_build:
        run_make(["seq", "omp", "mpi"])
        if args.include_cuda:
            run_make(["cuda"])

    raw_rows: list[dict[str, Any]] = []

    def append_case(
        experiment: str,
        backend: str,
        cmd: list[str],
        ranks: int,
        threads_per_rank: int,
        scale: int,
        n: int,
        k_val: int,
        m: int,
        t_val: int,
        env: dict[str, str] | None,
    ) -> None:
        total_runs = args.warmups + args.repeats
        for run_idx in range(total_runs):
            warmup = 1 if run_idx < args.warmups else 0
            result = run_case(cmd, args.timeout_s, env)
            row = {
                "experiment": experiment,
                "backend": backend,
                "run": run_idx + 1,
                "warmup": warmup,
                "ranks": ranks,
                "threads_per_rank": threads_per_rank,
                "scale": scale,
                "n": n,
                "K": k_val,
                "m": m,
                "T": t_val,
                "cmd": " ".join(cmd),
            }
            row.update(result)
            raw_rows.append(row)

    # Sequential baseline on strong-scaling workload.
    append_case(
        experiment="seq_strong_baseline",
        backend="seq",
        cmd=["./aco_vrp_seq", str(args.n), str(args.k), str(args.m), str(args.t)],
        ranks=1,
        threads_per_rank=1,
        scale=1,
        n=args.n,
        k_val=args.k,
        m=args.m,
        t_val=args.t,
        env=None,
    )

    # OpenMP strong scaling.
    for threads in omp_threads:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        append_case(
            experiment="omp_strong",
            backend="omp",
            cmd=[
                "./aco_vrp_omp",
                str(threads),
                str(args.n),
                str(args.k),
                str(args.m),
                str(args.t),
            ],
            ranks=1,
            threads_per_rank=threads,
            scale=threads,
            n=args.n,
            k_val=args.k,
            m=args.m,
            t_val=args.t,
            env=env,
        )

    # MPI+OpenMP strong scaling (fixed workload).
    for ranks in mpi_ranks:
        threads = args.mpi_threads_per_rank
        scale = ranks * threads
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        append_case(
            experiment="mpi_omp_strong",
            backend="mpi+omp",
            cmd=[
                "mpirun",
                "--oversubscribe",
                "-np",
                str(ranks),
                "./aco_vrp_mpi",
                str(threads),
                str(args.sync_every),
                str(args.n),
                str(args.k),
                str(args.m),
                str(args.t),
            ],
            ranks=ranks,
            threads_per_rank=threads,
            scale=scale,
            n=args.n,
            k_val=args.k,
            m=args.m,
            t_val=args.t,
            env=env,
        )

    # MPI+OpenMP weak scaling (ants proportional to workers).
    for ranks in mpi_ranks:
        threads = args.mpi_threads_per_rank
        scale = ranks * threads
        weak_m = args.weak_ants_per_worker * scale
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        append_case(
            experiment="mpi_omp_weak",
            backend="mpi+omp",
            cmd=[
                "mpirun",
                "--oversubscribe",
                "-np",
                str(ranks),
                "./aco_vrp_mpi",
                str(threads),
                str(args.sync_every),
                str(args.weak_n),
                str(args.weak_k),
                str(weak_m),
                str(args.weak_t),
            ],
            ranks=ranks,
            threads_per_rank=threads,
            scale=scale,
            n=args.weak_n,
            k_val=args.weak_k,
            m=weak_m,
            t_val=args.weak_t,
            env=env,
        )

    if args.include_cuda:
        append_case(
            experiment="cuda_fixed",
            backend="cuda",
            cmd=[
                "./aco_vrp_cuda",
                str(args.n),
                str(args.k),
                str(args.m),
                str(args.t),
            ],
            ranks=1,
            threads_per_rank=1,
            scale=1,
            n=args.n,
            k_val=args.k,
            m=args.m,
            t_val=args.t,
            env=None,
        )

    output_dir = Path(args.output_dir)
    raw_path = output_dir / "scaling_results_raw.csv"
    summary_path = output_dir / "scaling_summary.csv"
    report_path = output_dir / "scaling_report.md"

    raw_fields = [
        "experiment",
        "backend",
        "run",
        "warmup",
        "status",
        "return_code",
        "elapsed_s",
        "best_cost",
        "ranks",
        "threads_per_rank",
        "scale",
        "n",
        "K",
        "m",
        "T",
        "cmd",
        "stdout",
        "stderr",
    ]
    write_csv(raw_path, raw_rows, raw_fields)

    summary_rows = summarize(raw_rows)
    summary_fields = [
        "experiment",
        "backend",
        "ranks",
        "threads_per_rank",
        "scale",
        "n",
        "K",
        "m",
        "T",
        "samples",
        "mean_s",
        "std_s",
        "min_s",
        "max_s",
        "speedup",
        "efficiency",
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    build_report(report_path, args, raw_rows, summary_rows)

    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
