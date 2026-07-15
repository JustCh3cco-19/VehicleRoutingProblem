#!/usr/bin/env python3
"""
Run the practical experiment campaign for the VRP project.

The script orchestrates only existing make targets and solve scripts:
  - make solve_seq
  - make solve_mpi
  - make solve_cuda
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_NODES = 4
MAX_CPUS_PER_TASK = 32
MAX_TOTAL_CORES = MAX_NODES * MAX_CPUS_PER_TASK
MAX_RUNTIME_S = 300
DEFAULT_OPENMP_THREADS = "1,2,4,8,16,32"
DEFAULT_MPI_RANKS = "1,2,4"
DEFAULT_COMBINED_STRONG_PAIRS = "1x1,1x2,1x4,1x8,1x16,1x32,2x32,4x32"
DEFAULT_WEAK_HYBRID_PAIRS = "1x32,2x32,4x32"


@dataclass(frozen=True)
class ManifestRow:
    name: str
    instance_path: str
    n: int
    k: int
    m: int
    solver_seed: int


def parse_int_csv(text: str, field_name: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid integer in {field_name}: {token}") from exc
        if value <= 0:
            raise ValueError(f"{field_name} values must be > 0, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    return values


def parse_rank_thread_pairs(text: str, field_name: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"invalid pair in {field_name}: {token} (expected RxT)")
        try:
            ranks = int(parts[0].strip())
            threads = int(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"invalid pair in {field_name}: {token}") from exc
        if ranks <= 0 or threads <= 0:
            raise ValueError(f"{field_name} values must be > 0: {token}")
        pairs.append((ranks, threads))
    if not pairs:
        raise ValueError(f"{field_name} cannot be empty")
    return pairs


def validate_resource_limits(
    openmp_threads: list[int],
    mpi_rank_values: list[int],
    combined_pairs: list[tuple[int, int]],
    weak_hybrid_pairs: list[tuple[int, int]],
    mpi_thread_values: list[int],
) -> None:
    if max(openmp_threads) > MAX_CPUS_PER_TASK:
        raise ValueError(
            f"OpenMP threads exceed max cpus-per-task={MAX_CPUS_PER_TASK}: "
            f"{openmp_threads}"
        )
    if max(mpi_rank_values) > MAX_NODES:
        raise ValueError(
            f"MPI ranks/nodes exceed max nodes={MAX_NODES}: {mpi_rank_values}"
        )
    for threads in mpi_thread_values:
        if threads > MAX_CPUS_PER_TASK:
            raise ValueError(
                f"MPI OpenMP threads exceed max cpus-per-task={MAX_CPUS_PER_TASK}: "
                f"{threads}"
            )
    for field_name, pairs in (
        ("combined_strong_pairs", combined_pairs),
        ("weak_hybrid_pairs", weak_hybrid_pairs),
    ):
        for ranks, threads in pairs:
            total_cores = ranks * threads
            if ranks > MAX_NODES:
                raise ValueError(
                    f"{field_name}: {ranks} ranks exceed max nodes={MAX_NODES}"
                )
            if threads > MAX_CPUS_PER_TASK:
                raise ValueError(
                    f"{field_name}: {threads} threads exceed max "
                    f"cpus-per-task={MAX_CPUS_PER_TASK}"
                )
            if total_cores > MAX_TOTAL_CORES:
                raise ValueError(
                    f"{field_name}: {ranks}x{threads}={total_cores} exceeds "
                    f"max total cores={MAX_TOTAL_CORES}"
                )


def load_manifest(path: Path) -> dict[int, ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    by_n: dict[int, ManifestRow] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            by_n[n] = ManifestRow(
                name=row["name"],
                instance_path=row["instance_path"],
                n=n,
                k=int(row["K"]),
                m=int(row["m"]),
                solver_seed=int(row["solver_seed"]),
            )
    if not by_n:
        raise ValueError(f"manifest has no rows: {path}")
    return by_n


def ensure_present(label: str, values: list[int], available: set[int]) -> None:
    missing = sorted(set(values) - available)
    if missing:
        raise ValueError(
            f"{label}: missing n values in manifest: {missing}. "
            f"Available: {sorted(available)}"
        )


def clients_csv(values: list[int]) -> str:
    return ",".join(str(v) for v in values)


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    cp = subprocess.run(cmd, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}")


def run_make(target: str, variables: dict[str, str], dry_run: bool) -> None:
    cmd = ["make", target]
    for key, value in variables.items():
        cmd.append(f"{key}={value}")
    run_cmd(cmd, dry_run=dry_run)


def solve_dirs(root: Path, exp_name: str) -> dict[str, str]:
    return {
        "SOLVE_CSV_DIR": str(root / "solve_manifest" / "csv" / exp_name),
        "SOLVE_SOLUTIONS_DIR": str(root / "solve_manifest" / "solutions" / exp_name),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run practical OpenMP/MPI/CUDA experiments.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Root output dir. Default: results/practical_campaign/<timestamp>",
    )
    parser.add_argument("--manifest-dir", default="instances/test_aligned")
    parser.add_argument("--launcher", choices=["auto", "mpirun", "srun"], default="mpirun")

    parser.add_argument("--strong-n", type=int, default=16000)
    parser.add_argument("--weak-base-n", type=int, default=1000)
    parser.add_argument("--openmp-threads", default=DEFAULT_OPENMP_THREADS)
    parser.add_argument("--mpi-ranks", default=DEFAULT_MPI_RANKS)
    parser.add_argument(
        "--hybrid-pairs",
        default=DEFAULT_COMBINED_STRONG_PAIRS,
        help=(
            "Combined strong-scaling curve as RxT pairs. Default: "
            "1x1,1x2,1x4,1x8,1x16,1x32,2x32,4x32"
        ),
    )
    parser.add_argument(
        "--weak-hybrid-pairs",
        default=DEFAULT_WEAK_HYBRID_PAIRS,
        help="Weak hybrid RxT pairs (default: 1x32,2x32,4x32)",
    )

    parser.add_argument("--seq-repeats", type=int, default=5)
    parser.add_argument("--scaling-repeats", type=int, default=5)
    parser.add_argument("--quality-repeats", type=int, default=10)
    parser.add_argument("--cuda-repeats", type=int, default=5)

    parser.add_argument("--runtime-s", type=int, default=300)
    parser.add_argument("--stagnation-epochs", type=int, default=500)
    parser.add_argument("--min-rel-improvement", type=str, default="0.001")

    parser.add_argument("--mpi-strong-omp-threads", type=int, default=32)
    parser.add_argument("--mpi-weak-omp-threads", type=int, default=32)
    parser.add_argument("--quality-clients", default="2000,8000,16000")
    parser.add_argument("--quality-openmp-threads", type=int, default=32)
    parser.add_argument("--quality-mpi-ranks", type=int, default=4)
    parser.add_argument("--quality-mpi-threads", type=int, default=32)
    parser.add_argument("--quality-hybrid", default="4x32")

    parser.add_argument("--cuda-variant", default="v6")
    parser.add_argument("--cuda-clients", default="500,1000,2000,4000,8000,16000,32000,64000")
    parser.add_argument(
        "--cuda-cpu-comparison-clients",
        default="500,1000,2000,4000,8000,16000,32000",
        help="CPU sizes used only for SEQ/OpenMP vs CUDA comparisons",
    )

    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-weak", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument(
        "--only-cuda-sections",
        action="store_true",
        help="Run only CUDA-related sections (CUDA sweep + CUDA comparisons + optional CUDA quality).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.only_cuda_sections and args.skip_cuda:
        raise ValueError("--only-cuda-sections cannot be combined with --skip-cuda")
    if args.runtime_s <= 0 or args.runtime_s > MAX_RUNTIME_S:
        raise ValueError(f"runtime-s must be in the range 1..{MAX_RUNTIME_S}")

    openmp_threads = parse_int_csv(args.openmp_threads, "openmp_threads")
    mpi_ranks = parse_int_csv(args.mpi_ranks, "mpi_ranks")
    hybrid_pairs = parse_rank_thread_pairs(args.hybrid_pairs, "hybrid_pairs")
    weak_hybrid_pairs = parse_rank_thread_pairs(
        args.weak_hybrid_pairs, "weak_hybrid_pairs"
    )
    quality_clients = parse_int_csv(args.quality_clients, "quality_clients")
    cuda_clients = parse_int_csv(args.cuda_clients, "cuda_clients")
    cuda_cpu_comparison_clients = parse_int_csv(
        args.cuda_cpu_comparison_clients, "cuda_cpu_comparison_clients"
    )
    quality_hybrid_pair = parse_rank_thread_pairs(args.quality_hybrid, "quality_hybrid")[0]
    validate_resource_limits(
        openmp_threads,
        [*mpi_ranks, args.quality_mpi_ranks, quality_hybrid_pair[0]],
        hybrid_pairs,
        weak_hybrid_pairs,
        [
            args.mpi_strong_omp_threads,
            args.mpi_weak_omp_threads,
            args.quality_openmp_threads,
            args.quality_mpi_threads,
            quality_hybrid_pair[1],
        ],
    )

    if not args.results_root:
        tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("results") / "practical_campaign" / tag
    else:
        root = Path(args.results_root)

    manifest_dir = Path(args.manifest_dir)
    manifest_seq = manifest_dir / "manifest.csv"
    manifest_mpi = manifest_dir / "manifest_openmp_mpi.csv"
    manifest_cuda = manifest_dir / "manifest_cuda.csv"

    seq_by_n = load_manifest(manifest_seq)
    mpi_by_n = load_manifest(manifest_mpi)
    cuda_by_n = load_manifest(manifest_cuda)

    mpi_available = set(mpi_by_n.keys())
    seq_available = set(seq_by_n.keys())
    cuda_available = set(cuda_by_n.keys())

    if args.only_cuda_sections:
        ensure_present("cuda_clients(cuda manifest)", cuda_clients, cuda_available)
        ensure_present(
            "cuda_cpu_comparison_clients(seq manifest)",
            cuda_cpu_comparison_clients,
            seq_available,
        )
        ensure_present(
            "cuda_cpu_comparison_clients(mpi manifest)",
            cuda_cpu_comparison_clients,
            mpi_available,
        )
        if not args.skip_quality:
            ensure_present("quality_clients(cuda manifest)", quality_clients, cuda_available)
    else:
        ensure_present("strong_n", [args.strong_n], mpi_available)
        ensure_present("quality_clients", quality_clients, seq_available & mpi_available)
        if not args.skip_cuda:
            ensure_present("cuda_clients", cuda_clients, cuda_available)

        if not args.skip_weak:
            weak_openmp_clients = [args.weak_base_n * t for t in openmp_threads]
            weak_mpi_clients = [args.weak_base_n * r for r in mpi_ranks]
            weak_hybrid_clients = [
                args.weak_base_n * r for r, _ in weak_hybrid_pairs
            ]
            ensure_present("weak_openmp_clients", weak_openmp_clients, mpi_available)
            ensure_present("weak_mpi_clients", weak_mpi_clients, mpi_available)
            ensure_present("weak_hybrid_clients", weak_hybrid_clients, mpi_available)

    base_vars = {
        "RESULTS_ROOT": str(root),
        "SOLVE_MANIFEST": str(manifest_seq),
        "SOLVE_MANIFEST_MPI": str(manifest_mpi),
        "SOLVE_MANIFEST_CUDA": str(manifest_cuda),
    }

    if not args.skip_build:
        run_make("seq", base_vars, args.dry_run)
        run_make("openmp_mpi", base_vars, args.dry_run)
        if not args.skip_cuda:
            run_make("cuda", {**base_vars, "CUDA_VARIANT": args.cuda_variant}, args.dry_run)

    if not args.only_cuda_sections:
        # 1) Sequential baseline
        run_make(
            "solve_seq",
            {
                **base_vars,
                **solve_dirs(root, "exp_seq_baseline"),
                "SOLVE_SEQ_REPEATS": str(args.seq_repeats),
                "SOLVE_SEQ_RUNTIME_S": str(args.runtime_s),
                "SOLVE_SEQ_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_SEQ_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

        # 2) Strong OpenMP (rank=1, varying threads)
        for t in openmp_threads:
            run_make(
                "solve_mpi",
                {
                    **base_vars,
                    **solve_dirs(root, "exp_strong_openmp_practical"),
                    "SOLVE_CLIENTS": str(args.strong_n),
                    "SOLVE_MPI_RANKS": "1",
                    "SOLVE_MPI_OMP_THREADS": str(t),
                    "SOLVE_MPI_LAUNCHER": args.launcher,
                    "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                    "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                    "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                    "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                    "SOLVE_BATCH_ID": f"strong_openmp_n{args.strong_n}_r1_t{t}",
                },
                args.dry_run,
            )

        # 3) Strong MPI: 1/2/4 ranks, fixed 32 OpenMP threads per rank.
        for r in mpi_ranks:
            run_make(
                "solve_mpi",
                {
                    **base_vars,
                    **solve_dirs(root, "exp_strong_mpi_practical"),
                    "SOLVE_CLIENTS": str(args.strong_n),
                    "SOLVE_MPI_RANKS": str(r),
                    "SOLVE_MPI_OMP_THREADS": str(args.mpi_strong_omp_threads),
                    "SOLVE_MPI_LAUNCHER": args.launcher,
                    "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                    "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                    "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                    "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                    "SOLVE_BATCH_ID": f"strong_mpi_n{args.strong_n}_r{r}_t{args.mpi_strong_omp_threads}",
                },
                args.dry_run,
            )

        # 4) Combined strong curve: 1..32 cores on one node, then 64/128.
        for r, t in hybrid_pairs:
            run_make(
                "solve_mpi",
                {
                    **base_vars,
                    **solve_dirs(root, "exp_strong_hybrid_practical"),
                    "SOLVE_CLIENTS": str(args.strong_n),
                    "SOLVE_MPI_RANKS": str(r),
                    "SOLVE_MPI_OMP_THREADS": str(t),
                    "SOLVE_MPI_LAUNCHER": args.launcher,
                    "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                    "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                    "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                    "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                    "SOLVE_BATCH_ID": f"strong_hybrid_n{args.strong_n}_r{r}_t{t}",
                },
                args.dry_run,
            )

        # 5) Weak scaling (optional)
        if not args.skip_weak:
            for t in openmp_threads:
                n_val = args.weak_base_n * t
                run_make(
                    "solve_mpi",
                    {
                        **base_vars,
                        **solve_dirs(root, "exp_weak_openmp_practical"),
                        "SOLVE_CLIENTS": str(n_val),
                        "SOLVE_MPI_RANKS": "1",
                        "SOLVE_MPI_OMP_THREADS": str(t),
                        "SOLVE_MPI_LAUNCHER": args.launcher,
                        "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                        "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                        "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                        "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                        "SOLVE_BATCH_ID": f"weak_openmp_n{n_val}_r1_t{t}",
                    },
                    args.dry_run,
                )

            for r in mpi_ranks:
                n_val = args.weak_base_n * r
                run_make(
                    "solve_mpi",
                    {
                        **base_vars,
                        **solve_dirs(root, "exp_weak_mpi_practical"),
                        "SOLVE_CLIENTS": str(n_val),
                        "SOLVE_MPI_RANKS": str(r),
                        "SOLVE_MPI_OMP_THREADS": str(args.mpi_weak_omp_threads),
                        "SOLVE_MPI_LAUNCHER": args.launcher,
                        "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                        "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                        "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                        "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                        "SOLVE_BATCH_ID": f"weak_mpi_n{n_val}_r{r}_t{args.mpi_weak_omp_threads}",
                    },
                    args.dry_run,
                )

            for r, t in weak_hybrid_pairs:
                n_val = args.weak_base_n * r
                run_make(
                    "solve_mpi",
                    {
                        **base_vars,
                        **solve_dirs(root, "exp_weak_hybrid_practical"),
                        "SOLVE_CLIENTS": str(n_val),
                        "SOLVE_MPI_RANKS": str(r),
                        "SOLVE_MPI_OMP_THREADS": str(t),
                        "SOLVE_MPI_LAUNCHER": args.launcher,
                        "SOLVE_MPI_REPEATS": str(args.scaling_repeats),
                        "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                        "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                        "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                        "SOLVE_BATCH_ID": f"weak_hybrid_n{n_val}_r{r}_t{t}",
                    },
                    args.dry_run,
                )

    # 6) CUDA size sweep + seq/openmp comparisons
    if not args.skip_cuda:
        cuda_clients_csv = clients_csv(cuda_clients)
        run_make(
            "solve_cuda",
            {
                **base_vars,
                **solve_dirs(root, "exp_cuda_size_sweep_practical"),
                "SOLVE_CLIENTS": cuda_clients_csv,
                "SOLVE_CUDA_VARIANT": args.cuda_variant,
                "SOLVE_CUDA_REPEATS": str(args.cuda_repeats),
                "SOLVE_CUDA_RUNTIME_S": str(args.runtime_s),
                "SOLVE_CUDA_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_CUDA_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

        comparison_clients_csv = clients_csv(cuda_cpu_comparison_clients)
        run_make(
            "solve_seq",
            {
                **base_vars,
                **solve_dirs(root, "exp_seq_vs_cuda_practical"),
                "SOLVE_CLIENTS": comparison_clients_csv,
                "SOLVE_SEQ_REPEATS": str(args.cuda_repeats),
                "SOLVE_SEQ_RUNTIME_S": str(args.runtime_s),
                "SOLVE_SEQ_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_SEQ_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

        run_make(
            "solve_mpi",
            {
                **base_vars,
                **solve_dirs(root, "exp_openmp_vs_cuda_practical"),
                "SOLVE_CLIENTS": comparison_clients_csv,
                "SOLVE_MPI_RANKS": "1",
                "SOLVE_MPI_OMP_THREADS": str(args.quality_openmp_threads),
                "SOLVE_MPI_LAUNCHER": args.launcher,
                "SOLVE_MPI_REPEATS": str(args.cuda_repeats),
                "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                "SOLVE_BATCH_ID": f"openmp_vs_cuda_t{args.quality_openmp_threads}",
            },
            args.dry_run,
        )

    # 7) Quality campaign (optional)
    if (not args.skip_quality) and (not args.only_cuda_sections):
        quality_clients_csv = clients_csv(quality_clients)

        run_make(
            "solve_seq",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_seq"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_SEQ_REPEATS": str(args.quality_repeats),
                "SOLVE_SEQ_RUNTIME_S": str(args.runtime_s),
                "SOLVE_SEQ_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_SEQ_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

        run_make(
            "solve_mpi",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_openmp"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_MPI_RANKS": "1",
                "SOLVE_MPI_OMP_THREADS": str(args.quality_openmp_threads),
                "SOLVE_MPI_LAUNCHER": args.launcher,
                "SOLVE_MPI_REPEATS": str(args.quality_repeats),
                "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                "SOLVE_BATCH_ID": f"quality_openmp_t{args.quality_openmp_threads}",
            },
            args.dry_run,
        )

        run_make(
            "solve_mpi",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_mpi"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_MPI_RANKS": str(args.quality_mpi_ranks),
                "SOLVE_MPI_OMP_THREADS": str(args.quality_mpi_threads),
                "SOLVE_MPI_LAUNCHER": args.launcher,
                "SOLVE_MPI_REPEATS": str(args.quality_repeats),
                "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                "SOLVE_BATCH_ID": (
                    f"quality_mpi_r{args.quality_mpi_ranks}_t{args.quality_mpi_threads}"
                ),
            },
            args.dry_run,
        )

        run_make(
            "solve_mpi",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_hybrid"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_MPI_RANKS": str(quality_hybrid_pair[0]),
                "SOLVE_MPI_OMP_THREADS": str(quality_hybrid_pair[1]),
                "SOLVE_MPI_LAUNCHER": args.launcher,
                "SOLVE_MPI_REPEATS": str(args.quality_repeats),
                "SOLVE_MPI_RUNTIME_S": str(args.runtime_s),
                "SOLVE_MPI_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_MPI_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
                "SOLVE_BATCH_ID": f"quality_hybrid_r{quality_hybrid_pair[0]}_t{quality_hybrid_pair[1]}",
            },
            args.dry_run,
        )

    if (not args.skip_quality) and args.only_cuda_sections and (not args.skip_cuda):
        quality_clients_csv = clients_csv(quality_clients)
        run_make(
            "solve_cuda",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_cuda"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_CUDA_VARIANT": args.cuda_variant,
                "SOLVE_CUDA_REPEATS": str(args.quality_repeats),
                "SOLVE_CUDA_RUNTIME_S": str(args.runtime_s),
                "SOLVE_CUDA_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_CUDA_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

    if (not args.only_cuda_sections) and (not args.skip_quality) and (not args.skip_cuda):
        quality_clients_csv = clients_csv(quality_clients)
        run_make(
            "solve_cuda",
            {
                **base_vars,
                **solve_dirs(root, "exp_quality_cuda"),
                "SOLVE_CLIENTS": quality_clients_csv,
                "SOLVE_CUDA_VARIANT": args.cuda_variant,
                "SOLVE_CUDA_REPEATS": str(args.quality_repeats),
                "SOLVE_CUDA_RUNTIME_S": str(args.runtime_s),
                "SOLVE_CUDA_STAGNATION_EPOCHS": str(args.stagnation_epochs),
                "SOLVE_CUDA_MIN_REL_IMPROVEMENT": args.min_rel_improvement,
            },
            args.dry_run,
        )

    print(f"Campaign completed. Results root: {root}")
    print("Next step: python3 tools/python/summarize_practical_experiments.py --root", root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
