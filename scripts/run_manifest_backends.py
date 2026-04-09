#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


BEST_COST_RE = re.compile(r"best_cost=([0-9]*\.?[0-9]+)")
FINAL_COST_RE = re.compile(r"Final Best Cost:\s*([0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run manifest instances on pyvrp/seq/openmp-mpi/cuda and save results.")
    p.add_argument("--manifest", default="instances/test_aligned/manifest.csv")
    p.add_argument("--out-csv", default="results/manifest_backend_results.csv")
    p.add_argument("--logs-dir", default="results/manifest_backend_logs")
    p.add_argument("--backends", default="pyvrp,seq,mpi,cuda", help="Comma list: pyvrp,seq,mpi,cuda")
    p.add_argument("--limit", type=int, default=0, help="Process only first N manifest rows (0 = all)")
    p.add_argument("--build", action="store_true", help="Build required C helpers/binaries before running")
    p.add_argument("--pyvrp-timeout-s", type=float, default=10.0)
    p.add_argument("--solver-timeout-s", type=float, default=0.0)
    p.add_argument("--mpi-ranks", type=int, default=2)
    p.add_argument("--omp-threads", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--tau0", type=float, default=1.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--cuda-timeout-min", type=float, default=2.0)
    p.add_argument("--cuda-improvement-threshold", type=float, default=0.001)
    return p.parse_args()


def run_cmd(cmd: list[str], timeout_s: float, env: dict[str, str] | None = None) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=(timeout_s if timeout_s > 0 else None),
        env=env,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def parse_best_cost(text: str) -> str:
    m = BEST_COST_RE.search(text)
    if m:
        return m.group(1)
    m2 = FINAL_COST_RE.search(text)
    if m2:
        return m2.group(1)
    return ""


def ensure_build(backends: set[str]) -> None:
    targets: list[str] = []
    if "seq" in backends:
        targets.append("c_compare_case")
    if "mpi" in backends:
        targets.append("c_compare_case_mpi")
    if "cuda" in backends:
        targets.append("cuda")

    if not targets:
        return

    for tgt in targets:
        cp = subprocess.run(["make", tgt], capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            print(f"[WARN] build target failed: {tgt}")
            print(cp.stderr.strip() or cp.stdout.strip())


def pyvrp_cmd(instance_path: str, seed: int, timeout_s: float) -> list[str]:
    code = (
        "from pyvrp import read, Model, stop; "
        "inst=read(r'" + instance_path + "', round_func='round'); "
        "m=Model.from_data(inst); "
        "res=m.solve(stop.MaxRuntime(" + str(timeout_s) + "), seed=" + str(seed) + "); "
        "b=res.best; "
        "print(f'best_cost={b.distance():.6f}' if b.is_feasible() else 'best_cost=')"
    )
    py = "VRP/bin/python" if Path("VRP/bin/python").exists() else "python3"
    return [py, "-c", code]


def main() -> int:
    args = parse_args()
    root = Path.cwd()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")

    backends = {b.strip() for b in args.backends.split(",") if b.strip()}
    valid = {"pyvrp", "seq", "mpi", "cuda"}
    unknown = backends - valid
    if unknown:
        raise SystemExit(f"unknown backends: {sorted(unknown)}")

    if args.build:
        ensure_build(backends)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seq_bin = Path("tests/c_compare_case.out")
    mpi_bin = Path("tests/c_compare_case_mpi.out")
    cuda_bin = Path("aco_vrp_cuda.out")

    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))

    if args.limit > 0:
        data = data[: args.limit]

    total = len(data)
    run_idx = 0
    for rec in data:
        run_idx += 1
        name = rec["name"]
        print(f"[{run_idx}/{total}] {name}")

        n = int(rec["n"])
        k = int(rec["K"])
        m = int(rec["m"])
        t = int(rec["T"])
        solver_seed = int(rec["solver_seed"])
        instance_seed = int(rec["instance_seed"])
        layout_id = int(rec["layout_id"])
        inst = rec["instance_path"]

        # per-backend execution
        for backend in ["pyvrp", "seq", "mpi", "cuda"]:
            if backend not in backends:
                continue

            cmd: list[str] = []
            env = os.environ.copy()
            status = "ok"
            rc = -1
            elapsed = 0.0
            stdout = ""
            stderr = ""
            best_cost = ""

            if backend == "pyvrp":
                cmd = pyvrp_cmd(inst, solver_seed, args.pyvrp_timeout_s)
            elif backend == "seq":
                if not seq_bin.exists():
                    status = "missing_binary"
                else:
                    cmd = [
                        str(seq_bin),
                        str(n),
                        str(k),
                        str(m),
                        str(t),
                        str(solver_seed),
                        str(instance_seed),
                        str(layout_id),
                        str(args.alpha),
                        str(args.beta),
                        str(args.rho),
                        str(args.tau0),
                        str(args.q),
                    ]
            elif backend == "mpi":
                if not mpi_bin.exists():
                    status = "missing_binary"
                elif shutil.which("mpirun") is None:
                    status = "missing_mpirun"
                else:
                    env["OMP_NUM_THREADS"] = str(max(1, args.omp_threads))
                    cmd = [
                        "mpirun",
                        "-np",
                        str(max(1, args.mpi_ranks)),
                        str(mpi_bin),
                        str(n),
                        str(k),
                        str(m),
                        str(t),
                        str(solver_seed),
                        str(instance_seed),
                        str(layout_id),
                        str(args.alpha),
                        str(args.beta),
                        str(args.rho),
                        str(args.tau0),
                        str(args.q),
                    ]
            else:  # cuda
                if not cuda_bin.exists():
                    status = "missing_binary"
                else:
                    cmd = [
                        str(cuda_bin),
                        inst,
                        str(k),
                        str(args.cuda_timeout_min),
                        str(args.cuda_improvement_threshold),
                        str(solver_seed),
                    ]

            if status == "ok" and cmd:
                try:
                    rc, elapsed, stdout, stderr = run_cmd(cmd, args.solver_timeout_s, env)
                    if rc != 0:
                        status = "error"
                    best_cost = parse_best_cost(stdout)
                except subprocess.TimeoutExpired as exc:
                    rc = 124
                    elapsed = args.solver_timeout_s
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or "timeout"
                    status = "timeout"

            log_base = f"{name}_{backend}"
            (logs_dir / f"{log_base}.stdout.log").write_text(stdout or "", encoding="utf-8")
            (logs_dir / f"{log_base}.stderr.log").write_text(stderr or "", encoding="utf-8")

            rows.append(
                {
                    "name": name,
                    "profile": rec["profile"],
                    "backend": backend,
                    "instance_path": inst,
                    "n": str(n),
                    "K": str(k),
                    "m": str(m),
                    "T": str(t),
                    "solver_seed": str(solver_seed),
                    "instance_seed": str(instance_seed),
                    "layout_id": str(layout_id),
                    "status": status,
                    "return_code": str(rc),
                    "elapsed_s": f"{elapsed:.6f}",
                    "best_cost": best_cost,
                    "cmd": " ".join(cmd),
                }
            )

    fields = [
        "name",
        "profile",
        "backend",
        "instance_path",
        "n",
        "K",
        "m",
        "T",
        "solver_seed",
        "instance_seed",
        "layout_id",
        "status",
        "return_code",
        "elapsed_s",
        "best_cost",
        "cmd",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[DONE] wrote {out_csv} rows={len(rows)} ok={ok} failed={len(rows)-ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
