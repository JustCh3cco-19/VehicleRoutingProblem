#!/usr/bin/env python3
"""Generate a weak/strong scaling report from openmp+mpi progressive CSV results.

The script tries to compute canonical weak and strong scaling when enough data exist.
If the dataset does not contain multiple MPI ranks or repeated problem sizes,
it still generates diagnostic plots and a markdown report with clear limitations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scaling report and charts")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/scaling_progressive_openmp_mpi.csv"),
        help="Input CSV with progressive openmp+mpi results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory where report and plots will be written",
    )
    return parser.parse_args()


def _safe_local_slopes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 3:
        return np.array([]), np.array([])

    logx = np.log(x)
    logy = np.log(y)

    xs = []
    slopes = []
    for i in range(1, len(x) - 1):
        dx = logx[i + 1] - logx[i - 1]
        if dx == 0:
            continue
        slope = (logy[i + 1] - logy[i - 1]) / dx
        xs.append(x[i])
        slopes.append(slope)

    return np.array(xs), np.array(slopes)


def _save_runtime_plot(df_ok: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ok["n"], df_ok["elapsed_s"], marker="o", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("OpenMP+MPI - Runtime vs n")
    ax.set_xlabel("n (clienti)")
    ax.set_ylabel("tempo [s]")
    ax.grid(True, which="both", alpha=0.3)
    path = out_dir / "openmp_mpi_runtime_vs_n.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _save_throughput_plot(df_ok: pd.DataFrame, out_dir: Path) -> str:
    throughput = df_ok["n"] / df_ok["elapsed_s"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ok["n"], throughput, marker="o", linewidth=1.8, color="#d97706")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("OpenMP+MPI - Throughput (n / tempo)")
    ax.set_xlabel("n (clienti)")
    ax.set_ylabel("clienti/s (proxy)")
    ax.grid(True, which="both", alpha=0.3)
    path = out_dir / "openmp_mpi_throughput_vs_n.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _save_complexity_plot(df_ok: pd.DataFrame, out_dir: Path) -> tuple[str, float]:
    x = df_ok["n"].to_numpy(dtype=float)
    y = df_ok["elapsed_s"].to_numpy(dtype=float)

    slope_global = float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) >= 2 else float("nan")
    xs, slopes = _safe_local_slopes(x, y)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(xs) > 0:
        ax.plot(xs, slopes, marker="o", linewidth=1.8, color="#059669")
    ax.axhline(slope_global, linestyle="--", color="#111827", label=f"slope globale = {slope_global:.2f}")
    ax.set_xscale("log")
    ax.set_title("Esponente empirico locale: d log(t) / d log(n)")
    ax.set_xlabel("n (clienti)")
    ax.set_ylabel("esponente")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = out_dir / "openmp_mpi_empirical_exponent.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name, slope_global


def _save_memory_plot(df_ok: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ok["n"], df_ok["estimated_mem_gib"], marker="o", linewidth=1.8, color="#7c3aed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Memoria stimata vs n")
    ax.set_xlabel("n (clienti)")
    ax.set_ylabel("memoria stimata [GiB]")
    ax.grid(True, which="both", alpha=0.3)
    path = out_dir / "openmp_mpi_memory_vs_n.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _canonical_strong_scaling(df_ok: pd.DataFrame) -> pd.DataFrame:
    # Strong scaling needs same problem size and varying process counts.
    rows = []
    for n, g in df_ok.groupby("n"):
        ranks = g["mpi_ranks"].nunique()
        if ranks <= 1:
            continue
        g2 = g.sort_values("mpi_ranks")
        p0 = int(g2.iloc[0]["mpi_ranks"])
        t0 = float(g2.iloc[0]["elapsed_s"])
        for _, row in g2.iterrows():
            p = int(row["mpi_ranks"])
            tp = float(row["elapsed_s"])
            speedup = t0 / tp
            efficiency = speedup / (p / p0)
            rows.append(
                {
                    "n": int(n),
                    "p": p,
                    "t_p": tp,
                    "speedup": speedup,
                    "efficiency": efficiency,
                }
            )
    return pd.DataFrame(rows)


def _canonical_weak_scaling(df_ok: pd.DataFrame) -> pd.DataFrame:
    # Weak scaling needs approximately constant n/p and varying p.
    rows = []
    df2 = df_ok.copy()
    df2["n_per_rank"] = (df2["n"] / df2["mpi_ranks"]).round(0).astype(int)

    for npr, g in df2.groupby("n_per_rank"):
        ranks = g["mpi_ranks"].nunique()
        if ranks <= 1:
            continue
        g2 = g.sort_values("mpi_ranks")
        t_ref = float(g2.iloc[0]["elapsed_s"])
        p_ref = int(g2.iloc[0]["mpi_ranks"])
        for _, row in g2.iterrows():
            p = int(row["mpi_ranks"])
            tp = float(row["elapsed_s"])
            weak_eff = t_ref / tp
            rows.append(
                {
                    "n_per_rank": int(npr),
                    "p": p,
                    "t_p": tp,
                    "weak_efficiency": weak_eff,
                    "p_ref": p_ref,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    df_ok: pd.DataFrame,
    strong_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    slope_global: float,
    plot_files: list[str],
    out_dir: Path,
) -> Path:
    report_path = out_dir / "openmp_mpi_scaling_report.md"

    n_min = int(df_ok["n"].min()) if not df_ok.empty else 0
    n_max = int(df_ok["n"].max()) if not df_ok.empty else 0
    t_min = float(df_ok["elapsed_s"].min()) if not df_ok.empty else float("nan")
    t_max = float(df_ok["elapsed_s"].max()) if not df_ok.empty else float("nan")

    status_counts = df["status"].fillna("missing").value_counts().to_dict()
    status_lines = "\n".join(f"- {k}: {v}" for k, v in status_counts.items())

    if not df_ok.empty:
        thr = df_ok["n"] / df_ok["elapsed_s"]
        peak_thr = float(thr.max())
        peak_idx = int(thr.idxmax())
        peak_n = int(df_ok.loc[peak_idx, "n"])
    else:
        peak_thr = float("nan")
        peak_n = 0

    strong_note = (
        "Strong scaling canonico calcolabile (presenti n ripetuti con mpi_ranks diversi)."
        if not strong_df.empty
        else "Strong scaling canonico NON calcolabile: nel CSV non ci sono dimensioni n ripetute con mpi_ranks differenti."
    )
    weak_note = (
        "Weak scaling canonico calcolabile (presenti gruppi con n/mpi_ranks circa costante e mpi_ranks variabile)."
        if not weak_df.empty
        else "Weak scaling canonico NON calcolabile: nel CSV non ci sono gruppi con n/mpi_ranks costante e mpi_ranks differenti."
    )

    content = f"""# Report scaling OpenMP+MPI

## Dataset analizzato
- File: `{(out_dir / '..' / 'results').resolve()}` (origine: `results/scaling_progressive_openmp_mpi.csv`)
- Record totali: {len(df)}
- Record `status=ok`: {len(df_ok)}
- Intervallo n (`ok`): {n_min} .. {n_max}
- Intervallo tempo (`ok`): {t_min:.4f} s .. {t_max:.4f} s
- Distribuzione status:
{status_lines}

## Sintesi quantitativa
- Esponente empirico globale tempo vs n (fit log-log): **{slope_global:.2f}**
- Throughput massimo (proxy n/tempo): **{peak_thr:.2f} clienti/s** a **n={peak_n}**

## Strong scaling
{strong_note}

## Weak scaling
{weak_note}

## Interpretazione pratica
- Con questo CSV (progressivo su n e `mpi_ranks` costante), la lettura piu affidabile e il trend costo-tempo-memoria al crescere del problema.
- Per uno studio rigoroso strong scaling, serve fissare `n` e variare `mpi_ranks`.
- Per uno studio rigoroso weak scaling, serve mantenere circa costante `n/mpi_ranks` e variare `mpi_ranks`.

## Grafici generati
"""

    for name in plot_files:
        content += f"- `{name}`\n"

    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    numeric_cols = ["mpi_ranks", "n", "K", "m", "T", "estimated_mem_gib", "elapsed_s", "cost"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_ok = df[df["status"].eq("ok")].copy()
    df_ok = df_ok.dropna(subset=["n", "elapsed_s", "mpi_ranks"]).sort_values("n")

    if df_ok.empty:
        raise SystemExit("No usable rows with status=ok and elapsed_s available.")

    plot_files = []
    plot_files.append(_save_runtime_plot(df_ok, args.out_dir))
    plot_files.append(_save_throughput_plot(df_ok, args.out_dir))
    complexity_plot, slope_global = _save_complexity_plot(df_ok, args.out_dir)
    plot_files.append(complexity_plot)
    plot_files.append(_save_memory_plot(df_ok, args.out_dir))

    strong_df = _canonical_strong_scaling(df_ok)
    weak_df = _canonical_weak_scaling(df_ok)

    if not strong_df.empty:
        strong_df.to_csv(args.out_dir / "openmp_mpi_strong_scaling.csv", index=False)
    if not weak_df.empty:
        weak_df.to_csv(args.out_dir / "openmp_mpi_weak_scaling.csv", index=False)

    report_path = write_report(
        df=df,
        df_ok=df_ok,
        strong_df=strong_df,
        weak_df=weak_df,
        slope_global=slope_global,
        plot_files=plot_files,
        out_dir=args.out_dir,
    )

    print(f"Report written to: {report_path}")
    for p in plot_files:
        print(f"Plot written to: {args.out_dir / p}")


if __name__ == "__main__":
    main()
