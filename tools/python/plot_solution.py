#!/usr/bin/env python3
"""
Plot a VRP solution file produced by this repository.

Expected solution format:
  Route 1: 12 34 56
  Route 2:
  ...
  Cost: ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def maybe_tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def parse_coords(instance_path: Path) -> tuple[list[float], list[float]]:
    lines = instance_path.read_text(encoding="utf-8").splitlines()
    start = None
    dimension = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("DIMENSION"):
            m = re.search(r"(\d+)", s)
            if m:
                dimension = int(m.group(1))
        if s == "NODE_COORD_SECTION":
            start = i + 1
            break

    if start is None:
        raise ValueError(f"NODE_COORD_SECTION non trovato in {instance_path}")
    if dimension is None or dimension <= 0:
        raise ValueError(f"DIMENSION non valido in {instance_path}")

    xs: list[float] = []
    ys: list[float] = []
    for raw in lines[start:]:
        s = raw.strip()
        if not s or s.startswith("DEMAND_SECTION") or s.startswith("DEPOT_SECTION") or s.startswith("EOF"):
            break
        parts = s.split()
        if len(parts) < 3:
            continue
        # Manteniamo l'ordine dei record, coerente col parser C del progetto.
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))

    if len(xs) != dimension:
        raise ValueError(
            f"Coordinate lette ({len(xs)}) diverse da DIMENSION ({dimension}) in {instance_path}"
        )

    return xs, ys


def parse_routes(solution_path: Path) -> list[list[int]]:
    routes: list[list[int]] = []
    for raw in solution_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s.lower().startswith("route"):
            continue
        if ":" not in s:
            continue
        rhs = s.split(":", 1)[1].strip()
        if not rhs:
            routes.append([])
            continue
        nodes = [int(tok) for tok in rhs.split() if tok.isdigit()]
        routes.append(nodes)
    return routes


def infer_run_folder_name(solution_path: Path) -> str:
    # The full stem includes instance, backend, and run identifiers and avoids
    # collisions when plotting several backends for the same customer count.
    return re.sub(r"_solution$", "", solution_path.stem)


def infer_instance_for_solution(solution_path: Path, instances_dir: Path) -> Path | None:
    m = re.search(r"(n\d+_k\d+_s\d+)", solution_path.stem)
    if not m:
        return None
    candidate = instances_dir / f"{m.group(1)}.vrp"
    return candidate if candidate.exists() else None


def plot_one(
    instance_path: Path,
    solution_path: Path,
    out_path: Path,
    out_dir: Path,
    dpi: int,
    mode: str,
    style: str,
    show_labels: bool,
) -> list[Path]:
    xs, ys = parse_coords(instance_path)
    routes = parse_routes(solution_path)
    non_empty_routes = [r for r in routes if r]

    if not non_empty_routes:
        raise ValueError("Nessuna route non-vuota da plottare.")

    import matplotlib.pyplot as plt

    run_dir = out_dir / infer_run_folder_name(solution_path)
    routes_dir = run_dir / "routes"
    summary_dir = run_dir / "summary"
    routes_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    def draw_route_with_arrows(ax, seq: list[int], color: str) -> None:
        x_line = [xs[node] for node in seq]
        y_line = [ys[node] for node in seq]
        ax.plot(x_line, y_line, linewidth=1.8 if style == "presentation" else 0.9, alpha=0.95, color=color)
        step = 4 if style == "presentation" else 1
        for i in range(0, len(seq) - 1, step):
            x0, y0 = xs[seq[i]], ys[seq[i]]
            x1, y1 = xs[seq[i + 1]], ys[seq[i + 1]]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2 if style == "presentation" else 0.8),
                zorder=4,
            )

    def style_axes(ax) -> None:
        if style == "presentation":
            ax.set_facecolor("#f8fafc")
            ax.grid(color="#cbd5e1", alpha=0.9, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_color("#94a3b8")
            ax.tick_params(colors="#334155")
            ax.xaxis.label.set_color("#334155")
            ax.yaxis.label.set_color("#334155")
            ax.title.set_color("#0f172a")
        else:
            ax.grid(alpha=0.2)

    def plot_combined(target: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        palette = ["#ff4d3a", "#3b82f6", "#22c55e", "#f4b400", "#a855f7", "#00bcd4", "#ef4444", "#84cc16"]
        ax.scatter(xs[1:], ys[1:], s=8 if style == "presentation" else 4, alpha=0.08 if style == "presentation" else 0.2, color="gray", label="Clienti")
        ax.scatter(
            [xs[0]],
            [ys[0]],
            s=420 if style == "presentation" else 260,
            marker="*",
            color="#ff2d55",
            edgecolors="#fff176",
            linewidths=2.0 if style == "presentation" else 1.6,
            label="Depot",
            zorder=10,
        )

        for idx, route in enumerate(routes, start=1):
            if not route:
                continue
            seq = [0] + route + [0]
            color = palette[(idx - 1) % len(palette)]
            draw_route_with_arrows(ax, seq, color)
            ax.plot([], [], color=color, label=f"R{idx}")

        ax.set_title(f"{solution_path.name}\n{instance_path.name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        ax.legend(loc="best", fontsize=8, ncol=2, title="Legenda")
        fig.tight_layout()
        fig.savefig(target, dpi=dpi)
        plt.close(fig)

    def plot_split(target: Path) -> None:
        non_empty = [(i + 1, r) for i, r in enumerate(routes) if r]

        palette = ["#ff4d3a", "#3b82f6", "#22c55e", "#f4b400", "#a855f7", "#00bcd4", "#ef4444", "#84cc16"]
        cols = 2
        rows = (len(non_empty) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), squeeze=False)
        all_axes = [ax for row in axes for ax in row]

        for ax, (rid, route) in zip(all_axes, non_empty):
            color = palette[(rid - 1) % len(palette)]
            ax.scatter(xs[1:], ys[1:], s=5 if style == "presentation" else 3, alpha=0.06 if style == "presentation" else 0.15, color="gray", label="Clienti")
            ax.scatter(
                [xs[0]],
                [ys[0]],
                s=420 if style == "presentation" else 260,
                marker="*",
                color="#ff2d55",
                edgecolors="#fff176",
                linewidths=2.0 if style == "presentation" else 1.6,
                label="Depot",
                zorder=10,
            )
            seq = [0] + route + [0]
            draw_route_with_arrows(ax, seq, color)
            if show_labels and len(route) <= 40:
                for node in route:
                    ax.text(xs[node], ys[node], str(node), fontsize=7, color=color, ha="center", va="center")
            ax.plot([], [], color=color, label="Percorso route")
            ax.set_title(f"Route {rid} ({len(route)} clienti)")
            ax.set_aspect("equal", adjustable="box")
            style_axes(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="upper right", fontsize=8, title="Legenda")

        for ax in all_axes[len(non_empty):]:
            ax.axis("off")

        fig.suptitle(f"{solution_path.name} | {instance_path.name}", y=0.995)
        fig.tight_layout()
        fig.savefig(target, dpi=dpi)
        plt.close(fig)

    generated: list[Path] = []
    if mode in ("split", "both"):
        split_path = (
            run_dir / out_path.name
            if mode == "split"
            else summary_dir / (out_path.stem + "_split" + out_path.suffix)
        )
        plot_split(split_path)
        generated.append(split_path)
    if mode in ("combined", "both"):
        combined_path = (
            run_dir / out_path.name
            if mode == "combined"
            else summary_dir / (out_path.stem + "_combined" + out_path.suffix)
        )
        plot_combined(combined_path)
        generated.append(combined_path)

    per_route_items = [(i + 1, r) for i, r in enumerate(routes) if r]
    route_iter = maybe_tqdm(
        per_route_items,
        desc=f"routes {solution_path.name}",
        unit="route",
        leave=False,
    )

    for rid, route in route_iter:
        fig, ax = plt.subplots(figsize=(8, 8))
        color = "#3b82f6"
        ax.scatter(xs[1:], ys[1:], s=5 if style == "presentation" else 3, alpha=0.06 if style == "presentation" else 0.15, color="gray", label="Clienti")
        ax.scatter(
            [xs[0]],
            [ys[0]],
            s=420 if style == "presentation" else 260,
            marker="*",
            color="#ff2d55",
            edgecolors="#fff176",
            linewidths=2.0 if style == "presentation" else 1.6,
            label="Depot",
            zorder=10,
        )
        seq = [0] + route + [0]
        draw_route_with_arrows(ax, seq, color)
        if show_labels and len(route) <= 40:
            for node in route:
                ax.text(xs[node], ys[node], str(node), fontsize=8, color=color, ha="center", va="center")
        ax.plot([], [], color=color, label="Percorso route")
        ax.set_title(f"Route {rid} ({len(route)} clienti)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        ax.legend(loc="upper right", fontsize=8, title="Legenda")
        fig.tight_layout()
        per_route_path = routes_dir / f"route_{rid:02d}_clients_{len(route)}.png"
        fig.savefig(per_route_path, dpi=dpi)
        plt.close(fig)
        generated.append(per_route_path)

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot soluzione VRP da file .vrp + solution.txt")
    parser.add_argument("--instance", help="Path file istanza .vrp (modalita singola)")
    parser.add_argument("--solution", help="Path file soluzione *_solution.txt (modalita singola)")
    parser.add_argument("--out", default="overview.png", help="Nome output immagine overview (default: overview.png)")
    parser.add_argument("--dpi", type=int, default=160, help="DPI output (default: 160)")
    parser.add_argument(
        "--out-dir",
        default="results/solve_manifest/plots",
        help="Directory base output (default: results/solve_manifest/plots)",
    )
    parser.add_argument(
        "--mode",
        choices=("split", "combined", "both"),
        default="split",
        help="split: una subplot per route (default), combined: tutto insieme, both: entrambi",
    )
    parser.add_argument(
        "--style",
        choices=("default", "presentation"),
        default="default",
        help="Stile grafico: default o presentation (nodi numerati + frecce)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Mostra etichette numeriche nodi (auto-limitate alle route <= 40 clienti)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Plotta in batch tutte le soluzioni trovate in --solutions-glob",
    )
    parser.add_argument(
        "--solutions-glob",
        default="results/solve_manifest/solutions/*/*_solution.txt",
        help="Glob dei file soluzione usato con --all-runs",
    )
    parser.add_argument(
        "--instances-dir",
        default="instances/generated_benchmark",
        help="Directory istanze usata con --all-runs",
    )
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi deve essere positivo")

    out_path = Path(args.out)
    out_dir = Path(args.out_dir)

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "matplotlib non disponibile. Installa con: python -m pip install matplotlib"
        ) from exc

    generated: list[Path] = []
    if args.all_runs:
        instances_dir = Path(args.instances_dir)
        if not instances_dir.is_dir():
            raise ValueError(f"Directory istanze non trovata: {instances_dir}")
        if Path(args.solutions_glob).is_absolute():
            raise ValueError("--solutions-glob deve essere relativo alla root della repository")
        all_solutions = sorted(Path().glob(args.solutions_glob))
        if not all_solutions:
            raise ValueError(f"Nessuna soluzione trovata: {args.solutions_glob}")
        sol_iter = maybe_tqdm(all_solutions, desc="all-runs", unit="solution")

        for solution_path in sol_iter:
            instance_path = infer_instance_for_solution(solution_path, instances_dir)
            if instance_path is None:
                print(f"[skip] istanza non trovata per {solution_path}")
                continue
            try:
                generated.extend(
                    plot_one(
                        instance_path,
                        solution_path,
                        out_path,
                        out_dir,
                        args.dpi,
                        args.mode,
                        args.style,
                        args.show_labels,
                    )
                )
            except ValueError as exc:
                print(f"[skip] {solution_path}: {exc}")
    else:
        if not args.instance or not args.solution:
            raise ValueError("In modalita singola devi passare --instance e --solution.")
        single_items = [(Path(args.instance), Path(args.solution))]
        for instance_path, solution_path in maybe_tqdm(single_items, desc="single-run", unit="solution"):
            generated.extend(
                plot_one(
                    instance_path,
                    solution_path,
                    out_path,
                    out_dir,
                    args.dpi,
                    args.mode,
                    args.style,
                    args.show_labels,
                )
            )

    print("Plot salvati in:")
    for p in generated:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
