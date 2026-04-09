from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = ROOT / "instances/test_aligned/manifest.csv"
MANIFEST_OMP_MPI = ROOT / "instances/test_aligned/manifest_openmp_mpi.csv"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing manifest: {path}"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"Manifest is empty: {path}"
    return rows


def _parse_vrp_headers_and_counts(vrp_path: Path) -> tuple[dict[str, str], int, int, list[int], int]:
    headers: dict[str, str] = {}
    coord_count = 0
    demand_count = 0
    demand_values: list[int] = []
    depot_lines = 0

    section = "header"
    with vrp_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                section = "coords"
                continue
            if line == "DEMAND_SECTION":
                section = "demands"
                continue
            if line == "DEPOT_SECTION":
                section = "depot"
                continue
            if line == "EOF":
                section = "eof"
                continue

            if section == "header":
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()
            elif section == "coords":
                parts = line.split()
                assert len(parts) == 3, f"Bad NODE_COORD line in {vrp_path}: {line}"
                coord_count += 1
            elif section == "demands":
                parts = line.split()
                assert len(parts) == 2, f"Bad DEMAND line in {vrp_path}: {line}"
                demand_values.append(int(parts[1]))
                demand_count += 1
            elif section == "depot":
                depot_lines += 1

    return headers, coord_count, demand_count, demand_values, depot_lines


def _assert_manifest_row_matches_vrp(row: dict[str, str]) -> None:
    vrp_path = ROOT / row["instance_path"]
    assert vrp_path.exists(), f"Missing instance file: {vrp_path}"

    n = int(row["n"])
    k = int(row["K"])

    headers, coord_count, demand_count, demand_values, depot_lines = _parse_vrp_headers_and_counts(vrp_path)

    assert headers.get("NAME") == row["name"]
    assert headers.get("TYPE") == "CVRP"
    assert headers.get("EDGE_WEIGHT_TYPE") == "EUC_2D"
    assert int(headers.get("DIMENSION", "0")) == n + 1
    assert int(headers.get("VEHICLES", "0")) == k

    expected_capacity = math.ceil(n / k)
    assert int(headers.get("CAPACITY", "0")) == expected_capacity

    assert coord_count == n + 1
    assert demand_count == n + 1
    assert demand_values[0] == 0
    assert all(value == 1 for value in demand_values[1:])

    # Depot section is expected to contain: 1, -1
    assert depot_lines == 2


def test_manifests_exist_and_have_expected_row_count() -> None:
    rows_default = _read_manifest(MANIFEST_DEFAULT)
    rows_omp_mpi = _read_manifest(MANIFEST_OMP_MPI)

    assert len(rows_default) == 17
    assert len(rows_omp_mpi) == 17


def test_default_manifest_instances_are_valid() -> None:
    rows = _read_manifest(MANIFEST_DEFAULT)
    for row in rows:
        _assert_manifest_row_matches_vrp(row)


def test_openmp_mpi_manifest_instances_are_valid() -> None:
    rows = _read_manifest(MANIFEST_OMP_MPI)
    for row in rows:
        _assert_manifest_row_matches_vrp(row)


def test_manifests_reference_same_instance_set() -> None:
    rows_default = _read_manifest(MANIFEST_DEFAULT)
    rows_omp_mpi = _read_manifest(MANIFEST_OMP_MPI)

    key = lambda r: (r["name"], r["instance_path"], r["n"], r["K"], r["solver_seed"], r["instance_seed"], r["layout_id"])

    assert {key(r) for r in rows_default} == {key(r) for r in rows_omp_mpi}
