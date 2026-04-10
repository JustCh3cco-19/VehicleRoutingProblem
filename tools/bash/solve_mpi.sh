#!/usr/bin/env bash
set -uo pipefail

csv="${SOLVE_CSV_DIR}/manifest_openmp_mpi_per_instance_results.csv"
sol_dir="${SOLVE_SOLUTIONS_DIR}/mpi"
manifest="${SOLVE_MANIFEST_MPI}"
clients="${SOLVE_CLIENTS:-}"
limit="${SOLVE_LIMIT:-0}"
repeats="${SOLVE_MPI_REPEATS:-1}"
runtime_s="${SOLVE_MPI_RUNTIME_S:-0}"
stag_iters="${SOLVE_MPI_STAGNATION_EPOCHS:-0}"
improve_eps="${SOLVE_MPI_MIN_REL_IMPROVEMENT:-0.001}"
mpi_ranks="${SOLVE_MPI_RANKS:-2}"
omp_threads="${SOLVE_MPI_OMP_THREADS:-2}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi

echo "name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_kb,best_cost,error" > "$csv"

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      for run_id in $(seq 1 "$repeats"); do
        seed_run=$((solver_seed + run_id - 1))
        sol_file="${sol_dir}/${name}_mpi_run${run_id}_solution.txt"
        stats_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e,%M" -o "$stats_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_ITERS="$stag_iters" \
          ACO_SOLVER_IMPROVE_EPS="$improve_eps" \
          OMP_NUM_THREADS="$omp_threads" \
          mpirun -np "$mpi_ranks" ./aco_vrp_openmp_mpi.out "$instance_path" "$K" "$m" "$seed_run" </dev/null 2>&1)
        rc=$?

        elapsed="$(cut -d, -f1 "$stats_file" 2>/dev/null)"
        rss_kb="$(cut -d, -f2 "$stats_file" 2>/dev/null)"
        rm -f "$stats_file"
        [ -n "$elapsed" ] || elapsed=""
        [ -n "$rss_kb" ] || rss_kb=""

        printf '%s\n' "$out" > "$sol_file"
        if [ "$rc" -eq 0 ]; then
          cost="$(printf '%s\n' "$out" | sed -n 's/^best cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_kb,$cost," >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_kb,,$err" >> "$csv"
        fi
        echo "[mpi] $name run=$run_id done"
      done
    done

echo "wrote $csv"
