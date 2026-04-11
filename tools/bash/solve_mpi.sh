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
improve_rel_pct="${SOLVE_MPI_MIN_REL_IMPROVEMENT:-0.001}"
mpi_ranks="${SOLVE_MPI_RANKS:-2}"
omp_threads="${SOLVE_MPI_OMP_THREADS:-2}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi
improve_rel="$(awk "BEGIN { printf \"%.12g\", (${improve_rel_pct}) / 100.0 }")"

launcher_kind="mpirun"
launcher_cmd=(mpirun -np "$mpi_ranks")
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
  launcher_kind="srun"
  launcher_cmd=(srun --mpi=pmix -n "$mpi_ranks" --cpus-per-task "$omp_threads")
fi
echo "[mpi] launcher=${launcher_kind}"

header="name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_gb,best_cost,error"

# Prepare output CSV in advance, keeping existing rows for n values that are
# not part of the current execution. New rows are then appended live.
tmp_csv="$(mktemp)"
echo "$header" > "$tmp_csv"

if [ -f "$csv" ] && [ -s "$csv" ]; then
  selected_n_tmp="$(mktemp)"
  tail -n +2 "$manifest" \
    | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
    | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
    | awk -F, 'NF > 0 && $4 != "" { print $4 }' | sort -u > "$selected_n_tmp"

  if [ -s "$selected_n_tmp" ]; then
    awk -F, 'NR==FNR { drop[$1]=1; next } FNR==1 { next } !($4 in drop) { print $0 }' "$selected_n_tmp" "$csv" >> "$tmp_csv"
  else
    tail -n +2 "$csv" >> "$tmp_csv"
  fi

  rm -f "$selected_n_tmp"
fi

mv "$tmp_csv" "$csv"

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
          ACO_SOLVER_STAGNATION_EPOCHS="$stag_iters" \
          ACO_SOLVER_MIN_REL_IMPROVEMENT="$improve_rel" \
          OMP_NUM_THREADS="$omp_threads" \
          "${launcher_cmd[@]}" ./aco_vrp_openmp_mpi.out "$instance_path" "$K" "$m" "$seed_run" </dev/null 2>&1)
        rc=$?

        stats_line="$(grep -Eo '[0-9]+([.][0-9]+)?,[0-9]+' "$stats_file" | tail -n1)"
        elapsed="$(printf '%s' "$stats_line" | cut -d, -f1)"
        rss_kb="$(printf '%s' "$stats_line" | cut -d, -f2)"
        rm -f "$stats_file"
        [ -n "$elapsed" ] || elapsed=""
        rss_gb=""
        if [ -n "$rss_kb" ]; then
          rss_gb="$(awk "BEGIN {printf \"%.6f\", (${rss_kb})/1048576.0}")"
        fi

        printf '%s\n' "$out" > "$sol_file"
        if [ "$rc" -eq 0 ]; then
          cost="$(printf '%s\n' "$out" | sed -n 's/^best cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_gb,$cost," >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_gb,,$err" >> "$csv"
        fi
        echo "[mpi] $name run=$run_id done"
      done
    done

echo "wrote $csv"
