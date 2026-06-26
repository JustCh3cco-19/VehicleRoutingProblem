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
improve_rel_pct="${SOLVE_MPI_MIN_REL_IMPROVEMENT:-0.1}"
mpi_ranks="${SOLVE_MPI_RANKS:-2}"
omp_threads="${SOLVE_MPI_OMP_THREADS:-2}"
launcher_pref="${SOLVE_MPI_LAUNCHER:-auto}"
candidate_k="${SOLVE_CANDIDATE_K:-0}"
repro_mode="${SOLVE_REPRODUCIBILITY_MODE:-0}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi
improve_rel="$improve_rel_pct"

launcher_kind="mpirun"
launcher_cmd=(mpirun -np "$mpi_ranks")
if [ "$launcher_pref" = "srun" ]; then
  launcher_kind="srun"
  launcher_cmd=(srun --mpi=pmix -n "$mpi_ranks" --cpus-per-task "$omp_threads")
elif [ "$launcher_pref" = "auto" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    launcher_kind="srun"
    launcher_cmd=(srun --mpi=pmix -n "$mpi_ranks" --cpus-per-task "$omp_threads")
  fi
fi
echo "[mpi] launcher=${launcher_kind}"

header="name,profile,instance_path,n,K,m,candidate_k,solver_seed,reproducibility_mode,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_gb,best_cost,error"
header_v2="${header},mpi_ranks,omp_threads,batch_id"
batch_id="${SOLVE_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"

# Append-only behavior: keep all prior runs and append new rows.
if [ ! -f "$csv" ] || [ ! -s "$csv" ]; then
  echo "$header_v2" > "$csv"
else
  first_line="$(head -n1 "$csv" 2>/dev/null || true)"
  if [ "$first_line" = "$header" ]; then
    tmp_csv="$(mktemp)"
    echo "$header_v2" > "$tmp_csv"
    tail -n +2 "$csv" | awk 'NF > 0 { print $0 ",,,,,," }' >> "$tmp_csv"
    mv "$tmp_csv" "$csv"
  fi
fi

selected_instances="$(tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | wc -l)"
if [ -z "$selected_instances" ]; then
  selected_instances=0
fi
total_runs=$((selected_instances * repeats))
done_runs=0
run_duration_sum="0.0"
run_duration_count=0
run_rss_sum_gb="0.0"
run_rss_count=0
declare -A n_duration_sum
declare -A n_duration_count
declare -A n_rss_sum_gb
declare -A n_rss_count

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      for run_id in $(seq 1 "$repeats"); do
        seed_run=$((solver_seed + run_id - 1))
        done_runs=$((done_runs + 1))
        if [ "${n_duration_count[$n]:-0}" -gt 0 ]; then
          eta_run_s="$(awk "BEGIN {printf \"%.2f\", (${n_duration_sum[$n]}) / ${n_duration_count[$n]}}")"
        elif [ "$runtime_s" != "0" ]; then
          eta_run_s="$runtime_s"
        else
          eta_run_s="n/a"
        fi
        if [ "${n_rss_count[$n]:-0}" -gt 0 ]; then
          eta_mem_gb="$(awk "BEGIN {printf \"%.3f\", (${n_rss_sum_gb[$n]}) / ${n_rss_count[$n]}}")"
        else
          eta_mem_gb="n/a"
        fi
        echo "[mpi] ${name} run=${run_id} (${done_runs}/${total_runs})"
        echo "[mpi] run_stimata: eta_run_s=${eta_run_s} eta_mem_gb=${eta_mem_gb}"

        sol_file="${sol_dir}/${name}_mpi_run${run_id}_solution.txt"
        stats_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e,%M" -o "$stats_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_EPOCHS="$stag_iters" \
          ACO_SOLVER_MIN_REL_IMPROVEMENT="$improve_rel" \
          ACO_SOLVER_CANDIDATE_K="$candidate_k" \
          ACO_SOLVER_REPRODUCIBILITY_MODE="$repro_mode" \
          OMP_NUM_THREADS="$omp_threads" \
          "${launcher_cmd[@]}" ./aco_vrp_openmp_mpi.out "$instance_path" "$K" "$m" "$seed_run" </dev/null 2>&1)
        rc=$?

        stats_line="$(grep -Eo '[0-9]+([.][0-9]+)?,[0-9]+' "$stats_file" | tail -n1)"
        elapsed="$(printf '%s' "$stats_line" | cut -d, -f1)"
        rss_kb="$(printf '%s' "$stats_line" | cut -d, -f2)"
        rm -f "$stats_file"
        [ -n "$elapsed" ] || elapsed=""
        if printf '%s' "$elapsed" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
          run_duration_sum="$(awk "BEGIN {printf \"%.6f\", (${run_duration_sum}) + (${elapsed})}")"
          run_duration_count=$((run_duration_count + 1))
          n_duration_sum[$n]="$(awk "BEGIN {printf \"%.6f\", (${n_duration_sum[$n]:-0}) + (${elapsed})}")"
          n_duration_count[$n]=$(( ${n_duration_count[$n]:-0} + 1 ))
        fi
        rss_gb=""
        if [ -n "$rss_kb" ]; then
          rss_gb="$(awk "BEGIN {printf \"%.6f\", (${rss_kb})/1048576.0}")"
          if printf '%s' "$rss_gb" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
            run_rss_sum_gb="$(awk "BEGIN {printf \"%.6f\", (${run_rss_sum_gb}) + (${rss_gb})}")"
            run_rss_count=$((run_rss_count + 1))
            n_rss_sum_gb[$n]="$(awk "BEGIN {printf \"%.6f\", (${n_rss_sum_gb[$n]:-0}) + (${rss_gb})}")"
            n_rss_count[$n]=$(( ${n_rss_count[$n]:-0} + 1 ))
          fi
        fi

        printf '%s\n' "$out" > "$sol_file"
        if [ "$rc" -eq 0 ]; then
          cost="$(printf '%s\n' "$out" | sed -n 's/^best cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,$n,$K,$m,$candidate_k,$seed_run,$repro_mode,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_gb,$cost,,$mpi_ranks,$omp_threads,$batch_id" >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m,$candidate_k,$seed_run,$repro_mode,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_gb,,$err,$mpi_ranks,$omp_threads,$batch_id" >> "$csv"
        fi
        echo "[mpi] run_effettiva: elapsed_s=${elapsed:-n/a} mem_gb=${rss_gb:-n/a} status=$([ "$rc" -eq 0 ] && echo ok || echo error)"
        echo
      done
    done

echo "wrote $csv"
