#!/usr/bin/env bash
set -uo pipefail

csv="${SOLVE_CSV_DIR}/manifest_cuda_per_instance_results.csv"
sol_dir="${SOLVE_SOLUTIONS_DIR}/cuda"
manifest="${SOLVE_MANIFEST_CUDA:-${SOLVE_MANIFEST:-instances/test_aligned/manifest_cuda.csv}}"
clients="${SOLVE_CLIENTS:-}"
limit="${SOLVE_LIMIT:-0}"
repeats="${SOLVE_CUDA_REPEATS:-1}"
runtime_s="${SOLVE_SEQ_RUNTIME_EFFECTIVE:-0}"
stag_iters="${SOLVE_SEQ_STAGNATION_EPOCHS:-0}"
improve_rel_raw="${SOLVE_SEQ_MIN_REL_IMPROVEMENT:-0.001}"
cuda_profile="${SOLVE_CUDA_PROFILE:-0}"

# Prefer CUDA-specific controls when provided; keep seq vars as compatibility fallback.
if [ -n "${SOLVE_CUDA_RUNTIME_S:-}" ]; then
  runtime_s="${SOLVE_CUDA_RUNTIME_S}"
fi
if [ -n "${SOLVE_CUDA_STAGNATION_EPOCHS:-}" ]; then
  stag_iters="${SOLVE_CUDA_STAGNATION_EPOCHS}"
fi
if [ -n "${SOLVE_CUDA_MIN_REL_IMPROVEMENT:-}" ]; then
  improve_rel_raw="${SOLVE_CUDA_MIN_REL_IMPROVEMENT}"
fi

if [[ ! "$runtime_s" =~ ^[1-9][0-9]*$ ]] || [ "$runtime_s" -gt 300 ]; then
  echo "[ERROR] SOLVE_CUDA_RUNTIME_S deve essere compreso tra 1 e 300 secondi" >&2
  exit 2
fi
if [ "$repeats" -lt 1 ]; then
  repeats=1
fi
improve_rel="$(awk "BEGIN { v=${improve_rel_raw}; if (v > 1.0) v/=100.0; printf \"%.12g\", v }")"

header="name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,run_id,status,elapsed_s,best_cost,error"

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
        echo "[cuda][avvio] istanza=${name} | run=${run_id} | avanzamento_campagna=${done_runs}/${total_runs}"
        echo "[cuda][limiti] tempo_massimo=${runtime_s}s | epoche_stagnazione=${stag_iters} | durata_stimata=${eta_run_s}s | memoria_stimata=${eta_mem_gb}GiB"

        sol_file="${sol_dir}/${name}_cuda_run${run_id}_solution.txt"
        time_file="$(mktemp)"
        output_file="$(mktemp)"

        /usr/bin/time -f "%e,%M" -o "$time_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_EPOCHS="$stag_iters" \
          ACO_SOLVER_STAGNATION_ITERS="$stag_iters" \
          ACO_SOLVER_MIN_REL_IMPROVEMENT="$improve_rel" \
          ACO_SOLVER_IMPROVE_EPS="$improve_rel" \
          ACO_CUDA_PROFILE="$cuda_profile" \
          ./aco_vrp_cuda.out "$instance_path" "$K" "$m" "$seed_run" 2>&1 \
          | tee "$output_file" \
          | awk '/^\[progress\]/ { print; fflush(); }'
        rc=${PIPESTATUS[0]}
        out="$(cat "$output_file")"
        rm -f "$output_file"

        stats_line="$(grep -Eo '[0-9]+([.][0-9]+)?,[0-9]+' "$time_file" | tail -n1)"
        elapsed="$(printf '%s' "$stats_line" | cut -d, -f1)"
        rss_kb="$(printf '%s' "$stats_line" | cut -d, -f2)"
        rm -f "$time_file"
        [ -n "$elapsed" ] || elapsed=""
        if printf '%s' "$elapsed" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
          run_duration_sum="$(awk "BEGIN {printf \"%.6f\", (${run_duration_sum}) + (${elapsed})}")"
          run_duration_count=$((run_duration_count + 1))
          n_duration_sum[$n]="$(awk "BEGIN {printf \"%.6f\", (${n_duration_sum[$n]:-0}) + (${elapsed})}")"
          n_duration_count[$n]=$(( ${n_duration_count[$n]:-0} + 1 ))
        fi
        if [ -n "${rss_kb:-}" ]; then
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
          cost="$(printf '%s\n' "$out" | sed -n -e 's/^best cost: //p' -e 's/^Final Best Cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,ok,$elapsed,$cost," >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m,$seed_run,$instance_seed,$layout_id,$run_id,error,$elapsed,,$err" >> "$csv"
        fi
        if [ "$rc" -eq 0 ]; then
          echo "[cuda][completato] tempo_trascorso=${elapsed:-n/a}s | tempo_rimanente=0s | best_cost=${cost:-n/a} | stato=ok | memoria=${rss_gb:-n/a}GiB"
        else
          echo "[cuda][completato] tempo_trascorso=${elapsed:-n/a}s | tempo_rimanente=n/a | best_cost=n/a | stato=errore | memoria=${rss_gb:-n/a}GiB"
        fi
        echo
      done
    done

echo "wrote $csv"
