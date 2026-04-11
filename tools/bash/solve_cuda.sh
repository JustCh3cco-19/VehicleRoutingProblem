#!/usr/bin/env bash
set -uo pipefail

variant="${SOLVE_CUDA_VARIANT}"
csv="${SOLVE_CSV_DIR}/manifest_cuda_${variant}_per_instance_results.csv"
sol_dir="${SOLVE_SOLUTIONS_DIR}/cuda_${variant}"
manifest="${SOLVE_MANIFEST}"
clients="${SOLVE_CLIENTS:-}"
limit="${SOLVE_LIMIT:-0}"
repeats="${SOLVE_CUDA_REPEATS:-1}"
runtime_s="${SOLVE_SEQ_RUNTIME_EFFECTIVE:-0}"
stag_iters="${SOLVE_SEQ_STAGNATION_EPOCHS:-0}"
improve_eps="${SOLVE_SEQ_MIN_REL_IMPROVEMENT:-0.001}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi

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

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      for run_id in $(seq 1 "$repeats"); do
        seed_run=$((solver_seed + run_id - 1))
        done_runs=$((done_runs + 1))
        if [ "$run_duration_count" -gt 0 ]; then
          eta_run_s="$(awk "BEGIN {printf \"%.2f\", (${run_duration_sum}) / ${run_duration_count}}")"
        elif [ "$runtime_s" != "0" ]; then
          eta_run_s="$runtime_s"
        else
          eta_run_s="n/a"
        fi
        if [ "$run_rss_count" -gt 0 ]; then
          eta_mem_gb="$(awk "BEGIN {printf \"%.3f\", (${run_rss_sum_gb}) / ${run_rss_count}}")"
        else
          eta_mem_gb="n/a"
        fi
        echo "[cuda] ${name} run=${run_id} (${done_runs}/${total_runs}) eta_run_s=${eta_run_s} eta_mem_gb=${eta_mem_gb}"

        sol_file="${sol_dir}/${name}_cuda_${variant}_run${run_id}_solution.txt"
        time_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e,%M" -o "$time_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_ITERS="$stag_iters" \
          ACO_SOLVER_IMPROVE_EPS="$improve_eps" \
          ./aco_vrp_cuda.out "$instance_path" "$K" "$m" "$seed_run" 2>&1)
        rc=$?

        stats_line="$(grep -Eo '[0-9]+([.][0-9]+)?,[0-9]+' "$time_file" | tail -n1)"
        elapsed="$(printf '%s' "$stats_line" | cut -d, -f1)"
        rss_kb="$(printf '%s' "$stats_line" | cut -d, -f2)"
        rm -f "$time_file"
        [ -n "$elapsed" ] || elapsed=""
        if printf '%s' "$elapsed" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
          run_duration_sum="$(awk "BEGIN {printf \"%.6f\", (${run_duration_sum}) + (${elapsed})}")"
          run_duration_count=$((run_duration_count + 1))
        fi
        if [ -n "${rss_kb:-}" ]; then
          rss_gb="$(awk "BEGIN {printf \"%.6f\", (${rss_kb})/1048576.0}")"
          if printf '%s' "$rss_gb" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
            run_rss_sum_gb="$(awk "BEGIN {printf \"%.6f\", (${run_rss_sum_gb}) + (${rss_gb})}")"
            run_rss_count=$((run_rss_count + 1))
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
        echo "[cuda] $name run=$run_id done"
      done
    done

echo "wrote $csv"
