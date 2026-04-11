#!/usr/bin/env bash
set -uo pipefail

csv="${SOLVE_CSV_DIR}/manifest_pyvrp_per_instance_results.csv"
sol_dir="${SOLVE_SOLUTIONS_DIR}/pyvrp"
manifest="${SOLVE_MANIFEST}"
clients="${SOLVE_CLIENTS:-}"
limit="${SOLVE_LIMIT:-0}"
runtime_s="${SOLVE_PYVRP_RUNTIME_S:-10}"
seed="${SOLVE_PYVRP_SEED:-1234}"
py_bin="${PYTHON_BIN:-python3}"

if [ -x "VRP/bin/python" ]; then
  py_bin="VRP/bin/python"
fi

header="name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,status,elapsed_s,max_rss_gb,best_cost,error"

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
total_runs="$selected_instances"
done_runs=0
run_duration_sum="0.0"
run_duration_count=0
run_rss_sum_gb="0.0"
run_rss_count=0

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
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
      echo "[pyvrp] ${name} (${done_runs}/${total_runs}) eta_run_s=${eta_run_s} eta_mem_gb=${eta_mem_gb}"

      sol_file="${sol_dir}/${name}_pyvrp_solution.txt"
      rss_file="$(mktemp)"
      start_ns="$(date +%s%N)"
      out=$(/usr/bin/time -f "%M" -o "$rss_file" \
        "$py_bin" tools/python/solve_pyvrp_runner.py \
        "$instance_path" "$runtime_s" "$seed" "$sol_file" 2>&1)
      rc=$?
      end_ns="$(date +%s%N)"
      elapsed="$(awk "BEGIN {printf \"%.6f\", ($end_ns-$start_ns)/1000000000}")"
      if printf '%s' "$elapsed" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
        run_duration_sum="$(awk "BEGIN {printf \"%.6f\", (${run_duration_sum}) + (${elapsed})}")"
        run_duration_count=$((run_duration_count + 1))
      fi
      rss_kb="$(grep -Eo '[0-9]+' "$rss_file" | tail -n1)"
      rm -f "$rss_file"
      rss_gb=""
      if [ -n "$rss_kb" ]; then
        rss_gb="$(awk "BEGIN {printf \"%.6f\", (${rss_kb})/1048576.0}")"
        if printf '%s' "$rss_gb" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
          run_rss_sum_gb="$(awk "BEGIN {printf \"%.6f\", (${run_rss_sum_gb}) + (${rss_gb})}")"
          run_rss_count=$((run_rss_count + 1))
        fi
      fi

      if [ "$rc" -eq 0 ]; then
        cost="$(printf '%s\n' "$out" | sed -n 's/^best_cost=//p' | tail -n1)"
        [ -n "$cost" ] || cost=""
        echo "$name,$profile,$instance_path,$n,$K,$m,$solver_seed,$instance_seed,$layout_id,ok,$elapsed,$rss_gb,$cost," >> "$csv"
      else
        err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
        echo "$name,$profile,$instance_path,$n,$K,$m,$solver_seed,$instance_seed,$layout_id,error,$elapsed,$rss_gb,,$err" >> "$csv"
      fi
      echo "[pyvrp] $name done"
    done

echo "wrote $csv"
