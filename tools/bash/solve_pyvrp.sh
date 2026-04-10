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

echo "name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,status,elapsed_s,max_rss_kb,best_cost,error" > "$csv"

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      sol_file="${sol_dir}/${name}_pyvrp_solution.txt"
      rss_file="$(mktemp)"
      start_ns="$(date +%s%N)"
      out=$(/usr/bin/time -f "%M" -o "$rss_file" \
        "$py_bin" tools/python/solve_pyvrp_runner.py \
        "$instance_path" "$runtime_s" "$seed" "$sol_file" 2>&1)
      rc=$?
      end_ns="$(date +%s%N)"
      elapsed="$(awk "BEGIN {printf \"%.6f\", ($end_ns-$start_ns)/1000000000}")"
      rss_kb="$(cat "$rss_file" 2>/dev/null)"
      rm -f "$rss_file"

      if [ "$rc" -eq 0 ]; then
        cost="$(printf '%s\n' "$out" | sed -n 's/^best_cost=//p' | tail -n1)"
        [ -n "$cost" ] || cost=""
        echo "$name,$profile,$instance_path,$n,$K,$m,$solver_seed,$instance_seed,$layout_id,ok,$elapsed,$rss_kb,$cost," >> "$csv"
      else
        err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
        echo "$name,$profile,$instance_path,$n,$K,$m,$solver_seed,$instance_seed,$layout_id,error,$elapsed,$rss_kb,,$err" >> "$csv"
      fi
      echo "[pyvrp] $name done"
    done

echo "wrote $csv"
