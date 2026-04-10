#!/usr/bin/env bash
set -uo pipefail

csv="${SOLVE_CSV_DIR}/manifest_seq_per_instance_results.csv"
sol_dir="${SOLVE_SOLUTIONS_DIR}/seq"
manifest="${SOLVE_MANIFEST}"
clients="${SOLVE_CLIENTS:-}"
limit="${SOLVE_LIMIT:-0}"
seq_m="${SOLVE_SEQ_M:-0}"
repeats="${SOLVE_SEQ_REPEATS:-1}"
runtime_s="${SOLVE_SEQ_RUNTIME_EFFECTIVE:-0}"
stag_iters="${SOLVE_SEQ_STAGNATION_EPOCHS:-0}"
improve_rel_pct="${SOLVE_SEQ_MIN_REL_IMPROVEMENT:-0.001}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi
improve_rel="$(awk "BEGIN { printf \"%.12g\", (${improve_rel_pct}) / 100.0 }")"

echo "name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_kb,best_cost,error" > "$csv"

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      m_run="$m"
      if [ "$seq_m" -gt 0 ]; then
        m_run="$seq_m"
      fi

      for run_id in $(seq 1 "$repeats"); do
        seed_run=$((solver_seed + run_id - 1))
        sol_file="${sol_dir}/${name}_seq_run${run_id}_solution.txt"
        stats_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e,%M" -o "$stats_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_EPOCHS="$stag_iters" \
          ACO_SOLVER_MIN_REL_IMPROVEMENT="$improve_rel" \
          ./aco_vrp_seq.out "$instance_path" "$K" "$m_run" "$seed_run" 2>&1)
        rc=$?

        elapsed="$(cut -d, -f1 "$stats_file" 2>/dev/null)"
        rss_kb="$(cut -d, -f2 "$stats_file" 2>/dev/null)"
        rm -f "$stats_file"
        [ -n "$elapsed" ] || elapsed=""
        [ -n "$rss_kb" ] || rss_kb=""

        printf '%s\n' "$out" > "$sol_file"
        if [ "$rc" -eq 0 ]; then
          cost="$(printf '%s\n' "$out" | sed -n 's/^best cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,$n,$K,$m_run,$seed_run,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_kb,$cost," >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m_run,$seed_run,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_kb,,$err" >> "$csv"
        fi
        echo "[seq] $name run=$run_id done"
      done
    done

echo "wrote $csv"
