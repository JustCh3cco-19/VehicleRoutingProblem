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

header="name,profile,instance_path,n,K,m,solver_seed,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_gb,best_cost,error"
tmp_csv="$(mktemp)"
echo "$header" > "$tmp_csv"

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
          echo "$name,$profile,$instance_path,$n,$K,$m_run,$seed_run,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_gb,$cost," >> "$tmp_csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,$n,$K,$m_run,$seed_run,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_gb,,$err" >> "$tmp_csv"
        fi
        echo "[seq] $name run=$run_id done"
      done
    done

bash tools/bash/merge_results_csv_by_n.sh "$csv" "$tmp_csv"
rm -f "$tmp_csv"
echo "wrote $csv"
