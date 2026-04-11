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

tail -n +2 "$manifest" \
  | { if [ -n "$clients" ]; then awk -F, -v list="$clients" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($4 in wanted)'; else cat; fi; } \
  | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } \
  | while IFS=, read -r profile name instance_path n K m solver_seed instance_seed layout_id capacity_formula; do
      for run_id in $(seq 1 "$repeats"); do
        seed_run=$((solver_seed + run_id - 1))
        sol_file="${sol_dir}/${name}_cuda_${variant}_run${run_id}_solution.txt"
        time_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e" -o "$time_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_ITERS="$stag_iters" \
          ACO_SOLVER_IMPROVE_EPS="$improve_eps" \
          ./aco_vrp_cuda.out "$instance_path" "$K" "$m" "$seed_run" 2>&1)
        rc=$?

        elapsed="$(cat "$time_file" 2>/dev/null)"
        rm -f "$time_file"
        [ -n "$elapsed" ] || elapsed=""

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
