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
improve_rel_pct="${SOLVE_SEQ_MIN_REL_IMPROVEMENT:-0.1}"
candidate_k="${SOLVE_CANDIDATE_K:-0}"
repro_mode="${SOLVE_REPRODUCIBILITY_MODE:-0}"

if [ "$repeats" -lt 1 ]; then
  repeats=1
fi
commit_hash="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
compiler="$(gcc -dumpversion 2>/dev/null || echo "gcc")"
compiler_flags="-Wall -Wextra -std=c11 -Iinclude -O3"
cpu_model="$(lscpu | grep "Model name:" | sed 's/Model name:[[:space:]]*//' | xargs || hostname)"
gpu_model="n/a"
host_gpu="${cpu_model} / ${gpu_model}"

header="name,profile,instance_path,backend,commit_hash,compiler,compiler_flags,host_gpu,n,K,capacity,m,candidate_k,solver_seed,reproducibility_mode,instance_seed,layout_id,run_id,status,elapsed_s,max_rss_gb,best_cost,validation,exit_status,mpi_ranks,omp_threads,cuda_arch,timeout,stagnation,min_improvement,error"

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
      m_run="$m"
      if [ "$seq_m" -gt 0 ]; then
        m_run="$seq_m"
      fi

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
        echo "[seq] ${name} run=${run_id} (${done_runs}/${total_runs})"
        echo "[seq] run_stimata: eta_run_s=${eta_run_s} eta_mem_gb=${eta_mem_gb}"

        sol_file="${sol_dir}/${name}_seq_run${run_id}_solution.txt"
        stats_file="$(mktemp)"

        out=$(/usr/bin/time -f "%e,%M" -o "$stats_file" env \
          ACO_SOLVER_TIMEOUT_SECONDS="$runtime_s" \
          ACO_SOLVER_STAGNATION_EPOCHS="$stag_iters" \
          ACO_SOLVER_MIN_REL_IMPROVEMENT="$improve_rel" \
          ACO_SOLVER_CANDIDATE_K="$candidate_k" \
          ACO_SOLVER_REPRODUCIBILITY_MODE="$repro_mode" \
          ./aco_vrp_seq.out "$instance_path" "$K" "$m_run" "$seed_run" 2>&1)
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

        capacity="0"
        if [ -f "$instance_path" ]; then
          capacity="$(grep -i "^CAPACITY" "$instance_path" | awk -F: '{print $2}' | tr -d '[:space:]' || echo "0")"
        fi

        cand_k_eff="$(printf '%s\n' "$out" | sed -n 's/.*candidate_k=\([0-9]\+\).*/\1/p' | tail -n1)"
        if [ -z "$cand_k_eff" ]; then
          cand_k_eff="$candidate_k"
        fi

        validation="n/a"
        if [ "$rc" -eq 0 ]; then
          validation="valid"
        elif printf '%s\n' "$out" | grep -Eq "invalid solution|invalid cost|invalid solution cost"; then
          validation="invalid"
        fi

        printf '%s\n' "$out" > "$sol_file"
        if [ "$rc" -eq 0 ]; then
          cost="$(printf '%s\n' "$out" | sed -n 's/^best cost: //p' | tail -n1)"
          echo "$name,$profile,$instance_path,seq,$commit_hash,\"$compiler\",\"$compiler_flags\",\"$host_gpu\",$n,$K,$capacity,$m_run,$cand_k_eff,$seed_run,$repro_mode,$instance_seed,$layout_id,$run_id,ok,$elapsed,$rss_gb,$cost,$validation,$rc,1,1,n/a,$runtime_s,$stag_iters,$improve_rel," >> "$csv"
        else
          err="$(printf '%s' "$out" | tr '\n' ' ' | tr ',' ';')"
          echo "$name,$profile,$instance_path,seq,$commit_hash,\"$compiler\",\"$compiler_flags\",\"$host_gpu\",$n,$K,$capacity,$m_run,$cand_k_eff,$seed_run,$repro_mode,$instance_seed,$layout_id,$run_id,error,$elapsed,$rss_gb,,$validation,$rc,1,1,n/a,$runtime_s,$stag_iters,$improve_rel,$err" >> "$csv"
        fi
        echo "[seq] run_effettiva: elapsed_s=${elapsed:-n/a} mem_gb=${rss_gb:-n/a} status=$([ "$rc" -eq 0 ] && echo ok || echo error)"
        echo
      done
    done

echo "wrote $csv"
