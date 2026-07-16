#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <dest_csv> <new_csv>" >&2
  exit 2
fi

dest_csv="$1"
new_csv="$2"

if [ ! -f "$new_csv" ] || [ ! -s "$new_csv" ]; then
  echo "merge_results_csv_by_n: missing or empty new CSV: $new_csv" >&2
  exit 1
fi

header="$(head -n1 "$new_csv")"
if [ "$(awk -F, 'NR == 1 { print NF }' "$new_csv")" -lt 4 ]; then
  echo "merge_results_csv_by_n: expected at least four CSV columns: $new_csv" >&2
  exit 1
fi
if [ -f "$dest_csv" ] && [ -s "$dest_csv" ] && [ "$(head -n1 "$dest_csv")" != "$header" ]; then
  echo "merge_results_csv_by_n: CSV headers do not match" >&2
  exit 1
fi

dest_dir="$(dirname "$dest_csv")"
mkdir -p "$dest_dir"
tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

echo "$header" > "$tmp_out"

if [ -f "$dest_csv" ] && [ -s "$dest_csv" ]; then
  ns_tmp="$(mktemp)"
  trap 'rm -f "$tmp_out" "$ns_tmp"' EXIT
  tail -n +2 "$new_csv" | awk -F, 'NF > 0 && $4 != "" { print $4 }' | sort -u > "$ns_tmp"

  if [ -s "$ns_tmp" ]; then
    awk -F, 'NR==FNR { replace_n[$1]=1; next } FNR==1 { next } !($4 in replace_n) { print $0 }' "$ns_tmp" "$dest_csv" >> "$tmp_out"
  else
    tail -n +2 "$dest_csv" >> "$tmp_out"
  fi

  rm -f "$ns_tmp"
fi

tail -n +2 "$new_csv" >> "$tmp_out"
mv "$tmp_out" "$dest_csv"
