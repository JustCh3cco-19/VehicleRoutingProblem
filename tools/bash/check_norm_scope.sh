#!/usr/bin/env bash
set -u

status=0
tmp_file="$(mktemp)"

find src include -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' \) \
	| sort > "$tmp_file"

check_line_width()
{
	local found

	found=0
	while IFS= read -r file
	do
		awk '
		{
			line = $0
			gsub(/\t/, "    ", line)
			if (length(line) > 80)
			{
				printf("%s:%d:%d:%s\n", FILENAME, FNR, length(line), line)
				found = 1
			}
		}
		END { exit found }
		' "$file" || found=1
	done < "$tmp_file"
	if [ "$found" -ne 0 ]
	then
		printf '\n[norm] line width violations, tabs expanded to 4 spaces\n'
		status=1
	fi
}

check_multiline_macros()
{
	local output

	output="$(rg -n '#[[:space:]]*define .*\\$' src include \
		-g '*.c' -g '*.h' -g '*.cu' || true)"
	if [ -n "$output" ]
	then
		printf '%s\n' "$output"
		printf '\n[norm] multiline macro violations\n'
		status=1
	fi
}

check_function_counts()
{
	local file
	local count
	local output

	output=""
	while IFS= read -r file
	do
		case "$file" in
			*.h)
				continue
				;;
		esac
		count="$(rg -c '^(static )?(__device__ |__global__ )?[a-zA-Z_][a-zA-Z0-9_ *	]*	[a-zA-Z_][a-zA-Z0-9_]*\([^;]*$' "$file" || true)"
		if [ "$count" -gt 6 ]
		then
			output="${output}${file}:${count}"$'\n'
		fi
	done < "$tmp_file"
	if [ -n "$output" ]
	then
		printf '%s' "$output"
		printf '\n[norm] more than 6 function definitions per source file\n'
		status=1
	fi
}

check_camel_case_constants()
{
	local output

	output="$(rg -n '\bk[A-Z][A-Za-z0-9_]*\b' src include \
		-g '*.c' -g '*.h' -g '*.cu' || true)"
	if [ -n "$output" ]
	then
		printf '%s\n' "$output"
		printf '\n[norm] non-snake-case enum constants\n'
		status=1
	fi
}

check_scope_leaks()
{
	local output

	output="$(find experiments exp_dist_calc -type f \
		\( -name '*.c' -o -name '*.h' -o -name '*.cu' \) 2>/dev/null \
		| sort || true)"
	if [ -n "$output" ]
	then
		printf '[norm] excluded source trees present and ignored:\n'
		printf '%s\n' "$output" | sed -n '1,12p'
		if [ "$(printf '%s\n' "$output" | wc -l)" -gt 12 ]
		then
			printf '...\n'
		fi
	fi
}

check_scope_leaks
check_line_width
check_multiline_macros
check_function_counts
check_camel_case_constants
rm -f "$tmp_file"
exit "$status"
