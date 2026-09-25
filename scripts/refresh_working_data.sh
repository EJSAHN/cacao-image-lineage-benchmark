#!/usr/bin/env bash
set -euo pipefail
TARGET=${1:-${CACAO_WORKING_ROOT:-}}
[[ -n "$TARGET" ]] || { echo "Usage: $0 WORKING_ROOT" >&2; exit 2; }
[[ -d "$TARGET" ]] || { echo "Working directory does not exist: $TARGET" >&2; exit 2; }
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT=${CACAO_PROJECT_ROOT:-$PWD}/refresh_working_data_${STAMP}.tsv
printf 'path	files_touched	bytes	refreshed_at
' > "$REPORT"
files=$(find "$TARGET" -type f | wc -l)
bytes=$(du -sb "$TARGET" | awk '{print $1}')
find "$TARGET" -type f -print0 | xargs -0 -r touch -a -m
find "$TARGET" -type d -print0 | xargs -0 -r touch -a -m
printf '%s	%s	%s	%s
' "$TARGET" "$files" "$bytes" "$(date --iso-8601=seconds)" >> "$REPORT"
echo "refresh_report=$REPORT"
