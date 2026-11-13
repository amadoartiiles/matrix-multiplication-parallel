#!/usr/bin/env bash
# Full measurement campaign, from inputs to figures. Run from the repository root on AC power:
#
#     bash scripts/run_campaign.sh [WORK_DIR]
#
# WORK_DIR (default: $HOME/mm3_work) holds the inputs (~1.2 GB), the dumped results and the
# log; it should be a local folder outside any cloud-synchronised directory. On Windows the
# script keeps the system awake during the run (scripts/keep_awake.ps1).
set -euo pipefail
WORK="${1:-$HOME/mm3_work}"
PY="${PYTHON:-python}"
command -v py >/dev/null 2>&1 && PY="${PYTHON:-py -3.11}"
mkdir -p "$WORK"
LOG="$WORK/bench_log.txt"

stop_awake() { touch "$WORK/stop_keep_awake"; }
if command -v powershell >/dev/null 2>&1; then
    rm -f "$WORK/stop_keep_awake"
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 "$WORK/stop_keep_awake" &
    trap stop_awake EXIT
fi

{
echo "=== start $(date)"
if [ ! -f "$WORK/data/manifest.json" ]; then
    $PY scripts/generate_inputs.py --data "$WORK/data"
fi
$PY scripts/build.py
echo "=== benchmarks $(date)"
$PY scripts/run_benchmarks.py --data "$WORK/data" --outputs "$WORK/outputs" --resume
echo "=== validation $(date)"
$PY scripts/validate_outputs.py --data "$WORK/data" --outputs "$WORK/outputs"
$PY scripts/collect_env.py
$PY scripts/analyze.py
echo "=== CAMPAIGN DONE $(date)"
} 2>&1 | tee -a "$LOG"
