#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p benchmarks/results/logs
timestamp="$(date +%Y%m%dT%H%M%S)"
log_path="benchmarks/results/logs/benchmark_run_${timestamp}.log"

cache_dir="${HOME}/.cache/karaoke"
timeout_sec="5400"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  tools/run_benchmark_suite_bg.sh [runner args...]

Default behavior:
  - resumes latest run folder if present
  - uses cache dir ~/.cache/karaoke
  - timeout 5400s/song
  - default alignment mode (no whisper-map-lrc-dtw)
  - writes log to benchmarks/results/logs/benchmark_run_<timestamp>.log

Examples:
  tools/run_benchmark_suite_bg.sh
  tools/run_benchmark_suite_bg.sh --run-id timing_core_v1
  tools/run_benchmark_suite_bg.sh --resume-run-dir benchmarks/results/20260211T205447Z
  tools/run_benchmark_suite_bg.sh --rerun-failed
EOF
  exit 0
fi

cmd=(
  "./venv/bin/python"
  "-u"
  "tools/run_benchmark_suite.py"
  "--cache-dir"
  "$cache_dir"
  "--timeout-sec"
  "$timeout_sec"
  "--no-whisper-map-lrc-dtw"
  "--resume-latest"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

nohup "${cmd[@]}" >"$log_path" 2>&1 </dev/null &
pid=$!

echo "Started benchmark suite."
echo "PID: $pid"
echo "Log: $log_path"
echo "Follow progress with:"
echo "  tail -f $log_path"
