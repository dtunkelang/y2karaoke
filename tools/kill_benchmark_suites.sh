#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
fi

list_pids() {
  ps -ax -o pid=,command= | awk '
    /tools\/run_benchmark_suite\.py/ { print $1 }
    /y2karaoke\.cli generate/ && /--timing-report/ && /benchmarks\/results/ { print $1 }
  ' | sort -u
}

pids="$(list_pids)"
if [[ -z "${pids}" ]]; then
  echo "No benchmark suite processes found."
  exit 0
fi

echo "Benchmark suite processes:"
for pid in ${pids}; do
  ps -p "$pid" -o pid=,etime=,command=
done

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only; no processes were killed."
  exit 0
fi

echo "Sending SIGTERM..."
kill ${pids} || true
sleep 1

remaining="$(list_pids)"
if [[ -n "${remaining}" ]]; then
  echo "Forcing SIGKILL on remaining processes..."
  kill -9 ${remaining} || true
fi

final="$(list_pids)"
if [[ -n "${final}" ]]; then
  echo "Some benchmark processes are still running:"
  for pid in ${final}; do
    ps -p "$pid" -o pid=,etime=,command=
  done
  exit 1
fi

echo "All benchmark suite processes stopped."
