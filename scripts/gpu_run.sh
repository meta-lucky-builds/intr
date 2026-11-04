#!/usr/bin/env bash
# Thread P, one command. See docs/GPU_RUN.md. Safe to rerun: finished jobs are skipped.
#   bash scripts/gpu_run.sh            full run, detached from the terminal
#   bash scripts/gpu_run.sh --smoke    tiny run in the foreground, minutes on a CPU
#   bash scripts/gpu_run.sh --status   show progress
#   bash scripts/gpu_run.sh --repair-variant idm|state   pilot stage only: rung 0, pilot seed, this encoder objective
# Never add set -x here: this script's environment may hold a token.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${INTR_P_OUT:-$ROOT/results/thread_p_gpu}"
LOG="$OUT/run.log"
PIDFILE="$OUT/run.pid"
mkdir -p "$OUT"

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
  return 1
}

for arg in "$@"; do
  if [ "$arg" = "--status" ]; then
    if pid="$(running_pid)"; then echo "Running, pid $pid."; else echo "Not running."; fi
    tail -n 30 "$LOG" 2>/dev/null
    exit 0
  fi
done

SMOKE=0
for arg in "$@"; do [ "$arg" = "--smoke" ] && SMOKE=1; done

if [ "$SMOKE" = 0 ] && [ -z "${INTR_P_DETACHED:-}" ]; then
  if pid="$(running_pid)"; then
    echo "Already running, pid $pid. Watching its log. Ctrl-C stops watching, not the run."
    exec tail -n 40 -f "$LOG"
  fi
  INTR_P_DETACHED=1 nohup setsid bash "${BASH_SOURCE[0]}" "$@" >>"$LOG" 2>&1 </dev/null &
  PID=$!
  echo "$PID" >"$PIDFILE"
  echo "Started, pid $PID. It keeps running if this terminal closes."
  echo "  Watch:      tail -f $LOG"
  echo "  Status:     bash scripts/gpu_run.sh --status"
  echo "  Job logs:   $OUT/<run>/logs/"
  echo "  Stop:       kill $PID"
  echo "Watching now. Ctrl-C stops watching, not the run."
  tail -n +1 --pid="$PID" -f "$LOG"
  exit 0
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else
    echo "No python found."; exit 2
  fi
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is needed, found: $("$PY" --version 2>&1)"; exit 2
fi

has_module() { "$PY" -c "import $1" >/dev/null 2>&1; }

pick_torch_index() {
  # The newest PyTorch wheel index not above the driver's CUDA version. CPU wheels when there is no driver.
  local version=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    version="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1)"
  fi
  if [ -z "$version" ]; then echo cpu; return; fi
  local pair ver name
  for pair in 13.0:cu130 12.9:cu129 12.8:cu128 12.6:cu126 12.4:cu124 12.1:cu121 11.8:cu118; do
    ver="${pair%%:*}"; name="${pair##*:}"
    if [ "$(printf '%s\n%s\n' "$ver" "$version" | sort -V | head -n 1)" = "$ver" ]; then echo "$name"; fi
  done
}

if ! has_module torch; then
  echo "torch is missing. Installing a wheel that fits this machine."
  installed=0
  for index in $(pick_torch_index); do
    if "$PY" -m pip install --quiet torch --index-url "https://download.pytorch.org/whl/$index"; then
      echo "Installed torch from index $index."; installed=1; break
    fi
    echo "Index $index did not work, trying the next."
  done
  if [ "$installed" = 0 ]; then echo "Could not install torch. Install it by hand, then rerun this command."; exit 2; fi
else
  echo "torch is present and stays untouched: $("$PY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
fi
if ! has_module numpy; then "$PY" -m pip install --quiet numpy || { echo "Could not install numpy."; exit 2; }; fi
if ! has_module pytest; then "$PY" -m pip install --quiet pytest || { echo "Could not install pytest."; exit 2; }; fi

if [ "$SMOKE" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "nvidia-smi sees a GPU, but this torch cannot use CUDA: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "Nothing was changed. Use an environment with a CUDA build of torch, then rerun. Stopping before spending anything."
    exit 2
  fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT" || exit 2
"$PY" -m intr.pixel.runner --out "$OUT" "$@"
code=$?
[ "$SMOKE" = 0 ] && rm -f "$PIDFILE"
exit "$code"
