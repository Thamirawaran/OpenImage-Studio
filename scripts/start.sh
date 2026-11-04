#!/usr/bin/env bash
# Start OpenImage Studio with plain checks first.
#   scripts/start.sh [--open] [--lan] [--port 7860] [--profile group] [other lis.web flags]
#   --open  open the browser    --lan  let phones and other computers on your network open it
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail() { echo "Cannot start: $1" >&2; exit 1; }

[ -x .venv/bin/python ] || fail "not installed yet. Run scripts/install.sh first."
command -v nvidia-smi >/dev/null || fail "no NVIDIA driver found (nvidia-smi missing). This version needs an NVIDIA GPU."
.venv/bin/python - <<'PY' || exit 1
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("Cannot start: PyTorch sees no CUDA GPU. Run scripts/install.sh again.")
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 1e9:.1f} of {total / 1e9:.1f} GB free")
if free < 3e9:
    print("Note: under 3 GB of GPU memory is free; another program may be using it.")
PY
[ -f assets/weights/flux2-klein-4b/model_index.json ] || fail "the editing model is missing. Run scripts/install.sh."
[ -f assets/weights/sam2/config.json ] || fail "the selection model is missing. Run scripts/install.sh."
if [ -f assets/weights/sam3/config.json ]; then echo "Selection from words: SAM 3";
elif [ -f assets/weights/gdino-base/config.json ]; then echo "Selection from words: Grounding DINO";
else echo "Selection from words: off (run scripts/install.sh). Clicking and painting still work."; fi
ram_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576 + 0.5}' /proc/meminfo)
# Memory profile, unless --profile is given (README, Hardware and memory). The worker keeps the
# editing model on the GPU when enough video memory is free (resident, or fast), which is
# lossless and the quickest; otherwise it offloads by block:
#   24 GB of memory or more: group, lossless; the weights stay in the file cache.
#   15 to 23 GB:             group_fp8, nothing read from disk while editing; judged equal by eye.
#   less:                    group; the weights are read from disk on every edit, so it is slower.
AUTO_PROFILE=""
if ! printf '%s\n' "$@" | grep -q -- '--profile'; then
  if [ "$ram_gb" -ge 24 ]; then AUTO_PROFILE=auto
  elif [ "$ram_gb" -ge 15 ]; then AUTO_PROFILE=auto/group_fp8
  else AUTO_PROFILE=auto; echo "Note: ${ram_gb} GB of memory. Unless the GPU holds the model, it is read from disk during each edit, so edits take longer."; fi
  echo "Memory profile: $AUTO_PROFILE (${ram_gb} GB of memory; choose another with --profile)"
fi

source scripts/env.sh
ARGS=()
[ -n "$AUTO_PROFILE" ] && ARGS+=("--profile" "$AUTO_PROFILE")
LAN=0
PORT=7860
while [ $# -gt 0 ]; do
  case "$1" in
    --lan) LAN=1 ;;
    --port) PORT="$2"; ARGS+=("--port" "$2"); shift ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
if [ $LAN = 1 ]; then
  ARGS+=("--host" "0.0.0.0")
  IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  KEY=$(.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(9))")
  ARGS+=("--key" "$KEY")
  echo "Open on this computer: http://127.0.0.1:${PORT}"
  echo "Open on a phone or another computer on the same network: http://${IP}:${PORT}/?key=${KEY}"
  echo "Only devices that open that address (with its key) get in. Photos still stay on this computer."
else
  echo "Open http://127.0.0.1:${PORT} (only this computer can reach it; --lan allows other devices)"
fi
exec .venv/bin/python -m lis.web "${ARGS[@]}"
