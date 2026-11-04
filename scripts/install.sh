#!/usr/bin/env bash
# One-command install of OpenImage Studio on Linux with an NVIDIA GPU.
#
#   scripts/install.sh               install everything (about 25 GB of disk, mostly model weights)
#   scripts/install.sh --with-sam3   also SAM 3 for better selection from words (needs Hugging Face access, see README)
#   scripts/install.sh --no-desktop  do not add a launcher to the application menu
#
# Safe to run again: finished steps are skipped. The network is used only here, never while editing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
SAM3=0; DESKTOP=1; MODELS=1
for a in "$@"; do
  case "$a" in
    --with-sam3) SAM3=1 ;;
    --no-desktop) DESKTOP=0 ;;
    --skip-models) MODELS=0 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mNote:\033[0m %s\n' "$*"; }
die() { printf '\n\033[31mCannot install:\033[0m %s\n' "$*" >&2; exit 1; }
# Downloads are retried: slow or flaky connections are normal. Finished steps are skipped on a re-run.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
retry() {
  local n
  for n in 1 2 3 4; do
    "$@" && return 0
    [ $n -lt 4 ] && { warn "download failed (attempt $n of 4); trying again in $((n * 10)) s"; sleep $((n * 10)); }
  done
  die "a download kept failing. Check the internet connection and run scripts/install.sh again; finished steps are skipped."
}

step "Checking this computer"
[ "$(uname -s)" = Linux ] || die "this version runs on Linux with an NVIDIA GPU. Windows and macOS are not supported yet."
command -v nvidia-smi >/dev/null || die "no NVIDIA driver found (nvidia-smi is missing). Install the NVIDIA driver, version 550 or newer, then run this again."
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${DRIVER%%.*}" -ge 550 ] || die "NVIDIA driver $DRIVER is too old; 550 or newer is needed."
echo "GPU: $GPU, ${VRAM_MB} MB, driver $DRIVER"
[ "$VRAM_MB" -ge 3800 ] || die "the GPU has ${VRAM_MB} MB; at least 4 GB is needed."
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576 + 0.5}' /proc/meminfo)
echo "System memory: ${RAM_GB} GB"
[ "$RAM_GB" -ge 15 ] || warn "with ${RAM_GB} GB of memory the editor uses the low-memory mode; edits take longer."
NEED_GB=$([ $MODELS = 1 ] && echo 26 || echo 8)
HAVE_GB=$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1048576}')
HAVE_WEIGHTS=$([ -f assets/weights/flux2-klein-4b/model_index.json ] && echo 1 || echo 0)
if [ "$HAVE_WEIGHTS" = 0 ] && [ "$HAVE_GB" -lt "$NEED_GB" ]; then die "${HAVE_GB} GB free here; about ${NEED_GB} GB is needed."; fi
echo "Disk: ${HAVE_GB} GB free"

step "Python tools (uv)"
UV_BIN_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
if ! command -v uv >/dev/null; then
  if [ ! -x "$UV_BIN_DIR/uv" ]; then
    command -v curl >/dev/null || die "curl is needed to install uv."
    # Shell startup files are left alone; this script finds uv where it put it.
    retry sh -c "curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR='$UV_BIN_DIR' sh"
  fi
  export PATH="$UV_BIN_DIR:$PATH"
fi
uv --version
retry uv python install 3.12
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv

step "PyTorch with CUDA 12.8 (about 3 GB)"
# The cu128 index only: with PyPI as an extra index uv picks CUDA 13 builds that a 550 driver cannot run.
if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.__version__.endswith('+cu128') else 1)" 2>/dev/null; then
  retry uv pip install --python .venv torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python - <<'PY' || die "PyTorch cannot use the GPU. Check the NVIDIA driver (nvidia-smi) and run this again."
import sys, torch
assert torch.__version__.endswith("+cu128"), torch.__version__
ok = torch.cuda.is_available()
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0) if ok else "NOT AVAILABLE")
sys.exit(0 if ok else 1)
PY

step "The editor's libraries"
retry uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps -e .

if [ $MODELS = 1 ]; then
  step "Models (about 19 GB, downloaded once)"
  cp assets/manifest.json assets/manifest.pinned.json 2>/dev/null || true
  retry .venv/bin/python scripts/download_assets.py --only renderer sam
  retry .venv/bin/python scripts/download_assets.py --only grounder --grounder gdino-base
  if [ $SAM3 = 1 ]; then
    if .venv/bin/python scripts/download_assets.py --only grounder --grounder sam3; then echo "SAM 3 installed."
    else warn "SAM 3 could not be downloaded. Request access at https://huggingface.co/facebook/sam3, run '.venv/bin/hf auth login', then run this again with --with-sam3. Grounding DINO is used meanwhile."; fi
  fi
  step "Checking the downloaded files against the tested versions"
  .venv/bin/python scripts/compare_manifest.py assets/manifest.pinned.json assets/manifest.json || warn "some files differ from the versions tested; the editor still runs, but results may differ."
fi

step "Self-test (no GPU models)"
if .venv/bin/python -m pytest -q tests > install-selftest.log 2>&1; then tail -1 install-selftest.log; rm -f install-selftest.log
else tail -3 install-selftest.log; die "the self-test failed; the details are in $REPO/install-selftest.log. Please report them."; fi

chmod +x studio scripts/start.sh
if [ $DESKTOP = 1 ] && [ -d "$HOME/.local/share" ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/openimage-studio.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=OpenImage Studio
Comment=Edit photos on this computer by saying what to change
Exec=$REPO/studio --open
Icon=$REPO/lis/static/icon.svg
Terminal=true
Categories=Graphics;Photography;
DESK
  echo "Added OpenImage Studio to the application menu."
fi

printf '\n'
bold "Installed. Start it with:"
echo "  $REPO/studio --open"
echo "It opens http://127.0.0.1:7860 in your browser. Close the terminal window to stop it."
