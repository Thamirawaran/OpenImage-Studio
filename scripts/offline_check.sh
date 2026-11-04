#!/usr/bin/env bash
# Hard offline proof for the CLI path.
# Runs `lis edit` inside a user+network namespace with only the loopback interface up.
# There is no interface to leave through; the kernel refuses any outbound connect.
#
# Usage: scripts/offline_check.sh PHOTO "instruction" [extra lis edit args...]
#   e.g. scripts/offline_check.sh samples/cars.jpg "make the pink car copper" --click 1700,700
#   add --fake to prove the mechanism without models.
set -euo pipefail
cd "$(dirname "$0")/.."
photo="$1"; instruction="$2"; shift 2
mkdir -p runs/offline
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="runs/offline/$stamp.log"
py="${PYTHON:-$(command -v python)}"
if ! unshare -rn true 2>/dev/null; then
  echo "this kernel refuses unprivileged user namespaces (Ubuntu 23.10+ sets kernel.apparmor_restrict_unprivileged_userns=1)." >&2
  echo "run this proof on the GPU server (Ubuntu 22.04), or as root with: sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0" >&2
  exit 3
fi
{
  echo "== offline check $stamp"
  echo "host: $(hostname)  kernel: $(uname -r)  python: $py"
  echo "== interfaces inside the namespace:"
  unshare -rn sh -c 'ip link set lo up 2>/dev/null || true; ip -brief link'
  echo "== outbound connect attempt inside the namespace (must fail):"
  unshare -rn sh -c "ip link set lo up 2>/dev/null; $py - <<'PY'
import socket
try:
    socket.create_connection(('1.1.1.1', 443), timeout=3); print('UNEXPECTED: connect succeeded')
except OSError as e:
    print('connect refused by the kernel as expected:', e)
PY"
  echo "== lis edit inside the namespace:"
  set +e
  unshare -rn sh -c "ip link set lo up 2>/dev/null; set -a; . scripts/env.sh; set +a; PYTHONPATH=. $py -m lis.cli edit '$photo' --instruction \"$instruction\" $* --out runs/offline/$stamp.png"
  rc=$?
  set -e
  echo "== lis edit exit code: $rc"
} 2>&1 | tee "$log"
echo "log: $log"
