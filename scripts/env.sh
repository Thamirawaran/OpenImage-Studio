# Source before running anything in the product. Library-level offline configuration; not a firewall.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_TEMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/projects/.gradio"
export PYTHONUNBUFFERED=1
