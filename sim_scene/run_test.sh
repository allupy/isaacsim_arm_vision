#!/usr/bin/env bash
# Run a single closed-loop task in an already running feedback control container.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_python.sh" main.py --once "$@"
