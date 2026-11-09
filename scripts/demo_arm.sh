#!/usr/bin/env bash
# Launch the best tested local arm policy from any working directory.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec "$project_root/.venv/bin/python3" "$project_root/scripts/demo_arm_preacher.py" --view "$@"
