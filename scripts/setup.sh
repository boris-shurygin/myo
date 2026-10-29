#!/usr/bin/env bash
# Create the tested Python environment without depending on system ensurepip.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'uv is required. See docs/installation.md for installation commands.' >&2
    exit 1
fi

venv_path="$project_root/.venv"
if [[ -e "$venv_path" || -L "$venv_path" ]]; then
    if [[ ! -f "$venv_path/bin/activate" ]] || ! "$venv_path/bin/python3" -I -c \
        'import sys; sys.exit(not (sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix))' \
        >/dev/null 2>&1; then
        backup_path="$(mktemp -d "$project_root/.venv-backup.XXXXXXXX")"
        mv -- "$venv_path" "$backup_path/.venv"
        printf 'Previous environment preserved at %s/.venv\n' "$backup_path"
    fi
fi

if [[ ! -e "$venv_path" ]]; then
    uv venv --managed-python --python 3.12 --seed "$venv_path"
fi

# The explicit interpreter prevents installation into another active environment.
# Keep pip available for the subsequent python3 -m pip commands in the README.
uv pip install --python "$venv_path/bin/python3" 'pip>=25' --editable '.[dev]'

"$venv_path/bin/python3" -I -c \
    'import sys, myo, gymnasium, mujoco, numpy; print("Ready:", sys.executable, sys.version.split()[0])'
"$venv_path/bin/python3" -m pip check
printf '%s\n' 'Activate with: source .venv/bin/activate' \
    'Then run: python3 -m myo rollout --seconds 10 --controller heuristic'
