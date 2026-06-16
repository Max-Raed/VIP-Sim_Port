#!/usr/bin/env bash
#
# One-command VipSim shader-equivalence validation.
#
# Reproduces the full Tier-2 pixel-fidelity validation from the repo alone —
# no Unity required (the Unity reference renders + burst captures are committed
# under vipsim_assets/unity_refs/). Sets up a virtualenv, installs deps, then
# runs the static + time-varying (burst) validators and builds the visuals.
#
# Usage:
#   ./validate.sh              # full run
#   ./validate.sh --no-venv    # use the current Python env, skip venv setup
#
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="python3"

if [[ "${1:-}" != "--no-venv" ]]; then
  if [[ ! -d .venv ]]; then
    echo ">> Creating virtualenv at .venv"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY="python"
  echo ">> Installing dependencies (requirements.txt)"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
fi

echo
echo "============================================================"
echo " 1/5  Static / single-frame filters"
echo "============================================================"
"$PY" validation_scripts/validate_all_filters.py

echo
echo "============================================================"
echo " 2/5  Burst — flickering_stars"
echo "============================================================"
"$PY" validation_scripts/validate_flickering_stars_burst.py

echo
echo "============================================================"
echo " 3/5  Burst — wiggle"
echo "============================================================"
"$PY" validation_scripts/validate_wiggle_burst.py

echo
echo "============================================================"
echo " 4/5  Burst — nystagmus"
echo "============================================================"
"$PY" validation_scripts/validate_nystagmus_burst.py

echo
echo "============================================================"
echo " 5/5  Build burst visuals (grid.png + loop.gif per filter)"
echo "============================================================"
"$PY" validation_scripts/build_burst_visuals.py

echo
echo "============================================================"
echo " DONE. Outputs:"
echo "   $ROOT/abs_diff_out/validation_summary.{json,csv}   (per-filter scores)"
echo "   $ROOT/abs_diff_out/unityVSpython/compare/          (side-by-side strips)"
echo "   $ROOT/abs_diff_out/*_burst/                         (burst results + visuals)"
echo " Write-up: $ROOT/vipsim_equivalence_report.md"
echo "============================================================"
