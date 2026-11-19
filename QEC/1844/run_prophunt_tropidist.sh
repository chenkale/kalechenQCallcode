#!/usr/bin/env bash
# One-shot runner: PropHunt venv drives the script; tropidist venv is used
# internally for circuit-distance certification.
#
# Usage:
#   ./run_prophunt_tropidist.sh --skip-optimize
#   ./run_prophunt_tropidist.sh --max-iter 10 --samples 200 --threads 8
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PROPHUNT_PY="${REPO}/tools/PropHunt/.venv/bin/python"
TROPIDIST_PY="${REPO}/tools/tropidist/.venv/bin/python"
SCRIPT="${HERE}/prophunt_tropidist.py"

die() { echo "error: $*" >&2; exit 1; }

[[ -x "${PROPHUNT_PY}" ]] || die "PropHunt venv missing: ${PROPHUNT_PY}"
[[ -x "${TROPIDIST_PY}" ]] || die "tropidist venv missing: ${TROPIDIST_PY}"
[[ -f "${SCRIPT}" ]] || die "script missing: ${SCRIPT}"

export PROPHUNT_PYTHON="${PROPHUNT_PY}"
export TROPIDIST_PYTHON="${TROPIDIST_PY}"

echo "PropHunt python : ${PROPHUNT_PY}"
echo "tropidist python: ${TROPIDIST_PY}"
echo "running: ${SCRIPT} $*"
echo

exec "${PROPHUNT_PY}" "${SCRIPT}" "$@"
