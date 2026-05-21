#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OLD_PATH="simulationMath1.py"
NEW_PATH="ros_abm/backends/legacy/runtime/simulation_math1.py"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="python"
fi

PRE_TESTS=(
  "tests/test_import_smoke_legacy_modules.py"
  "tests/test_guardrail_runtime_semantics.py"
  "tests/test_output_layout_contract.py"
  "tests/test_postprocess_hook_invocation.py"
  "tests/test_path_cwd_contracts.py"
  "tests/test_settings_contracts.py"
)

POST_TESTS=(
  "tests/test_simulation_math1_shim.py"
  "tests/test_import_smoke_legacy_modules.py"
  "tests/test_namespace_imports.py"
  "tests/test_guardrail_runtime_semantics.py"
  "tests/test_output_layout_contract.py"
  "tests/test_postprocess_hook_invocation.py"
  "tests/test_path_cwd_contracts.py"
  "tests/test_settings_contracts.py"
)

echo "[stage5] repo: ${REPO_ROOT}"
echo "[stage5] python: ${PYTHON}"
echo "[stage5] step 1/5: pre-move validation"
"${PYTHON}" -m pytest -q -rs "${PRE_TESTS[@]}"

echo "[stage5] step 2/5: move simulationMath1.py into package namespace"
mkdir -p "$(dirname "${NEW_PATH}")"
if [[ -f "${OLD_PATH}" && ! -f "${NEW_PATH}" ]]; then
  mv "${OLD_PATH}" "${NEW_PATH}"
elif [[ ! -f "${OLD_PATH}" && -f "${NEW_PATH}" ]]; then
  echo "[stage5] legacy file already moved; refreshing root shim only"
elif [[ -f "${OLD_PATH}" && -f "${NEW_PATH}" ]]; then
  echo "[stage5] both legacy and canonical files exist; preserving canonical file and refreshing root shim"
else
  echo "[stage5] error: neither ${OLD_PATH} nor ${NEW_PATH} exists" >&2
  exit 1
fi

cat > "${OLD_PATH}" <<'PY'
"""Legacy compatibility shim for simulation math utilities."""

from ros_abm.backends.legacy.runtime.simulation_math1 import *  # noqa: F401,F403
PY

echo "[stage5] step 3/5: post-move validation"
"${PYTHON}" -m pytest -q -rs "${POST_TESTS[@]}"

echo "[stage5] step 4/5: migration status"
"${PYTHON}" - <<'PY'
from pathlib import Path

old_path = Path("simulationMath1.py")
new_path = Path("ros_abm/backends/legacy/runtime/simulation_math1.py")
print(f"[stage5] root shim exists: {old_path.exists()}")
print(f"[stage5] canonical module exists: {new_path.exists()}")
if old_path.exists():
    print(f"[stage5] root shim bytes: {old_path.stat().st_size}")
if new_path.exists():
    print(f"[stage5] canonical module bytes: {new_path.stat().st_size}")
PY

echo "[stage5] step 5/5: complete"
