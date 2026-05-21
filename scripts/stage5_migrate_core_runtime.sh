#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="python"
fi

run_tests() {
  local label="$1"
  shift
  echo "[stage5-core] ${label}"
  "${PYTHON}" -m pytest -q -rs "$@"
}

echo "[stage5-core] repo: ${REPO_ROOT}"
echo "[stage5-core] python: ${PYTHON}"

PRE_TESTS=(
  "tests/test_import_smoke_legacy_modules.py"
  "tests/test_guardrail_runtime_semantics.py"
  "tests/test_output_layout_contract.py"
  "tests/test_postprocess_hook_invocation.py"
  "tests/test_path_cwd_contracts.py"
  "tests/test_settings_contracts.py"
  "tests/test_namespace_imports.py"
  "tests/test_simulation_math1_shim.py"
)

run_tests "step 1/8 preflight" "${PRE_TESTS[@]}"

echo "[stage5-core] step 2/8 ensure Stage 5A simulationMath1 migration"
"${SCRIPT_DIR}/stage5_migrate_simulationmath1.sh"

echo "[stage5-core] step 3/8 move Model_initialParameters.py to package path"
OLD_MODEL_INIT="Model_initialParameters.py"
NEW_MODEL_INIT="ros_abm/backends/legacy/runtime/model_initial_parameters.py"
mkdir -p "$(dirname "${NEW_MODEL_INIT}")"
if [[ -f "${OLD_MODEL_INIT}" && ! -f "${NEW_MODEL_INIT}" ]]; then
  mv "${OLD_MODEL_INIT}" "${NEW_MODEL_INIT}"
elif [[ ! -f "${OLD_MODEL_INIT}" && -f "${NEW_MODEL_INIT}" ]]; then
  echo "[stage5-core] Model_initialParameters canonical file already exists"
elif [[ -f "${OLD_MODEL_INIT}" && -f "${NEW_MODEL_INIT}" ]]; then
  echo "[stage5-core] preserving existing canonical Model_initialParameters; refreshing root shim"
else
  echo "[stage5-core] error: Model_initialParameters missing from both old and canonical locations" >&2
  exit 1
fi

cat > "${OLD_MODEL_INIT}" <<'PY'
"""Legacy compatibility shim for model initialization wiring."""

from ros_abm.backends.legacy.runtime.model_initial_parameters import *  # noqa: F401,F403
PY

MID_TESTS=(
  "tests/test_settings_contracts.py"
  "tests/test_path_cwd_contracts.py"
  "tests/test_guardrail_runtime_semantics.py"
  "tests/test_import_smoke_legacy_modules.py"
)
run_tests "step 4/8 post-Model_initialParameters validation" "${MID_TESTS[@]}"

echo "[stage5-core] step 5/8 move Model_run.py to package path"
OLD_MODEL_RUN="Model_run.py"
NEW_MODEL_RUN="ros_abm/cli/model_run.py"
mkdir -p "$(dirname "${NEW_MODEL_RUN}")"
if [[ -f "${OLD_MODEL_RUN}" && ! -f "${NEW_MODEL_RUN}" ]]; then
  mv "${OLD_MODEL_RUN}" "${NEW_MODEL_RUN}"
elif [[ ! -f "${OLD_MODEL_RUN}" && -f "${NEW_MODEL_RUN}" ]]; then
  echo "[stage5-core] Model_run canonical file already exists"
elif [[ -f "${OLD_MODEL_RUN}" && -f "${NEW_MODEL_RUN}" ]]; then
  echo "[stage5-core] preserving existing canonical Model_run; refreshing root shim"
else
  echo "[stage5-core] error: Model_run missing from both old and canonical locations" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
from pathlib import Path

path = Path("ros_abm/cli/model_run.py")
src = path.read_text(encoding="utf-8")
if "def main(" not in src:
    marker = "if __name__ == '__main__':"
    idx = src.find(marker)
    if idx == -1:
        raise SystemExit("Unable to locate Model_run script entrypoint")
    prefix = src[:idx].rstrip()
    tail = src[idx:].splitlines()
    body_lines = tail[1:]
    normalized = []
    for line in body_lines:
        if line.startswith("    "):
            normalized.append(line[4:])
        elif line.startswith("\t"):
            normalized.append(line[1:])
        else:
            normalized.append(line)
    body = "\n".join(normalized).rstrip()
    lines = [prefix, "", "", "def main():"]
    if body:
        for line in body.splitlines():
            lines.append(f"    {line}" if line else "")
    else:
        lines.append("    pass")
    lines.extend(["", "", "if __name__ == '__main__':", "    main()", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
PY

cat > "${OLD_MODEL_RUN}" <<'PY'
"""Legacy compatibility shim for simulation launcher."""

from ros_abm.cli.model_run import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
PY

echo "[stage5-core] step 6/8 add package wrappers for root-authoritative core runtime modules"
cat > "ros_abm/backends/legacy/runtime/simulation.py" <<'PY'
"""Package wrapper for the legacy Simulation runtime module."""

from Simulation import *  # noqa: F401,F403
PY

cat > "ros_abm/backends/legacy/runtime/simulation_objects.py" <<'PY'
"""Package wrapper for the legacy simulationObjects runtime module."""

from simulationObjects import *  # noqa: F401,F403
PY

cat > "ros_abm/backends/legacy/ode/fba_ode.py" <<'PY'
"""Package wrapper for the legacy FBA_ode runtime module."""

from FBA_ode import *  # noqa: F401,F403
PY

FINAL_TESTS=(
  "tests/test_import_smoke_legacy_modules.py"
  "tests/test_namespace_imports.py"
  "tests/test_count_directory.py"
  "tests/test_count_directory_shim.py"
  "tests/test_fba_ode_moles_contract.py"
  "tests/test_simulation_math1_shim.py"
  "tests/test_stage5_runtime_shims.py"
  "tests/test_guardrail_runtime_semantics.py"
  "tests/test_output_layout_contract.py"
  "tests/test_postprocess_hook_invocation.py"
  "tests/test_path_cwd_contracts.py"
  "tests/test_settings_contracts.py"
)
run_tests "step 7/8 final validation" "${FINAL_TESTS[@]}"

echo "[stage5-core] step 8/8 done"
echo "[stage5-core] created/refreshed:"
echo "  - ${OLD_MODEL_INIT}"
echo "  - ${NEW_MODEL_INIT}"
echo "  - ${OLD_MODEL_RUN}"
echo "  - ${NEW_MODEL_RUN}"
echo "  - ros_abm/backends/legacy/runtime/simulation.py"
echo "  - ros_abm/backends/legacy/runtime/simulation_objects.py"
echo "  - ros_abm/backends/legacy/ode/fba_ode.py"
