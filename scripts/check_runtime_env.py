"""Check that the legacy runtime environment can import hard dependencies."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_version(package_name):
    module = import_module(package_name)
    try:
        package_version = version(package_name)
    except PackageNotFoundError:
        package_version = getattr(module, "__version__", "unknown")
    print(f"{package_name}: {package_version}")


def main():
    print(f"sys.executable: {sys.executable}")
    print(f"python: {platform.python_version()}")

    for package_name in ("numpy", "scipy", "networkx", "torch"):
        _print_version(package_name)

    import IntC_diff  # noqa: F401
    import ExtC_gradient  # noqa: F401

    print("runtime environment OK")


if __name__ == "__main__":
    main()
