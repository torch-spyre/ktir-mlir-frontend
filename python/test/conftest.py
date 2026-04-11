import os
import sys
from pathlib import Path

try:
    import mlir_ktdp  # noqa: F401
except ImportError:
    # Not pip-installed — fall back to the bare cmake build output tree.
    _repo = Path(__file__).resolve().parent.parent.parent
    _build = Path(os.environ.get("KTIR_BUILD_DIR", _repo / "build"))
    _pkg = _build / "python_packages" / "ktdp"
    if _pkg.exists():
        sys.path.insert(0, str(_pkg))
