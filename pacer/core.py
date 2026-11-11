"""Optional C++17 hot path (pacer/csrc/core.cpp). Falls back to Python when not built."""
try:
    from . import _core  # noqa: F401
    HAVE_CORE = True
except ImportError:  # pragma: no cover - no compiler / not built
    _core = None
    HAVE_CORE = False
