"""The Rust push/fold solver (rust/, built with `cargo build --release`), called through ctypes."""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from mbr.ranges import N_CLASSES

LIB_PATH = Path(__file__).resolve().parent.parent / "rust" / "target" / "release" / "libmbr_solver.so"

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


def enabled():
    """The Rust solver is used when it is built, unless MBR_SOLVER=python."""
    return os.environ.get("MBR_SOLVER", "rust") != "python" and LIB_PATH.exists()


@lru_cache(maxsize=1)
def _lib():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.mbr_solve.restype = ctypes.c_int
    lib.mbr_solve.argtypes = [
        ctypes.c_size_t, _f64, _f64, _f64, _f64, _f64, _f64, _f64, _f64, _f64,
        ctypes.c_void_p, ctypes.c_void_p, _f64, _f64, _f64, _f64, ctypes.c_int,
    ]
    return lib


def solve(m, w, eq, tie, cc, v_steal, v_iwin, v_jwin, v_tie, v3, tensors, d_all, params, strategies=None):
    """Run the solver; returns (strategies, ev_play, ev_fold) over the decision nodes: shoves (m - 1),
    first calls (one per pair), overcalls (one per triple, only with tensors).

    With `strategies` (nodes x 169) only evaluate them instead of solving."""
    n = N_CLASSES
    np_ = m * (m - 1) // 2
    nt = len(v3)
    nodes = (m - 1) + np_ + nt
    strategy = np.zeros((nodes, n)) if strategies is None else np.array(strategies, dtype=np.float64)
    ev_play = np.zeros((nodes, n))
    ev_fold = np.zeros(nodes)
    arr = lambda x, dtype=np.float64: np.ascontiguousarray(x, dtype=dtype)
    keep = []  # keep the arrays behind raw pointers alive during the call
    if tensors is not None:
        tensors = [arr(t, np.float32) for t in tensors]
        d_all = arr(d_all, np.float32)
        ptrs = (ctypes.c_void_p * len(tensors))(*[t.ctypes.data for t in tensors])
        keep += [tensors, d_all, ptrs]
        t3_ptr, d_ptr = ctypes.cast(ptrs, ctypes.c_void_p), d_all.ctypes.data
    else:
        t3_ptr = d_ptr = None
    v3_arr = arr(v3).reshape(-1) if nt else np.zeros(1)
    code = _lib().mbr_solve(
        m, arr(w), arr(eq), arr(tie), arr(cc), arr(v_steal), arr(v_iwin).reshape(-1), arr(v_jwin).reshape(-1),
        arr(v_tie).reshape(-1), v3_arr,
        t3_ptr, d_ptr, arr(params), strategy, ev_play, ev_fold, int(strategies is not None),
    )
    if code != 0:
        raise RuntimeError(f"mbr_solve failed with code {code}")
    return strategy, ev_play, ev_fold
