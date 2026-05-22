# pyright: reportInvalidTypeForm=false

"""
Pack-2 register-resident 16x16 tile with the same POTRF unroll optimisations as `_tile16.py`.

This module fuses two prior optimisations:

  1. **Pack-2 layout.** A single 32-lane CUDA warp processes two independent 16x16 tiles in
     parallel (lanes 0-15 = tile A, lanes 16-31 = tile B). When invoked from a kernel launched
     with ``block_dim=32`` and a per-half-warp ``i_b`` (e.g. ``i_b = i // 16``), every method
     correctly handles two envs per warp with no cross-tile contamination. Per-tile FFMA
     instructions retire at the full 32-lane warp throughput, closing the sub-warp execution
     penalty documented in ``doc/cholesky_mjw_vs_gs_2026may21.md`` (T17 + T18).

  2. **POTRF unroll** (ported from ``genesis/utils/_tile16.py`` on origin/main, PR #2820).
     The outer ``k`` and inner ``j`` loops of ``cholesky_`` are wrapped in ``qd.static(...)``
     so the ``if k > j`` predicates fold at compile time and register access collapses to a
     single field reference (vs. the 16-deep cascade emitted by ``_get_col(k)``). A per-lane
     running ``my_norm_sq`` makes each diagonal update O(1) instead of O(k). The off-diagonal
     ``dot`` accumulator is split into two interleaved partial sums (``dot0`` / ``dot1``) so
     the FMA dependency chain depth is halved, exposing more instruction-level parallelism.
     Inline ``if k == N: self.rN = val`` setattr chains replace ``_set_col(k, val)`` so writes
     also avoid the 16-way runtime conditional. The same inlined register-cascade pattern is
     applied to ``_load*``, ``_store*``, ``eye_``, and ``_ger_sub`` (SYRK), which is the
     highest-frequency tile op in the dex_hand Cholesky factor (10 ``_ger_sub`` calls per
     factor at n_dofs=60, vs. 4 ``cholesky_`` calls).

When invoked from a kernel launched with ``block_dim=16`` (the legacy single-env layout) the
per-warp ``tile_base`` is always 0 and ``local == tid``, so the pack-2 codepath degenerates to
the upstream single-tile-per-warp behavior. That is the property we rely on to run the same
tile class in both code paths if needed.

Two quadrants-side miscompilation traps avoided in this file (both produce silently-wrong
codegen as of commit ``b0906ebd`` on the cluster image, recorded in
``doc/cholesky_mjw_vs_gs_2026may21.md`` T18 step 5):

  - ``tid & qd.i32(~15)`` does *not* mask correctly. Use ``(tid >> qd.i32(4)) << qd.i32(4)``.
  - ``qd.i32(env_pair_base * 2)`` does *not* multiply correctly when ``env_pair_base`` is a
    kernel-loop i32. Use ``env_pair_base << qd.i32(1)`` instead. (Not used directly here but
    relevant to call sites that address two envs per warp.)
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

# Import the upstream OuterProduct / VecSliceProxy classes so that `qd.outer(...)` and 2D/3D
# vector slices coming from kernel code (which use the upstream classes) are correctly
# recognized by our local _augassign / _resolve_vec_proxy. (Mixing the local copies would
# silently break `tile -= qd.outer(v, v)` because the isinstance check would fail.)
from quadrants.lang.simt._tile16 import _OuterProduct as _UpstreamOuterProduct
from quadrants.lang.simt._tile16 import _VecSliceProxy as _UpstreamVecSliceProxy
from quadrants.lang.simt._tile16 import _tile16_cache as _upstream_tile16_cache

if _TYPE_CHECKING:

    class _Tile16x16Proto:  # noqa: E303
        """Static type stub so pyright sees Tile16x16Pack2 methods correctly."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_Tile16x16Proto", lower: bool = True) -> None: ...  # noqa: E704
        def _load(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _load3d(self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _get_col(self, k: Any) -> Any: ...  # noqa: E704
        def _set_col(self, k: Any, val: Any) -> None: ...  # noqa: E704
        def _ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def _trsm(self, L: "_Tile16x16Proto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_Tile16x16Proto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 16

# Field-name lookup table for direct register access in qd.static-unrolled loops. Used via `self._r(k)` (defined below)
# which is just `getattr(self, _REGS[k])`. With a python-int `k` (which is what `qd.static(range(16))` binds inside its
# body) this collapses to a single field-reference AST node, vs. the 16-way `if k == 0: val = self.r0; ...` cascade
# emitted by a dynamic `_get_col(k)` call.
_REGS = tuple(f"r{i}" for i in range(_TILE))


class _OuterProduct:
    """Deferred outer product proxy for use with augmented assignment on Tile16x16Pack2."""

    _qd_is_deferred = True

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b

    def __add__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")

    def __radd__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")


def outer(a: Any, b: Any) -> _OuterProduct:
    """Create a deferred outer product for use with Tile16x16Pack2 augmented assignment.

    Usage::

        t -= qd.outer(a, b)   # equivalent to t._ger_sub(a, b)
        t -= qd.outer(v, v)   # symmetric case (a == b)
    """
    return _OuterProduct(a, b)


class _DeferredProxyMixin:
    """Raises clear errors if a deferred tile proxy is accidentally used as a value."""

    _proxy_description = "Tile proxy"

    def _misuse(self, op: str = "used") -> NoReturn:
        raise TypeError(
            f"{self._proxy_description} was {op}, but it is only valid in tile operations"
        )

    def __add__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __radd__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __sub__(self, other: Any) -> NoReturn:
        self._misuse("subtracted")

    def __mul__(self, other: Any) -> NoReturn:
        self._misuse("multiplied")

    def __getitem__(self, key: Any) -> NoReturn:
        self._misuse("subscripted")

    def __repr__(self) -> str:
        return f"<{self._proxy_description} — not a value>"


class _TileSliceProxy(_DeferredProxyMixin):
    """Deferred 2D/3D array slice for tile load/store."""

    _qd_is_deferred = True
    _proxy_description = "Array slice proxy (arr[r0:r1, c0:c1])"

    def __init__(
        self, arr: Any, row_start: Any, row_stop: Any, col_start: Any, col_stop: Any, batch_idx: Any = None
    ) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop
        self.batch_idx = batch_idx

    def _assign(self, tile: Any) -> None:
        if self.batch_idx is not None:
            tile._store3d(self.arr, self.batch_idx, self.row_start, self.row_stop, self.col_start, self.col_stop)
        else:
            tile._store(self.arr, self.row_start, self.row_stop, self.col_start, self.col_stop)


class _VecSliceProxy(_DeferredProxyMixin):
    """Deferred column-vector load from a 2D/3D array."""

    _qd_is_deferred = True
    _proxy_description = "Vec slice proxy (arr[r0:r1, col])"

    def __init__(self, arr: Any, row_start: Any, row_stop: Any, col: Any, batch_idx: Any = None) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col = col
        self.batch_idx = batch_idx


class _TileRefProxy:
    """Proxy returned by tile[:] for the LHS of a load assignment."""

    _qd_is_deferred = True

    def __init__(self, tile: Any) -> None:
        self.tile = tile

    def _assign(self, value: Any) -> None:
        if isinstance(value, _TileSliceProxy):
            if value.batch_idx is not None:
                self.tile._load3d(
                    value.arr, value.batch_idx, value.row_start, value.row_stop, value.col_start, value.col_stop
                )
            else:
                self.tile._load(value.arr, value.row_start, value.row_stop, value.col_start, value.col_stop)
        else:
            raise TypeError(f"Tile16x16Pack2[:] can only be assigned from an array slice, got {type(value)}")


# Per-dtype class cache. Independent of quadrants' own Tile16x16 cache so this
# module never mutates upstream state for its primary key.
_tile16_cache: dict[Any, type] = {}


def _make_tile16x16(dtype=None) -> "type[_Tile16x16Proto]":
    """Build (and memoize) a Tile16x16Pack2 dataclass with the optimised cholesky_."""
    if dtype is None:
        dtype = qd.f32
    cached = _tile16_cache.get(dtype)
    if cached is not None:
        return cached  # pyright: ignore[reportReturnType]
    cls = _make_tile16x16_class(dtype)
    _tile16_cache[dtype] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile16x16_class(dtype):
    class _Tile16x16Pack2:
        """A 16x16 tile distributed one row per subgroup half-warp lane, held in 16 scalar registers.

        Under ``block_dim=32`` the same warp processes two independent tiles in parallel
        (lanes 0-15 = tile A, lanes 16-31 = tile B), with per-method ``tile_base`` derived
        from the lane id. Under ``block_dim=16`` the implementation degenerates to the
        upstream single-tile-per-warp behavior (``tile_base == 0``, ``local == tid``).
        """

        r0: dtype
        r1: dtype
        r2: dtype
        r3: dtype
        r4: dtype
        r5: dtype
        r6: dtype
        r7: dtype
        r8: dtype
        r9: dtype
        r10: dtype
        r11: dtype
        r12: dtype
        r13: dtype
        r14: dtype
        r15: dtype

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array. Each lane loads arr[row_start + local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(16)):
                    if col_start + j < col_stop:
                        val = arr[row, col_start + j]
                        if j == 0:
                            self.r0 = val
                        if j == 1:
                            self.r1 = val
                        if j == 2:
                            self.r2 = val
                        if j == 3:
                            self.r3 = val
                        if j == 4:
                            self.r4 = val
                        if j == 5:
                            self.r5 = val
                        if j == 6:
                            self.r6 = val
                        if j == 7:
                            self.r7 = val
                        if j == 8:
                            self.r8 = val
                        if j == 9:
                            self.r9 = val
                        if j == 10:
                            self.r10 = val
                        if j == 11:
                            self.r11 = val
                        if j == 12:
                            self.r12 = val
                        if j == 13:
                            self.r13 = val
                        if j == 14:
                            self.r14 = val
                        if j == 15:
                            self.r15 = val

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array. Each lane loads arr[batch, row_start+local, col_start:col_stop].

            For pack-2: ``batch`` should be per-half-warp (e.g. each half-warp computes its own
            env index from ``i_b = i // 16`` with ``block_dim=32``). Each lane reads its env.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(16)):
                    if col_start + j < col_stop:
                        val = arr[batch, row, col_start + j]
                        if j == 0:
                            self.r0 = val
                        if j == 1:
                            self.r1 = val
                        if j == 2:
                            self.r2 = val
                        if j == 3:
                            self.r3 = val
                        if j == 4:
                            self.r4 = val
                        if j == 5:
                            self.r5 = val
                        if j == 6:
                            self.r6 = val
                        if j == 7:
                            self.r7 = val
                        if j == 8:
                            self.r8 = val
                        if j == 9:
                            self.r9 = val
                        if j == 10:
                            self.r10 = val
                        if j == 11:
                            self.r11 = val
                        if j == 12:
                            self.r12 = val
                        if j == 13:
                            self.r13 = val
                        if j == 14:
                            self.r14 = val
                        if j == 15:
                            self.r15 = val

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array. Each lane stores to arr[row_start + local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(16)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self._r(j)

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Store to a 3D array. Each lane stores to arr[batch, row_start+local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(16)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self._r(j)

        @qd.func
        def eye_(self):
            """Set this tile to the 16x16 identity matrix (per half-warp sets its own identity)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            for j in qd.static(range(16)):
                val = qd.cast(1.0, dtype) if local == j else qd.cast(0.0, dtype)
                if j == 0:
                    self.r0 = val
                if j == 1:
                    self.r1 = val
                if j == 2:
                    self.r2 = val
                if j == 3:
                    self.r3 = val
                if j == 4:
                    self.r4 = val
                if j == 5:
                    self.r5 = val
                if j == 6:
                    self.r6 = val
                if j == 7:
                    self.r7 = val
                if j == 8:
                    self.r8 = val
                if j == 9:
                    self.r9 = val
                if j == 10:
                    self.r10 = val
                if j == 11:
                    self.r11 = val
                if j == 12:
                    self.r12 = val
                if j == 13:
                    self.r13 = val
                if j == 14:
                    self.r14 = val
                if j == 15:
                    self.r15 = val

        @qd.func
        def _get_col(self, k):
            """Return the value of register (column) k."""
            val = qd.cast(0.0, dtype)
            if k == 0:
                val = self.r0
            if k == 1:
                val = self.r1
            if k == 2:
                val = self.r2
            if k == 3:
                val = self.r3
            if k == 4:
                val = self.r4
            if k == 5:
                val = self.r5
            if k == 6:
                val = self.r6
            if k == 7:
                val = self.r7
            if k == 8:
                val = self.r8
            if k == 9:
                val = self.r9
            if k == 10:
                val = self.r10
            if k == 11:
                val = self.r11
            if k == 12:
                val = self.r12
            if k == 13:
                val = self.r13
            if k == 14:
                val = self.r14
            if k == 15:
                val = self.r15
            return val

        @qd.func
        def _set_col(self, k, val):
            """Set register (column) k to val."""
            if k == 0:
                self.r0 = val
            if k == 1:
                self.r1 = val
            if k == 2:
                self.r2 = val
            if k == 3:
                self.r3 = val
            if k == 4:
                self.r4 = val
            if k == 5:
                self.r5 = val
            if k == 6:
                self.r6 = val
            if k == 7:
                self.r7 = val
            if k == 8:
                self.r8 = val
            if k == 9:
                self.r9 = val
            if k == 10:
                self.r10 = val
            if k == 11:
                self.r11 = val
            if k == 12:
                self.r12 = val
            if k == 13:
                self.r13 = val
            if k == 14:
                self.r14 = val
            if k == 15:
                self.r15 = val

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T (per half-warp).

            This is the SYRK update inside blocked Cholesky and runs 10 times per factor at
            n_dofs=60 (vs. 4 times for cholesky_), so its FFMA pipeline efficiency dominates
            the overall factor cost. Pack-2 closes the sub-warp execution penalty (16 active
            lanes -> 32) and the inlined register-cascade kills the 16-deep `_set_col(j)` cascade.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            for j in qd.static(range(16)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(tile_base + qd.i32(j)))
                val = self._r(j) - a * bc
                if j == 0:
                    self.r0 = val
                if j == 1:
                    self.r1 = val
                if j == 2:
                    self.r2 = val
                if j == 3:
                    self.r3 = val
                if j == 4:
                    self.r4 = val
                if j == 5:
                    self.r5 = val
                if j == 6:
                    self.r6 = val
                if j == 7:
                    self.r7 = val
                if j == 8:
                    self.r8 = val
                if j == 9:
                    self.r9 = val
                if j == 10:
                    self.r10 = val
                if j == 11:
                    self.r11 = val
                if j == 12:
                    self.r12 = val
                if j == 13:
                    self.r13 = val
                if j == 14:
                    self.r14 = val
                if j == 15:
                    self.r15 = val

        @qd.func
        def cholesky_(self, eps):
            """In-place 16x16 Cholesky factorization via subgroup shuffles (per half-warp).

            On return, the lower triangle holds L such that A = L @ L^T. Diagonal clamped to
            sqrt(max(value, eps)) for numerical stability. Combines:

              - ``qd.static`` outer and inner loops so ``if k > j`` predicates fold at compile
                time and register access collapses to single field references.
              - Per-lane running ``my_norm_sq`` so each diagonal step is O(1) rather than O(k).
              - Split ``dot0`` / ``dot1`` accumulators to halve the FMA dep-chain depth.
              - Pack-2 shuffle indices: ``tile_base + qd.i32(k)`` so each half-warp broadcasts
                within its own tile, with two tiles' broadcasts dispatched in a single
                ``__shfl_sync(0xFFFFFFFF, ...)`` instruction.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(16)):
                diag_val = qd.cast(0.0, dtype)
                if local == k:
                    diag_val = qd.sqrt(qd.max(self._r(k) - my_norm_sq, eps))
                    if k == 0:
                        self.r0 = diag_val
                    if k == 1:
                        self.r1 = diag_val
                    if k == 2:
                        self.r2 = diag_val
                    if k == 3:
                        self.r3 = diag_val
                    if k == 4:
                        self.r4 = diag_val
                    if k == 5:
                        self.r5 = diag_val
                    if k == 6:
                        self.r6 = diag_val
                    if k == 7:
                        self.r7 = diag_val
                    if k == 8:
                        self.r8 = diag_val
                    if k == 9:
                        self.r9 = diag_val
                    if k == 10:
                        self.r10 = diag_val
                    if k == 11:
                        self.r11 = diag_val
                    if k == 12:
                        self.r12 = diag_val
                    if k == 13:
                        self.r13 = diag_val
                    if k == 14:
                        self.r14 = diag_val
                    if k == 15:
                        self.r15 = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(tile_base + qd.i32(k)))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(16)):
                    if k > j:
                        my_col = self._r(j)
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(tile_base + qd.i32(k)))
                        if j % 2 == 0:
                            dot0 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                        else:
                            dot1 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                dot = dot0 + dot1

                new_val = qd.cast(0.0, dtype)
                if local > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self._r(k) - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    if k == 0:
                        self.r0 = new_val
                    if k == 1:
                        self.r1 = new_val
                    if k == 2:
                        self.r2 = new_val
                    if k == 3:
                        self.r3 = new_val
                    if k == 4:
                        self.r4 = new_val
                    if k == 5:
                        self.r5 = new_val
                    if k == 6:
                        self.r6 = new_val
                    if k == 7:
                        self.r7 = new_val
                    if k == 8:
                        self.r8 = new_val
                    if k == 9:
                        self.r9 = new_val
                    if k == 10:
                        self.r10 = new_val
                    if k == 11:
                        self.r11 = new_val
                    if k == 12:
                        self.r12 = new_val
                    if k == 13:
                        self.r13 = new_val
                    if k == 14:
                        self.r14 = new_val
                    if k == 15:
                        self.r15 = new_val
                if local > k:  # type: ignore[reportOperatorIssue]
                    my_norm_sq += new_val * new_val

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self), per half-warp.

            L is a Tile16x16Pack2 holding the lower-triangular Cholesky factor (from cholesky_).
            On return, self holds the solution X.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            for c in range(16):
                dot = qd.cast(0.0, dtype)
                for j in range(16):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(tile_base + qd.i32(c)))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(tile_base + qd.i32(c)))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place."""
            if not lower:
                raise TypeError("Tile16x16Pack2.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        # See main `_tile16.py`: the AST transformer's external-function check exempts callees
        # whose `__module__` starts with `"quadrants."`. We rewrite the `__module__` of helper
        # methods after class definition to restore parity with stock `qd.simt.Tile16x16`.
        solve_triangular_.__module__ = "quadrants.gen.tile16_pack2"

        def _r(self, k):
            """Direct field read by python-int index. Used at qd.static-unrolled call sites to bypass the 16-way
            ``_get_col(k)`` cascade: with ``k`` a python int (from ``qd.static(range(16))``),
            ``getattr(self, _REGS[k])`` is evaluated by the AST transformer at build time and returns a single
            field-reference expression."""
            return getattr(self, _REGS[k])

        _r.__module__ = "quadrants.gen.tile16_pack2"

        @qd.func
        def _resolve_vec2d(self, arr: qd.template(), row_start, row_stop, col):
            """Load one scalar per lane from a 2D array column, clamped to array bounds (per half-warp local row)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + local < row_stop:
                v = arr[row_start + local, col]
            return v

        @qd.func
        def _resolve_vec3d(self, arr: qd.template(), batch, row_start, row_stop, col):
            """Load one scalar per lane from a 3D array column, clamped to array bounds (per half-warp local + batch)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + local < row_stop:
                v = arr[batch, row_start + local, col]
            return v

        def _resolve_vec_proxy(self, proxy: Any) -> Any:
            """Materialize a vec slice proxy into a scalar by dispatching to _resolve_vec2d or _resolve_vec3d."""
            if proxy.batch_idx is not None:
                return self._resolve_vec3d(proxy.arr, proxy.batch_idx, proxy.row_start, proxy.row_stop, proxy.col)
            return self._resolve_vec2d(proxy.arr, proxy.row_start, proxy.row_stop, proxy.col)

        def _augassign(self, other: Any, op: str) -> None:
            """Handle augmented assignment (e.g. tile -= qd.outer(a, b)).

            Resolves _VecSliceProxy arguments and dispatches to _ger_sub. Only 'Sub' is supported.
            Accepts both the local and upstream OuterProduct / VecSliceProxy classes so this tile works
            transparently with ``qd.outer(...)``.
            """
            if isinstance(other, (_OuterProduct, _UpstreamOuterProduct)):
                if op == "Sub":
                    a_orig = other.a
                    b_orig = other.b
                    vec_proxy_types = (_VecSliceProxy, _UpstreamVecSliceProxy)
                    a = self._resolve_vec_proxy(a_orig) if isinstance(a_orig, vec_proxy_types) else a_orig
                    b = (
                        a
                        if (b_orig is a_orig)
                        else (self._resolve_vec_proxy(b_orig) if isinstance(b_orig, vec_proxy_types) else b_orig)
                    )
                    self._ger_sub(a, b)
                else:
                    raise TypeError(f"Tile16x16Pack2: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile16x16Pack2: unsupported augmented assignment with {type(other)}")

    # StructType.__call__ already defaults missing args to 0, so Tile() produces a zero-initialized tile.
    result = qd.dataclass(_Tile16x16Pack2)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]

    # Also register this pack-2 class into quadrants' upstream `_tile16_cache` so the slice-dispatch
    # in `quadrants/lang/simt/tile_slicing.py` recognizes `tile[:]` as a tile-ref and
    # `arr[batch, r:r2, c:c2]` as a tile-slice. Using `("pack2", dtype)` keeps both this and the
    # upstream `Tile16x16[dtype]` alive in the same kernel.
    _upstream_tile16_cache[("pack2", dtype)] = result
    return result


class _Tile16x16Pack2Proxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``Tile16x16Pack2.zeros(dtype=qd.f32)`` inside a kernel. The dtype is resolved at kernel
    compilation time, defaulting to the compile config's ``default_fp`` if omitted.
    """

    SIZE = _TILE

    @staticmethod
    def _resolve(dtype):
        from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
        from quadrants.lang.exception import (  # pylint: disable=import-outside-toplevel
            QuadrantsSyntaxError,
        )

        arch = impl.current_cfg().arch
        if arch in (qd.cpu, qd.x64, getattr(qd, "arm64", None)):
            raise QuadrantsSyntaxError(
                f"Tile16x16Pack2 requires a GPU backend (cuda, metal, vulkan, amdgpu). Current arch is {arch}."
            )
        if dtype is None:
            dtype = impl.get_runtime().default_fp
        return _make_tile16x16(dtype)

    def zeros(self, *, dtype=None):
        """Zero-initialized tile."""
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        """Identity tile (diagonal = 1, rest = 0)."""
        return self._resolve(dtype).eye()


# Re-declare the proxy constructors as belonging to a quadrants.* module so the AST transformer's
# external-function check (which exempts callees whose `__module__` starts with `"quadrants."`) does
# not warn that they are not @qd.func when invoked from inside a kernel.
_Tile16x16Pack2Proxy.zeros.__module__ = "quadrants.gen.tile16_pack2"
_Tile16x16Pack2Proxy.eye.__module__ = "quadrants.gen.tile16_pack2"


Tile16x16Pack2 = _Tile16x16Pack2Proxy()


# Eagerly register pack-2 tile classes for the standard float dtypes so quadrants' slice-dispatch
# recognizes `tile[:]` and `arr[batch, r:r2, c:c2]` on the very first statement of the first tiled
# kernel that uses them. Without this, the dispatch can briefly see an empty upstream cache (the
# cache is only populated on the first `Tile16x16Pack2.zeros/.eye(dtype=...)` call, which races the
# subscript build inside the same statement / kernel).
for _dt in (qd.f32, qd.f64):
    try:
        _make_tile16x16(_dt)
    except Exception:  # pylint: disable=broad-except
        # Some quadrants builds may restrict dataclass construction outside an active runtime; skip
        # silently and rely on the lazy path. Any failure here just reverts to the lazy registration.
        pass
