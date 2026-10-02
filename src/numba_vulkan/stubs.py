"""Functions that only have a meaning inside a kernel."""

import sys
import types


def global_id(axis):
    """Index of the current invocation along ``axis`` of the dispatch grid.

    Parameters
    ----------
    axis : int
        Constant ``0``, ``1`` or ``2``.

    Returns
    -------
    int
        Like ``cuda.grid``, this can exceed the requested extent because the
        grid is rounded up to whole workgroups, so kernels must bounds-check.
    """
    raise NotImplementedError("global_id() can only be called inside a kernel")


def local_id(axis):
    """Index of the current invocation within its workgroup along ``axis``.

    Like ``cuda.threadIdx``.

    Parameters
    ----------
    axis : int
        Constant ``0``, ``1`` or ``2``.

    Returns
    -------
    int
    """
    raise NotImplementedError("local_id() can only be called inside a kernel")


def group_id(axis):
    """Index of the current workgroup along ``axis``, like ``cuda.blockIdx``.

    Parameters
    ----------
    axis : int
        Constant ``0``, ``1`` or ``2``.

    Returns
    -------
    int
    """
    raise NotImplementedError("group_id() can only be called inside a kernel")


def local_size(axis):
    """Number of invocations per workgroup along ``axis``, like ``cuda.blockDim``.

    Parameters
    ----------
    axis : int
        Constant ``0``, ``1`` or ``2``.

    Returns
    -------
    int
    """
    raise NotImplementedError("local_size() can only be called inside a kernel")


def num_groups(axis):
    """Number of workgroups along ``axis`` of the dispatch, like ``cuda.gridDim``.

    Parameters
    ----------
    axis : int
        Constant ``0``, ``1`` or ``2``.

    Returns
    -------
    int
    """
    raise NotImplementedError("num_groups() can only be called inside a kernel")


def barrier():
    """Wait until every invocation of the workgroup has reached this point.

    Like ``cuda.syncthreads``: writes to shared and global memory made before
    the barrier are visible to the whole workgroup after it. All invocations
    of a workgroup must reach the same barrier, so it must not be called
    under a condition that differs between them.
    """
    raise NotImplementedError("barrier() can only be called inside a kernel")


syncthreads = barrier


def _module(name, doc, functions):
    """A module object holding stub functions, like ``numba.cuda.atomic``.

    Numba types attributes of module objects by their values, so
    ``nv.atomic.add`` is typed like a plain global function.
    """
    module = types.ModuleType(f"numba_vulkan.{name}", doc)
    # Numba looks functions' modules up by name.
    sys.modules[module.__name__] = module
    for function in functions:
        function.__module__ = module.__name__
        function.__qualname__ = function.__name__
        setattr(module, function.__name__, function)
    return module


def _shared_array(shape, dtype):
    """Allocate an array in memory shared by the invocations of a workgroup.

    Parameters
    ----------
    shape : int or tuple of int
        Constant shape.
    dtype : numpy dtype or Numba type
        Constant element type.

    Returns
    -------
    array
        Uninitialised. Every workgroup has its own copy, which lives as
        long as the workgroup; use `barrier` between writing and reading
        it from different invocations.
    """
    raise NotImplementedError("shared.array() can only be called inside a kernel")


_shared_array.__name__ = "array"
shared = _module(
    "shared", "Workgroup-shared memory, like ``numba.cuda.shared``.", [_shared_array]
)


def _atomic(name, doc):
    """Create the stub of one atomic operation."""

    def stub(array, index, value):
        raise NotImplementedError(f"atomic.{name}() can only be called inside a kernel")

    stub.__name__ = name
    stub.__doc__ = doc
    return stub


_ATOMIC_DOC = """Atomically {what} ``array[index]``; return its previous value.

Parameters
----------
array : array
    A writable array argument or shared array of 32- or 64-bit integers or
    floats{floats}.
index : int or tuple of int
    Position of the element.
value : scalar
    The operand.

Returns
-------
scalar
    The element's value before the operation.
"""


def _cas(array, index, expected, value):
    """Atomically replace ``array[index]`` if it equals `expected`.

    Parameters
    ----------
    array : array
        A writable array argument or shared array of ``int32``,
        ``uint32`` or ``float32``.
    index : int or tuple of int
        Position of the element.
    expected, value : scalar
        The value the element must have, and its replacement.

    Returns
    -------
    scalar
        The element's value before; the replacement happened if it equals
        `expected`. Floats are compared by their bit patterns.
    """
    raise NotImplementedError("atomic.cas() can only be called inside a kernel")


_cas.__name__ = "cas"
atomic = _module(
    "atomic",
    "Atomic operations on array elements, like ``numba.cuda.atomic``.",
    [
        _atomic("add", _ATOMIC_DOC.format(what="add `value` to", floats="")),
        _atomic("sub", _ATOMIC_DOC.format(what="subtract `value` from", floats="")),
        _atomic("max", _ATOMIC_DOC.format(what="raise to at least `value`", floats="")),
        _atomic("min", _ATOMIC_DOC.format(what="lower to at most `value`", floats="")),
        _atomic(
            "and_",
            _ATOMIC_DOC.format(
                what="bitwise-and `value` into", floats=" (integers only)"
            ),
        ),
        _atomic(
            "or_",
            _ATOMIC_DOC.format(
                what="bitwise-or `value` into", floats=" (integers only)"
            ),
        ),
        _atomic(
            "xor",
            _ATOMIC_DOC.format(
                what="bitwise-xor `value` into", floats=" (integers only)"
            ),
        ),
        _atomic("exch", _ATOMIC_DOC.format(what="replace by `value`", floats="")),
        _cas,
    ],
)
