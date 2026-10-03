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


def _local_array(shape, dtype):
    """Allocate an array private to the current invocation.

    Like ``numba.cuda.local.array``; ``np.empty``, ``np.zeros``,
    ``np.ones`` and ``np.full`` with a constant shape do the same inside
    kernels.

    Parameters
    ----------
    shape : int or tuple of int
        Constant shape.
    dtype : numpy dtype or Numba type
        Constant element type.

    Returns
    -------
    array
        Uninitialised. It lives as long as the invocation.
    """
    raise NotImplementedError("local.array() can only be called inside a kernel")


_local_array.__name__ = "array"
local = _module(
    "local", "Invocation-private memory, like ``numba.cuda.local``.", [_local_array]
)


def _subgroup(name, doc):
    """Create the stub of one subgroup function."""

    def stub(*args):
        raise NotImplementedError(
            f"subgroup.{name}() can only be called inside a kernel"
        )

    stub.__name__ = name
    stub.__doc__ = doc
    return stub


_REDUCE_DOC = """The {what} of `value` over the active invocations of the subgroup.

Parameters
----------
value : int or float
    A 32- or 64-bit integer or float.

Returns
-------
int or float
    The same for every active invocation, of the type of `value`.
"""
_SCAN_DOC = """The {what} of `value` over the active invocations up to this one.

Invocations are ordered by `lane`; the {kind} scan {includes} the value of
the invocation itself.

Parameters
----------
value : int or float
    A 32- or 64-bit integer or float.

Returns
-------
int or float
    Of the type of `value`.{empty}
"""
_SHUFFLE_DOC = """`value` of the invocation {which} in the subgroup.

Parameters
----------
value : int or float
    A 32- or 64-bit integer or float.
{argument} : int
    {meaning}

Returns
-------
int or float
    Undefined if that invocation is not active or does not exist.
"""
_ID_DOC = """{what}

Returns
-------
int
"""


def _subgroup_functions():
    """All stubs of `subgroup`, generated from their descriptions."""
    functions = [
        _subgroup(
            "size",
            _ID_DOC.format(
                what="Number of invocations a subgroup can hold.\n\n"
                "Drivers may run fewer: Intel's runs many kernels with 16 or 8\n"
                "invocations per subgroup while reporting 32, and `count` then\n"
                "says how many subgroups a workgroup has."
            ),
        ),
        _subgroup(
            "lane",
            _ID_DOC.format(
                what="Index of the current invocation within its subgroup "
                "(``cuda.laneid``)."
            ),
        ),
        _subgroup(
            "id", _ID_DOC.format(what="Index of the subgroup within its workgroup.")
        ),
        _subgroup("count", _ID_DOC.format(what="Number of subgroups per workgroup.")),
    ]
    whats = {"sum": "sum", "prod": "product", "min": "minimum", "max": "maximum"}
    for op, what in whats.items():
        functions.append(_subgroup(op, _REDUCE_DOC.format(what=what)))
        for kind, includes, empty in (
            ("inclusive", "includes", ""),
            (
                "exclusive",
                "leaves out",
                (
                    " The first invocation receives the identity of the\n"
                    "    operation (0 for a sum, 1 for a product, the largest value\n"
                    "    for a minimum, the smallest for a maximum)."
                ),
            ),
        ):
            functions.append(
                _subgroup(
                    f"{kind}_{op}",
                    _SCAN_DOC.format(
                        what=what, kind=kind, includes=includes, empty=empty
                    ),
                )
            )
    functions += [
        _subgroup(
            "any",
            "Whether `predicate` holds for any active invocation of the subgroup.\n\n"
            "Parameters\n----------\npredicate : bool\n\nReturns\n-------\nbool\n",
        ),
        _subgroup(
            "all",
            "Whether `predicate` holds for all active invocations of the subgroup."
            "\n\nParameters\n----------\npredicate : bool\n\nReturns\n-------\nbool\n",
        ),
        _subgroup(
            "elect",
            "Whether this is the active invocation of the subgroup with the "
            "lowest `lane`.\n\nReturns\n-------\nbool\n",
        ),
        _subgroup(
            "ballot",
            "The active invocations of the subgroup for which `predicate` holds."
            "\n\nParameters\n----------\npredicate : bool\n\nReturns\n-------\n"
            "tuple of int\n    Four ``uint32`` words; bit ``k % 32`` of word "
            "``k // 32`` stands\n    for the invocation with `lane` ``k``.\n",
        ),
        _subgroup(
            "ballot_count",
            "Number of active invocations of the subgroup for which `predicate` "
            "holds.\n\nParameters\n----------\npredicate : bool\n\nReturns\n"
            "-------\nint\n",
        ),
        _subgroup(
            "broadcast",
            _SHUFFLE_DOC.format(
                which="with `lane` ``lane``",
                argument="lane",
                meaning="The same for every invocation of the subgroup.",
            ),
        ),
        _subgroup(
            "broadcast_first",
            "`value` of the active invocation with the lowest `lane`.\n\n"
            "Parameters\n----------\nvalue : int or float\n\nReturns\n-------\n"
            "int or float\n",
        ),
        _subgroup(
            "shuffle",
            _SHUFFLE_DOC.format(
                which="with `lane` ``lane``",
                argument="lane",
                meaning="May differ between invocations.",
            ),
        ),
        _subgroup(
            "shuffle_xor",
            _SHUFFLE_DOC.format(
                which="whose `lane` is that of this one xor ``mask``",
                argument="mask",
                meaning="The same for every invocation of the subgroup.",
            ),
        ),
        _subgroup(
            "shuffle_up",
            _SHUFFLE_DOC.format(
                which="``delta`` lanes below this one",
                argument="delta",
                meaning="The same for every invocation of the subgroup.",
            ),
        ),
        _subgroup(
            "shuffle_down",
            _SHUFFLE_DOC.format(
                which="``delta`` lanes above this one",
                argument="delta",
                meaning="The same for every invocation of the subgroup.",
            ),
        ),
    ]
    return functions


subgroup = _module(
    "subgroup",
    "Operations across the invocations of a subgroup (a warp, in CUDA's terms).\n\n"
    "A workgroup runs as subgroups of `subgroup.size` invocations that execute\n"
    "together. These functions combine or exchange values within one; they need\n"
    "the device's support for subgroup operations, and they must be reached by\n"
    "the invocations that take part, as in CUDA's ``*_sync`` functions.",
    _subgroup_functions(),
)
