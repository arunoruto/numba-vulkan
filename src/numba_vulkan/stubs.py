"""Functions that only have a meaning inside a kernel."""


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
