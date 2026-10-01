"""Typing of Vulkan-specific functions."""

from numba.core import types
from numba.core.typing.templates import AbstractTemplate, Registry, signature

from numba_vulkan import stubs

registry = Registry()


@registry.register_global(stubs.global_id)
class GlobalId(AbstractTemplate):
    """Typing of `numba_vulkan.stubs.global_id`.

    Only a literal axis of 0, 1 or 2 is accepted, because the axis selects
    a component of a SPIR-V built-in at compile time.
    """

    def generic(self, args, kws):
        # Only literal axes type-check; Numba retries with literal types.
        """Type a call.

        Parameters
        ----------
        args : tuple of numba.types.Type
            Positional argument types.
        kws : dict
            Keyword argument types.

        Returns
        -------
        numba.core.typing.Signature or None
            ``int32(literal)`` for a valid literal axis, otherwise ``None``,
            which makes Numba retry with literal argument types.
        """
        if len(args) == 1 and not kws and isinstance(args[0], types.IntegerLiteral):
            if args[0].literal_value in (0, 1, 2):
                return signature(types.int32, args[0])
