"""Data models: how Vulkan-target types are represented in LLVM IR."""

import functools

from numba.core import types
from numba.core.datamodel.registry import DataModelManager, register
from numba.core.extending import models

from numba_vulkan.vktypes import VulkanArray, VulkanDispatcherType

vulkan_data_manager = DataModelManager()
register_model = functools.partial(register, vulkan_data_manager)


@register_model(VulkanArray)
class VulkanArrayModel(models.StructModel):
    """Array metadata only; the data is reached through the type's binding.

    Member names match Numba's own array model so that its ``shape``,
    ``size``, ``len()``... implementations work unchanged. There is
    deliberately no ``data`` member: Numba code that needs the data pointer
    fails at compile time instead of producing an invalid shader.

    Parameters
    ----------
    dmm : numba.core.datamodel.DataModelManager
        The data model manager.
    fe_type : VulkanArray
        The array type being modelled.
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("shape", types.UniTuple(types.intp, fe_type.ndim)),
            ("strides", types.UniTuple(types.intp, fe_type.ndim)),
        ]
        super().__init__(dmm, fe_type, members)


register_model(VulkanDispatcherType)(models.OpaqueModel)
