# spmd_types package
from __future__ import annotations

import sys
from types import ModuleType

# Collectives and operations
from sixlib.spmd_types._checker import (  # noqa: F401
    _SpmdTypeBackwardCompatibleMode as SpmdTypeMode,
    assert_local_type,
    assert_type,
    is_type_checking,
    mutate_type,
    no_typecheck,
    register_autograd_function,
    register_local_autograd_function,
    trace,
    typecheck,
)
from sixlib.spmd_types._collectives import (  # noqa: F401
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
    unshard,
)
from sixlib.spmd_types._dist import set_dist  # noqa: F401
from sixlib.spmd_types._dtensor import (  # noqa: F401
    dtensor_placement_to_spmd_type,
    spmd_redistribute,
    spmd_type_to_dtensor_placement,
)
from sixlib.spmd_types._local import (  # noqa: F401
    convert,
    invariant_to_replicate,
    reinterpret,
    shard,
)
from sixlib.spmd_types._mesh_axis import MeshAxis  # noqa: F401
from sixlib.spmd_types._scalar import Scalar  # noqa: F401
from sixlib.spmd_types._traceback import traceback_filtering  # noqa: F401
from sixlib.spmd_types._type_attr import get_local_type  # noqa: F401

# Types
from sixlib.spmd_types.types import (  # noqa: F401
    DimSharding,
    I,
    Invariant,
    LocalSpmdType,
    normalize_axis,
    P,
    Partial,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    R,
    Replicate,
    S,
    Shard,
    SpmdTypeError,
    TensorSharding,
    V,
    Varying,
)


class _SpmdTypesModule(ModuleType):
    """Module wrapper that provides TYPE_CHECKING as a dynamic attribute.

    Accessing ``sixlib.spmd_types.TYPE_CHECKING`` returns True when a
    ``typecheck()`` context is active and False otherwise.  This uses the
    sys.modules replacement trick (same pattern as torch._dynamo.config)
    so that a simple attribute read works like a function call.
    """

    @property
    def TYPE_CHECKING(self) -> bool:
        return is_type_checking()


# Replace this module in sys.modules so that attribute access on the
# module object goes through _SpmdTypesModule.__getattr__ / properties.
_self = sys.modules[__name__]
_obj = _SpmdTypesModule(__name__)
_obj.__dict__.update(_self.__dict__)
_obj.__file__ = _self.__file__
_obj.__package__ = _self.__package__
_obj.__path__ = _self.__path__  # type: ignore[attr-defined]
_obj.__spec__ = _self.__spec__
sys.modules[__name__] = _obj
