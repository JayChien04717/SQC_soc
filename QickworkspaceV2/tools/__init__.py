"""Tools package."""
from .fitting import *  # noqa

_SYSTEM_TOOL_EXPORTS = {
    "get_next_filename_labber",
    "hdf5_generator",
    "config_to_yaml",
    "auto_unit",
}


def __getattr__(name):
    if name in _SYSTEM_TOOL_EXPORTS:
        from .system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml, auto_unit

        exports = {
            "get_next_filename_labber": get_next_filename_labber,
            "hdf5_generator": hdf5_generator,
            "config_to_yaml": config_to_yaml,
            "auto_unit": auto_unit,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
