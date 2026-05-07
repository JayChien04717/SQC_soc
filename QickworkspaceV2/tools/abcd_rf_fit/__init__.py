"""ABCD RF fitting utilities with lazy public exports."""

_LAZY_EXPORTS = {
    "fit_signal": ".abcd_rf_fit",
    "get_abcd": ".abcd_rf_fit",
    "analyze": ".abcd_rf_fit",
    "plot": ".plot",
    "get_synthetic_signal": ".synthetic_signal",
    "ResonatorParams": ".resonators",
    "get_fit_function": ".resonators",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
