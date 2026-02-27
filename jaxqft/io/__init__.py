"""I/O helpers for SciDAC/ILDG LIME files."""

from importlib import import_module

_EXPORTS = {
    "LimeRecord",
    "ScidacFieldMeta",
    "iter_lime_records",
    "list_lime_records",
    "read_lime_record_data",
    "find_scidac_fields",
    "decode_scidac_field",
    "decode_scidac_gauge",
    "decode_scidac_momentum",
    "decode_scidac_pseudofermion",
}


def __getattr__(name):
    if name in _EXPORTS:
        mod = import_module(".lime", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = sorted(_EXPORTS)
