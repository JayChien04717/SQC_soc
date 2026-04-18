import datetime
import math
import os
import pprint
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from addict import Dict as AddictDict

try:
    import Labber
except ImportError:
    print("No Labber module")


# =============================================================================
# HDF5 / File Management Utilities
# =============================================================================


def get_next_filename_labber(
    dest_path: str, exp_name: str, yoko_value: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate the next HDF5 filename for a Labber-compatible log file.

    Files are saved inside a date-structured subdirectory under *dest_path*
    (``<dest_path>/<YYYY>/<MM>/Data_<MMDD>/``).  In normal (index) mode the
    function scans the entire *dest_path* tree for existing files matching
    ``<exp_name>_NNN.hdf5`` and returns a path with the next available index.
    In Yoko mode it builds a filename from the measured output value.

    Parameters
    ----------
    dest_path : str
        Root directory where experiment data is stored.
    exp_name : str
        Base name of the experiment (e.g. ``"s008_T1_ge_Q1"``).
    yoko_value : dict, optional
        If provided, must contain ``"value"`` (numeric) and ``"unit"`` (str)
        keys.  The filename encodes the Yokogawa output level instead of an
        auto-increment index.

    Returns
    -------
    file_path : str
        Full path (without ``.hdf5`` extension) ready to pass to
        :func:`hdf5_generator`.

    Raises
    ------
    ValueError
        If *yoko_value* is provided but is missing the ``"value"`` or
        ``"unit"`` key.
    """
    # 1. Ensure dest_path is absolute and create today's save directory
    dest_path = os.path.abspath(dest_path)
    yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
    save_path = os.path.join(dest_path, yy, mm, f"Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)

    # 2. Check Yoko mode.
    if yoko_value is not None:
        try:
            value = yoko_value["value"]
            value = auto_unit(value)
            unit = yoko_value["unit"]
            filename = f"{exp_name}_{value['value']:.2f}{value['unit']}{unit}"
            return os.path.join(save_path, filename)
        except KeyError:
            raise ValueError(
                "yoko_value dictionary must contain 'value' and 'unit' keys"
            )

    else:
        # 3. Normal (index) mode
        max_index = 0
        pattern = re.compile(rf"^{re.escape(exp_name)}_(\d+)\.hdf5$")

        for root, dirs, files in os.walk(dest_path):
            for f in files:
                match = pattern.match(f)
                if match:
                    current_index = int(match.group(1))
                    if current_index > max_index:
                        max_index = current_index

        next_index = max_index + 1
        final_filename = f"{exp_name}_{next_index:03d}"
        return os.path.join(save_path, final_filename)


def hdf5_generator(
    filepath: str,
    x_info: dict,
    z_info: dict,
    y_info: dict = None,
    comment=None,
    tag=None,
):
    """
    Create a Labber-compatible HDF5 log file and write measurement data.

    Wraps ``Labber.createLogFile_ForData`` to handle both 1-D (x only) and
    2-D (x + y sweep) datasets.  The signal channel is always stored as a
    complex vector.

    Parameters
    ----------
    filepath : str
        Destination file path (without ``.hdf5`` extension).
    x_info : dict
        Step-channel descriptor for the fast (x) axis.  Must contain at
        minimum ``"name"``, ``"unit"``, and ``"values"`` keys as required by
        Labber.
    z_info : dict
        Log-channel descriptor for the measured signal.  Must contain
        ``"name"``, ``"unit"``, and ``"values"`` keys.  ``"complex"`` and
        ``"vector"`` flags are set automatically.
    y_info : dict, optional
        Step-channel descriptor for the slow (y) axis.  When supplied the
        data in ``z_info["values"]`` is iterated row by row.
    comment : str, optional
        Free-text comment stored in the log file header.
    tag : str, optional
        Labber tag string associated with the log file.
    """
    np.float = float
    np.bool = bool
    zdata = z_info["values"]
    z_info.update({"complex": True, "vector": False})

    log_channels = [z_info]
    step_channels = list(filter(None, [x_info, y_info]))

    fObj = Labber.createLogFile_ForData(filepath, log_channels, step_channels)
    if y_info:
        for trace in zdata:
            fObj.addEntry({z_info["name"]: trace})
    else:
        fObj.addEntry({z_info["name"]: zdata})

    if comment:
        fObj.setComment(comment)
    if tag:
        fObj.setTags(tag)


# =============================================================================
# Config Serialization Utilities
# =============================================================================


def clean_config(config):
    """
    Recursively deep-copy and clean a config dict for serialization.

    - **Auto-removes** any value that is a QICK sweep parameter
      (``QickParam`` or subclass), so callers never need to manually
      ``pop()`` sweep keys before saving.
    - Converts ``numpy`` scalars / arrays to Python builtins.
    - Converts ``complex`` numbers to ``{"real": ..., "imag": ...}``.
    - Converts ``addict.Dict`` to plain ``dict``.

    The original *config* is **not** mutated.

    Parameters
    ----------
    config : dict or addict.Dict
        Arbitrary nested configuration mapping to clean.

    Returns
    -------
    cleaned : dict
        A plain Python ``dict`` (nested) with all non-serializable objects
        replaced by their JSON-compatible equivalents.
    """
    # Lazy import – avoids hard dependency on qick at module level
    try:
        from qick.asm_v2 import QickParam

        _qick_param_types = (QickParam,)
    except ImportError:
        _qick_param_types = ()

    def _clean(obj):
        if isinstance(obj, (dict, AddictDict)):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if not (_qick_param_types and isinstance(v, _qick_param_types))
            }
        elif isinstance(obj, (list, tuple)):
            return [_clean(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return obj

    return _clean(config)


def config_to_yaml(config: dict) -> str:
    """
    Clean a config dict and serialize it to a YAML string.

    Internally calls :func:`clean_config` before dumping, so QICK sweep
    parameters and NumPy objects are handled automatically.

    Parameters
    ----------
    config : dict
        Raw experiment configuration (may contain NumPy scalars,
        ``QickParam`` values, or ``addict.Dict`` nodes).

    Returns
    -------
    yaml_str : str
        Human-readable YAML representation of the cleaned configuration.
    """
    return yaml.dump(
        clean_config(config),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


class ExperimentConfig:
    """
    Manager for nested multi-qubit experiment configurations.

    Provides unified access, per-qubit extraction, structured updates,
    and YAML / Python-file export for a list of qubit configuration dicts.

    Features
    --------
    - Unified (mux) config with dot-notation via :class:`addict.Dict`.
    - Per-qubit flat extraction via :meth:`get_qubit`.
    - Smart key search and recursive update via :meth:`update`.
    - YAML export with human-readable spacing via :meth:`to_yaml`.

    Parameters
    ----------
    data : list of dict or dict
        One configuration dict per qubit, or a single dict.
    keys_to_unify : list of str, optional
        Keys whose values should be collapsed to a scalar when all qubits
        share the same value.  Defaults to
        ``["reps", "res_length", "ro_length", "trig_time", "relax_delay"]``.
    """

    def __init__(
        self, data: Union[List, Dict], keys_to_unify: Optional[List[str]] = None
    ):
        self._raw_list = data
        self.keys_to_unify = keys_to_unify or [
            "reps",
            "res_length",
            "ro_length",
            "trig_time",
            "relax_delay",
        ]

        # Create Name Map for quick lookup
        self._name_map = {}
        if isinstance(self._raw_list, list):
            for idx, cfg in enumerate(self._raw_list):
                name = cfg.get("name")
                if name:
                    self._name_map[name] = idx

        # Lazy refresh: only rebuild unified_config when accessed after mutation
        self._dirty = True
        self._unified_cache = None

    @property
    def unified_config(self) -> AddictDict:
        """
        Lazy-computed unified configuration across all qubits.

        Rebuilds only when the underlying data has been mutated since the
        last access (controlled by the ``_dirty`` flag).

        Returns
        -------
        unified : AddictDict
            Flat mapping where list-valued keys hold one entry per qubit and
            scalar-valued keys have been unified when all qubits agree.
        """
        if self._dirty or self._unified_cache is None:
            raw_collected = self._collect_all_key_values(self._raw_list)
            self._unified_cache = self._refine_cfg(raw_collected)
            self._dirty = False
        return self._unified_cache

    def _mark_dirty(self) -> None:
        """Mark the unified config cache as stale so it is rebuilt on next access."""
        self._dirty = True

    def __repr__(self) -> str:
        names = [cfg.get("name", f"idx{i}") for i, cfg in enumerate(self._raw_list)]
        return f"ExperimentConfig(qubits={names})"

    def get_qubit(self, q_id: Union[int, str]) -> AddictDict:
        """
        Return a flat configuration dict for a single qubit.

        List-valued keys in the unified config are sliced to the element
        corresponding to *q_id*; scalar keys are passed through unchanged.

        Parameters
        ----------
        q_id : int or str
            Qubit index (0-based) or qubit name (e.g. ``"Q1"``).

        Returns
        -------
        cfg : AddictDict
            Flat dot-access configuration for the requested qubit.
        """
        indices = self._resolve_indices(q_id)
        idx = indices[0]

        selected = AddictDict()
        for key, value in self.unified_config.items():
            if isinstance(value, list):
                if idx < len(value):
                    selected[key] = value[idx]
            else:
                selected[key] = value
        return selected

    def muxconfig(
        self,
        qb_list: List[Union[int, str]],
        mux_ro_ch_start: int = 2,
        mux_gen: int = 12,
    ) -> AddictDict:
        """
        Extract a mux-ready configuration for a subset of qubits.

        For each key in :attr:`unified_config`:

        - If the value is a list, selects only the elements at the indices
          corresponding to the qubits in *qb_list* (preserving order).
        - If the value is a scalar, passes it through unchanged.

        Additionally injects mux-specific keys:

        - ``mux_ro_chs``: readout channels mapped from qubit index
          (e.g. Q1 → 2, Q2 → 3, … with default *mux_ro_ch_start* = 2).
        - ``gen_mask``: raw qubit indices (e.g. Q1 → 0, Q3 → 2, Q5 → 4).
        - ``mux_gen``: generator tile number.
        - ``mux_ro_phases``: list of zeros, one per qubit.
        - ``mixer_freq``: rounded mean of ``res_freq_ge`` (if present).

        Parameters
        ----------
        qb_list : list of int or str
            Qubit identifiers, e.g. ``["Q1", "Q3"]`` or ``[0, 2]``.
        mux_ro_ch_start : int, optional
            Readout channel number for the first qubit (index 0).
            Default is ``2``.
        mux_gen : int, optional
            Generator tile number to assign to ``mux_gen``.  Default is ``12``.

        Returns
        -------
        cfg : AddictDict
            Mux-ready config with list keys sliced and mux keys injected.
        """
        indices = self._resolve_indices(qb_list)

        selected = AddictDict()
        for key, value in self.unified_config.items():
            if isinstance(value, list):
                selected[key] = [value[i] for i in indices if i < len(value)]
            else:
                selected[key] = value

        # Mux-specific keys
        selected["mux_ro_chs"] = [i + mux_ro_ch_start for i in indices]
        selected["gen_mask"] = list(indices)
        selected["mux_gen"] = mux_gen
        selected["mux_ro_phases"] = [0] * len(indices)
        # Calculate mixer_freq as the rounded mean of res_freq_ge
        if "res_freq_ge" in selected and selected["res_freq_ge"]:
            selected["mixer_freq"] = int(round(np.mean(selected["res_freq_ge"])))

        return selected

    def to_yaml_mux(self, qb_list: List[Union[int, str]], **kwargs) -> str:
        """
        Extract a mux-ready config while preserving the nested dict structure.

        Unlike :meth:`muxconfig`, which flattens all keys, this method walks
        the first qubit's raw nested config as a template and fills in
        per-qubit values (or unified scalars) at each leaf position.

        Parameters
        ----------
        qb_list : list of int or str
            Qubit identifiers (names or 0-based indices).
        **kwargs
            Forwarded to :meth:`muxconfig`; supports ``mux_gen`` and
            ``mux_ro_ch_start``.

        Returns
        -------
        yaml_str : str
            Human-readable YAML string with the nested mux configuration.
        """
        indices = self._resolve_indices(qb_list)

        def _get_mux_val(key, path_parts):
            vals = []
            for idx in indices:
                cfg = self._raw_list[idx]
                curr = cfg
                for p in path_parts:
                    if isinstance(curr, dict):
                        curr = curr.get(p)
                    else:
                        curr = None
                        break
                vals.append(curr)

            vals = [v for v in vals if v is not None]
            if not vals:
                return None

            # Unify if possible
            if key in self.keys_to_unify:
                unique = set(v for v in vals if not isinstance(v, (dict, list)))
                if len(unique) == 1:
                    return list(unique)[0]
            return vals

        def _recursive_fill(template_node, path=[]):
            if isinstance(template_node, dict):
                res = {}
                for k, v in template_node.items():
                    if isinstance(v, dict):
                        res[k] = _recursive_fill(v, path + [k])
                    else:
                        res[k] = _get_mux_val(k, path + [k])
                return res
            return template_node

        # Use first config as template
        template = self._clean_data(self._raw_list[indices[0]])
        mux_nested = _recursive_fill(template)

        # Mux Overrides
        mux_gen = kwargs.get("mux_gen", 12)
        mux_ro_ch_start = kwargs.get("mux_ro_ch_start", 2)

        if "name" in mux_nested:
            all_names = [self._raw_list[i].get("name", f"Q{i}") for i in indices]
            mux_nested["name"] = ", ".join(all_names)

        if "ch" in mux_nested:
            mux_nested["ch"].pop("res_ch", None)
            mux_nested["ch"].pop("ro_ch", None)
            mux_nested["ch"]["mux_res_ch"] = mux_gen
            mux_nested["ch"]["mux_ro_ch"] = [i + mux_ro_ch_start for i in indices]

        # Flat stats
        flat = self.muxconfig(qb_list, **kwargs)
        if "gen_mask" in flat:
            mux_nested["gen_mask"] = flat["gen_mask"]
        if "mixer_freq" in flat:
            mux_nested["mixer_freq"] = flat["mixer_freq"]

        return self._dump_dict_with_spacing(mux_nested)

    def update_mux(
        self,
        param: Union[str, Dict[str, Any]],
        value: Union[float, List] = None,
        q_index: Union[int, str, List] = None,
    ) -> None:
        """
        Update parameters for multiple qubits (alias for :meth:`update`).

        When *value* is a list of the same length as the number of target
        qubits, each qubit receives the corresponding element.

        Parameters
        ----------
        param : str or dict
            Key path string or flat dictionary — passed directly to
            :meth:`update`.
        value : float or list, optional
            Value(s) to set.
        q_index : int, str, or list, optional
            Target qubit identifier(s).  ``None`` broadcasts to all qubits.
        """
        self.update(param, value, q_index)

    def update(
        self,
        param: Union[str, Dict[str, Any]],
        value: Any = None,
        q_index: Union[int, str, List] = None,
    ) -> None:
        """
        Unified update method supporting three addressing modes.

        Mode 1 — Dictionary merge
            ``update(flat_dict, q_index="Q1")``
            Merges a flat dictionary into the nested structure of the target
            qubit(s).

        Mode 2 — Auto-search (plain key, no dots)
            ``update("qb_freq_ge", 5000, "Q1")``
            Locates the key anywhere inside the nested config and updates it.

        Mode 3 — Explicit dot-path
            ``update("qb.qb_freq_ge", 5000, "Q1")``
            Navigates to the exact nested location using dot notation.

        When *value* is a list of the same length as the resolved target
        indices, each qubit receives the corresponding element (distribute
        mode); otherwise the same *value* is written to every target qubit.

        Parameters
        ----------
        param : str or dict
            Either a key-path string (e.g. ``"qb_freq_ge"`` or
            ``"qb.qb_freq_ge"``) or a flat dictionary to merge.
        value : any, optional
            Value to set when *param* is a string.  Ignored for dict mode.
        q_index : int, str, or list, optional
            Target qubit identifier(s).  ``None`` targets all qubits.

        Raises
        ------
        TypeError
            If *param* is neither a string nor a dict.
        """
        target_indices = self._resolve_indices(q_index)

        # --- Mode 1: Dictionary Merge ---
        if isinstance(param, dict):
            flat_config = param
            updated_count = 0
            for idx in target_indices:
                raw_nested_cfg = self._raw_list[idx]
                for k, v in flat_config.items():
                    if self._recursive_update(raw_nested_cfg, k, v):
                        updated_count += 1
            if q_index is not None:
                print(
                    f"Merged dictionary into {q_index}. Updated {updated_count} parameters."
                )

        # --- Mode 2: Auto-search (plain key, no dots) ---
        elif isinstance(param, str) and "." not in param:
            leaf_key = param

            is_list_val = isinstance(value, (list, np.ndarray))
            should_distribute = is_list_val and (len(value) == len(target_indices))

            # Check PER-QUBIT whether the key exists in its nested config.
            # Any qubit that is missing the key AND is not being updated gets
            # None padded, so the unified_config list length stays consistent.
            has_key = [
                self._find_key_path(self._raw_list[j], leaf_key) is not None
                for j in range(len(self._raw_list))
            ]

            for j in range(len(self._raw_list)):
                if not has_key[j] and j not in target_indices:
                    # Missing key, not being updated → pad with None at top level
                    self._raw_list[j][leaf_key] = None

            # Write value(s) for target qubits
            for i, cfg_idx in enumerate(target_indices):
                cfg = self._raw_list[cfg_idx]
                val_to_set = value[i] if should_distribute else value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()

                if not has_key[cfg_idx]:
                    # Key didn't exist yet in this qubit: put at top level
                    cfg[leaf_key] = val_to_set
                else:
                    # Update wherever the key already lives
                    self._recursive_update(cfg, leaf_key, val_to_set)

        # --- Mode 3: Explicit Path Update (dot notation) ---
        elif isinstance(param, str):
            key_path = param
            keys = key_path.split(".")
            leaf_key = keys[-1]

            is_list_val = isinstance(value, (list, np.ndarray))
            should_distribute = is_list_val and (len(value) == len(target_indices))

            # ── Per-qubit key existence check ──────────────────────────────────
            # Check PER-QUBIT whether the key already exists at the given path.
            # Any qubit missing the key that is NOT being updated gets None,
            # keeping the unified_config list length consistent with other keys.
            def _key_exists_in(nested, path_keys):
                """Return True if the leaf key is reachable via path_keys."""
                curr = nested
                for k in path_keys[:-1]:
                    if not isinstance(curr, dict) or k not in curr:
                        return False
                    curr = curr[k]
                return isinstance(curr, dict) and path_keys[-1] in curr

            has_key = [
                _key_exists_in(self._raw_list[j], keys)
                for j in range(len(self._raw_list))
            ]

            # Pre-populate None for qubits that are missing the key and not being updated
            for j in range(len(self._raw_list)):
                if not has_key[j] and j not in target_indices:
                    cfg_j = self._raw_list[j]
                    node = cfg_j
                    for k in keys[:-1]:
                        if isinstance(node, dict):
                            node = node.setdefault(k, {})
                        else:
                            node = getattr(node, k)
                    if isinstance(node, dict):
                        node[leaf_key] = None
                    else:
                        setattr(node, leaf_key, None)

            # ── Write target value(s) ──────────────────────────────────────────
            for i, cfg_idx in enumerate(target_indices):
                cfg = self._raw_list[cfg_idx]
                target = cfg

                # Traverse to parent dict
                for k in keys[:-1]:
                    if isinstance(target, dict):
                        if k not in target:
                            warnings.warn(
                                f"Creating new nested key '{k}' in path '{key_path}'",
                                stacklevel=2,
                            )
                        target = target.setdefault(k, {})
                    else:
                        target = getattr(target, k)

                # Determine value
                val_to_set = value[i] if should_distribute else value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()

                # Set value
                if isinstance(target, dict):
                    target[leaf_key] = val_to_set
                else:
                    setattr(target, leaf_key, val_to_set)

        else:
            raise TypeError(
                "First argument must be a string (key path) or a dict (config)."
            )

        self._mark_dirty()

    def _find_key_path(self, nested, target_key, _path=()) -> Optional[tuple]:
        """
        Recursively search *nested* for *target_key* and return its key path.

        Parameters
        ----------
        nested : dict or list
            The nested structure to search.
        target_key : str
            The leaf key to locate.
        _path : tuple, optional
            Accumulated path of ancestor keys (used internally for recursion).

        Returns
        -------
        path : tuple of str or None
            Tuple of keys leading to *target_key*, or ``None`` if not found.

        Examples
        --------
        >>> cfg = {'qb': {'qb_freq_ge': 5000}}
        >>> self._find_key_path(cfg, 'qb_freq_ge')
        ('qb', 'qb_freq_ge')
        """
        if isinstance(nested, dict):
            if target_key in nested:
                return _path + (target_key,)
            for k, v in nested.items():
                if isinstance(v, (dict, list)):
                    result = self._find_key_path(v, target_key, _path + (k,))
                    if result is not None:
                        return result
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, (dict, list)):
                    result = self._find_key_path(item, target_key, _path)
                    if result is not None:
                        return result
        return None

    def _recursive_update(self, nested_data, target_key, new_value) -> bool:
        """
        Recursively locate *target_key* in *nested_data* and update its value.

        Parameters
        ----------
        nested_data : dict or list
            Nested configuration structure to search.
        target_key : str
            Key to update.
        new_value : any
            Replacement value.

        Returns
        -------
        found : bool
            ``True`` if *target_key* was found and updated at least once.
        """
        found = False
        if isinstance(nested_data, dict):
            if target_key in nested_data:
                val_to_set = new_value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()

                if nested_data[target_key] != val_to_set:
                    nested_data[target_key] = val_to_set
                    return True
                return False

            for v in nested_data.values():
                if isinstance(v, (dict, list)):
                    if self._recursive_update(v, target_key, new_value):
                        found = True
        elif isinstance(nested_data, list):
            for item in nested_data:
                if isinstance(item, (dict, list)):
                    if self._recursive_update(item, target_key, new_value):
                        found = True
        return found

    def read_config(self, q_id: Union[int, str]) -> Dict:
        """
        Return the cleaned nested configuration dict for a single qubit.

        Parameters
        ----------
        q_id : int or str
            Qubit index (0-based) or qubit name (e.g. ``"Q1"``).

        Returns
        -------
        cfg : dict
            Plain Python dict with NumPy scalars and ``QickParam`` objects
            removed (via :meth:`_clean_data`).
        """
        indices = self._resolve_indices(q_id)
        target_idx = indices[0]
        raw_nested_cfg = self._raw_list[target_idx]
        clean_data = self._clean_data(raw_nested_cfg)
        return clean_data

    def save_to_py(self, filename: str = "latest_cfg.py") -> None:
        """
        Export the full configuration list to a Python source file.

        The output file defines ``DATA_PATH`` and ``config_list`` using
        :func:`pprint.pprint` for readable formatting.

        Parameters
        ----------
        filename : str, optional
            Destination file path.  Defaults to ``"latest_cfg.py"``.
        """
        from .system_cfg import DATA_PATH

        clean_data = self._clean_data(self._raw_list)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Auto-generated configuration file\n")
            f.write(f"DATA_PATH = r'{DATA_PATH}'\n\n")
            f.write("config_list = ")
            pprint.pprint(clean_data, stream=f, width=120, sort_dicts=False)
            f.write("\n")
        print(f"Configuration saved to {filename}")

    # Backward-compatible alias
    read_qubit_config = read_config

    def save_qubit_config(
        self,
        q_id: Union[int, str],
        filename: Optional[str] = None,
        var_name: str = "config",
    ) -> None:
        """
        Export the nested configuration of a specific qubit to a Python file.

        Parameters
        ----------
        q_id : int or str
            Qubit index (0-based) or qubit name (e.g. ``"Q1"``).
        filename : str, optional
            Destination file path.  Defaults to ``"<name>_config.py"`` where
            *name* is taken from the qubit's ``"name"`` field.
        var_name : str, optional
            Python variable name to use in the output file.  Default is
            ``"config"``.
        """
        # 1. Resolve index and get nested config
        indices = self._resolve_indices(q_id)
        target_idx = indices[0]
        raw_nested_cfg = self._raw_list[target_idx]

        # 2. Clean data
        clean_data = self._clean_data(raw_nested_cfg)

        # 3. Determine filename
        if filename is None:
            name = clean_data.get("name", f"Q{target_idx}")
            filename = f"{name}_config.py"

        # 4. Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Auto-generated configuration file for {clean_data.get('name', q_id)}\n"
            )
            f.write("from addict import Dict\n\n")
            f.write(f"{var_name} = ")
            pprint.pprint(clean_data, stream=f, width=120, sort_dicts=False)
            f.write("\n")

        print(f"Saved {q_id} configuration to {filename}")

    def to_yaml(self, q_id: Union[int, str] = None) -> str:
        """
        Serialize the configuration to a human-readable YAML string.

        Parameters
        ----------
        q_id : int or str, optional
            Qubit identifier (name or 0-based index).  When provided, only
            that qubit's config is serialized.  When ``None`` (default), all
            qubits are serialized as a YAML list.

        Returns
        -------
        yaml_str : str
            YAML-formatted configuration string.

        Raises
        ------
        ValueError
            If *q_id* is provided but does not match any known qubit.
        """
        if q_id is not None:
            indices = self._resolve_indices(q_id)
            if not indices:
                raise ValueError(f"No qubit found for identifier {q_id}")
            target_idx = indices[0]
            clean_data = self._clean_data(self._raw_list[target_idx])
            yaml_str = self._dump_dict_with_spacing(clean_data)
        else:
            clean_data = self._clean_data(self._raw_list)
            if isinstance(clean_data, list):
                # Dump each item individually to control spacing
                yaml_parts = []
                for item in clean_data:
                    # Dump as a list of one item to preserve the "- " prefix for the first key
                    # But we want custom spacing inside the dict too.
                    # So we construct the string manually for the list item structure

                    # 1. Dump the first key-value pair with "- " prefix
                    # We need to handle the dict content with spacing

                    # Alternative approach:
                    # Use yaml.dump for the structure but post-process? No, too risky.
                    # Let's use the helper.

                    part = self._dump_dict_with_spacing(item, is_list_item=True)
                    yaml_parts.append(part)

                # 2 empty lines between list items -> 3 newlines
                yaml_str = "\n\n\n".join(yaml_parts) + "\n"
            else:
                yaml_str = self._dump_dict_with_spacing(clean_data)

        return yaml_str

    def _dump_dict_with_spacing(
        self, data: dict, is_list_item: bool = False, indent: int = 0
    ) -> str:
        """
        Recursively serialize *data* to YAML with context-aware blank lines.

        Simple lists are rendered in flow style (``[1, 2, 3]``).  A blank line
        is inserted between adjacent keys when either the current or the next
        value is a dict or list (complex type).

        Parameters
        ----------
        data : dict
            Mapping to serialize.
        is_list_item : bool, optional
            When ``True``, the first key is prefixed with ``"- "`` to produce
            valid YAML list-item syntax.  Default is ``False``.
        indent : int, optional
            Current indentation level (2 spaces per level).  Default is ``0``.

        Returns
        -------
        yaml_str : str
            YAML-formatted string for *data*.
        """
        if not isinstance(data, dict):
            if isinstance(data, list) and all(
                not isinstance(x, (dict, list)) for x in data
            ):
                return yaml.dump(data, default_flow_style=True, sort_keys=False).strip()
            return yaml.dump(data, default_flow_style=False, sort_keys=False).strip()

        parts = []
        keys = list(data.keys())

        for i, key in enumerate(keys):
            val = data[key]

            # 1. Prepare prefix for current line
            if is_list_item and i == 0:
                # First line of a list item: "- key: value"
                line_pfx = ("  " * indent) + "- "
                eff_indent = indent + 1  # Sub-lines should align with key
            elif is_list_item:
                # Subsequent lines of a list item: "  key: value"
                line_pfx = ("  " * indent) + "  "
                eff_indent = indent + 1
            else:
                # Standard dict line: "key: value"
                line_pfx = "  " * indent
                eff_indent = indent

            # 2. Dump content
            if isinstance(val, dict):
                parts.append(f"{line_pfx}{key}:")
                parts.append(self._dump_dict_with_spacing(val, indent=eff_indent + 1))
            elif isinstance(val, list) and all(
                not isinstance(x, (dict, list)) for x in val
            ):
                # Force flow-style for simple lists
                list_str = yaml.dump(
                    val, default_flow_style=True, sort_keys=False
                ).strip()
                parts.append(f"{line_pfx}{key}: {list_str}")
            else:
                # Scalar or complex object
                # We use yaml.dump for single item to get "key: value"
                # then we adjust indentation for multi-line values if any.
                dumped = yaml.dump(
                    {key: val}, default_flow_style=False, sort_keys=False
                ).strip()
                lines = dumped.split("\n")
                parts.append(f"{line_pfx}{lines[0]}")
                for line in lines[1:]:
                    parts.append(f"{'  ' * eff_indent}{line}")

            # 3. Spacing logic
            if i < len(keys) - 1:
                next_key = keys[i + 1]
                next_val = data[next_key]
                # Add empty line if current or next item is complex
                is_curr_complex = isinstance(val, (dict, list))
                is_next_complex = isinstance(next_val, (dict, list))
                if is_curr_complex or is_next_complex:
                    parts.append("")

        return "\n".join(parts)

    def to_yaml_file(self, filename: str, q_id: Union[int, str] = None):
        """
        Serialize the configuration to YAML and write it to a file.

        Parameters
        ----------
        filename : str
            Destination file path.
        q_id : int or str, optional
            Qubit identifier to filter the output.  ``None`` (default)
            exports all qubits.
        """
        yaml_str = self.to_yaml(q_id=q_id)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            print(f"Configuration saved to {filename}")

    def _resolve_indices(self, q_identifier) -> List[int]:
        """
        Resolve a qubit identifier to a list of integer indices.

        Parameters
        ----------
        q_identifier : int, str, list, or None
            Qubit name, 0-based integer index, list of either, or ``None``
            to select all qubits.

        Returns
        -------
        indices : list of int
            Resolved integer indices into ``_raw_list``.

        Raises
        ------
        ValueError
            If a string identifier is not found in the name map.
        TypeError
            If *q_identifier* is an unsupported type.
        """
        if q_identifier is None:
            return list(range(len(self._raw_list)))
        if isinstance(q_identifier, (int, np.integer)):
            return [int(q_identifier)]
        if isinstance(q_identifier, str):
            if q_identifier in self._name_map:
                return [self._name_map[q_identifier]]
            else:
                raise ValueError(f"Qubit name '{q_identifier}' not found.")
        if isinstance(q_identifier, list):
            resolved = []
            for x in q_identifier:
                if isinstance(x, str) and x in self._name_map:
                    resolved.append(self._name_map[x])
                elif isinstance(x, (int, np.integer)):
                    resolved.append(int(x))
                else:
                    raise ValueError(f"Invalid identifier: {x}")
            return resolved
        raise TypeError("Invalid q_index type.")

    def _clean_data(self, data: Any) -> Any:
        """
        Recursively convert *data* to plain Python types for serialization.

        Removes ``QickParam`` values, converts ``addict.Dict`` to ``dict``,
        NumPy arrays to lists, NumPy scalars to Python scalars, and complex
        numbers to ``{"real": ..., "imag": ...}`` dicts.

        Parameters
        ----------
        data : any
            Arbitrary nested structure to clean.

        Returns
        -------
        cleaned : any
            Equivalent structure using only plain Python types.
        """
        try:
            from qick.asm_v2 import QickParam

            _qp = (QickParam,)
        except ImportError:
            _qp = ()

        def _clean(obj):
            if isinstance(obj, (dict, AddictDict)):
                return {
                    k: _clean(v)
                    for k, v in obj.items()
                    if not (_qp and isinstance(v, _qp))
                }
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            return obj

        return _clean(data)

    def _collect_all_key_values(self, data) -> defaultdict:
        """
        Recursively collect all leaf values grouped by key from *data*.

        Parameters
        ----------
        data : dict, addict.Dict, or list
            Nested configuration structure (typically ``_raw_list``).

        Returns
        -------
        result : defaultdict of list
            Mapping from each leaf key to a list of all values found for
            that key across all nested dicts.
        """
        result = defaultdict(list)
        if isinstance(data, (dict, AddictDict)):
            for key, value in data.items():
                if isinstance(value, (dict, list, AddictDict)):
                    nested_results = self._collect_all_key_values(value)
                    for k, v_list in nested_results.items():
                        result[k].extend(v_list)
                else:
                    result[key].append(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list, AddictDict)):
                    nested_results = self._collect_all_key_values(item)
                    for k, v_list in nested_results.items():
                        result[k].extend(v_list)
        return result

    def _refine_cfg(self, collected_data) -> AddictDict:
        """
        Collapse single-valued lists to scalars and wrap result in AddictDict.

        For keys listed in :attr:`keys_to_unify`, if all collected values are
        identical, the list is replaced by that single scalar value.

        Parameters
        ----------
        collected_data : defaultdict of list
            Output of :meth:`_collect_all_key_values`.

        Returns
        -------
        refined : AddictDict
            Unified config where unifiable keys hold scalars and the rest
            hold lists (one entry per qubit).
        """
        refined_dict = AddictDict(collected_data)
        for key, value_list in refined_dict.items():
            if (
                key in self.keys_to_unify
                and isinstance(value_list, list)
                and value_list
            ):
                unique_values = set(value_list)
                if len(unique_values) == 1:
                    single_value = unique_values.pop()
                    if isinstance(single_value, np.generic):
                        single_value = single_value.item()
                    refined_dict[key] = single_value
        return refined_dict

    def __getitem__(self, item):
        if isinstance(item, str):
            # If it's a known qubit name, return its flat config
            if item in self._name_map:
                return self.get_qubit(item)
            # Otherwise, look up in unified config
            return self.unified_config.get(item)
        if isinstance(item, int):
            return self.get_qubit(item)
        raise TypeError("Index must be int or str")


# =============================================================================
# Helper Utilities
# =============================================================================


def auto_unit(value, base_unit=""):
    """
    Scale a numeric value to the most appropriate SI metric prefix.

    Determines the best SI prefix (pico through giga) by inspecting the
    magnitude of *value* and returns both the scaled value and the combined
    prefix + *base_unit* string.

    Parameters
    ----------
    value : float or array-like
        Numeric value(s) to scale.
    base_unit : str, optional
        Physical unit string to append after the metric prefix (e.g. ``"Hz"``
        or ``"V"``).  Default is ``""`` (no unit).

    Returns
    -------
    result : dict
        Dictionary with two keys:

        ``"value"`` : np.ndarray
            Array of scaled values (same shape as *value*).
        ``"unit"`` : str
            Metric-prefixed unit string (e.g. ``"MHz"`` or ``"mV"``).

    Examples
    --------
    >>> auto_unit(5_000_000, "Hz")
    {'value': array([5.]), 'unit': 'MHz'}
    """
    prefixes = {
        -12: "p",  # pico
        -9: "n",  # nano
        -6: "u",  # micro
        -3: "m",  # milli
        0: "",  # base
        3: "k",  # kilo
        6: "M",  # mega
        9: "G",  # giga
    }

    arr = np.array(value, dtype=float)

    maxval = np.max(np.abs(arr))
    if maxval == 0:
        exp = 0
    else:
        exp = int(math.floor(math.log10(maxval) / 3) * 3)
        exp = max(min(exp, 9), -12)

    scaled_value = arr / (10**exp)
    prefix = prefixes[exp]

    return {"unit": f"{prefix}{base_unit}", "value": scaled_value}
