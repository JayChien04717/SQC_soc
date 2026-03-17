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
    Generates the next HDF5 filename.
    Files are saved in the directory for the current date.
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
    Create a Labber-compatible LogFile for data.
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
    Clean a config dict and dump it to a YAML string (PyYAML).

    Drop-in replacement for the old ``yml_comment()`` from yamltool.
    """
    return yaml.dump(
        clean_config(config),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


class ExperimentConfig:
    """
    A wrapper class to manage, query, and update nested experiment configurations.

    Features:
    - Unified access (Mux config) with dot notation support (Addict).
    - Single qubit extraction (flat dict) with dot notation support.
    - Unified update method for both single values and dictionary merging.
    - Export to Python files (Full config or Single Qubit).
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
        """Lazy-computed unified configuration. Only rebuilds when dirty."""
        if self._dirty or self._unified_cache is None:
            raw_collected = self._collect_all_key_values(self._raw_list)
            self._unified_cache = self._refine_cfg(raw_collected)
            self._dirty = False
        return self._unified_cache

    def _mark_dirty(self) -> None:
        """Mark the unified config cache as stale."""
        self._dirty = True

    def __repr__(self) -> str:
        names = [cfg.get("name", f"idx{i}") for i, cfg in enumerate(self._raw_list)]
        return f"ExperimentConfig(qubits={names})"

    def get_qubit(self, q_id: Union[int, str]) -> AddictDict:
        """
        Retrieve a flattened configuration for a specific Qubit as an AddictDict.
        Allows access via `config.name` or `config['name']`.
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
        Extract a mux-ready config for a subset of qubits.

        For each key in unified_config:
        - If the value is a list, select only the elements at the indices
          corresponding to the qubits in qb_list (preserving order).
        - If the value is a scalar, keep it as-is.

        Additionally adds:
        - ``mux_ro_chs``: readout channels mapped from qubit index
          (e.g. Q1→2, Q2→3, ..., Q6→7 with default mux_ro_ch_start=2).
        - ``gen_mask``: the raw qubit indices (e.g. Q1→0, Q3→2, Q5→4).

        Parameters
        ----------
        qb_list : List[Union[int, str]]
            Qubit identifiers, e.g. ['Q1', 'Q3', 'Q5'] or [0, 2, 4].
        mux_ro_ch_start : int, optional
            The readout channel number corresponding to the first qubit (index 0).
            Default is 2  (i.e. Q1→2, Q2→3, ...).

        Returns
        -------
        AddictDict
            A config dict where list-valued keys have been sliced to only the
            selected qubits, with ``mux_ro_chs`` and ``gen_mask`` injected.
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
        Extract a mux-ready config while preserving the nested structure.
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
        Update parameters for multiple qubits.
        If `value` is a list/array of the same length as the number of target qubits,
        each target qubit receives the corresponding value from the list.
        """
        self.update(param, value, q_index)

    def update(
        self,
        param: Union[str, Dict[str, Any]],
        value: Any = None,
        q_index: Union[int, str, List] = None,
    ) -> None:
        """
        Unified update method.

        Mode 1: Dictionary Merge
            update(flat_dict, q_index="Q1")
            Merges a flat dictionary into the nested structure of the target qubit(s).

        Mode 2: Path Update
            update("res.res_freq_ge", 5000, "Q1")
            Updates a specific nested key using dot notation string.

        Parameters
        ----------
        param : Union[str, Dict]
            Either a key path string (e.g., 'res.freq') or a flat dictionary.
        value : Any, optional
            The value to set if param is a string. Ignored if param is a dict.
        q_index : Union[int, str, List], optional
            The target qubits. If None, targets all (broadcast/distribute).
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

        # --- Mode 2: Path Update ---
        elif isinstance(param, str):
            key_path = param
            keys = key_path.split(".")

            is_list_val = isinstance(value, (list, np.ndarray))
            should_distribute = is_list_val and (len(value) == len(target_indices))

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
                    target[keys[-1]] = val_to_set
                else:
                    setattr(target, keys[-1], val_to_set)

        else:
            raise TypeError(
                "First argument must be a string (key path) or a dict (config)."
            )

        self._mark_dirty()

    def _recursive_update(self, nested_data, target_key, new_value) -> bool:
        """Helper to recursively find a key in nested structure and update it."""
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
        indices = self._resolve_indices(q_id)
        target_idx = indices[0]
        raw_nested_cfg = self._raw_list[target_idx]
        clean_data = self._clean_data(raw_nested_cfg)
        return clean_data

    def save_to_py(self, filename: str = "latest_cfg.py") -> None:
        """Export full configuration list to file."""
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
        Export the configuration of a specific qubit to a Python file.
        Saves the original nested dictionary structure.
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
        Convert the configuration to YAML format.
        Args:
            q_id: Optional qubit identifier (name or index) to filter the output.
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
        Helper to dump a dictionary with conditional spacing and recursive structure.
        - If a value is a dict, it recurses.
        - Simple lists are forced to flow-style [1, 2, 3].
        - Proper spacing is applied between complex segments.
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
        Convert the configuration to YAML format and save to a file.
        Args:
            filename: The file path to save to.
            q_id: Optional qubit identifier (name or index) to filter the output.
        """
        yaml_str = self.to_yaml(q_id=q_id)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            print(f"Configuration saved to {filename}")

    def _resolve_indices(self, q_identifier) -> List[int]:
        """Resolve indices from int, str, list or None."""
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
        """Recursively clean data (Addict->dict, Numpy->native, filter QickParam)."""
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
        """Recursively collect all values for the same key."""
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
        """Refine list to scalar and return as AddictDict."""
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
    Automatically scale a value and add a metric prefix (e.g., k, M, G, m, u, n).
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
