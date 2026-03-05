import datetime
import json
import math
import os
import pprint
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import h5py
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


def get_next_filename(base_path: str, exp_name: str, suffix: str = ".h5") -> str:
    """
    Generate a unique filename for an experiment, ensuring no duplicates.
    Files are saved in a directory structure: base_path/YYYY/MM/MM-DD/
    """
    today = datetime.date.today()
    year, month, day = today.strftime("%Y"), today.strftime("%m"), today.strftime("%d")
    date_path = f"{month}-{day}"

    experiment_path = os.path.join(base_path, year, month, date_path)
    os.makedirs(experiment_path, exist_ok=True)

    i = 1
    while True:
        fname = f"{exp_name}_{i}{suffix}"
        full_path = os.path.join(experiment_path, fname)
        if not os.path.exists(full_path):
            return full_path
        i += 1


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


def saveh5(
    file_path: str,
    data_dict: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save experiment data to an HDF5 file with structured groups for x/y/z axes.
    """
    with h5py.File(file_path, "w") as f:
        param_grp = f.create_group("parameter")
        data_grp = f.create_group("data")

        if "x_name" in data_dict and "x_value" in data_dict:
            x_grp = param_grp.create_group(data_dict["x_name"])
            x_grp.create_dataset("x_axis_value", data=data_dict["x_value"])

        if "y_name" in data_dict and "y_value" in data_dict:
            y_grp = param_grp.create_group(data_dict["y_name"])
            y_grp.create_dataset("y_axis_value", data=data_dict["y_value"])

        if "z_name" in data_dict and "z_value" in data_dict:
            data_grp.create_dataset(data_dict["z_name"], data=data_dict["z_value"])
        if "experiment_name" in data_dict:
            f.attrs["experiment_name"] = data_dict["experiment_name"]
        if config:
            f.attrs["config"] = json.dumps(config)
        if result:
            f.attrs["result"] = json.dumps(result)


def saveshot(
    file_path: str,
    data_dict: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save experiment data to an HDF5 file, dumping all keys in data_dict to the 'data' group.
    """
    with h5py.File(file_path, "w") as f:
        data_grp = f.create_group("data")

        for key, value in data_dict.items():
            data_grp.create_dataset(key, data=value)

        if "experiment_name" in data_dict:
            f.attrs["experiment_name"] = data_dict["experiment_name"]
        if config:
            f.attrs["config"] = json.dumps(config)
        if result:
            f.attrs["result"] = json.dumps(result)


def read_h5_file(file_path: str) -> Dict[str, Any]:
    """
    Read experiment data from an HDF5 file.
    """
    data = {}

    with h5py.File(file_path, "r") as f:
        param_grp = f["parameter"]
        data_grp = f["data"]

        x_name, y_name = None, None
        x_value, y_value = None, None

        for key in param_grp.keys():
            subgroup = param_grp[key]
            if "x_axis_value" in subgroup:
                x_name = key
                x_value = subgroup["x_axis_value"][:]
            elif "y_axis_value" in subgroup:
                y_name = key
                y_value = subgroup["y_axis_value"][:]

        if x_name and x_value is not None:
            data["x_name"] = x_name
            data["x_value"] = np.asarray(x_value)
        else:
            raise ValueError("No x-axis data found in the HDF5 file.")

        if y_name and y_value is not None:
            data["y_name"] = y_name
            data["y_value"] = np.asarray(y_value)
        else:
            data["y_name"] = None
            data["y_value"] = None

        z_name = next(iter(data_grp.keys()), None)
        if z_name:
            data["z_name"] = z_name
            data["z_value"] = np.asarray(data_grp[z_name][:])
        else:
            raise ValueError("No z-axis data found in the HDF5 file.")

        data["experiment_name"] = (
            json.loads(f.attrs["experiment_name"])
            if "experiment_name" in f.attrs
            else None
        )
        data["config"] = json.loads(f.attrs["config"]) if "config" in f.attrs else None
        data["result"] = json.loads(f.attrs["result"]) if "result" in f.attrs else None

    return data


# =============================================================================
# Config / Dictionary Utilities
# =============================================================================


def update_python_dict(
    file_path: str, updates: Dict[str, Union[Any, Dict[int, Any]]]
) -> None:
    """
    Update dictionary values inside a Python config file while preserving formatting.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    inside_target: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        for full_key, new_value in updates.items():
            if "." in full_key:
                dict_name, key = full_key.split(".", 1)
            else:
                continue

            if stripped.startswith(f"{dict_name} = {{"):
                inside_target = dict_name

            if inside_target and re.match(rf'^\s*"{key}"\s*:\s*', stripped):
                if isinstance(new_value, dict):
                    match = re.search(rf'"{key}"\s*:\s*(\[[^\]]*\])', stripped)
                    if match:
                        old_list = eval(match.group(1))
                        for idx, val in new_value.items():
                            if 0 <= idx < len(old_list):
                                old_list[idx] = val
                        new_list_str = str(old_list).replace("'", "")
                        line = re.sub(
                            rf'"{key}"\s*:\s*\[[^\]]*\]',
                            f'"{key}": {new_list_str}',
                            line,
                        )

                else:
                    line = re.sub(
                        rf'("{key}"\s*:\s*)[^,]*',
                        lambda m: f"{m.group(1)}{new_value}",
                        line,
                    )

        new_lines.append(line)

        if inside_target and stripped == "}":
            inside_target = None

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def collect_all_key_values(data):
    """Recursively collects values for the same key."""
    result = defaultdict(list)
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                nested_results = collect_all_key_values(value)
                for k, v_list in nested_results.items():
                    result[k].extend(v_list)
            else:
                result[key].append(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                nested_results = collect_all_key_values(item)
                for k, v_list in nested_results.items():
                    result[k].extend(v_list)
    return result


def refine_cfg(
    collected_data: defaultdict,
    keys_to_unify: List[str] = [
        "reps",
        "res_length",
        "ro_length",
        "trig_time",
        "relax_delay",
    ],
) -> Dict[str, Any]:
    """Unifies specified keys into scalars if values are identical."""
    refined_dict = dict(collected_data)
    for key, value_list in refined_dict.items():
        if key in keys_to_unify and isinstance(value_list, list) and value_list:
            unique_values = set(value_list)
            if len(unique_values) == 1:
                single_value = unique_values.pop()
                if isinstance(single_value, np.generic):
                    single_value = single_value.item()
                refined_dict[key] = single_value
    return refined_dict


def select_config_idx(config: dict, idx: int) -> dict:
    """Select configuration for a specific index."""
    selected = {}
    for key, value in config.items():
        if isinstance(value, list):
            if idx < len(value):
                selected[key] = value[idx]
            else:
                raise IndexError(f"Index {idx} out of range for key '{key}'")
        else:
            selected[key] = value
    return selected


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
        self,
        data: Union[List, Dict],
        keys_to_unify: Optional[List[str]] = None
    ):
        self._raw_list = data
        self.keys_to_unify = keys_to_unify or [
            "reps", "res_length", "ro_length", "trig_time", "relax_delay"
        ]

        # Create Name Map for quick lookup
        self._name_map = {}
        if isinstance(self._raw_list, list):
            for idx, cfg in enumerate(self._raw_list):
                name = cfg.get("name")
                if name:
                    self._name_map[name] = idx

        self._refresh()

    def _refresh(self) -> None:
        """Recalculate the unified configuration after updates."""
        self._raw_collected = self._collect_all_key_values(self._raw_list)
        # Unified config is now an AddictDict for dot notation
        self.unified_config = self._refine_cfg(self._raw_collected)

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

        self._refresh()

    def _recursive_update(self, nested_data, target_key, new_value) -> bool:
        """Helper to recursively find a key in nested structure and update it."""
        found = False
        if isinstance(nested_data, dict):
            if target_key in nested_data:
                old_val = nested_data[target_key]
                val_to_set = new_value
                if isinstance(val_to_set, np.generic):
                    val_to_set = val_to_set.item()

                if old_val != val_to_set:
                    nested_data[target_key] = val_to_set
                    return True
                return False

            for _, v in nested_data.items():
                if isinstance(v, (dict, list)):
                    if self._recursive_update(v, target_key, new_value):
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
        clean_data = self._clean_data(self._raw_list)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Auto-generated configuration file\n")
            f.write("from addict import Dict\n\n")
            f.write("config_list = ")
            pprint.pprint(clean_data, stream=f, width=120, sort_dicts=False)
            f.write("\n")
        print(f"Configuration saved to {filename}")

    def read_qubit_config(
        self,
        q_id: Union[int, str],
    ) -> Dict:
        indices = self._resolve_indices(q_id)
        target_idx = indices[0]
        raw_nested_cfg = self._raw_list[target_idx]
        clean_data = self._clean_data(raw_nested_cfg)
        return clean_data

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

    def _dump_dict_with_spacing(self, data: dict, is_list_item: bool = False) -> str:
        """
        Helper to dump a dictionary with conditional spacing.
        - If a value is a dict (or next value is a dict), use 1 empty line separator.
        - If both are scalars, use 0 empty lines (single newline).
        If is_list_item is True, the first key is prefixed with "- ".
        """
        if not isinstance(data, dict):
            return yaml.dump(data, default_flow_style=False, sort_keys=False).strip()

        parts = []
        keys = list(data.keys())
        
        for i, key in enumerate(keys):
            val = data[key]
            single_item = {key: val}
            dumped = yaml.dump(single_item, default_flow_style=False, sort_keys=False).strip()
            
            if i == 0 and is_list_item:
                part = "- " + dumped
            else:
                if is_list_item:
                    # Indent by 2 spaces
                    lines = dumped.split('\n')
                    indented_lines = ["  " + line for line in lines]
                    part = "\n".join(indented_lines)
                else:
                    part = dumped
            
            # Determine separator
            if i < len(keys) - 1:
                next_key = keys[i+1]
                next_val = data[next_key]
                
                # Check if current or next value is a complex structure (dict or list)
                # In this config, we mostly care about dicts.
                is_complex = isinstance(val, (dict, list))
                is_next_complex = isinstance(next_val, (dict, list))
                
                if is_complex or is_next_complex:
                    separator = "\n\n"
                else:
                    separator = "\n"
            else:
                separator = ""
            
            parts.append(part + separator)

        return "".join(parts)

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
        """Recursively clean data (Addict->dict, Numpy->native)."""
        if isinstance(data, (dict, AddictDict)):
            return {k: self._clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data(v) for v in data]
        elif isinstance(data, np.generic):
            return data.item()
        return data

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
