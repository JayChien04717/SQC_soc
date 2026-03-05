from addict import Dict
import json
from ruamel.yaml import YAML
import ruamel.yaml
import numpy
import sys
from collections.abc import MutableMapping
from io import StringIO
import numpy as np

yml = YAML(pure=True)
yml.indent(mapping=4, sequence=4, offset=2)
yml.compact(seq_seq=False, seq_map=False)
yml.default_flow_style = False
yml.preserve_quotes = True
yml.explicit_start = True


def read_yml(data_pth, verbose=False):
    with open(data_pth, "r") as file:
        config = yml.load(file)
    if verbose:
        yml.dump(config, sys.stdout)
    return config


def read_yml2(data_pth, verbose=False):
    with open(data_pth, "r") as file:
        if verbose:
            yml.dump(yml.load(file), sys.stdout)
        data = yml.load(file)
        globals().update(data)


def save_yml(data_name: str, data: dict, show_config: bool = False, save_py=False):
    with open(data_name + ".yaml", "w") as file:
        yml.dump(data, file)
    if save_py:
        with open(data_name + ".py", "w", encoding="utf-8") as file:
            file.write(json.dumps(data, sort_keys=False, indent=4))
    if show_config:
        yaml_str = StringIO()
        yml.dump(data, yaml_str)
        print(yaml_str.getvalue())
    print("Data saved!")


def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {convert_to_builtin(k): convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_builtin(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    else:
        return obj


def yml_comment(config: dict):
    config_clean = convert_to_builtin(config)
    stream = StringIO()
    yml.dump(config_clean, stream)
    return stream.getvalue()


def find_key_in_dict(config: dict, target_key):
    result = []

    def recursive_search(config, path=[]):
        if isinstance(config, dict):
            for key, value in config.items():
                if key == target_key:
                    result.append(path + [key])
                if isinstance(value, dict):
                    recursive_search(value, path + [key])
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        recursive_search(item, path + [key, index])

    recursive_search(config)
    return result


def update_dict_value(config, target_key, new_value):
    current = config
    path = find_key_in_dict(config, target_key)[0]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = new_value


def flatten_dict(
    config: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_addict(config: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    items = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return Dict(items)


############ add represent np float #####################
def represent_numpy_float64(self, value):
    return self.represent_float(value)  # alternatively dump as a tagged float


def represent_numpy_int64(self, value):
    return self.represent_int(value)  # alternatively dump as a tagged int


def represent_numpy_array(self, array, flow_style=None):
    tag = "!numpy.ndarray"
    value = []
    node = ruamel.yaml.nodes.SequenceNode(tag, value, flow_style=flow_style)
    for elem in array:
        node_elem = self.represent_data(elem)
        value.append(node_elem)
    if flow_style is None:
        node.flow_style = True
    return node


yml.Representer.add_representer(numpy.ndarray, represent_numpy_array)
yml.Representer.add_representer(numpy.float64, represent_numpy_float64)
yml.Representer.add_representer(numpy.int64, represent_numpy_int64)

############################################################
