import pickle
import importlib
import dataclasses

import torch

import gstaichi as ti
import genesis as gs

ti.init(arch=ti.cpu)


def uncook_dtype(dtype_str):
    return getattr(ti, dtype_str)


def pickle_to_py(pkl):
    if isinstance(pkl, tuple):
        res = []
        for _child in pkl:
            res.append(pickle_to_py(_child))
        return tuple(res)
    elif isinstance(pkl, (int, float, bool)):
        return pkl
    elif isinstance(pkl, str):
        print("str", pkl)
        asdf
    elif isinstance(pkl, dict):
        if "classtype" in pkl:
            class_type = pkl["classtype"]
            print("class_type", class_type)
            print(pkl.keys())
            if class_type == "torch":
                print('torch type')
                dtype = pkl["dtype"]
                shape = pkl["shape"]
                data = pkl["data"]
                res = torch.zeros(shape, dtype=dtype)
                res.copy_(data)
                print("torch res", res)
                return res
            elif class_type == "ndarray":
                print('ndarray type')
                classname = pkl["classname"]
                dtype_str = pkl["element_type"]
                dtype = uncook_dtype(dtype_str)
                shape = pkl["shape"]
                element_shape = pkl["element_shape"]
                data = pkl["data"]
                print(classname, dtype, shape, element_shape)
                if len(element_shape) == 0:
                    res = ti.ndarray(dtype=dtype, shape=shape)
                elif len(element_shape) == 1:
                    element_type = ti.types.vector(element_shape[0], dtype)
                    print('element_type', element_type)
                    res = ti.ndarray(element_type, shape=shape)
                elif len(element_shape) == 2:
                    ...
                    asdfds
                else:
                    raise Exception(f"Unhandled element shape len {len(element_shape)}")
                # res = torch.zeros(shape, dtype=dtype)
                res.from_numpy(pkl["data"])
                print("ndarray res", res)
                # print(res.to_numpy())
                return res
            elif class_type == "dataclasses.dataclass":
                print('dataclasses.dataclass')
                fields_dict = pickle_to_py(pkl["fields"])
                classname = pkl["classname"]
                class_path = pkl["classpath"]
                print('classname', classname)
                print('class_path', class_path)
                module_path, _, cls_name = class_path.rpartition('.')
                mod = importlib.import_module(module_path)
                cls = getattr(mod, cls_name)
                if not dataclasses.is_dataclass(cls):
                    raise TypeError(f"{class_path} not a dataclass")
                res = cls(**fields_dict)
                print("res", res)
                return res
            else:
                raise Exception(f"unhandled classtype {class_type}")
        else:
            res = {}
            print('dict keys', pkl.keys())
            for k, _pkl in pkl.items():
                v = pickle_to_py(_pkl)
                res[k] = v
            return res
    else:
        raise Exception(f"unhandled type {type(pkl)}")


def init_genesis(gs):
    gs._initialized = True
    gs.use_ndarray = True
    gs.ti_float = ti.f32
    gs.use_fastcache = True


def load_pickle(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        data = pickle.load(f)
    print('data', data)
    py = pickle_to_py(data)
    return py
