import gstaichi as ti
import pickle
import torch
import dataclasses
import os


def convert_to_python_objs(params):
    if isinstance(params, tuple):
        res = []
        for param in params:
            _res = convert_to_python_objs(param)
            res.append(_res)
        return tuple(res)
    elif isinstance(params, torch.Tensor):
        print('torch.tensor')
        res = {
            "classname": params.__class__.__name__,
            "classtype": "torch",
            "dtype": params.dtype,
            "shape": tuple(params.shape),
        }
        print(res)
        res["data"] = params.detach().cpu()
        return res
    # elif hasattr(params, "_is_taichi_class"):
    #     print("is data oriented")
    elif dataclasses.is_dataclass(params):
        print("is py dataclass")
        res_fields = {}
        res = {
            "classname": params.__class__.__name__,
            "classpath": f"{params.__class__.__module__}.{params.__class__.__qualname__}",
            "classtype": "dataclasses.dataclass",
        }
        print(res)
        res["fields"] = res_fields
        for field in dataclasses.fields(params):
            field_name = field.name
            field_val = getattr(params, field_name)
            print("field_name", field_name)
            res_fields[field_name] = convert_to_python_objs(field_val)
        return res
    elif isinstance(params, (float, int, bool)):
        return params
    elif isinstance(params, (ti.VectorNdarray, ti.ScalarNdarray)):
        print("got ndarray")
        res = {
            "classname": params.__class__.__name__,
            "classtype": "ndarray",
            "shape": params.shape,
            "element_type": str(params.dtype),
            "element_shape": params.element_shape,
            "data": params.to_numpy()
        }
        res_ = dict(res)
        del res_["data"]
        print("res", res_)
        return res
    else:
        raise Exception("unhandled type", type(params))


def dump_params(pickle_filepath, params):
    print("dumpign params")
    python_objs = convert_to_python_objs(params)
    # print("python_objs", python_objs)
    # print("python_objs[0]", python_objs[0])
    with open(pickle_filepath, "wb") as fh:
        pickle.dump(python_objs, fh)
    os.system(f"ls -lh {pickle_filepath}")


