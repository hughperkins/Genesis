import dataclasses
import gstaichi as ti
import torch
import sys
sys.path.append("tools")
import params_pickler
import params_unpickler


def test_params_pickler_torch():
    t = torch.zeros((3, 2), dtype=torch.int32)
    t[0, 0] = 5
    t[1, 1] = 3
    print('t', t)
    py_objs = params_pickler.convert_to_python_objs(t)
    print(py_objs)
    t2 = params_unpickler.pickle_to_py(py_objs)
    print('t2', t2)


def test_params_pickler_ndarray_scalar():
    t = ti.ndarray(ti.i32, (3, 2))
    t[0, 0] = 5
    t[1, 1] = 3
    print('t', t.to_numpy())
    py_objs = params_pickler.convert_to_python_objs(t)
    print('py_objs', py_objs)
    t2 = params_unpickler.pickle_to_py(py_objs)
    print('t2', t2, t2.dtype)
    print(t2.to_numpy())


def test_params_pickler_ndarray_vector():
    vec3 = ti.types.vector(3, ti.i32)
    t = ti.ndarray(vec3, (3, 2))
    t[0, 0] = (3, 4, 5)
    t[1, 1] = (10, 11, 12)
    print('t', t.to_numpy())
    py_objs = params_pickler.convert_to_python_objs(t)
    print('py_objs', py_objs)
    t2 = params_unpickler.pickle_to_py(py_objs)
    print('t2', t2, t2.dtype)
    print(t2.to_numpy())


@dataclasses.dataclass
class MyDataclass:
    foo: ti.types.NDArray
    bar: ti.types.NDArray
    an_int: int
    a_float: float
    a_bool: bool


def test_params_pickle_dataclass() -> None:
    foo = ti.ndarray(ti.i32, (3,))
    bar = ti.ndarray(ti.i32, (5,))
    an_int = 123
    a_float = 4.56
    a_bool = True
    my_dataclass = MyDataclass(foo=foo, bar=bar, an_int=an_int, a_float=a_float, a_bool=a_bool)
    t = my_dataclass
    print('t', t)
    py_objs = params_pickler.convert_to_python_objs(t)
    print('py_objs', py_objs)
    t2 = params_unpickler.pickle_to_py(py_objs)
    print('t2', t2)
    # print(t2.to_numpy())
