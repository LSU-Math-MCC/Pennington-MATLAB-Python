from functools import reduce
from typing import Iterable, Mapping, List, Dict

import pandas as pd

from utilities.common_functions import copy_remove_duplicates

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 199
pd.options.display.width = None


# print([[3]] * 121)
# exit(1)

# option grid - dataframe with __data__ column
# option obj - dict or list

def require_tuple(obj):
    if isinstance(obj, str):
        return (obj,)
    assert isinstance(obj, Iterable)
    return tuple(obj)

def compose_option_grid(option_obj, cnames=None):
    '''
    Example: option_obj={'common', 'all', 'none'}, cnames=measurements

       measurements __data__
    0       common   common
    1          all      all
    2         none     none
    '''
    if isinstance(option_obj, pd.DataFrame):
        return option_obj
    if cnames is None or "__cols" in option_obj:
        cnames = require_tuple(option_obj["__cols"])
    else:
        cnames = require_tuple(cnames)
    dict = {}
    for col in cnames:
        dict[col] = []
    data = []
    if not isinstance(option_obj, Mapping):
        assert len(cnames) == 1
        option_obj = {str(item): item for item in option_obj}
    for key, value in option_obj.items():
        if key != "__cols":
            key = require_tuple(key)
            for idx in range(len(key)):
                dict[cnames[idx]].append(key[idx])
            data.append(value)
    return pd.DataFrame({**dict, "__data__": data})

def add_option_columns(cnames, cvalue_dict):
    cnames = require_tuple(cnames)
    combined = pd.DataFrame()
    for cvalues, option_grid in cvalue_dict.items():
        cvalues = require_tuple(cvalues)
        assert len(cnames) == len(cvalues)
        assert isinstance(option_grid, pd.DataFrame)
        combined = combined.append(option_grid.assign(
            **{cname: [cvalue] * len(option_grid) for cname, cvalue in zip(cnames, cvalues)}))
    return combined.reset_index(inplace=True, drop=True)

def compose_option_grids(cvalue_to_option_dicts):
    dfs = []
    for cvalue, option_dict in cvalue_to_option_dicts.items():
        assert isinstance(option_dict, Mapping)
        dfs.append(compose_option_grid(option_dict, cvalue))
    return dfs

def __cross_product(fn, df1, df2):
    '''
    Example:
    fn = <function compose_param_grid.<locals>.<lambda>.<locals>.<lambda> at 0x0000016458EBAEA0>
    df1=
      BMI      __data__
    0   N  {'BMI': 'N'}
    1   Y  {'BMI': 'Y'}
    df2=
      Age      __data__
    0   N  {'Age': 'N'}
    1   Y  {'Age': 'Y'}

    product=
      BMI Age                  __data__
    0   N   N  {'BMI': 'N', 'Age': 'N'}
    1   N   Y  {'BMI': 'N', 'Age': 'Y'}
    2   Y   N  {'BMI': 'Y', 'Age': 'N'}
    3   Y   Y  {'BMI': 'Y', 'Age': 'Y'}
    '''
    df1 = df1.assign(__merge__=1)
    df2 = df2.assign(__merge__=1)
    product = df1.merge(df2, on="__merge__")
    product["__data__"] = [fn(x, y) for x, y in zip(product["__data___x"].values, product["__data___y"].values)]
    product.drop(["__merge__", "__data___x", "__data___y"], inplace=True, axis=1)
    return product

# Makes options into df
def combine_options(input, *other_option_objs, additional_options=[], remove_empty=True):
    if len(other_option_objs) > 0:
        input = [input] + list(other_option_objs)
    if isinstance(input, Iterable) and isinstance(next(iter(input)), pd.DataFrame):
        grids = input
    elif isinstance(input, Mapping):
        grids = compose_option_grids(input)
    else:
        assert "unknown input for combine_options"
    combined = reduce(lambda df1, df2: __cross_product(lambda x, y: x + y, df1, df2), grids)
    if remove_empty:
        empty_idxs = [idx for idx, data in combined["__data__"].iteritems() if len(data) == 0]
        combined.drop(empty_idxs, inplace=True)
    if len(additional_options) > 0:
        def add_additional(existing):
            for additional in additional_options:
                if additional not in existing:
                    existing.append(additional)
            return existing
        combined["__data__"] = combined["__data__"].apply(add_additional)
    combined.reset_index(inplace=True, drop=True)
    return combined

def compose_param_grid(param_option_dict):
    '''
    Enumerate 'decision tree' branches as rows of a pd DataFrame from a param_option_dict.

    :param param_option_dict: Dictionary with keys as categories and values as options for those categories
    :return: Pandas DataFrame with columns from keys in param_option_dict and a __data__ column with the row data as a dict
    '''
    dfs = []
    for key, option_obj in param_option_dict.items():
        if not isinstance(option_obj, pd.DataFrame):
            # convert into dataframe
            option_obj = compose_option_grid(option_obj=option_obj, cnames=key)
        # check that it's a dataframe. assert aborts and gives an error if it's false
        assert isinstance(option_obj, pd.DataFrame)
        # writes dfs -  a list with vales from __data__ and keys from dict in {} pandas
        dfs.append(option_obj.assign(__data__=option_obj["__data__"].apply(lambda val: {key: val})))
    if len(dfs) == 0:
        return pd.DataFrame()
    return reduce(lambda df1, df2: __cross_product(lambda x, y: {**x, **y}, df1, df2), dfs)


def iter_params(param_option_dict, noniterative_params={}):
    '''
    Return output of compose_param_grid as iterable param_maps and column_maps for batch runs.

    :param param_option_dict: Dict of parameter options passed to compose_param_grid
    :param noniterative_params: Dict of parameter options that stay constant such as column scalars
    :return: param_maps, column_maps (iterable over rows of compose_param_grid(param_option_dict), both dicts, param_maps has key/value pairs for noniterative_params, column maps does not)
    '''
    grid = compose_param_grid(param_option_dict)
    if len(grid) == 0:
        yield noniterative_params, {}
    else:
        for idx, row in grid.iterrows():
            yield {**noniterative_params, **row["__data__"]}, row.drop("__data__").to_dict()

# dfs = {
#     ("bmi1", "bmi2"): {
#         ("N", 1): [],
#         ("Y", 2): ["bmi"]
#     },
#     "age": {
#         "N": [],
#         "Y": ["age"]
#     },
#     "measurements": tiered({
#         "none": [],
#         "lengths": ["height", "upper_arm_length", "upper_leg_length"],
#         "circumferences": ["waist_circumference", "arm_circumference", "calf_circumference", "thigh_circumference"]
#     }),
# }
# # for df in dfs:
# #     print(df)
# print("")
# print("")
# giant = combine_options(dfs)
# print(giant)
# print("")
# print(compose_option_grid(["a", "b", "c"], "n2ame"))
#
#
# grid = {"feature": giant, "f2": [1, 2]}
#
# for data, params in iter_params(grid):
#     print(f"{data}, {params}\n")
#
#
# params = dict(
#     param1=["red", "green", "orange"],
#     param2={
#         "zero": 0,
#         "one": 1,
#         "two": 2
#     }
# )
#
# for data, params in iter_params(params):
#     print(f"{data}, {params}\n")
#
# print(list(["s", "s"]))


# Dict from typing
def tiered(dict, combine_names=True):
    if not isinstance(dict, Dict):
        raise AssertionError("dict must be Dict")
    out = {}
    cumulative_name = ""
    cumulative_values = []
    # dict.items() returns key-value pairs from dictionary
    for name, list_value in dict.items():
        if not isinstance(list_value, List):
            raise AssertionError("value must be list")
        if len(cumulative_values) == 0:
            cumulative_name = name
        else:
            cumulative_name += "," + name
        cumulative_values = list_value + cumulative_values
        out[cumulative_name if combine_names else name] = copy_remove_duplicates(cumulative_values)
    return out