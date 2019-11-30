from typing import Mapping, Iterable, List, Tuple
from joblib import Parallel

import pandas as pd
from sklearn.model_selection import ParameterGrid

'''
Type-Casting Functions
'''


def require_tuple(obj):
    if isinstance(obj, str):
        return (obj,)
    assert isinstance(obj, Iterable)
    return tuple(obj)


def require_dict(obj):
    if isinstance(obj, dict):
        return obj


def require_enumerable(item):
    if isinstance(item, (List, Tuple)):
        return item
    return [item]


'''
List and DataFrame Manipulation
'''


# Partition Python list with optional offset.
def partition(list, n, offset=0):
    return [list[i:i + n] for i in range(0, len(list), n - offset)]


def append_dict(df, dict):
    # append dictionary to dataframe
    dict_df = pd.DataFrame(data={key: {0: value} for key, value in dict.items()})
    return pd.concat([df, dict_df], sort=False)


def df_append(master_df, df, column_map):
    df = df.copy()
    for column_name, column_value in column_map.items():
        df[column_name] = column_value
    if master_df is not None:
        master_df = master_df.append(df)
        master_df.reset_index(drop=True, inplace=True)
    else:
        master_df = df
    return master_df


def df_reorder_columns(df, send_front=[], send_back=[]):
    middle = [col for col in df.columns if col not in send_front and col not in send_back]
    return df.reindex(columns=send_front + middle + send_back, copy=False)


def dataframe_with_structure(dataframe, ndarray):
    '''
    Takes an ndarray and converts it to a dataframe
    :param dataframe:
    :param ndarray:
    :return:
    '''
    if isinstance(dataframe, pd.Series):  # checking for type series
        dataframe = dataframe.to_frame()  # converts series into a dataframe
    dataframe = dataframe.copy() # third possible irrelevant copy function?
    if (len(ndarray.shape) == 1):  # checks ndarray is vector (1-tensor)
        ndarray = ndarray.reshape(-1, 1) # reshapes array to be of option(-1) and size 1// -1 autofills based of the previous shape
    if len(dataframe.index) != ndarray.shape[0] or len(dataframe.columns) != ndarray.shape[1]:
        raise Exception("wrong shape")
    dataframe[dataframe.columns] = ndarray
    return dataframe


'''
Runtime Control
'''


def resolve_delayed(delayed_iter):
    # Resolve delayed statements with multicore processing
    delayed_list = list(delayed_iter)
    return list(Parallel(n_jobs=3)(delayed_list)) # Parallel - n_jobs = -1, use all cpu's

'''
Dot-Notation and (unused) Parameter Functions
'''


# compact way to use dot notation
class Map():
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def collect(self, exclude_keys=[]):
        all_values = []
        Map.__collect(self.__dict__, exclude_keys, all_values)
        return all_values

    @staticmethod
    def __collect(dict, exclude_keys, l):
        for key, val in dict.items():
            if key in exclude_keys:
                continue
            if isinstance(val, str):
                l.append(val)
            else:
                Map.__collect(val.__dict__, exclude_keys, l)

# https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
def adjusted_r2_score(r2_score, num_features, num_samples):
    return 1 - (1 - r2_score ** 2) * ((num_samples - 1) / (num_samples - num_features - 1))

def attach_params(df, params): # function for attaching new paramaters into the sci function
    for key, value in params.iteritems():
        df[key] = str(value)

class ParamStitcher:
    def __init__(self, param_grid, static_params={}):
        param_tuple_grid = {}

        for param_name, dict_value in param_grid.items():
            column_names = [param_name]
            # if "|" in param_name:
            #     column_values = param_name[param_name.index("|") + 1:]
            #     param_name = param_name[:param_name.index("|")]
            #     param_display_names[param_name] = column_values
            # else:
            #     param_display_names[param_name] = param_name

            if isinstance(dict_value, Mapping):
                if "__cols" in dict_value:
                    column_names = require_enumerable(dict_value["__cols"])
                param_tuple_grid[param_name] = []
                for column_values, param_value in dict_value.items():
                    if column_values is "__cols":
                        continue
                    column_kvs = {column_name: str(column_value) for column_name, column_value in zip(column_names, require_enumerable(column_values))}
                    param_tuple_grid[param_name].append((param_value, column_kvs))
            else:
                param_tuple_grid[param_name] = [(entry, {param_name: str(entry)}) for entry in dict_value]



        def split(combination):
            param_map = {}
            column_map = {}
            for key, pair in combination.items():
                param_map[key] = pair[0]
                for column_name, column_value in pair[1].items():
                    column_map[column_name] = column_value
            return {**static_params, **param_map}, column_map

        combinations = [{key: combination[key] for key in param_grid.keys()} for combination in ParameterGrid(param_tuple_grid)]
        self.params = [split(combination) for combination in combinations]
        self.master_df = None


    def __iter__(self):
        return iter(self.params)

    def attach(self, df, column_map):
        df = df.copy()
        for column_name, column_value in column_map.items():
            df[column_name] = column_value
        if self.master_df is not None:
            self.master_df = self.master_df.append(df)
            self.master_df.reset_index(drop=True, inplace=True)
        else:
            self.master_df = df


def param_combinations(static_params, param_grid, fn):
    param_display_names = {}
    display_names_grid = {}
    values_grid = {}
    for param_name, value in param_grid.items():
        if "|" in param_name:
            param_display_name = param_name[param_name.index("|")+1:]
            param_name = param_name[:param_name.index("|")]
            param_display_names[param_name] = param_display_name
        else:
            param_display_names[param_name] = param_name
        if isinstance(value, Mapping):
            display_names_grid[param_name] = []
            values_grid[param_name] = []
            for param_display_name, param_value in value.items():
                display_names_grid[param_name].append(param_display_name)
                values_grid[param_name].append(param_value)
        else:
            display_names_grid[param_name] = value
            values_grid[param_name] = value

    ParameterGrid(display_names_grid)
    names_params = [{**static_params, **entry} for entry in ParameterGrid(display_names_grid)]
    objects_params = [{**static_params, **entry} for entry in ParameterGrid(values_grid)]

    master_df = None

    for name_params, object_params in zip(names_params, objects_params):
        ret = fn(object_params)
        if isinstance(ret, pd.DataFrame):
            df = ret.copy()
            for param_name, column_name in param_display_names.items():
                df[column_name] = str(name_params[param_name])
            if master_df is not None:
                master_df = master_df.append(df)
            else:
                master_df = df
    return master_df.reset_index(drop=True)


def copy_remove_duplicates(list):
    unique = set()
    ret = []
    for item in list:
        if item not in unique:
            unique.add(item)
            ret.append(item)
    return ret

