import re
import numpy as np


# Partition Python list with optional offset.
def partition(list, n, offset=0):
    return [list[i:i + n] for i in range(0, len(list), n - offset)]


# Create discrete class from numeric segmentation
def discrete_class(df, cname, segmentation, classnames=None, defaultclass='nan'):
    if classnames is None:
        classnames = list(range(len(segmentation) + 1))
    assert np.all(np.diff(segmentation) > 0), 'Segmentation must be strictly increasing.'
    assert len(classnames) == len(segmentation) + 1, \
        'Number of classes must be one more than the number of floats in segmentation.'

    internal_bounds = partition(segmentation, 2, 1)[:-1]
    conditions = [lambda x: x < segmentation[0]]
    conditions += [lambda x, l = lower, u = upper: (l <= x) & (x < u)
                   for lower, upper in internal_bounds]
    conditions += [lambda x: segmentation[-1] <= x]
    condlist = lambda x: [cnd(x) for cnd in conditions]

    return np.select(condlist(df[cname]), classnames, default=defaultclass)


def standardize_subject_ids(series):
    seen = []
    id_len = len("02ADL0153")  # standard length of ids

    def map_name(name):
        if name is not str:
            name = str(name)
        name = name.upper()
        id = name[0:id_len]  # takes name column and extracts ids for beginning of string
        seen.append(id)
        if len(name) > id_len:
            suffix = name[id_len:id_len + 2]  # next 2 characters after id
            if suffix == "_A":
                return id
            elif suffix == "_B":
                return id + "_2"
        return id if seen.count(id) == 1 else f"{id}_{seen.count(id)}"
        # possibilities: id, id_2, id_n

    return series.apply(map_name)


def cut_subject_ids(series):
    id_len = len("02ADL0153")
    def map_name(name):
        name = name.upper()
        id = name[0:id_len]
        return id
    return series.apply(map_name)


# Remove all elements not 'kept' in some column
def column_filter(cname, keep):
    def ret(df, feature_columns):
        if cname in feature_columns:
            feature_columns.remove(cname)
        return df[df[cname] == keep]
    return ret

def mean_body_part_transformer(diff=False):
    def ret(df, feature_columns):
        for left_feature, right_feature, root_feature in __left_right_generator(feature_columns):
            avg_feature = "avg" + root_feature
            diff_feature = "diff" + root_feature
            feature_columns.remove(left_feature)
            feature_columns.remove(right_feature)
            feature_columns.append(avg_feature)
            df[avg_feature] = (df[left_feature] + df[right_feature]) / 2
            df[diff_feature] = (df[right_feature] - df[avg_feature]) / df[avg_feature]
            if diff:
                feature_columns.append(diff_feature)
            df.drop([left_feature, right_feature], axis=1, inplace=True)
    return ret

def __left_right_generator(feature_columns):
    fc = feature_columns.copy()
    for left_feature in fc:
        left_prefix = None
        right_prefix = None
        for (l, r) in [("l", "r"), ("left", "right")]:
            if left_feature.startswith(l) and r + left_feature[len(l):] in fc:
                left_prefix = l
                right_prefix = r
                break
        if right_prefix is None:
            continue

        root_feature = left_feature[len(left_prefix):]
        right_feature = right_prefix + root_feature
        yield left_feature, right_feature, root_feature

def age_transformer(cname_in, cname_out="age"):
    def ret(df, feature_columns):
        df[cname_out] = [119 - int(birthdate[-2:]) for birthdate in df[cname_in]]  # "28-Oct-92" => "92" => (119 - 92) => 27
    return ret

def bmi_transformer(df, feature_columns):
    df["BMI"] = df["Weight1KG"] / (df["HeightCMAvg"] / 100.) ** 2

def average_transformer(new_cname, cname_regex, series_method="mean"):
    def ret(df, feature_columns):
        cnames = [cname for cname in df.columns.values if re.match(cname_regex, cname)]
        df[new_cname] = df.apply(lambda row: getattr(row[cnames], series_method)(), axis='columns')
    return ret