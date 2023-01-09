import collections
import re

import tensorflow as tf


def init_tvars_from_checkpoint(tvars, init_checkpoint, warm_start_vars, mapping_fn=None):
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
            if warm_start_vars == '*' or re.match(warm_start_vars, name) is not None:
                name_to_variable[name] = var

    maps = [collections.OrderedDict()]

    for name in name_to_variable:
        if mapping_fn:
            key = mapping_fn(name)
        else:
            key = name
        written = False
        for assignment_map in maps:
            if key not in assignment_map:
                assignment_map[key] = name_to_variable[name]
                written = True
                break
            else:
                continue
        if not written:
            assignment_map = collections.OrderedDict()
            assignment_map[key] = name_to_variable[name]
            maps.append(assignment_map)

        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for assignment_map in maps:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    print("*" * 30 + " 未载入下列参数 " + "*" * 30)
    for var in tvars:
        if var.name not in initialized_variable_names:
            print(f"name = {var.name}, shape = {var.shape}")

    print("*" * 30 + " 载入下列参数 " + "*" * 30)
    for var in tvars:
        if var.name in initialized_variable_names:
            print(f"name = {var.name}, shape = {var.shape}")

    print("*" * 30 + f" 载入{len(name_to_variable)}参数 " + "*" * 30)
