from random import Random
from sklearn.model_selection import ParameterGrid
from typing import Tuple
import uwndc19.hp_tuning.mutation_funcs


def get_mutations(mutations_grid: list):
    mutations = []
    for mutation_grid in mutations_grid:
        func_name = list(mutation_grid.keys())[0]
        params_grid = mutation_grid[func_name]

        for params in ParameterGrid(params_grid):
            mutations.append((func_name, params))

    Random(123).shuffle(mutations)

    return mutations


def mutate_config(config: dict, mutation: Tuple):
    config = dict(config)
    func_name, params = mutation

    # changes config in-place
    getattr(uwndc19.hp_tuning.mutation_funcs, func_name)(config, **params)

    return config


def generate_mutation_name(mutation: Tuple):
    func_name, params = mutation

    name_parts = [func_name]
    for key, value in params.items():
        name_parts.append('%s=%s' % (key, str(value)))

    return '-'.join(name_parts)
