import itertools

from hyperopt import hp
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from experiment.pipeline.space import *

def expand_params(operation, operator):
    try:
        params = globals()['params_{}'.format(type(operator).__name__)]()
        expanded_params = {}
        for param_name, param_val in params.iteritems():
            expanded_params['{}__{}'.format(operation, param_name)] = param_val
        return expanded_params
    except Exception as e:
        return {} # TODO: Warning in verbose mode


def generate_grid(prototype):
    final_grid = []
    elements = [zip([k] * len(o), o) for k,o in prototype.iteritems()]
    for element in itertools.product(*elements):
        config = dict(element)
        params = {}
        for operation, operator in config.iteritems():
            config[operation] = [operator] # Trick since ScikitLearn requires list
            if operator is not None:
                params.update(expand_params(operation, operator))
        config.update(params)
        final_grid.append(config)
    return final_grid


def pretty_config(conf):
    print_conf = {}
    for k,v in conf.iteritems():
        if '__' in k:
            print_conf[k] = v
        elif v == 'NoneType':
            print_conf[k] = None
        else:
            if isinstance(v, list):
                print_conf[k] = type(v[0]).__name__
            else:
                print_conf[k] = type(v).__name__
    return print_conf


def pretty_print_grid(grid):
    for conf in grid:
        print_conf = pretty_config(conf)
        print(print_conf)


def generate_domain_space(prototype):
    domain_space = {}
    for operation, operators in prototype.iteritems():
        operators_space = []
        for operator in operators:
            label = '{}_{}'.format(operation, type(operator).__name__ if operator is not None else 'NoneType')
            params = expand_params(operation, operator)
            operator_config = {}
            for k, v in params.iteritems():
                operator_config[k] = hp.choice('{}_{}'.format(label, k), v)
            operators_space.append((label, operator_config))
        domain_space[operation] = hp.choice(operation, operators_space)
    return domain_space

