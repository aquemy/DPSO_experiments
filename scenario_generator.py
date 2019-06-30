import os
import copy
import re
from collections import OrderedDict

datasets = ['wine', 'digits', 'breast', 'iris']
algorithms = ['RandomForest', 'DecisionTree', 'SVM', 'NeuralNet']
policies = ['iterative', 'split', 'adaptive', 'joint']

policies_config = {
    'iterative': {
        'step_algorithm': 15, 
        'step_pipeline': 15, 
        'reset_trial': False
    },
    'split': {
        'step_pipeline': 30
    },
    'adaptive': {
        'initial_step_time': 15,
        'reset_trial': False,
        'reset_trials_after': 2
    },
    'joint': {}
}

base = OrderedDict([
    ('title', 'Random Forest on Wine with Iterative policy'),
    ('setup', {
        'policy': 'iterative', 
        'runtime': 300, 
        'algorithm': 'RandomForest', 
        'dataset': 'wine'
    }),
    ('control', {
        'seed': 42
    }), 
    ('policy', {})
])

def __write_scenario(path, scenario):
    try:
        print('   -> {}'.format(path))
        with open(path, 'w') as f:
            for k,v in scenario.iteritems():
                if isinstance(v, str):
                    f.write('{}: {}\n'.format(k, v))
                else:
                    f.write('{}:\n'.format(k))
                    for i,j in v.iteritems():
                        f.write('  {}: {}\n'.format(i,j))
    except Exception as e:
        print(e)

for dataset in datasets:
    print('# DATASET: {}'.format(dataset))
    for algorithm in algorithms:
        print('## ALGORITHM: {}'.format(algorithm))
        for policy in policies:
            scenario = copy.deepcopy(base)
            scenario['setup']['dataset'] = dataset
            scenario['setup']['algorithm'] = algorithm
            scenario['setup']['policy'] = policy
            scenario['policy'] = copy.deepcopy(policies_config[policy])
            a = re.sub(r"(\w)([A-Z])", r"\1 \2", algorithm)
            b = ''.join([c for c in algorithm if c.isupper()]).lower()
            scenario['title'] = '{} on {} with {} policy'.format(
                a, 
                dataset.title(), 
                policy.title()
            )

            if policy == 'split':
                runtime = scenario['setup']['runtime']
                step = policies_config['split']['step_pipeline']
                ranges = [i for i in xrange(0, runtime+step, step)]
                for r in ranges:
                    scenario['policy']['step_pipeline'] = r
                    path = os.path.join('./scenarios', '{}_{}_{}_{}.yaml'.format(b, dataset, policy, r))
                    __write_scenario(path, scenario)
            else:
                path = os.path.join('./scenarios', '{}_{}_{}.yaml'.format(b, dataset, policy))
                __write_scenario(path, scenario)

