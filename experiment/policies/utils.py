from experiment.policies.iterative import Iterative
from experiment.policies.split import Split
from experiment.policies.adaptive import Adaptive
from experiment.policies.joint import Joint

def initiate(name, config):
    policies = {
        'iterative': Iterative,
        'split': Split,
        'adaptive': Adaptive,
        'joint': Joint
    }
    if name in policies:
        return policies[name](config)
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(policies.keys())
        ))
        exit(1)  # TODO: Throw exception