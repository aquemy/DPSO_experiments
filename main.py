from experiment.utils import datasets, scenarios, serializer, cli
from experiment import policies

import json
import sys
import os
import json

from sklearn.model_selection import train_test_split

def main(args):
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario)
    print('SCENARIO:\n {}'.format(json.dumps(scenario, indent=4, sort_keys=True)))
    
    X, y = datasets.load(scenario['setup']['dataset'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.4,
        stratify=y,
        random_state=scenario['control']['seed']
    )

    policy = policies.initiate(scenario['setup']['policy'], config)
    policy.run(X,y)

    serializer.serialize_results(scenario, policy)

if __name__ == "__main__":
    args = cli.parse_args()
    main(args)