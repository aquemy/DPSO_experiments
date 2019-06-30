from datetime import datetime
import json
import os

def serialize_results(scenario, policy):
    
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    results = {
        'scenario': scenario,
        'context': policy.context
    }
    path = os.path.join('results', '{}_{}.json'.format(scenario['file_name'], now))
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)