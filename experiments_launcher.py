import os
import yaml
from six import iteritems
import time
import subprocess
import datetime

from prettytable import PrettyTable
from tqdm import tqdm

SCENARIO_PATH = './scenarios'
RESULT_PATH = './results'
GLOBAL_SEED = 42

def yes_or_no(question):
    while True:
        reply = str(raw_input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

# Gather list of scenarios
scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
result_list = [p for p in os.listdir(RESULT_PATH) if '.json' in p]
scenarios = {}

# Determine which one have no result files
for scenario in scenario_list:
    base_scenario = scenario.split('.yaml')[0]
    if scenario not in scenarios:
        scenarios[scenario] = {'results': None, 'path': scenario}
    for result in result_list:
        base_result = result.split('.json')[0]
        if base_result.startswith(base_scenario):
            scenarios[scenario]['results'] = result
            date = base_result.split(base_scenario + '_')[-1].replace('_', ' ')
            scenarios[scenario]['results_date'] = date

# Calculate total amount of time
total_runtime = 0
for path, scenario in iteritems(scenarios):
    with open(os.path.join(SCENARIO_PATH, path), 'r') as f:
        details = None
        try:
            details = yaml.safe_load(f)
        except Exception:
            details = None
            scenario['status'] = 'Invalid YAML'
        if details is not None:
            try:
                runtime = details['setup']['runtime']
                scenario['status'] = 'Ok'
                scenario['runtime'] = runtime
                if scenario['results'] is None:
                    total_runtime += runtime
            except:
                scenario['status'] = 'No runtime info'

# Display list of scenario to be run
invalid_scenarios = {k:v for k,v in iteritems(scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'RUNTIME',  'STATUS', 'COMPLETION', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row([v['path'], str(v['runtime']) + 's', v['status'], v['results_date'], v['results']])

to_run = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'RUNTIME', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['runtime']) + 's', v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print('TOTAL RUNTIME: {} ({}s)'.format(datetime.timedelta(seconds=total_runtime), total_runtime))
print

# Ask if the user wants, because time will be long
answer = yes_or_no("The total runtime is {}. Are you sure you want to run all scenarios?".format(
    datetime.timedelta(seconds=total_runtime))  )
print

if answer:
    with tqdm(total=total_runtime) as pbar:  
        for info in to_run.values():
            base_scenario = info['path'].split('.yaml')[0]
            pbar.set_description("Running scenario {}\n\r".format(info['path']))
            cmd = 'python ./main.py -s {} -c control.seed={}'.format(
                os.path.join(SCENARIO_PATH, info['path']), 
                GLOBAL_SEED)
            with open(os.path.join(RESULT_PATH, '{}_stdout.txt'.format(base_scenario)), "a") as log_out:
                with open(os.path.join(RESULT_PATH, '{}_stderr.txt'.format(base_scenario)), "a") as log_err:
                    subprocess.call(cmd, shell=True, stdout=log_out, stderr=log_err)
            pbar.update(info['runtime'])