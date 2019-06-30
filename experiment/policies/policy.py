from experiment.objective import get_baseline_score

import json

class Policy(object):
    def __init__(self, config):
        self.compute_baseline = True
        self.config = config
        self.context = {
            'iteration': 0,
            'history_hash': [],
            'history_index': {},
            'history': [],
            'max_history_score': 0.,
            'max_history_score_std': 0.,
            'max_history_step': 'baseline',
            'best_config': {},
        }

    def __compute_baseline(self, X, y):
        baseline_score, baseline_score_std = get_baseline_score(
            self.config['algorithm'], 
            X, 
            y, 
            self.config['seed'])
        self.context['baseline_score'] = baseline_score
        self.context['baseline_score_std'] = baseline_score_std

    def run(self, X, y):
        if self.compute_baseline:
            self.__compute_baseline(X, y)

    def display_step_results(self, best_config):
        print('{} STEP RESULT {}'.format('#' * 20, '#' * 20))
        print('BEST PIPELINE:\n {}'.format(json.dumps(best_config['pipeline'], indent=4, sort_keys=True),))
        print('BEST ALGO CONFIG:\n {}'.format(json.dumps(best_config['algorithm'], indent=4, sort_keys=True)))
        print('BEST SCORE: {} ({})'.format(best_config['score'], best_config['score_std']))
        print('#' * 50)
