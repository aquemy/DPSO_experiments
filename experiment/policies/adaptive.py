from experiment.policies.policy import Policy
from experiment.pipeline.prototype import DOMAIN_SPACE as PIPELINE_SPACE
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_pipeline, objective_algo

import functools
import time
import json

from hyperopt import tpe, fmin, Trials


class Adaptive(Policy):

    def __init__(self, config):
        super(Adaptive, self).__init__(config)
        self.history_steps = []
        self.history_steps_score = []
        self.current_steptime = {
            'algorithm': self.config['initial_step_time'],
            'pipeline': self.config['initial_step_time']
        }
        self.reset_trials_next_time = {
            'algorithm': False,
            'pipeline': False
        }

    def __determine_last_step(self, from_date):
        step_res = [p for p in self.context['history'] if p['start_time'] > from_date]
        index_best_earlier_config = self.context['history_index'][step_res[0]['config_hash']['config']] - 1
        best_earlier_config = self.context['history'][index_best_earlier_config] if index_best_earlier_config > 0 else None
        self.history_steps.append((step_res, best_earlier_config))

    def __score_last_step(self):
        total_pos_variation = 0.
        best_earlier_score = self.history_steps[-1][1]['max_history_score'] if self.history_steps[-1][1] is not None else self.context['baseline_score']
        for c in self.history_steps[-1][0]:
            if c['score'] > best_earlier_score:
                total_pos_variation += c['score'] - best_earlier_score
        self.history_steps_score.append(total_pos_variation)
        return total_pos_variation

    def __update_timestep(self, step_type):
        # Get the iterations from the last step
        if self.history_steps_score[-1] > 0:
            print('--> [POLICY ACTION] DOUBLE TIME FOR {} from {}s to {}s.'.format(
                step_type.upper(), 
                self.current_steptime[step_type], 
                self.current_steptime[step_type] * 2)
            )
            self.current_steptime[step_type] = 2 * self.current_steptime[step_type]
        else:
            l = self.config['reset_trials_after']
            '''
            if len(self.history_steps_score) >= l:
                stationary = all([s == 0. for s in self.history_steps_score[-l:]])
                if stationary:
                    print('--> [POLICY ACTION] RESET TRIALS FOR {}'.format(step_type.upper()))
                    self.reset_trials_next_time[step_type] = True
            '''
            if self.current_steptime[step_type] > self.config['initial_step_time']:
                print('--> [POLICY ACTION] DIVIDE TIME FOR {} from {}s to {}s.'.format(
                step_type.upper(), 
                self.current_steptime[step_type], 
                self.current_steptime[step_type] // 2)
                )
                self.current_steptime[step_type] = self.current_steptime[step_type] // 2

        return

    def __reset_trials(self, step_type):
        if self.config['reset_trial']:
            return True
        if self.reset_trials_next_time[step_type]:
            self.reset_trials_next_time[step_type] = False
            return True
        return False

    def run(self, X, y):
        super(Adaptive, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials_pipelines = Trials()
        trials_algo = Trials()
        timeout = False
        start = time.time()
        remaining = self.config['time']
        while not timeout:
            ellapsed = time.time() - start
            remaining = self.config['time'] - ellapsed
            timeout = remaining < 0
            print('## Data Pipeline')
            step_start = time.time()
            if self.__reset_trials('pipeline'):
                trials_pipelines = Trials()
            obj_pl = functools.partial(objective_pipeline,
                    current_algo_config=current_algo_configuration,
                    algorithm=self.config['algorithm'],
                    X=X,
                    y=y,
                    context=self.context,
                    config=self.config)
            try:
                fmin(
                    fn=obj_pl, 
                    space=PIPELINE_SPACE, 
                    algo=tpe.suggest, 
                    max_evals=None,
                    max_time=self.current_steptime['pipeline'],     
                    trials=trials_pipelines,
                    show_progressbar=False,
                    verbose=0
                )
            except Exception as e:
                pass
            best_config = self.context['best_config']
            current_pipeline_configuration = best_config['pipeline']
            super(Adaptive, self).display_step_results(best_config)
            self.__determine_last_step(step_start)
            self.__score_last_step()
            self.__update_timestep('pipeline')


            print('## Algorithm')
            step_start = time.time()
            if self.__reset_trials('algorithm'):
                trials_algo = Trials()
            obj_algo = functools.partial(objective_algo, 
                    current_pipeline_config=current_pipeline_configuration,
                    algorithm=self.config['algorithm'],
                    X=X,
                    y=y,
                    context=self.context,
                    config=self.config)
            try:
                fmin(fn=obj_algo, 
                    space=ALGORITHM_SPACE.get_domain_space(self.config['algorithm']), 
                    algo=tpe.suggest, 
                    max_evals=None,
                    max_time=self.current_steptime['algorithm'],
                    trials=trials_algo,
                    show_progressbar=False,
                    verbose=0
                )
            except Exception as e:
                pass
            best_config = self.context['best_config']
            current_pipeline_configuration = best_config['pipeline']
            super(Adaptive, self).display_step_results(best_config)
            self.__determine_last_step(step_start)
            self.__score_last_step()
            self.__update_timestep('algorithm')
        self.context['current_steptime'] = self.current_steptime
        self.context['history_steps_score'] = self.history_steps_score