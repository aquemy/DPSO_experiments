from experiment.policies.policy import Policy
from experiment.pipeline.prototype import DOMAIN_SPACE as PIPELINE_SPACE
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_pipeline, objective_algo

import functools

from hyperopt import tpe, fmin, Trials


class Iterative(Policy):

    def run(self, X, y):
        super(Iterative, self).run(X, y)
        ranges = [i for i in xrange(0, self.config['time'] / 
            (self.config['step_pipeline']+self.config['step_algorithm']))]
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials_pipelines = Trials()
        trials_algo = Trials()
        reset_trial = False
        for r in ranges:
            print('## Data Pipeline')
            if reset_trial:
                trials_pipelines = Trials()
            obj_pl = functools.partial(objective_pipeline,
                    current_algo_config=current_algo_configuration,
                    algorithm=self.config['algorithm'],
                    X=X,
                    y=y,
                    context=self.context,
                    config=self.config)
            fmin(
                fn=obj_pl, 
                space=PIPELINE_SPACE, 
                algo=tpe.suggest, 
                max_evals=None,
                max_time=self.config['step_pipeline'],     
                trials=trials_pipelines,
                show_progressbar=False,
                verbose=0
            )
            best_config = self.context['best_config']
            current_pipeline_configuration = best_config['pipeline']
            super(Iterative, self).display_step_results(best_config)

            print('## Algorithm')
            if reset_trial:
                trials_algo = Trials()
            obj_algo = functools.partial(objective_algo, 
                    current_pipeline_config=current_pipeline_configuration,
                    algorithm=self.config['algorithm'],
                    X=X,
                    y=y,
                    context=self.context,
                    config=self.config)
            fmin(fn=obj_algo, 
                space=ALGORITHM_SPACE.get_domain_space(self.config['algorithm']), 
                algo=tpe.suggest, 
                max_evals=None,
                max_time=self.config['step_algorithm'],
                trials=trials_algo,
                show_progressbar=False,
                verbose=0
            )
            best_config = self.context['best_config']
            current_pipeline_configuration = best_config['pipeline']
            super(Iterative, self).display_step_results(best_config)