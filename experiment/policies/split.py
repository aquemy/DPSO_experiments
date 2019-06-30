from experiment.policies.policy import Policy
from experiment.pipeline.prototype import DOMAIN_SPACE as PIPELINE_SPACE
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_pipeline, objective_algo

import functools

from hyperopt import tpe, fmin, Trials


class Split(Policy):

    def run(self, X, y):
        super(Split, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials_pipelines = Trials()
        trials_algo = Trials()
        if self.config['step_pipeline'] > 0:
            print('## Data Pipeline')
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
            super(Split, self).display_step_results(best_config)
            
        if self.config['time'] - self.config['step_pipeline'] > 0:
            print('## Algorithm')
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
                max_time=self.config['time'] - self.config['step_pipeline'],
                trials=trials_algo,
                show_progressbar=False,
                verbose=0
            )

        best_config = self.context['best_config']
        super(Split, self).display_step_results(best_config)
        current_pipeline_configuration = best_config['pipeline']