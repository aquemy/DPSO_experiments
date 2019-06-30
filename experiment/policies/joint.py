from experiment.policies.policy import Policy
from experiment.pipeline.prototype import DOMAIN_SPACE as PIPELINE_SPACE
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_joint

import functools

from hyperopt import tpe, fmin, Trials
import hyperopt.pyll.stochastic

class Joint(Policy):

    def run(self, X, y):
        super(Joint, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials = Trials()
        algorithm = self.config['algorithm']
        space = {
            'pipeline': PIPELINE_SPACE,
            'algorithm': ALGORITHM_SPACE.get_domain_space(algorithm),
        }
        obj_pl = functools.partial(objective_joint,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        fmin(
            fn=obj_pl, 
            space=space, 
            algo=tpe.suggest, 
            max_evals=None,
            max_time=self.config['time'],     
            trials=trials,
            show_progressbar=False,
            verbose=0
        )

        best_config = self.context['best_config']
        super(Joint, self).display_step_results(best_config)
        current_pipeline_configuration = best_config['pipeline']