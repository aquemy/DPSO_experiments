from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
#from sklearn.decomposition import TruncatedSVD as PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer

from utils import generate_domain_space

PROTOTYPE = {
    "rebalance": [None, NearMiss(), CondensedNearestNeighbour(), SMOTE()],
    "normalizer": [None, StandardScaler(), PowerTransformer(), MinMaxScaler(), RobustScaler()],
    "features": [None, PCA(), SelectKBest(), FeatureUnion([("pca", PCA()), ("selectkbest", SelectKBest())])]
}

DOMAIN_SPACE = generate_domain_space(PROTOTYPE)

def get_baseline():
    baseline = {}
    for k in PROTOTYPE.keys():
        baseline[k] = ('{}_NoneType'.format(k), {})
    return baseline

def pipeline_conf_to_full_pipeline(args, algorithm, seed, algo_config):
        if args == {}:
            args = get_baseline()
        op_to_class = {'pca': PCA, 'selectkbest': SelectKBest}
        parts = ['rebalance', 'normalizer', 'features']
        operators = []
        for part in parts:
            item = args[part]
            if 'NoneType' in item[0]:
                continue
            else:
                params =  {k.split('__', 1)[-1]:v for k,v in item[1].iteritems()}
                if item[0] == 'features_FeatureUnion':
                    fparams = {'pca':{}, 'selectkbest':{}}
                    for p,v in params.iteritems():
                        op = p.split('__')[0]
                        pa = p.split('__')[1]
                        if op not in fparams:
                            fparams[op] = {}
                        fparams[op][pa] = v 
                    oparams = []
                    for p,v in fparams.iteritems():
                        oparams.append((p, op_to_class[p](**v)))
                    operator = FeatureUnion(oparams)
                    operators.append((part, operator))
                else:
                    operator = globals()[item[0].split('_',1)[-1]](**params)
                    operators.append((part, operator))

        clf = algorithm(random_state=seed, **algo_config)
        return Pipeline(operators + [("classifier", clf)]), operators