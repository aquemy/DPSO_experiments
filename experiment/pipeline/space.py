
def params_NearMiss():
    return {
        'n_neighbors': [1,2,3]
    }

def params_CondensedNearestNeighbour():
    return {
        'n_neighbors': [1,2,3]
    }

def params_SMOTE():
    return {
        'k_neighbors': [5,6,7]
    }

def params_StandardScaler():
    return {
        'with_mean': [True, False],
        'with_std': [True, False]
    }

def params_RobustScaler():
    return {
        'quantile_range':[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)],
        'with_centering': [True, False],
        'with_scaling': [True, False]
    }

def params_PCA():
    return {
        'n_components':[1, 2, 3, 4],
    }

def params_TruncatedSVD():
    return {
        'n_components':[1, 2, 3, 4],
    }

def params_SelectKBest():
    return {
        'k':[1, 2, 3, 4],
    }

def params_FeatureUnion():
    return {
        'pca__n_components':[1, 2, 3, 4],
        'selectkbest__k':[1, 2, 3, 4]
    }
