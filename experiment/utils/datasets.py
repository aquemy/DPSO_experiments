from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_covtype
#import echr

def load(name):
    loader = {
        'breast': breast_cancer,
        'iris': iris,
        'wine': wine,
        'digits': digits,
        'covtype': covtype,
        #'echr_article_1': echr.binary.get_dataset(article='1', flavors=[echr.Flavor.desc]).load
    }
    if name in loader:
        return loader[name]()
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(loader.keys())
        ))
        exit(1)  # TODO: Throw exception

def breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target

def iris():
    data = load_iris()
    return data.data, data.target

def wine():
    data = load_wine()
    return data.data, data.target

def digits():
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def covtype():
    data = fetch_covtype
    return data.data, data.target