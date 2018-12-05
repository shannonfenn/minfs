import numpy as np
from pytest import fixture


@fixture(params=[1, 2, 3, 4, 5])
def instance(request):
    filename_base = 'minfs/test/instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        feature_sets = data['feature_sets']
    return features, target, feature_sets
