import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.stats import entropy
# from minfs.fs_solver_numberjack as fss
# import minfs.fs_solver_ortools as fss
import minfs.fs_solver_cplex as cpx
import minfs.fs_solver_localsolver as lcs
import minfs.fs_solver_raps as raps
import minfs.fs_solver_greedy as greedy


def diversity(patterns):
    total = 0
    for p0, p1 in combinations(patterns, 2):
        total += np.sum(x != y for x, y in zip(p0, p1))
    return total


def feature_diversity(all_features, fs_indices):
    return diversity(all_features[:, fs_indices].T)


def pattern_diversity(all_features, fs_indices):
    return diversity(all_features[:, fs_indices])


def feature_set_entropy(all_features, fs_indices):
    fs = all_features[:, fs_indices]
    counts = defaultdict(int)
    for pattern in fs:
        counts[tuple(pattern)] += 1
    return entropy(list(counts.values()), base=2)


def unique_pattern_count(all_features, fs_indices):
    fs = all_features[:, fs_indices]
    counts = defaultdict(int)
    for pattern in fs:
        counts[tuple(pattern)] = 1
    return len(counts)


def best_feature_set(features, target, metric='cardinality>first',
                     solver='cplex', params={}):
    ''' Takes a featureset matrix and target vector and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
    target      - <1D numpy array> of the same number of rows as features
    metric      - <string> which metric to use to pick best feature set.
    returns     - <1D numpy array> feature indices representing best FS
                  according to given metric.'''
    if solver == 'cplex':
        fss = cpx
    elif solver == 'localsolver':
        fss = lcs
    elif solver == 'raps':
        fss = raps
    elif solver == 'greedy':
        fss = greedy
    else:
        raise ValueError('Invalid solver: {}'.format(solver))

    if metric == 'cardinality>first':
        try:
            fs = fss.single_minimum_feature_set(features, target, **params)
        except Exception as e:
            print(e)
            raise e
        return fs, 0
    else:
        feature_sets = fss.all_minimum_feature_sets(features, target, **params)
        if len(feature_sets) == 0:
            # No feature sets found - likely due to constant target
            return [], None
        elif metric == 'cardinality>random':
            rand_index = np.random.randint(len(feature_sets))
            return feature_sets[rand_index], 0
        elif metric == 'cardinality>entropy':
            entropies = [feature_set_entropy(features, fs)
                         for fs in feature_sets]
            best_fs = np.argmax(entropies)
            return feature_sets[best_fs], entropies[best_fs]
        elif metric == 'cardinality>feature_diversity':
            feature_diversities = [feature_diversity(features, fs)
                                   for fs in feature_sets]
            best_fs = np.argmax(feature_diversities)
            return feature_sets[best_fs], feature_diversities[best_fs]
        elif metric == 'cardinality>pattern_diversity':
            pattern_diversities = [pattern_diversity(features, fs)
                                   for fs in feature_sets]
            best_fs = np.argmax(pattern_diversities)
            return feature_sets[best_fs], pattern_diversities[best_fs]
        else:
            raise ValueError('Invalid fs selection metric : {}'.format(
                metric))


def ranked_feature_sets(features, targets, metric='cardinality>first',
                        solver='cplex', params={}, prior_solns=None):
    ''' Takes a featureset matrix and target matrix and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
    targets     - <2D numpy array> in example x feature format.
    metric      - <string> which metric to use to pick best feature set.
    returns     - (<permutation>, <list of 1D numpy arrays>)
                    * ranking of targets
                    * feature indices representing best FS for each target
                      according to given metric.'''

    Nt = targets.shape[1]
    feature_sets = np.empty(Nt, dtype=list)
    cardinalities = np.zeros(Nt)
    secondary_scores = np.zeros(Nt)

    for i in range(Nt):
        if prior_solns is not None:
            params['prior_soln'] = prior_solns[i]
        fs, score = best_feature_set(features, targets[:, i], metric,
                                     solver, params)
        feature_sets[i] = fs
        cardinalities[i] = len(fs)
        secondary_scores[i] = score

    # sort by first minimising cardinality and then maximising score
    # (keys are reversed in numpy lexsort)
    order = np.lexsort((-secondary_scores, cardinalities))

    rank = order_to_ranking_with_ties(
        order, lambda i1, i2: (cardinalities[i1] == cardinalities[i2] and
                               secondary_scores[i1] == secondary_scores[i2]))

    return rank, feature_sets


def order_to_ranking_with_ties(order, tied):
    # build ranking from the above order (with ties)
    ranking = np.zeros(len(order), dtype=int)

    # check ranking from 2nd element in order
    # first element is rank 0 automatically by np.zeros
    for i in range(1, len(order)):
        i1 = order[i-1]
        i2 = order[i]
        if tied(i1, i2):
            # tie - continuing using current ranking
            ranking[i2] = ranking[i1]
        else:
            ranking[i2] = i
    return ranking
