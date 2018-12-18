import random
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.stats import entropy
# from minfs.fs_solver_numberjack as fss
# import minfs.fs_solver_ortools as fss
import minfs.fs_solver_cplex as cpx
# import minfs.fs_solver_localsolver as lcs
import minfs.fs_solver_raps as raps
import minfs.fs_solver_greedy as greedy


def diversity(patterns):
    total = 0
    for p0, p1 in combinations(patterns, 2):
        total += np.sum(x != y for x, y in zip(p0, p1))
    return total


def hypercube_entropy(X):
    counts = defaultdict(int)
    for pattern in X:
        counts[tuple(pattern)] += 1
    return entropy(list(counts.values()), base=2)


def unique_pattern_count(X):
    counts = defaultdict(int)
    for pattern in X:
        counts[tuple(pattern)] = 1
    return len(counts)


def missing_pattern_count(X):
    return 2**X.shape[1] - unique_pattern_count(X)


def binary_entropy(v):
    p = sum(v) / len(v)
    if p == 0 or p == 1:
        return 0.0
    else:
        return - p*np.log2(p) - (1-p)*np.log2(1-p)


def info_gain(x, y):
    ent = binary_entropy(y)
    y0 = y[x == 0]
    y1 = y[x == 1]
    ent0 = binary_entropy(y0)
    ent1 = binary_entropy(y1)
    return ent - len(y0)/len(y)*ent0 - len(y1)/len(y)*ent1


def information_gain_score(X, y):
    return sum(info_gain(x, y) for x in X.T) / X.shape[1]


def is_valid(X, y):
    X0 = X[y == 0, :]
    X1 = X[y == 1, :]
    X0 = X0.view([('', X0.dtype)] * X0.shape[1])
    X1 = X1.view([('', X1.dtype)] * X1.shape[1])
    return 0 == len(np.intersect1d(X0, X1))


def best_feature_set(features, target, metric='cardinality>first',
                     solver='cplex', solver_params={}, prior_soln=None):
    ''' Takes a featureset matrix and target vector and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
    target      - <1D numpy array> of the same number of rows as features
    metric      - <string> which metric to use to pick best feature set.
    returns     - <1D numpy array> feature indices representing best FS
                  according to given metric.'''
    if solver == 'cplex':
        fss = cpx
    # elif solver == 'localsolver':
    #     fss = lcs
    elif solver == 'raps':
        fss = raps
    elif solver == 'greedy':
        fss = greedy
    else:
        raise ValueError('Invalid solver: {}'.format(solver))

    if metric == 'cardinality>first':
        try:
            fs = fss.single_minfs(features, target, prior_soln=prior_soln,
                                  **solver_params)
        except Exception as e:
            print(e)
            raise e
        return fs, 0, 1
    else:
        feature_sets = fss.all_minfs(features, target, prior_soln=prior_soln,
                                     **solver_params)
        if len(feature_sets) == 0:
            # No feature sets found - likely due to constant target
            return [], None, 1

        elif metric == 'cardinality>random':
            rand_index = random.randrange(len(feature_sets))
            return feature_sets[rand_index], 0, len(feature_sets)

        elif metric == 'cardinality>hypercube_entropy':
            entropies = [hypercube_entropy(features[:, fs])
                         for fs in feature_sets]
            best_fs = np.argmax(entropies)
            return feature_sets[best_fs], entropies[best_fs], len(feature_sets)

        elif metric == 'cardinality>feature_diversity':
            # feature diversity can be found by pattern diversity of X.transp
            scores = [diversity(features[:, fs].T) for fs in feature_sets]
            best_fs = np.argmax(scores)
            return feature_sets[best_fs], scores[best_fs], len(feature_sets)

        elif metric == 'cardinality>pattern_diversity':
            scores = [diversity(features[:, fs]) for fs in feature_sets]
            best_fs = np.argmax(scores)
            return feature_sets[best_fs], scores[best_fs], len(feature_sets)

        elif metric == 'cardinality>silhouette':
            scores = [metric.silhouette_score(features[:, fs], target)
                      for fs in feature_sets]
            best_fs = np.argmax(scores)
            return feature_sets[best_fs], scores[best_fs], len(feature_sets)

        elif metric == 'cardinality>information_gain':
            scores = [information_gain_score(features[:, fs], fs)
                      for fs in feature_sets]
            best_fs = np.argmax(scores)
            return feature_sets[best_fs], scores[best_fs], len(feature_sets)

        elif metric == 'cardinality>missing_patterns':
            scores = [missing_pattern_count(features[:, fs], fs)
                      for fs in feature_sets]
            best_fs = np.argmin(scores)
            return feature_sets[best_fs], scores[best_fs], len(feature_sets)

        else:
            raise ValueError('Invalid fs selection metric : {}'.format(
                metric))


def ranked_feature_sets(features, targets, metric='cardinality>first',
                        solver='cplex', solver_params={}, prior_solns=None):
    ''' Takes a featureset matrix and target matrix and finds a minimum FS.
    features    - <2D numpy array> in example x feature format.
                - OR a iterable of such arrays of length = |targets|
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

    if isinstance(features, np.ndarray):
        # generator to give the same feature matrix for all targets
        X = (features for i in range(Nt))
    else:
        X = features
        assert len(X) == Nt

    for i, x in enumerate(X):
        prior = prior_solns[i] if prior_solns is not None else None
        fs, score, _ = best_feature_set(x, targets[:, i], metric, solver,
                                        solver_params, prior)
        feature_sets[i] = fs
        cardinalities[i] = len(fs)
        secondary_scores[i] = score

    # sort by first minimising cardinality and then maximising score
    # (keys are reversed in numpy lexsort)
    order = np.lexsort((-secondary_scores, cardinalities))

    rank = order_to_ranking_with_ties(
        order, lambda i1, i2: (cardinalities[i1] == cardinalities[i2] and
                               secondary_scores[i1] == secondary_scores[i2]))

    return rank, feature_sets, secondary_scores


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


def inverse_permutation(permutation):
    inverse = np.zeros_like(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


def rank_with_ties_broken(ranking_with_ties):
    new_ranking = np.zeros_like(ranking_with_ties)
    ranks, counts = np.unique(ranking_with_ties, return_counts=True)
    for rank, count in zip(ranks, counts):
        indices = np.where(ranking_with_ties == rank)
        perm = list(range(rank, rank+count))
        random.shuffle(perm)
        new_ranking[indices] = perm
    return new_ranking


def order_from_rank(ranking_with_ties):
    ''' Converts a ranking with ties into an ordering,
        breaking ties with uniform probability.'''
    ranking_without_ties = rank_with_ties_broken(ranking_with_ties)
    # orders and rankings are inverse when no ties are present
    return inverse_permutation(ranking_without_ties)
