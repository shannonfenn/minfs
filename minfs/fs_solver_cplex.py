"""
  MIP for Boolean feature selection.

  Author: Shannon Fenn (shannon.fenn@gmail.com)
"""
import cplex
from cplex.exceptions import CplexSolverError
import numpy as np


def build_minfs_model(features, target, prior_soln, debug=False):
    model = cplex.Cplex()

    # prevent multithreading
    model.parameters.threads.set(1)

    if not debug:
        # stop cplex chatter
        model.set_results_stream(None)
        model.set_warning_stream(None)
        model.set_error_stream(None)
        model.set_log_stream(None)

    coverage = sparse_coverage(features, target)
    Np = len(coverage)
    Nf = features.shape[1]

    model.objective.set_sense(model.objective.sense.minimize)

    obj_coefficients = [1] * Nf
    lbounds = [0] * Nf
    ubounds = [1] * Nf
    types = 'I' * Nf
    # feature_ids = list(range(Nf))

    sense = 'G' * Np
    rhs = [1] * Np

    model.variables.add(obj=obj_coefficients, lb=lbounds, ub=ubounds, types=types)

    # rows = [[feature_ids, row] for row in coverage]
    rows = coverage

    model.linear_constraints.add(lin_expr=rows, senses=sense, rhs=rhs)

    if prior_soln is not None:
        x_ = np.zeros(Nf)
        x_[prior_soln] = 1
        model.MIP_starts.add([list(range(Nf)), x_.tolist()],
                             model.MIP_starts.effort_level.check_feasibility)

    return model


def single_minfs(features, target, prior_soln=None,
                 timelimit=None, debug=False):
    if np.all(target) or not np.any(target):
        # constant target - no solutions
        return []

    model = build_minfs_model(features, target, prior_soln, debug)

    if timelimit is not None:
        model.parameters.timelimit.set(timelimit)

    model.solve()

    status = model.solution.get_status()

    if status in [model.solution.status.MIP_optimal,
                  model.solution.status.MIP_time_limit_feasible]:
        x = model.solution.get_values()
        fs = sorted(np.where(x)[0])
    else:
        fs = []

    return fs


def all_minfs(features, target, intensity=2, prior_soln=None,
              timelimit=None, debug=False):
    if np.all(target) or not np.any(target):
        # constant target - no solutions
        return []

    model = build_minfs_model(features, target, prior_soln, debug)

    if timelimit is not None:
        model.parameters.timelimit.set(timelimit)

    # guide suggests this to get all solutions:
    # https://www.ibm.com/support/knowledgecenter/ru/SSSA5P_12.4.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/discr_optim/soln_pool/18_howTo.html
    model.parameters.mip.pool.absgap.set(0)
    model.parameters.mip.pool.intensity.set(0)
    model.parameters.mip.limits.populate.set(2100000000)

    try:
        model.populate_solution_pool()
    except CplexSolverError:
        print("Exception raised during populate")
        return

    # Print information about other solutions
    print()
    numsol = model.solution.pool.get_num()
    print("The solution pool contains %d solutions." % numsol)

    numsolreplaced = model.solution.pool.get_num_replaced()
    print("%d solutions were removed due to the solution pool "
          "relative gap parameter." % numsolreplaced)

    numsoltotal = numsol + numsolreplaced
    print("In total, %d solutions were generated." % numsoltotal)

    meanobjval = model.solution.pool.get_mean_objective_value()
    print("The average objective value of the solutions is %.10g." %
          meanobjval)

    status = model.solution.get_status()
    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", status, ":", end=' ')
    # the following line prints the corresponding string
    print(model.solution.status[status])

    if status not in [model.solution.status.optimal_populated,
                      model.solution.status.optimal_populated_tolerance,
                      model.solution.status.MIP_optimal,
                      model.solution.status.MIP_time_limit_feasible]:
        print('Warning (all_minimum_feature_sets): possible non-optimal solutions.')
    F = []
    for i in range(numsol):
        x = model.solution.pool.get_values(i)
        F.append(sorted(np.where(x)[0]))
    return F


# def all_minimum_feature_sets(features, target):
#     if np.all(target) or not np.any(target):
#         # constant target - no solutions
#         return []
#     coverage = build_coverage(features, target)
#     return all_minfs(coverage)


def coverage(features, target):
    Ne, Nf = features.shape
    class_0_indices = np.flatnonzero(target == 0)
    class_1_indices = np.flatnonzero(target)
    num_eg_pairs = class_0_indices.size * class_1_indices.size
    coverage_matrix = np.zeros((num_eg_pairs, Nf), dtype=float)
    i = 0
    for i0 in class_0_indices:
        for i1 in class_1_indices:
            for f in range(Nf):
                if features[i0, f] != features[i1, f]:
                    coverage_matrix[i, f] = 1
            i += 1
    return coverage_matrix


def sparse_coverage(features, target):
    Ne, Nf = features.shape
    class_0_indices = np.flatnonzero(target == 0)
    class_1_indices = np.flatnonzero(target)
    coverage = []
    for i0 in class_0_indices:
        for i1 in class_1_indices:
            ind = np.nonzero(features[i0, :] != features[i1, :])[0].tolist()
            val = [1]*len(ind)
            coverage.append(cplex.SparsePair(ind, val))
    return coverage





# def all_kfs(coverage, k):
#     Np, Nf = coverage.shape
#     solver = pywrapcp.Solver("all_kfs")

#     # decision variable
#     x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

#     # constraints
#     # all pairs must be covered by at least one feature
#     for p in range(Np):
#         b = [x[f] for f in range(Nf) if coverage[p][f]]
#         solver.Add(solver.SumGreaterOrEqual(b, 1))
#     # only k features may be selected
#     solver.Add(solver.SumEquality(x, k))

#     # solution and search
#     solution = solver.Assignment()
#     solution.Add(x)
#     # solution.AddObjective(k)

#     collector = solver.AllSolutionCollector(solution)
#     db = solver.Phase(x,
#                       solver.INT_VAR_DEFAULT,
#                       solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, collector)

#     if solution_found:
#         # collect all feature sets
#         numSol = collector.SolutionCount()
#         feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
#                         for i in range(numSol)]

#         return feature_sets
#     else:
#         return []


# def all_minfs(coverage):
#     Np, Nf = coverage.shape
#     solver = pywrapcp.Solver("all_minfs")

#     # decision variable
#     x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

#     # constraints
#     # all pairs must be covered by at least one feature
#     for p in range(Np):
#         b = [x[f] for f in range(Nf) if coverage[p][f]]
#         solver.Add(solver.SumGreaterOrEqual(b, 1))

#     # objective
#     # minimise number of features
#     k = solver.Sum(x)
#     objective = solver.Minimize(k, 1)

#     # find min k
#     solution = solver.Assignment()
#     solution.Add(x)
#     solution.AddObjective(k)

#     collector = solver.LastSolutionCollector(solution)
#     db = solver.Phase(x + [k], solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, [collector, objective])

#     if not solution_found:
#         return []

#     # find all kfs
#     # only k features may be selected
#     min_k = collector.ObjectiveValue(0)
#     solver.Add(solver.SumEquality(x, min_k))

#     # solution and search
#     collector = solver.AllSolutionCollector(solution)
#     db = solver.Phase(x, solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, collector)

#     if not solution_found:
#         return []

#     # collect all feature sets
#     numSol = collector.SolutionCount()
#     feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
#                     for i in range(numSol)]

#     return feature_sets
