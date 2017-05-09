# minfs
Some simple python tools for solving the Minimum-Feature-Set problem.

Exact solvers using: 
 - CPLEX (best performance)
 - localsolver (horrible performance)
 - numberjack (quite poor)
 - Googles ortools (good performance but slower than with CPLEX)

Heuristic solvers:
 - Greedy (Fast but often suboptimal)
 - Meta-RaPS (Slower but scales much better than the exact solvers and with better results than the greedy heuristic)
 
The Meta-RaPS heuristic reduces the problem to the equivalent Set Cover Problem and solves with [this](http://www.sciencedirect.com/science/article/pii/S0377221705008313) method.
