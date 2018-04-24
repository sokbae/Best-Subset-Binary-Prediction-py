# Best-Subset-Binary-Prediction

Python package implementation of the best subset maximum score binary prediction method proposed by Chen and Lee (2017). Description of this prediction method and its computation details can be found in the paper:

Chen, Le-Yu and Lee, Sokbae (November 2017), "Best Subset Binary Prediction". The latest version of this paper can be found in this repository.

# Main functions
- max_score_constr_fn: used to compute the the best subset maximum score prediction rule via the mixed integer optimization (MIO) approach.
- warm_start_max_score: implements warm-start strategy by refining the input parameter space to improve the MIO computational performance.

# Examples
Included in the package are 2 examples from the original paper.
- Simulation
- Horowitz

## Requirements
Requires python 3 and the python Gurobi solver (available free for academic purposes).
