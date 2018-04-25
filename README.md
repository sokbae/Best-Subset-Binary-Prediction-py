# Best-Subset-Binary-Prediction

Python package implementation of the best subset maximum score binary prediction method proposed by Chen and Lee (2017). Description of this prediction method and its computation details can be found in the paper:

Chen, Le-Yu and Lee, Sokbae (November 2017), ["Best Subset Binary Prediction"](https://arxiv.org/pdf/1610.02738.pdf).

## Installation
1. Install via pip:
```
pip install bsbp
```

2. Download package and navigate inside the root directory. Run the following code in terminal:
```
python setup.py
```
or
```
pip install -e .
```

## Main functions
- max_score_constr_fn:
  Used to compute the the best subset maximum score prediction rule via the mixed integer optimization (MIO) approach.
- warm_start_max_score:
  Implements warm-start strategy by refining the input parameter space to improve the MIO computational performance.
- cv_best_subset_maximum_score:
  Implements cross validation best subset binary prediction and computes the optimal q value.
## Examples
Included in the package are 2 examples from the original paper.

###Simulation
Implements best subset approach to simulated data.
```
from bsbp.simulation import simulation
simulation()
```
Command line:
```
python -m simulation *args
```
where `\*args` are optional arguments. Use option `-h` to list optional arguments.

###Horowitz
Implements best subset approach to Horowitz (1993) data.
```
from bsbp.transportation import transportation
transportation()
```
Implement best subset approach with cross validation to Horowitz (1993) data.
```
from bsbp.transportation import transportationcv
transportationcv()
```

Command line:
```
python -m transportation (or transportationcv) *args
```

where `\*args` are optional arguments. Use option `-h` to list optional arguments.


## Requirements
Requires python 3.6 and the python Gurobi solver (available free for academic purposes).
