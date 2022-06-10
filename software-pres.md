---
date: Copenhagen, 13 June 2022
title: Software for estimating life-cycle models
author: Hans-Martin von Gaudecker
organization: Universität Bonn & IZA
---

### Ferrall’s (2020) Tale of Two Papers

- Thomas MaCurdy (1981). “An Empirical Model of Labor Supply in a Life-Cycle Setting,” Journal of Political Economy 89, 6, 1059-1085.
- Kenneth Wolpin (1984). “An Estimable Dynamic Stochastic Model of Fertility and Child Mortality,” Journal of Political Economy 92, 5, 852-874.

### Ferrall’s (2020) Tale of Two Papers

- Both papers required a huge amount of customised code in the early 1980s
- MaCurdy (1981) approximates a structural model so that it has closed form; nowadays Panel IV
- Wolpin (1984) uses full-blown structural estimation of discrete choice model; almost requires same amount of coding today

### Why is this a problem?

- Time to PhD / content thereof
- Fixed cost of entering the field later
- Software quality, code sharing culture
- Division of labor difficult (counterexample: develop treatmentment effect estimator and supply R package)

Σ Much slower progress than in other fields

### Libraries for solving / estimating structural models

- QuantEcon
- Heterogeneous Agents Resources and toolKit (HARK)
- consav
- niqlow
- respy

### Some observations

- Programmer / user distinction ?
- Exploitation of computational tricks ?
- How to make the most of different researcher profiles ?

### A new attempt

- People
- Prior code
- Building blocks
- Discussion

### OSE Group

- Janoś Gabler, Moritz Mendel, Tobias Raabe, Sebastian Gsell, Christian Zimpelmann, Annica Gehlen, Klara Röhrl, Tim Mensinger, Max Blesch
- Partnering with CS departments
- All open source, happy to involve others!

### Prior code: Learning experiences

- Code for French, von Gaudecker & Jones — Solving life-cycle models on GPUs and some negative examples
- skillmodels — Speeding up code with Numpy, Numba, and Jax; automatic differentiation
- respy — Highly optimized solution of Eckstein-Keane-Wolpin models
- econ-project-templates — Sensible ways to structure research projects and pipelines
- GETTSIM, pytask — DAGs and introspection
- SID — Structural CoViD-19 infection model, real-world & real-time test of the general approach

<!--
### But but but … Python is slow!

- General purpose programming language

- Not targeted at numerical computing

  - Pro: Use one language for many different things (data management in Fortran? Webscraping in Julia?)
  - Con: Some things are not what you expect (no log() function in main namespace)

- Is it very hard to write

  ```py
  from numba import jit

  @jit
  def nfpx(...):
  ...

  ```

  ?

### Speed of languages

- Native Python is slow for structural econometrics
- But with appropriate libraries, it is not: Jax, Numba, Numpy, …
- Unless you are an expert Fortran/C programmer, execution speed in Python or Julia will be faster (not talking about using modern hardware yet…)
- Full development cycle is much faster:
  - Prototyping
  - Testing
  - Optimizing
  - Adjusting for changes in model
-->

### OOP vs functional

- OOP: Hide state in objects grouping data and functions (methods)
- Functional: Eliminate state by relying on pure functions alone
  - No side effects
  - Only depend on inputs
  - Mostly atomic operations + JIT compilers
- OOP: Overriding things via subclassing always ends up a mess in complex projects
- Functional + DAG/introspection: Replacing functions by others with compatible interfaces is very natural

### Ecosystem for estimating structural life-cycle model

- Utilities used throughout: pybaum, dags
- Function optimisation, standard errors, sensitivity analysis — estimagic
- Depiction of Taxes & Transfers system — GETTSIM, OpenFisca
- Running tasks in a complex project — pytask
- Solution of dynamic programming problem — lcm

### Some helpers

- [pybaum](https://github.com/OpenSourceEconomics/pybaum): (Fully) flexible specifications of parameters

  - Modelled after Jax' pytrees
  - Very natural to express structure of parameters

- [dags](https://github.com/OpenSourceEconomics/dags): automate program execution

  - introspect of function arguments
  - build and execute DAG from that

- [pytask](https://github.com/pytask-dev/): dags on steroids for running pipeline
  - files & functions as primitives
  - allows mixing in R, Julia, Stata; compilation of LaTeX documents, ...

### dags

```py
def utility(consumption, leisure, leisure_weight):
    return consumption + leisure_weight * leisure


def leisure(working_hours):
    return 24 - working_hours


def consumption(working_hours, wage):
    return wage * working_hours


def unrelated(working_hours):
    raise NotImplementedError()
```

### dags

```py
model = dags.concatenate_functions(
    functions=[utility, unrelated, leisure, consumption],
    targets=["utility", "consumption"],
    return_type="dict"
)

model(wage=5, working_hours=8, leisure_weight=2)
{'utility': 72, 'consumption': 40}
```

### pybaum

Specify your parameters as:

```python
start_params = {
    "Type 0": {
        "β": 0.95,
        "ɣ": 2.0
    },
    "Type 1": {
        "β": 0.98,
        "ɣ": 2.0
    },
    "consumption_floor": 3_000,
}
```

### estimagic

- Collection of optimizers with
  - Unified and well designed interface
  - Flexible handling of constraints (fixing parameters, bounds, adding-up)
  - Real time dashboard
- Includes all optimisation algorithms from scipy
- Has specialized optimizers for nonlinear least squares (e.g., POUNDerS)
  (Parallelized) numerical differentiation
- Inference and sensitivity analysis for ML / GMM / MSM

### Start parameters

```py
start*params = pd.DataFrame(
data=np.arange(5) + 1, columns=["value"], index=[f"x*{i}" for i in range(5)]
)

    value

x_0     1
x_1     2
x_2     3
x_3     4
x_4     5
```

### Start parameters

- Can be DataFrame with any index
  “value” column is mandatory
  “lower_bound”, “upper_bound” are optional

### Minimise the sphere function

```py
res = minimize(
    criterion=sphere,
    params=start_params,
    algorithm="scipy_lbfgsb",
    logging="sphere_lbfgsb.db",
)

print(res)
```

### Result

```
{
'solution*x': array([
8.64900661e-07,
-4.24445584e-07,
-2.40790567e-07,
-2.27446962e-07,
2.74273971e-07
]),
'solution_criterion': 1.1131456360722564e-12,
'solution_derivative': array([
1.73129144e-06,
-8.47401052e-07,
-4.80091018e-07,
-4.53403807e-07,
5.50038057e-07
]),
'solution_hessian': None,
'n_criterion_evaluations': 3,
'n_derivative_evaluations': None,
'n_iterations': 2,
'success': True,
'reached_convergence_criterion': None,
'message': 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT*<=\_PGTOL',
'solution_params':
lower_bound upper_bound value
x_0 -inf inf 8.649007e-07
x_1 -inf inf -4.244456e-07
x_2 -inf inf -2.407906e-07
x_3 -inf inf -2.274470e-07
x_4 -inf inf 2.742740e-07
}
```

### Using the estimagic dashboard

- Track evolution of parameters and criterion
- Check evolution ex post
- Can do more advanced stuff

<!-- estimagic dashboard -->

### LCM: Why?

- Field matures; both models and solution methods become more and more complex:
  - More and more difficult to be an expert in both
  - More important to build on existing models instead of starting from scratch for each project
- HPC becomes more accessible through libraries
  - More flexibility
  - GPUs / TPUs better suited than large-scale MPI

### LCM: Target users / developers

- Developers of new algorithms (sensible benchmark free lunch, available for downstream users)
- Frontier researchers
- Economists in policy analysis

### LCM development: Guiding principles

- Ease of swapping different parts for custom implementations — Modularity
- Extensive unit testing
- Make use of tools actively developed in ML/AI community as much as possible
- Allow for different solutions with respect to the computation — memory usage trade-off out of the box

### LCM development: Early implementation choices

- Functional code organized by DAGs / introspection
  - Extremely flexible
  - No need to worry about execution order from user perspective
- Speeding up based on JAX, Numba
  - JAX great for computations on arrays, seamless choice of CPU/GPU/TPU
  - Numba great for filtering / refinement operations (=logical → CPU)
  - No need to take a stance on loop vs. vectorisation

### LCM development: Current state

- Efficient interpolation
- DAG (see test)
  - Define functions
  - Specify inputs
  - Specify outputs
  - Execution order automatically determined
- All functions in DAG can be
  - differentiated
  - inverted (experimental)
  - vectorised and evaluated efficiently on grids

### LCM development: Next steps

- Get prototype to run (did not decide on first state-space implementation yet)
- Add realistic models as testcases (niqlow has a great selection)
- Extend to frontier models as test cases

### Discussion

- Suggestions for development?
- What would be needed to make you use this ecosystem?
- What would make you contribute? What could that be?
