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
- Division of labor difficult (counterexample: develop treatment effect estimator and supply R package)

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

### pybaum

Specify your parameters as:

```python
start_params = {
    "preferences": {
        "leisure_weight": 0.9,
        "ces": 0.5
    },
    "work": {
        "hourly_wage": 25,
        "hours": 2_000
    },
    "time_budget": 24 * 7 * 365,
    "consumption_floor": 3_000,
}
```

### dags

```py
def utility(consumption, leisure, params):
    ɑ = params["preferences"]["leisure_weight"]
    ɣ = params["preferences"]["ces"]
    c = (1 - ɑ) ** (1 / ɣ) * consumption ** ((ɣ - 1) / ɣ)
    l = ɑ ** (1 / ɣ) * leisure ** ((ɣ - 1) / ɣ)
    return (c + l) ** (ɣ / (ɣ - 1))

def leisure(params):
    return params["time_budget"] - params["work"]["hours"]


def income(params):
    return params["work"]["hourly_wage"] * params["work"]["hours"]

def consumption(income, params):
    c_min = params["consumption_floor"]
    return income if income > c_min else c_min

def unrelated(working_hours):
    raise NotImplementedError()
```

### dags

```py
model = dags.concatenate_functions(
    functions=[utility, unrelated, leisure, consumption, income],
    targets=["utility", "consumption"],
    return_type="dict"
)
```

### (aside: estimagic)

See notebook

### LCM: Basic building blocks

1. pybaum — Flexible structuring of parameters, data
2. dags — has functions on individual states (all scalars)
   - Not shown: Partial parameters that do not change in estimation right in beginning
3. Dispatchers — Leverage Jax to vectorize scalar functions on almost arbitrary state-space

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

### Applied user

- Supplies economic primitives on per-state basis (i.e. everything scalar)
  - Utility functions
  - Constraints
  - Descriptions of states and choice options
  - State transitions based on states and choices (including filters that induce sparsity)

### LCM

- Builds state space representation
- Builds derived economic functions (e.g. value function, derivatives, ...) (still on scalars)
- Infers parameters (anything that is not state or choice), yields template
- Vectorises derived functions on state space
- Builds solution, simulation and likelihood functions

### Expert user

- Supplies same things as applied user
- Thanks to dags implementation, may implement custom functions for anything
  - State space representation
  - Derived economic functions
  - Vectorization operators directly

### State space representation

- dense variables / Cartesian grid
  - Memory efficient during calculation
  - Memory hungry when storing value / policy functions
  - Fast computations
- contingent variables / combined grid
  - Memory efficient when storing value / policy functions
  - Save computations

### Abstracting from type of variables / grid

- Dispatcher / gridmap decorator provides abstraction during computation
- LCM value / policy function provides abstraction during lookup (WIP, but basically solved)
- No performance penalty

### LCM benchmarks

- See notebook

### LCM: Roadmap

- Most building blocks are done & tested
- DC-EGM example with brute force should be working in September
- Implement DC-EGM by November (one continuous choice, arbitrary number of DC)
- Have frontier models running by Christmas

### Discussion

- Suggestions for development?
- Interested in using (parts of) ecosystem?
- Interested in helping develop (parts of) ecosystem?
