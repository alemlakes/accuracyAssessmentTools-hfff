# Accuracy Assessment Tools

The simplest way to run this is to open the Jupyter Notebook
`AccuracyAssessmentTools-hfff.ipynb` in Google Colab and follow the instructions
in the notebook.

Open directly in Colab:
https://colab.research.google.com/github/alemlakes/accuracyAssessmentTools-hfff/blob/main/AccuracyAssessmentTools-hfff.ipynb

Requires pandas and numpy.

## Purpose

This README mirrors the notebook workflow for running map-accuracy
assessments with the `acc_assessment` package across probabilistic and crisp
methods.

## How to use

1. Install the package from the repo root.
2. Run the integrated four-method workflow below for an end-to-end example.
3. Then run method-specific examples as needed.

## Expected inputs (integrated example)

For the main integrated workflow, inputs are:

1. A map probability table with one row per sampled point, containing:
    - `id` (point identifier),
    - `strata` (sampling stratum),
    - one column per map class with probabilities that sum to 1 per row.
2. A reference probability table with matching `id` rows and the same class
    probability columns.
3. A strata population table with columns `strata` and `population`.

The integrated workflow runs GUE and MCEM directly on the probabilistic
tables, then hardens them with `argmax` to run Stehman and Olofsson on the
same shared assumptions.

For crisp-only workflows, provide a table with map and reference class labels
and a class/strata population dictionary whose keys match those labels.

To verify the package is working in your environment, run:

`pytest`

## Integrated Four-Method Example (Shared Data)

This mirrors the main end-to-end example in
`AccuracyAssessmentTools-hfff.ipynb`: use one shared probabilistic dataset,
run **GUE** and **MCEM** directly, then harden to crisp labels and run
**Stehman** and **Olofsson** for direct comparison.

Workflow:
1. Set three input file paths (map probabilities, reference probabilities, and strata populations).
2. Load the three files and derive shared inputs.
3. Run GUE (analytical) and MCEM (simulation) on probabilistic inputs.
4. Convert probabilities to crisp classes using `argmax`.
5. Run Stehman and Olofsson on the hardened table.

After running `pip install .` from the repo root:

```python
import pandas as pd
from acc_assessment.gue import GUE
from acc_assessment.mcem import MCEM
from acc_assessment.stehman import Stehman
from acc_assessment.olofsson import Olofsson

# 1) Point to the same three input files used in the notebook example
map_file_same = "./tests/map_data_table.csv"
ref_file_same = "./tests/ref_data_table.csv"
strata_file_same = "./tests/strata_population_table.csv"

# 2) Load shared probabilistic inputs
prob_map_same = pd.read_csv(map_file_same)
prob_ref_same = pd.read_csv(ref_file_same)
strata_population_df = pd.read_csv(strata_file_same)

strata_population_same = dict(
    zip(strata_population_df["strata"], strata_population_df["population"])
)
class_cols_same = [c for c in prob_map_same.columns if c not in ["strata", "id"]]
target_class_same = class_cols_same[0]
N_same = sum(strata_population_same.values())

# 3) Probabilistic methods on shared inputs
gue_same = GUE(
    map_data=prob_map_same,
    ref_data=prob_ref_same,
    strata_col="strata",
    id_col="id",
    strata_population=strata_population_same,
)

mcem_same = MCEM(
    map_data=prob_map_same,
    ref_data=prob_ref_same,
    strata_col="strata",
    id_col="id",
    strata_population=strata_population_same,
    n_simulations=3000,
    random_state=42,
)

print("=== GUE (analytical) ===")
print("Overall Accuracy:", gue_same.overall_accuracy())
print(
    f"Reference Area ({target_class_same}):",
    gue_same.area(target_class_same, reference=True),
)

print("\n=== MCEM (simulation) ===")
print("Overall Accuracy:", mcem_same.overall_accuracy())
print(
    f"Reference Area ({target_class_same}):",
    mcem_same.area(target_class_same, reference=True),
)

# 4) Harden probabilities to crisp labels (argmax), then run crisp methods
map_crisp_labels = prob_map_same[class_cols_same].idxmax(axis=1)
ref_crisp_labels = prob_ref_same[class_cols_same].idxmax(axis=1)

crisp_same = pd.DataFrame(
    {
        "id": prob_map_same["id"],
        "Map class": map_crisp_labels,
        "Reference class": ref_crisp_labels,
    }
)

mapped_props_same = crisp_same["Map class"].value_counts(normalize=True)
mapped_population_same = {k: float(v * N_same) for k, v in mapped_props_same.items()}

crisp_same_stehman = crisp_same.copy()
crisp_same_stehman["strata"] = crisp_same_stehman["Map class"]

stehman_same = Stehman(
    data=crisp_same_stehman,
    strata_col="strata",
    map_col="Map class",
    ref_col="Reference class",
    strata_population=mapped_population_same,
)

olofsson_same = Olofsson(
    crisp_same,
    mapped_population_same,
    map_col="Map class",
    ref_col="Reference class",
)

print("\n=== Stehman (crisp) ===")
print("Overall Accuracy:", stehman_same.overall_accuracy())
print(
    f"Reference Area ({target_class_same}):",
    stehman_same.area(target_class_same, reference=True),
)

print("\n=== Olofsson (crisp) ===")
print("Overall Accuracy:", olofsson_same.overall_accuracy())
print(f"Reference Area ({target_class_same}):", olofsson_same.area(target_class_same))
```

## GUE Usage Example

`GUE` expects map and reference inputs as class-probability tables (one row per
sample point), plus a stratum column and a shared point id column.

If probabilities are crisp, `GUE` returns the same accuracy and area estimates
as `Stehman`.

```python
import pandas as pd
from acc_assessment import GUE

map_data = pd.DataFrame(
    {
        "forest": [0.9, 0.2, 0.6],
        "water": [0.1, 0.8, 0.4],
        "strata": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

ref_data = pd.DataFrame(
    {
        "forest": [1.0, 0.3, 0.5],
        "water": [0.0, 0.7, 0.5],
        "strata": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

strata_population = {1: 5000, 2: 3000}

assessment = GUE(
    map_data,
    ref_data,
    strata_col="strata",
    id_col="id",
    strata_population=strata_population,
)

print(assessment.overall_accuracy())
print(assessment.error_matrix())
```

## MCEM Usage Example

`MCEM` runs Monte Carlo simulations by sampling crisp classes from map and
reference probabilities at each point, then reports the mean estimate and a
95% percentile interval.

This simulation distribution captures both data uncertainty and sampling
uncertainty in a single result.

```python
import pandas as pd
from acc_assessment import MCEM

map_data = pd.DataFrame(
    {
        "forest": [0.9, 0.2, 0.6],
        "water": [0.1, 0.8, 0.4],
        "strata": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

ref_data = pd.DataFrame(
    {
        "forest": [1.0, 0.3, 0.5],
        "water": [0.0, 0.7, 0.5],
        "strata": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

strata_population = {1: 5000, 2: 3000}

assessment = MCEM(
    map_data,
    ref_data,
    strata_col="strata",
    id_col="id",
    strata_population=strata_population,
    n_simulations=10000,
)

overall_mean, overall_ci = assessment.overall_accuracy()
print(overall_mean, overall_ci)
print(assessment.users_accuracy("forest"))
print(assessment.producers_accuracy("forest"))
print(assessment.area("forest"))
```
