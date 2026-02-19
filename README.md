# Accuracy Assessment Tools

The simplest way to run this is to open the Jupyter Notebook
`AccuracyAssessmentTools-hfff.ipynb` in Google Colab and follow the instructions
in the notebook.

Open directly in Colab:
https://colab.research.google.com/github/alemlakes/accuracyAssessmentTools-hfff/blob/main/AccuracyAssessmentTools-hfff.ipynb

Requires pandas and numpy.

Input data is expected to be in a table with a column containing the mapped
values for each point and a column containing the reference value for each
point. Each row of the table should be a separate point.

Additional data such as mapped proportions or strata proportions should be
given as a dictionary whose keys match the labels used in the columns e.g.
if your class labels are 0, 1, 2, ... then the keys of the given dictionary
should also be 0, 1, 2, ...

Running each file will print test values from the corresponding paper to
verify that the math is being done properly, e.g.

`python naive_acc_assessment.py`

## Usage Example

After running `pip install . ` from the root directory:

```python
import pandas as pd
from acc_assessment.olofsson import Olofsson

data = pd.read_csv("/path/to/file/containing/assessment/points.csv")

mapped_areas = {"forest": 200000, "deforestation": 1000}

assessment = Olofsson(
    data, "name of map value col", "name of ref value col",
    mapped_areas)

print(assessment.overall_accuracy())
print(assessment.users_accuracy("forest"))
```

## GUE Usage Example

`GUE` expects map and reference inputs as class-probability tables (one row per
sample point), plus a strata column and a shared point id column.

If probabilities are crisp, `GUE` returns the same accuracy and area estimates
as `Stehman`.

```python
import pandas as pd
from acc_assessment import GUE

map_data = pd.DataFrame(
    {
        "forest": [0.9, 0.2, 0.6],
        "water": [0.1, 0.8, 0.4],
        "stratum": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

ref_data = pd.DataFrame(
    {
        "forest": [1.0, 0.3, 0.5],
        "water": [0.0, 0.7, 0.5],
        "stratum": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

strata_population = {1: 5000, 2: 3000}

assessment = GUE(
    map_data,
    ref_data,
    strata_col="stratum",
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
        "stratum": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

ref_data = pd.DataFrame(
    {
        "forest": [1.0, 0.3, 0.5],
        "water": [0.0, 0.7, 0.5],
        "stratum": [1, 1, 2],
        "id": [0, 1, 2],
    }
)

strata_population = {1: 5000, 2: 3000}

assessment = MCEM(
    map_data,
    ref_data,
    strata_col="stratum",
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
