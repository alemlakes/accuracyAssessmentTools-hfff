# Preparation Helper README

This guide documents the preparation helper in:

- `src/acc_assessment/standardizer.py`

It is useful when your input labels are not yet in the standard probabilistic
format expected by the integrated workflow and by `GUE`/`MCEM`.

Interactive notebook guide:

- `ProbabilityStandardizer-Guide.ipynb`

## What the helper does

The helper provides two core tools:

- `ProbStandardizer`: converts non-standard confidence formats into row-wise
  class probabilities.
- `verify_standard_structure`: checks whether class probability columns exist,
  are numeric, and sum to 1 per row within a tolerance.

## Standard target format

Output tables should contain:

- one column per class probability (e.g., `Forest`, `Water`, `Agriculture`),
- each row summing to `1.0` (within tolerance),
- optional metadata columns such as `id`.

For integrated map/reference workflows, keep class columns in the same order
across both tables (for example, do not use `A,B,C` in one file and `B,A,C`
in the other).

## Main API

### `ProbStandardizer(class_names, id_col="id", require_unique_id=False)`

Initialize with the class names used in your project.

When `require_unique_id=True`, conversion methods enforce that the `id` column
exists and has no duplicates.

## Which mode should I use?

Use this order when deciding mode:

1. **`from_crisp`**: you have exactly one class label per row.
2. **`from_confidence`**: you have one class label plus a numeric confidence value in `[0, 1]`.
3. **`from_binary_confidence`**: you have one class label plus a confidence flag.
4. **`from_likert`**: you have per-class rating/score columns.
5. **`from_votes`**: you have one row per point with per-class integer vote totals.
6. **`from_multi_interpreter_vectors`**: you have multiple rows per point, one probability vector per interpreter, and you want the per-point average vector.

## How each mode is converted (math)

### Crisp labels (`from_crisp`)

Each row has one class label. Output is binary confidence values of 1 and 0 across class columns:

- selected class = `1.0`
- all other classes = `0.0`

Example with class order `[A, B, C]` and label `C` gives `[0, 0, 1]`.

### Numeric confidence (`from_confidence`)

Each row has a label and a numeric confidence value in `[0, 1]`.

- selected class gets that confidence value
- remaining probability mass is split evenly across other classes

For $K$ classes, selected confidence $p_s$:

$$
p_{other} = \frac{1 - p_s}{K - 1}
$$

### Binary confidence (`from_binary_confidence`)

Each row has a label and a confidence flag.

- selected class gets `high_p` (or `low_p` when not confident)
- remaining probability mass is split evenly across other classes

For $K$ classes, selected probability $p_s$:

$$
p_{other} = \frac{1 - p_s}{K - 1}
$$

### Likert scores (`from_likert`) and votes (`from_votes`)

These two modes use the **same transformation** in this implementation:
row-wise normalization of class columns.

For row values $x_1, x_2, \ldots, x_K$:

$$
p_k = \frac{x_k}{\sum_{j=1}^{K} x_j}
$$

So yes: mathematically, Likert and votes are equivalent here; they differ by
data meaning (ratings vs tallies), not by conversion formula.

### Vote tables (`from_votes`)

`from_votes` treats each row as a per-point tally over classes and row-normalizes
that tally into probabilities.

### Multi-interpreter vectors (`from_multi_interpreter_vectors`)

This mode is different from vote counting. It expects multiple rows per point,
each containing a class vector from one interpreter. The method first
row-normalizes each interpreter vector and then averages those vectors within
each point id.

For point $i$ with $n_i$ interpreters and per-interpreter vectors
$p^{(1)}, p^{(2)}, \ldots, p^{(n_i)}$:

$$
\bar{p}_i = \frac{1}{n_i} \sum_{m=1}^{n_i} p^{(m)}
$$

If a strata column is present, it must be consistent across all rows for the
same point id.

### `from_crisp(df, label_col="label", id_col=None)`

Converts a single crisp class label per row into binary confidence values of 1 and 0 across class columns
(for example, `Forest` becomes `[1, 0, 0]` across class columns).

### `from_confidence(df, label_col="label", confidence_col="confidence", id_col=None)`

Converts one selected class label plus numeric confidence into probabilities.

Required columns:

- `label_col` (default `"label"`)
- `confidence_col` (default `"confidence"`) with values in `[0, 1]`.

### `from_binary_confidence(...)`

Converts one-label + confidence flag format into probabilities.

Required columns:

- `label_col` (default `"label"`)
- `is_confident_col` (default `"is_confident"`)

By default:

- confident label gets `high_p=0.99`
- non-confident label gets `low_p=0.40`
- remaining probability mass is distributed evenly across other classes.

### `from_likert(df, id_col=None)`

Treats class columns as scores and normalizes each row to probabilities.

### `from_votes(df, id_col=None)`

Treats class columns as vote totals and normalizes each row.

### `from_multi_interpreter_vectors(df, id_col=None)`

Treats repeated rows with the same id as per-interpreter class vectors,
normalizes each row, and then averages those vectors within each id.

### `verify_standard_style(df, tolerance=0.001)`

Checks if a dataframe is already in standard probabilistic style.

## Example

```python
import pandas as pd
from acc_assessment.standardizer import ProbStandardizer

raw = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "Forest": [5, 2, 1],
        "Water": [1, 5, 2],
        "Agriculture": [0, 1, 7],
    }
)

standardizer = ProbStandardizer(
    class_names=["Forest", "Water", "Agriculture"],
    id_col="id",
)

prob_df = standardizer.from_votes(raw)
print(standardizer.verify_standard_style(prob_df))  # True
```

## Common validation errors

The helper raises clear errors for common data issues, including:

- missing required class columns,
- non-numeric or missing class values,
- rows with non-positive total before normalization,
- duplicate or missing id column when `require_unique_id=True`,
- missing/unknown labels in crisp mode,
- missing required columns in binary-confidence mode,
- labels not present in `class_names`.

## Typical workflow position

1. Read your raw confidence/count table.
2. Convert with `ProbStandardizer`.
3. Validate with `verify_standard_style`.
4. Save and use the standardized tables in downstream assessments.
