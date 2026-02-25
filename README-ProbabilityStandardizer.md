# Preparation Helper README

This guide documents the preparation helper in:

- `src/acc_assessment/standardizer.py`

It is useful when your input labels are not yet in the standard probabilistic
format expected by the integrated workflow and by `GUE`/`MCEM`.

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
- optional metadata columns such as `id` and `strata`.

## Main API

### `ProbStandardizer(class_names, id_col="id", strata_col="strata")`

Initialize with the class names used in your project.

### `from_likert(df, id_col=None, strata_col=None)`

Treats class columns as scores and normalizes each row to probabilities.

### `from_counts(df, id_col=None, strata_col=None)`

Treats class columns as vote/count totals and normalizes each row.

### `from_multi_interpreter(df, id_col=None, strata_col=None)`

Alias of `from_counts` for multi-interpreter vote tables.

### `from_binary_confidence(...)`

Converts one-label + confidence flag format into probabilities.

Required columns:

- `label_col` (default `"label"`)
- `is_confident_col` (default `"is_confident"`)

By default:

- confident label gets `high_p=0.99`
- non-confident label gets `low_p=0.40`
- remaining probability mass is distributed evenly across other classes.

### `verify_standard_style(df, tolerance=0.001)`

Checks if a dataframe is already in standard probabilistic style.

## Example

```python
import pandas as pd
from acc_assessment.standardizer import ProbStandardizer

raw = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "strata": ["a", "a", "f"],
        "Forest": [5, 2, 1],
        "Water": [1, 5, 2],
        "Agriculture": [0, 1, 7],
    }
)

standardizer = ProbStandardizer(
    class_names=["Forest", "Water", "Agriculture"],
    id_col="id",
    strata_col="strata",
)

prob_df = standardizer.from_counts(raw)
print(standardizer.verify_standard_style(prob_df))  # True
```

## Common validation errors

The helper raises clear errors for common data issues, including:

- non-numeric or missing class values,
- rows with non-positive total before normalization,
- missing required columns in binary-confidence mode,
- labels not present in `class_names`.

## Typical workflow position

1. Read your raw confidence/count table.
2. Convert with `ProbStandardizer`.
3. Validate with `verify_standard_style`.
4. Save and use the standardized tables in downstream assessments.
