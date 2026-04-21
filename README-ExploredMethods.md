# Explored Methods

This document summarizes the explored method numbering used in the Sime branch and clarifies which entries are primary standardization methods versus comparison or aggregation outputs.

## Executive Summary

The explored methods fall into three groups:

1. **Primary reference-standardization methods**: Methods 1-3
2. **Ensemble or comparison outputs**: `cbind.123`, Methods 4-5
3. **Direct evaluator aggregation baseline**: Method 7

These methods are not all operating at the same stage of the workflow. Methods 1-3 transform one interpreter's uncertain assessment into a class-probability vector. Methods 4-5 combine outputs from earlier methods. Method 7 averages across evaluators directly.

## Method Summary

### Method 1: Binary Heuristic Thresholding

- Assigns a hard label and a binary confidence flag based on interpreter certainty.
- Maps naturally to `ProbabilityStandardizer.from_binary_confidence()`.
- Produces highly certain, nearly one-hot vectors.

**Interpretation:**
Use when uncertainty should act as a strict quality gate. If any major hindrance is present, confidence is sharply reduced.

### Method 2: Continuous Weighted Confidence Index (CWCI)

- Uses the interpreter's continuous confidence to weight the assigned class.
- Maps naturally to `ProbabilityStandardizer.from_confidence()`.
- Produces graded probabilistic vectors reflecting uncertainty.

**Interpretation:**
Use when confidence should vary smoothly rather than collapse to a high/low rule.

### Method 3: Probabilistic Discounting

- Adjusts class probabilities by discounting based on image/context quality.
- Typically implemented by discounting the original vector and then normalizing through `from_counts()` or an equivalent counts-style pathway.
- Produces probabilities diluted toward uniformity when uncertainty is high.

**Interpretation:**
Use when the original interpreter proportions should be preserved as much as possible, but weakened under hindrance.

### `cbind.123`

- Stores the vectors from Methods 1-3 side-by-side for each observation.
- Does **not** average them.

**Interpretation:**
This is best treated as a comparison artifact rather than a standalone statistical method. It is useful for inspecting how the three primary methods differ on the same point.

### Method 4: `averageof123`

- Computes the simple average of the three probability vectors from Methods 1-3 for each class.

For each class $k$:

$$
p_k^{(4)} = \frac{p_k^{(1)} + p_k^{(2)} + p_k^{(3)}}{3}
$$

**Interpretation:**
This is an ensemble of the three primary standardization methods. It reflects the idea that no single uncertainty model should dominate.

### Method 5: Adjusted `averageof123` (help/hinder dilution)

- Starts with the averaged vector from Method 4.
- Further adjusts that vector using interpreter `Helped` / `Hinder` fields.
- Dilutes the vector toward uniformity as uncertainty increases.

**Interpretation:**
This is an ensemble-plus-discount method. It combines the stability of averaging with an additional uncertainty penalty.

### Method 6: Unused / Reserved

- No method is currently assigned to this number.

**Interpretation:**
This is simply a numbering placeholder.

### Method 7: Simple 12-Month Average

- For each point, averages the 12-month confidence values across all evaluators.
- Output form:
  - `tracker`
  - `Growth_avg`
  - `Loss_avg`
  - `Stable_avg`
  - `n_evaluators`

**Interpretation:**
This is a direct evaluator-aggregation baseline. It differs from Methods 1-5 because it is primarily about combining multiple evaluators rather than transforming one evaluator's uncertainty metadata.

## Conceptual Distinction

Two different questions are mixed into this numbering system:

1. **How should a single interpreter's uncertainty be converted into a probability vector?**
   - Methods 1-3
2. **How should multiple candidate vectors or multiple evaluators be combined?**
   - `cbind.123`, Methods 4-5, Method 7

This distinction is important. Method 7 is not a direct competitor to Method 2 in the same sense, because they act at different stages of the pipeline.

## Recommended Framing

For future documentation, the cleanest interpretation is:

- **Primary methods**: 1, 2, 3
- **Comparison / ensemble outputs**: `cbind.123`, 4, 5
- **Evaluator aggregation baseline**: 7

It is also helpful to describe `cbind.123` as a comparison table rather than a true standalone method.

## Relation to HFFF and the Probability Standardizer

Within this branch, HFFF is the umbrella framework for producing standardized reference-side probability vectors. The explored methods above are candidate ways of generating those vectors before using downstream estimators such as GUE or MCEM.

In practical terms:

- Method 1 aligns with `from_binary_confidence()`
- Method 2 aligns with `from_confidence()`
- Method 3 aligns with `from_counts()` after discounting
- Method 7 aligns with averaging across evaluators prior to downstream probabilistic assessment

## Summary

The explored-method numbering is coherent if read as a working inventory rather than a single unified taxonomy. Methods 1-3 are the core standardization candidates. Methods 4-5 are ensemble variants. Method 7 is a direct multi-evaluator baseline. `cbind.123` is best understood as a comparison object rather than a method.