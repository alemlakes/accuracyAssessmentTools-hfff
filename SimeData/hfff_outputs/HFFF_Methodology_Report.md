# HFFF Methodology Output Report

## Overview
This report summarizes the results of applying three interpreter confidence standardization methodologies to the full referenceIntObservationsFull.csv dataset, as well as a composite output. The methods are:

1. **Binary Heuristic Thresholding**
2. **Continuous Weighted Confidence Index (CWCI)**
3. **Probabilistic Discounting via Uniform Redistribution**
4. **Composite** (all three vectors side-by-side)

## Output Files
All output files are located in this directory and are named as follows:

- 1.hfff_binary.csv
- 2.hfff_cwci.csv
- 3.hfff_discount.csv
- 4.hfff_composite.csv

## Method Summaries

### 1. Binary Heuristic Thresholding
- Assigns a hard label and confidence flag based on interpreter certainty.
- Produces highly certain vectors (mostly 0/1 or near-0/1 values).
- Pros: Simple, interpretable, robust to ambiguity.
- Cons: Discards nuanced confidence information, may overstate certainty.

### 2. Continuous Weighted Confidence Index (CWCI)
- Uses interpreter's continuous confidence to weight the assigned class.
- Produces more graded, probabilistic outputs.
- Pros: Retains more information, reflects uncertainty.
- Cons: Sensitive to subjective confidence scaling.

### 3. Probabilistic Discounting
- Adjusts class probabilities by discounting based on image/interpretation quality factors.
- Pros: Incorporates context and uncertainty from multiple sources.
- Cons: May dilute strong signals if discounting is aggressive.

### 4. Composite
- Combines all three vectors for each point for comparison and downstream analysis.

## Recommendation
- **CWCI** (2.hfff_cwci.csv) is recommended for most analyses, as it balances interpretability and information retention.
- Use **Binary** (1.hfff_binary.csv) for strict, high-confidence applications.
- Use **Discounting** (3.hfff_discount.csv) when context-based uncertainty is critical.
- The **Composite** file (4.hfff_composite.csv) enables side-by-side comparison.

## Notes
- All files are HFFF-compatible and ready for downstream use.
- See generate_hfff_vectors.py for code and logic details.
