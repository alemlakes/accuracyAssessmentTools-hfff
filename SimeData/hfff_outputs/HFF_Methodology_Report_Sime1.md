
# HFF_Methodology_Report_Sime1.md
These examples illustrate the range of harmony and disagreement between methods, the nature of typical clusters, and the types of outliers present in the dataset.

## Interpretation of Composite Results for Sime1 Dataset

### Compatibility and Contradiction Between Methods

The three methods (Binary Heuristic, CWCI, Probabilistic Discounting) are generally compatible in their broad assignment of dominant classes, but they differ in how they express uncertainty and handle ambiguous or low-confidence points:

- **Binary Heuristic** produces highly certain, almost one-hot vectors (e.g., [0.005, 0.005, 0.99]), reflecting a strict threshold for confidence. This method rarely expresses ambiguity, so most points are assigned nearly all probability to a single class.
- **CWCI** (Continuous Weighted Confidence Index) provides more graded probabilities, reflecting the interpreter's confidence. For many points, this method gives a higher probability to the dominant class but allows for more uncertainty (e.g., [0.09, 0.09, 0.82]).
- **Probabilistic Discounting** can further flatten the probabilities, especially when image or context quality is poor, sometimes resulting in nearly uniform distributions (e.g., [0.333, 0.333, 0.333]) or shifting probability away from the dominant class.

**Contradictions:**
- There are few outright contradictions (e.g., one method assigning high probability to Growth, another to Loss), but there are cases where the degree of certainty differs substantially. For example, a point may be nearly certain in Binary but much less so in Discounting.
- In rare cases, Discounting or CWCI may elevate a non-dominant class due to low confidence or high uncertainty, but the dominant class usually remains the same.

### Outliers and Method-Specific Sensitivity

- **Outliers** are points where one method gives a very different result from the others. For example, some points are assigned [0.3, 0.4, 0.3] by Binary (indicating ambiguity), while CWCI or Discounting may push the probabilities closer to uniform or shift the dominant class.
- Some points that are highly certain in Binary (e.g., [0.99, 0.005, 0.005]) are much less certain in CWCI or Discounting, indicating that the interpreter's confidence or the image/context quality was low.
- Conversely, some points that are ambiguous in Binary (e.g., [0.3, 0.4, 0.3]) may be resolved more decisively by CWCI or Discounting if the interpreter's confidence or context is strong.

### Method Compatibility

- The methods are **not fully interchangeable**: Binary is best for strict, high-confidence labeling; CWCI for nuanced, confidence-weighted analysis; Discounting for incorporating external uncertainty.
- For most points, the dominant class is consistent across methods, but the degree of certainty varies.
- The composite file enables identification of points where methods diverge, which may warrant further review or targeted analysis.

### Summary

- **No major contradictions** between methods, but significant differences in certainty and handling of ambiguous points.
- **Outliers** exist and are identifiable in the composite file—these are points where method choice could affect downstream analysis.
- **Recommendation:** Use the composite to flag points with high disagreement for review, and select the method that best matches your risk tolerance and analytic goals for each use case.

# Appendix: Point-Level and Cluster Analysis

## Five Random Points: Method Harmony

1. **ID 87950922**
	- Binary: [0.005, 0.005, 0.99]
	- CWCI: [0.09, 0.09, 0.82]
	- Discount: [0.0, 0.0, 1.0]
	- *Harmony*: All methods agree on Stable as dominant, with Binary most certain, Discount fully certain, and CWCI showing some uncertainty.

2. **ID 81395819**
	- Binary: [0.3, 0.4, 0.3]
	- CWCI: [0.25, 0.5, 0.25]
	- Discount: [0.333, 0.333, 0.333]
	- *Harmony*: All methods reflect ambiguity, with Discounting flattening the probabilities most. No contradiction, but high uncertainty.

3. **ID 84152401**
	- Binary: [0.005, 0.005, 0.99]
	- CWCI: [0.125, 0.125, 0.75]
	- Discount: [0.0, 0.0, 1.0]
	- *Harmony*: All methods agree on Stable, with Binary and Discounting most certain, CWCI less so.

4. **ID 29398292**
	- Binary: [0.005, 0.005, 0.99]
	- CWCI: [0.146, 0.146, 0.708]
	- Discount: [0.083, 0.233, 0.683]
	- *Harmony*: All methods agree on Stable, but Discounting and CWCI show more uncertainty and some probability for Loss.

5. **ID 34090186**
	- Binary: [0.3, 0.3, 0.4]
	- CWCI: [0.153, 0.153, 0.693]
	- Discount: [0.223, 0.223, 0.553]
	- *Harmony*: All methods agree on Stable, but Binary is more ambiguous, CWCI and Discounting less so.

## Three Representative Clusters

**Cluster 1: High Certainty, High Agreement**
- Example: ID 87950922, 84152401, 97187302
- *Pattern*: All methods assign nearly all probability to one class (usually Stable), with only minor differences in uncertainty.
- *Grouping logic*: Points where all methods' dominant class probability >0.7 and agree on the class.

**Cluster 2: Ambiguous/Uncertain Points**
- Example: ID 81395819, 34090186, 29398292
- *Pattern*: All methods show ambiguity, with probabilities spread across classes. Discounting may flatten further.
- *Grouping logic*: Points where no class exceeds 0.5 in any method, and all methods show similar ambiguity.

**Cluster 3: Moderate Certainty, Method-Dependent**
- Example: ID 29398292, 66411237, 23083100
- *Pattern*: Binary is certain, but CWCI and Discounting show more uncertainty or shift probability to other classes.
- *Grouping logic*: Points where Binary is certain, but other methods are not, or vice versa.

## Five Outliers: Method Disagreement

1. **ID 54194878**
	- Binary: [0.3, 0.4, 0.3]
	- CWCI: [0.208, 0.583, 0.208]
	- Discount: [0.0, 1.0, 0.0]
	- *Outlier*: Discounting is fully certain for Loss, while Binary and CWCI are ambiguous.

2. **ID 24933449**
	- Binary: [0.99, 0.005, 0.005]
	- CWCI: [0.833, 0.083, 0.083]
	- Discount: [1.0, 0.0, 0.0]
	- *Outlier*: All methods agree on Growth, but Discounting is fully certain, Binary nearly so, CWCI less so.

3. **ID 79842933**
	- Binary: [0.4, 0.3, 0.3]
	- CWCI: [0.152, 0.424, 0.424]
	- Discount: [0.418, 0.25, 0.332]
	- *Outlier*: Binary is ambiguous, CWCI splits between Loss and Stable, Discounting is more even.

4. **ID 26822782**
	- Binary: [0.005, 0.99, 0.005]
	- CWCI: [0.077, 0.847, 0.077]
	- Discount: [0.033, 0.933, 0.033]
	- *Outlier*: All methods agree on Loss, but Discounting and CWCI are less certain than Binary.

5. **ID 13068541**
	- Binary: [0.3, 0.4, 0.3]
	- CWCI: [0.271, 0.458, 0.271]
	- Discount: [0.25, 0.482, 0.268]
	- *Outlier*: All methods are ambiguous, but Discounting and CWCI are more certain for Loss than Binary.

---
