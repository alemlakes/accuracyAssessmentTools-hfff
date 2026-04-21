# Map vs. Reference Uncertainty: Conceptual Overview

## Executive Summary

Probabilistic accuracy assessment requires handling two fundamentally distinct sources of uncertainty:
- **Map uncertainty**: How confident is the automated classifier about its predictions?
- **Reference uncertainty**: How confident are human interpreters in their ground-truth labels?

Both must be standardized into per-point class probability vectors and propagated through estimators like GUE (Generalized Unbiased Estimator) and MCEM (Monte Carlo Error Matrix) to produce unbiased area estimates and accuracy metrics.

---

## Map Uncertainty

### What It Is

Map uncertainty represents the classifier's own doubt about the class assignment at each spatial point. Instead of predicting a single hard class label, modern ML systems output *probability distributions* across classes.

**Examples:**
- Logits from a neural network softmax layer: `[0.1, 0.7, 0.2]` → `[P(Deforestation), P(Natural), P(Antrópico)]`
- Post-hoc calibrated probabilities from an ensemble model
- Per-class confidence scores from any probabilistic classifier

### Sources of Map Uncertainty

1. **Model architecture limitations**: Neural networks with limited capacity struggle with ambiguous pixels
2. **Training data noise**: If training data contains mislabeled examples, the model learns unreliable decision boundaries
3. **Spectral ambiguity**: Some land-cover types have overlapping reflectance signatures (e.g., sparse shrub vs. degraded forest)
4. **Mixed pixels**: Sub-pixel boundaries (a pixel contains both forest and agriculture) are inherently ambiguous
5. **Temporal dynamics**: Seasonal phenology can make the same location look different at different times
6. **Sensor artifacts**: Atmospheric effects, sensor saturation, and noise introduce variability

### How the Codebase Handles It

**ProbabilityStandardizer methods for map data:**
- `from_likert()` → Normalizes raw Likert/rating-scale scores
- `from_counts()` → Normalizes vote tallies or ensemble results
- Already in probability form → Just validate structure with `verify_standard_structure()`

**Integration into estimation:**
- Map probabilities are passed directly to `GUE` and `MCEM`
- Both estimators treat map probabilities as **model outputs** — they don't modify them, only propagate their uncertainty downstream

---

## Reference Uncertainty

### What It Is

Reference uncertainty represents the human interpreter's confidence (or lack thereof) in their assigned ground-truth label at each point. A hard label like "Stable" obscures the interpreter's true doubt, so reference uncertainty is captured as:
1. **Per-class probability distributions** (e.g., `[P(Growth), P(Loss), P(Stable)]`)
2. **Metadata about why** the interpreter was uncertain (clouds, boundary effects, sensor limits)

### Sources of Reference Uncertainty

1. **Atmospheric occlusion**
   - Dense clouds, aerosol haze, smoke obscure the ground
   - Interpreter cannot see clearly enough to decide confidently

2. **Spatial edge effects**
   - Sample point sits exactly on a boundary (ecotone) between two classes
   - Minor image misregistration between satellite dates can artificially create or erase change

3. **Spectral capacity limitations**
   - Sensor lacks sufficient radiometric depth to detect subtle changes (e.g., selective logging vs. natural thinning)
   - Ecological signal is overwhelmed by instrument noise

4. **Temporal ambiguity**
   - Change event timing unclear (did loss occur in month 1 or month 11?)
   - Multiple small changes vs. single large change hard to distinguish

5. **Cognitive friction / decision hesitation**
   - Interpreter spends hours deliberating and adding qualitative caveats
   - Behavior proxy: elapsed time correlates with uncertainty

### How the Codebase Handles It

**Reference uncertainty is captured in three dimensions:**

1. **Base probability vectors** (raw interpreter input)
   - Rec_Prob_* = Most recent interpreter's assessment
   - Ave_Prob_* = Average across all interpreters for that point
   - Interpreted as integers on 0–100 scale, normalized to [0,1]

2. **Uncertainty covariates** (JSON-structured metadata)
   - **Image_Condition**: Atmospheric/radiometric quality impact
   - **Spatial_Context**: Boundary/registration effects
   - **Spectral_Capacity**: Sensor capability limitations
   - Each encoded as `{"Helped": X%, "Hinder": Y%, "NAN": Z%}`

3. **Qualitative comments**
   - Free-text interpreter justification (e.g., "clouds made it impossible to see")
   - Diagnostic bridge between statistical probability and human reasoning

### Standardization Methods: Three Approaches

Reference data must be converted into HFFF-compliant probability vectors using one of three methods:

#### **Method 1: Binary Heuristic Thresholding** 
Maps to `ProbabilityStandardizer.from_binary_confidence()`

- **Principle**: Any single environmental failure is a total failure
- **Algorithm**: Extract max Hinder across covariates; if max Hinder ≥ threshold, flag as "not confident"
- **Output**: Boolean flag → high_p (e.g., 0.99) or low_p (e.g., 0.40) for primary class
- **Pros**: Simple, conservative, prevents noisy data from corrupting training
- **Cons**: Destroys nuance; ignores Helped metadata; threshold is arbitrary

#### **Method 2: Continuous Weighted Confidence Index (CWCI)**
Maps to `ProbabilityStandardizer.from_confidence()`

- **Principle**: Confidence is a continuous balance of Helped vs. Hinder across all three covariates
- **Algorithm**:
  1. For each covariate, compute net score: `S = (Helped - Hinder) / 100`
  2. Average across three covariates: `S_avg = (S_Image + S_Spatial + S_Spectral) / 3`
  3. Normalize to [0,1] via min-max: `p_confidence = (S_avg + 1) / 2`
  4. Assign this confidence to the primary class; spread remainder evenly across others
- **Output**: Graded probability vector (e.g., `[0.1, 0.8, 0.1]`)
- **Pros**: Highest information retention; fluid uncertainty representation; handles both Helped and Hinder
- **Cons**: Assumes linearity/equivalence of covariates; sensitive to NAN interpretation

#### **Method 3: Probabilistic Discounting via Uniform Redistribution**
Maps to `ProbabilityStandardizer.from_counts()` (pre-discounted)

- **Principle**: Compress the interpreter's original probability vector toward Maximum Entropy (uniform distribution) proportional to max hindrance
- **Algorithm**:
  1. Extract max Hinder: `H_max = max(H_Image, H_Spatial, H_Spectral)`
  2. Compute discount factor: `d = H_max / 100`
  3. Compress: `p_adjusted = (1 - d) * p_original + (d / num_classes)`
- **Output**: Discounted probability vector (e.g., `[0.15, 0.55, 0.30]`)
- **Pros**: Respects original interpreter variance; graceful degradation; Bayesian flavor; preserves relative class preference
- **Cons**: Injects impossible probabilities into background classes; ignores Helped metadata

---

## Multi-Interpreter Aggregation

When multiple interpreters assess the same point:

1. **Standardize each independently** using one of the three methods above
2. **Stack the standardized vectors** (one row per interpreter)
3. **Row-normalize across interpreters** using `ProbabilityStandardizer.from_multi_interpreter()`

This produces a **consensus** probability vector that:
- Preserves disagreement (if interpreters split, output probabilities reflect it)
- Prevents any single noisy interpretation from dominating
- Properly weights interpreters who agree vs. disagree

---

## Integration: From Uncertainty to Estimation

### Workflow for GUE (Analytical)

```
Map Data                    Reference Data
    ↓                            ↓
[per-point probs]          [standardized probs]
    ↓                            ↓
    └────→ GUE Estimator ←───────┘
           • Computes expected confusion matrix
           • Calculates area-unbiased accuracy
           • No simulation needed (fast)
```

### Workflow for MCEM (Monte Carlo)

```
Map Data                    Reference Data
    ↓                            ↓
[per-point probs]          [standardized probs]
    ↓                            ↓
    └────→ MCEM Estimator ←──────┘
           • Sample hard classes N times
           • Run crisp Stehman estimator each time
           • Average results → CIs from distribution
           • Slower but very transparent
```

---

## Key Conceptual Differences

| Aspect | Map Uncertainty | Reference Uncertainty |
|--------|-----------------|----------------------|
| **Origin** | ML model outputs | Human interpreter judgment |
| **Semantics** | Classifier confidence | Observer confidence |
| **Typical source** | Training data noise, spectral ambiguity | Clouds, boundaries, sensor limits |
| **How to capture** | Model native probabilities | Metadata + qualitative comments |
| **Standardization** | Usually already in [0,1] form | Requires one of three methods (Binary, CWCI, Discounting) |
| **Role in estimation** | Propagated as given | Can be weighted/discounted | 
| **Multi-variant** | Ensemble averaging | Multi-interpreter consensus |

---

## Why Separate Map and Reference?

Design clarity:
- **Map side** focuses on: calibration, ensemble post-hoc methods, temporal aggregation
- **Reference side** focuses on: human confidence encoding, multi-interpreter voting, environmental metadata

Statistical correctness:
- Both uncertainty sources must be honored to achieve unbiased area estimation
- GUE and MCEM are designed to propagate both simultaneously
- Ignoring either will bias results

Interpretability:
- Separating the two allows downstream users to understand *which* uncertainty drove each decision
- Enables transparency: "Did my model fail, or did the interpreter?"

---

## Recommended Reading

1. **For map uncertainty**: See your model's documentation (softmax calibration, ensemble methods)
2. **For reference uncertainty**: See `Confidence Assessment for HFFF Framework.txt` for deep methodology
3. **For GUE**: See `src/acc_assessment/gue.py` docstrings
4. **For MCEM**: See `src/acc_assessment/mcem.py` docstrings
5. **For standardization**: See `ProbabilityStandardizer` in `src/acc_assessment/standardizer.py`
