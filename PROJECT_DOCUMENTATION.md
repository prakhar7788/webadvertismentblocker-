# Web Advertisement Blocker — Project Documentation
### G13 · UCI Internet Advertisements · P13

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [Baseline Model — DummyClassifier](#6-baseline-model--dummyclassifier)
7. [Algorithm 1 — Logistic Regression](#7-algorithm-1--logistic-regression)
8. [Algorithm 2 — Bernoulli Naive Bayes](#8-algorithm-2--bernoulli-naive-bayes)
9. [Algorithm 3 — Random Forest](#9-algorithm-3--random-forest)
10. [Algorithm 4 — Support Vector Machine (SVM)](#10-algorithm-4--support-vector-machine-svm)
11. [Results Comparison](#11-results-comparison)
12. [Feature Importance and Interpretation](#12-feature-importance-and-interpretation)
13. [Limitations](#13-limitations)
14. [File and Notebook Index](#14-file-and-notebook-index)

---

## 1. Problem Statement

The goal is to decide automatically whether an image on a web page is an
**advertisement**, exactly as an ad blocker would, without hiding real (non-ad)
content.

This is a **binary classification** problem:

| Class | Label | Meaning |
|---|---|---|
| `ad` | 1 (positive) | The image is an advertisement |
| `nonad` | 0 (negative) | The image is regular page content |

**Why precision matters more than recall.**  A *false positive* (a real non-ad
image predicted to be an ad) means useful content is blocked — this is the worse
mistake.  A *false negative* (a real ad that slips through) is less harmful; the
user just sees one more ad.  We therefore **optimise precision on the ad class** and
report recall alongside it, subject to a stated false-positive rate (FPR) cap of
**≤ 5 %**.

---

## 2. Dataset

**Source:** UCI Machine Learning Repository — Internet Advertisements dataset
(Nicholas Kushmerick, 1998).  
**File:** `ad.data` / `ml/dataset/ad.data`

### Dimensions

| Property | Value |
|---|---|
| Total rows (raw) | 3,279 |
| Duplicate rows | 856 |
| Rows after deduplication | **2,423** |
| Total features (columns) | 1,558 |
| Continuous features | 3 (`height`, `width`, `aratio`) |
| Binary features | 1,555 (0/1 word-presence flags) |
| Target column | 1 (`ad` / `nonad`) |

### Feature Groups

| Group | Columns | Description |
|---|---|---|
| Continuous | 0–2 | Pixel height, pixel width, aspect ratio (width/height) of the image |
| Local (binary) | 3 | Whether the image URL is on the same host as the page |
| URL words | 4–495 | Presence of tokens in the image URL |
| Original-URL words | 496–557 | Presence of tokens in the page URL |
| Alt-text words | 558–944 | Presence of tokens in the image's alt attribute |
| Anchor-URL words | 945–1,557 | Presence of tokens in the surrounding anchor tag's URL |

The binary features are bag-of-words indicators extracted from URL text, alt text
and anchor text — sparse features typical of text-based classification.

### Missing Values

| Column | Count | Percentage |
|---|---|---|
| `height` (col 0) | 903 | 27.54 % |
| `width` (col 1) | 901 | 27.48 % |
| `aratio` (col 2) | 910 | 27.75 % |
| `local` (col 3) | 15 | 0.46 % |
| All binary cols | 0 | 0 % |
| **Total missing values** | **2,729** | — |
| **Rows with ≥ 1 missing value** | **920** | **28.06 %** |

Missing values appear only in the continuous and local columns; all 1,555 binary
features are complete.  Missing values are encoded as `?` in the raw file.

---

## 3. Exploratory Data Analysis (EDA)

All EDA was performed in `ml/eda.ipynb`.

### 3.1 Class Balance

**Raw data (3,279 rows):**

| Class | Count | Share |
|---|---|---|
| `nonad` | 2,820 | 86.0 % |
| `ad` | 459 | 14.0 % |

**After deduplication (2,423 rows):**

| Class | Count | Share |
|---|---|---|
| `nonad` | 2,037 | 84.1 % |
| `ad` | 386 | 15.9 % |

The dataset is **imbalanced** (~6:1 nonad-to-ad ratio), which means accuracy is a
misleading metric (a classifier that always predicts "nonad" achieves 84% accuracy).
All models address this with `class_weight="balanced"` and are evaluated on
precision/recall rather than accuracy.

### 3.2 Missing Values

920 rows (28 %) have at least one missing value.  Missing values are confined to
the three continuous geometry columns (height, width, aratio) and the local column.
No binary feature has any missing values.  Missing values are encoded as `?` in the
source file; `pandas.read_csv` reads them as `NaN` when `na_values="?"` is set.

### 3.3 Duplicate Rows

856 rows (26.1 % of the raw data) are exact duplicates.  These are removed before
splitting, reducing the dataset to 2,423 unique rows.  Keeping duplicates would
allow the same image to appear in both the training and test sets, inflating
apparent performance.

### 3.4 Continuous Feature Statistics (after deduplication, before imputation)

| Feature | Non-null Count | Mean | Std | Min | Median | Max | Potential Outliers (IQR) |
|---|---|---|---|---|---|---|---|
| height | 1,936 | 67.68 | 56.41 | 1 | 60 | 640 | 76 |
| width | 1,940 | 160.80 | 126.79 | 1 | 115 | 640 | 232 |
| aratio | 1,931 | 3.73 | 4.34 | 0 | 2.27 | 60.0 | 18 |

Width has the most potential outliers; `aratio` has the heaviest right tail.  After
binarising by class, ad images tend to be narrow and tall (high aspect ratio),
while nonad images span a wider range of dimensions.

### 3.5 Binary Features Most Associated with Ads

For each binary feature the ad-occurrence rate when the token is **present** was
compared with the rate when it is **absent**.  The features with the largest
positive rate-difference are (top-10 by rate-difference in percentage points):

| Feature | Ad rate when present | Ad rate when absent | Rate difference |
|---|---|---|---|
| `ancurl*familyid` | 100 % | ~14 % | ~85.6 pp |
| `ancurl*groupid` | 100 % | ~14 % | ~85.6 pp |
| `ancurl*tagvalues` | 100 % | ~14 % | ~85.6 pp |
| `ancurl*adclick` | ~99 % | ~14 % | ~85 pp |
| `ancurl*cid` | ~99 % | ~14 % | ~85.3 pp |
| `url*ads+media` | ~99 % | ~14 % | ~85 pp |
| `ancurl*click` | ~98 % | ~14 % | ~84 pp |
| `url*advertising` | ~98 % | ~14 % | ~84 pp |
| `ancurl*http+www` | ~98 % | ~14 % | ~84 pp |
| `ancurl*adid` | ~97 % | ~14 % | ~83 pp |

These features are click-through and ad-serving URL patterns from 1998 ad networks.
When one of these tokens appears in an anchor-URL or image-URL, the image is almost
certainly an ad.

---

## 4. Preprocessing Pipeline

All preprocessing is **leak-free**: the imputer and scaler are fitted on the
training fold only and applied to validation and test folds without re-fitting.
Preprocessing lives inside a `scikit-learn Pipeline` / `ColumnTransformer` so this
guarantee holds during cross-validation as well as final training.

### 4.1 Split Strategy

A **three-way stratified split** is used:

```
Full deduplicated data  (2,423 rows)
        │
        ├─ 80 % fit set  (1,938 rows)
        │       │
        │       ├─ 75 % train      (1,453 rows, 60 % of total)
        │       └─ 25 % validation (  485 rows, 20 % of total)
        │
        └─ 20 % test set  (485 rows, 20 % of total)
```

`stratify=y` is passed to both `train_test_split` calls so each split preserves
the ~16 % ad fraction.  The **test set is used once**, at the very end, after all
hyperparameter choices and threshold selection are finalised.

### 4.2 ColumnTransformer

The 1,558 feature columns are handled differently:

| Group | Transformer | Reason |
|---|---|---|
| **Continuous** (height, width, aratio) | `SimpleImputer(strategy="mean")` → `StandardScaler()` | Fill missing values with the training-set mean; scale to zero mean and unit variance so gradient-based models (Logistic Regression) and distance-based models (SVM) are not dominated by the scale of these three columns |
| **Binary** (1,555 columns) | `SimpleImputer(strategy="most_frequent")` | Fill the rare missing values (only col 3) with the most common value; leave all others unscaled — they are already 0/1 |

```
ColumnTransformer
├── "continuous"  →  SimpleImputer(mean) → StandardScaler()  →  cols: [height, width, aratio]
└── "binary"      →  SimpleImputer(most_frequent)             →  cols: [binary_1 … binary_1555]
```

**Why scale only the three continuous columns?**  StandardScaler on binary 0/1
columns does not help (they are already bounded) and distorts the natural binary
interpretation expected by BernoulliNB and tree-based models.  SVM and Logistic
Regression need the continuous features scaled but are unaffected by unscaled
binary features that are already on the same 0/1 scale.

---

## 5. Evaluation Methodology

### 5.1 Metric

**Primary metric: precision on the ad class.**

$$\text{Precision} = \frac{TP}{TP + FP}$$

where TP = true positives (ads correctly classified as ads) and FP = false positives
(non-ads incorrectly classified as ads, i.e. real content that is blocked).

**Secondary metric: recall on the ad class.**

$$\text{Recall} = \frac{TP}{TP + FN}$$

where FN = false negatives (ads missed by the classifier).

**Reported alongside:**

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{False-Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

Accuracy is reported for reference but is not the selection criterion.

### 5.2 Why Not Accuracy?

With an 84 % majority class a "predict nonad always" classifier reaches 84 %
accuracy while being useless.  Precision directly measures the ad blocker's
reliability: "of everything I blocked, what fraction was actually an ad?"

### 5.3 Five-Fold Stratified Cross-Validation

Before touching the test set, every model is evaluated with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the **training
data only**.  Results are reported as:

```
mean ± standard deviation   across the 5 folds
```

Stratification ensures each fold preserves the ~16 % ad fraction.

### 5.4 Hyperparameter Tuning with GridSearchCV

Hyperparameters are selected using `GridSearchCV` with the same 5-fold CV on the
**training data only** (`refit="precision"`).  The best estimator is refit on the
full training set.  The validation and test sets are never seen during tuning.

### 5.5 Threshold Selection

All models produce a **probability score** for the ad class.  Rather than applying
the default 0.5 cut-off, we search over all candidate thresholds on the
**validation split** to find the one that:

1. Keeps the validation **FPR ≤ 5 %** (stated limit: at most 1 in 20 non-ad images
   is blocked), **and**
2. **Maximises precision** on the ad class, **subject to a recall floor of 0.50**
   (the model must still catch at least half the ads).

If no threshold meets the recall floor under the FPR cap, the fallback is the
FPR-eligible threshold with the highest recall.

The selected threshold is then applied **once** to the test set.

---

## 6. Baseline Model — DummyClassifier

**Purpose:** establish a floor that any real model must beat.

**Strategy:** `DummyClassifier(strategy="stratified")` — predicts classes randomly
according to the training-set class prior (~16 % ad, ~84 % nonad).

**Mathematics:** this produces a precision of approximately the ad fraction (~0.16)
and an equally poor recall, since predictions are made independently of features.

**5-fold CV results on training data:**

| Metric | Mean | SD |
|---|---|---|
| Precision (ad) | 0.1347 | ± 0.0666 |
| Recall (ad) | 0.1417 | ± 0.0692 |

All four models comfortably exceed this floor.

---

## 7. Algorithm 1 — Logistic Regression

**Notebook:** `ml/LogisticRegression.ipynb`

### 7.1 Background

Logistic Regression is a **linear discriminative model** that estimates the
probability that an input $\mathbf{x}$ belongs to class 1 (ad):

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

where $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ is the sigmoid (logistic) function,
$\mathbf{w} \in \mathbb{R}^d$ is the weight vector and $b$ is the bias.

### 7.2 Training Objective

The model is trained by **minimising the regularised binary cross-entropy loss**:

$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

where:
- $\hat{p}_i = P(y=1 \mid \mathbf{x}_i)$ is the predicted probability for sample $i$
- $\lambda$ is the regularisation strength (inverse of scikit-learn's `C`)
- The $L_2$ penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$ shrinks large weights toward
  zero, reducing overfitting on the high-dimensional feature space

The optimiser used (scikit-learn default for large `max_iter`) is **lbfgs** (Limited
memory BFGS), a quasi-Newton second-order method that approximates the Hessian to
converge faster than plain gradient descent.

### 7.3 Decision Rule

For a threshold $\tau$ chosen on the validation set:

$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

### 7.4 Class Imbalance

`class_weight="balanced"` is passed, which re-weights each training sample by
$\dfrac{n}{2 \cdot n_k}$ where $n_k$ is the count of class $k$.  This prevents the
majority class from dominating the gradient updates.

### 7.5 Hyperparameters (current notebook)

| Parameter | Value | Reason |
|---|---|---|
| `max_iter` | 2000 | Ensures convergence on 1,558 features |
| `class_weight` | `"balanced"` | Counteracts the 6:1 class imbalance |
| `random_state` | 42 | Reproducibility |

> **Note:** The LogReg notebook does not yet include `GridSearchCV` for `C`
> (regularisation strength). Adding a grid over `C = [0.01, 0.1, 1.0, 10]` is the
> recommended next improvement.

### 7.6 Results

| Metric | Value |
|---|---|
| Validation threshold selected | 0.5708 |
| Validation FPR | 4.41 % |
| Validation precision | 0.7955 |
| Validation recall | 0.9091 |
| **Test FPR** | **1.72 %** (7 / 408 non-ads blocked) |
| **Test precision** | **0.9041** |
| **Test recall** | **0.8571** |
| **Test F1** | **0.8800** |
| Test accuracy | 96.3 % |

---

## 8. Algorithm 2 — Bernoulli Naive Bayes

**Notebook:** `naivevayesclassifier.ipynb`

### 8.1 Background

Bernoulli Naive Bayes is a **generative probabilistic model** that uses Bayes'
theorem under the **naive independence assumption**: features are assumed independent
given the class.

### 8.2 Bayes' Theorem

For a feature vector $\mathbf{x} = (x_1, x_2, \dots, x_d)$:

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}$$

Since $P(\mathbf{x})$ is constant across classes, classification reduces to:

$$\hat{y} = \arg\max_{k \in \{0,1\}} P(y=k) \prod_{j=1}^{d} P(x_j \mid y=k)$$

### 8.3 Bernoulli Likelihood

For the **Bernoulli** variant each feature $x_j$ is binary (present/absent, after
binarisation at threshold `binarize`):

$$P(x_j \mid y=k) = p_{jk}^{x_j} \cdot (1 - p_{jk})^{1 - x_j}$$

where $p_{jk} = P(x_j = 1 \mid y = k)$ is estimated from training data.

### 8.4 Laplace Smoothing

Raw counts would assign zero probability to unseen feature–class combinations,
causing the product to collapse to zero.  **Laplace (additive) smoothing** with
parameter $\alpha$ avoids this:

$$\hat{p}_{jk} = \frac{N_{jk} + \alpha}{N_k + 2\alpha}$$

where $N_{jk}$ = number of training samples of class $k$ where feature $j$ is
present, $N_k$ = total training samples of class $k$.

### 8.5 Log-Space Computation

In practice the product over 1,558 features is computed in **log-space** to prevent
numerical underflow:

$$\log P(y=k \mid \mathbf{x}) \propto \log P(y=k) + \sum_{j=1}^{d} \left[ x_j \log p_{jk} + (1-x_j) \log(1-p_{jk}) \right]$$

### 8.6 Log-Likelihood Ratio for Interpretation

The importance of feature $j$ for the ad class is measured by the
**log-likelihood ratio** (log of the Bayes factor):

$$\text{LLR}_j = \log \frac{P(x_j=1 \mid \text{ad})}{P(x_j=1 \mid \text{nonad})} = \log p_{j,\text{ad}} - \log p_{j,\text{nonad}}$$

A large positive LLR means the feature is a strong indicator of an ad.

### 8.7 Binarisation

Before the Bernoulli likelihood is applied, continuous features (height, width,
aratio — already standardised to zero mean by `StandardScaler`) are binarised at
the `binarize` threshold.  With `binarize=0.5` (selected by GridSearch), a
standardised continuous value above 0.5 standard deviations from the mean is
treated as "present", below as "absent".

### 8.8 Hyperparameter Tuning (GridSearchCV)

| Parameter | Grid | Best value |
|---|---|---|
| `alpha` (Laplace smoothing) | [0.01, 0.1, 0.5, 1.0] | **1.0** |
| `binarize` (threshold) | [0.0, 0.5] | **0.5** |

Grid searched with 5-fold CV, refit on best precision.

### 8.9 Results

| Metric | Baseline | Default NB (CV) | Tuned NB (CV) | Test (once) |
|---|---|---|---|---|
| Precision (ad) | 0.1347 ± 0.0666 | 0.9381 ± 0.0404 | **0.9393 ± 0.0390** | **0.9800** |
| Recall (ad) | 0.1417 ± 0.0692 | 0.7069 ± 0.0775 | 0.7155 ± 0.0724 | 0.6364 |

| Test metric | Value |
|---|---|
| Threshold (validation-selected) | 0.9998 |
| Validation FPR | 0.245 % |
| Validation precision | 0.9818 |
| Validation recall | 0.7013 |
| **Test FPR** | **0.245 %** (1 / 408 non-ads blocked) |
| **Test precision** | **0.9800** |
| **Test recall** | **0.6364** |
| **Test F1** | **0.7717** |
| Test accuracy | 94.0 % |

**Note on the extreme threshold (~0.9998):** NaiveBayes probabilities are
over-confident (the independence assumption ignores correlated features), so the
raw `predict_proba` output is concentrated near 0 and 1.  The threshold search
corrects for this automatically.

---

## 9. Algorithm 3 — Random Forest

**Notebook:** `ml/randomforest.ipynb`

### 9.1 Background

A Random Forest is an **ensemble of decision trees** trained using bootstrap
aggregation (bagging) combined with random feature selection at each split.

### 9.2 Decision Tree Split Criterion

Each tree is built by recursively splitting nodes to maximise **Gini impurity
reduction** (the default in scikit-learn):

$$\text{Gini}(t) = 1 - \sum_{k} p_k^2$$

where $p_k$ is the fraction of class-$k$ samples at node $t$.  The best split
$(j^*, \theta^*)$ is:

$$\arg\max_{j,\theta} \left[ \text{Gini}(t) - \frac{|t_L|}{|t|}\,\text{Gini}(t_L) - \frac{|t_R|}{|t|}\,\text{Gini}(t_R) \right]$$

where $t_L$ and $t_R$ are the left and right child nodes.

### 9.3 Random Feature Selection

At each split, only a **random subset of $\sqrt{d}$ features** are considered
(where $d = 1,558$ here).  This decorrelates the trees and reduces variance without
increasing bias much.

### 9.4 Bootstrap Aggregation (Bagging)

Each tree is trained on a **bootstrap sample** (random sample with replacement) of
the training data.  For classification the forest aggregates by **majority vote**:

$$\hat{y} = \text{mode}\left\{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \right\}$$

For probability estimates, votes are averaged:

$$\hat{p}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{p}_b(\mathbf{x})$$

### 9.5 Feature Importance

The **mean decrease in Gini impurity** across all trees and all splits on feature
$j$ gives the feature importance:

$$\text{Importance}(j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b : \text{split on } j} \Delta\text{Gini}(t)$$

Higher values indicate the feature is more discriminative.

### 9.6 Class Imbalance

`class_weight="balanced"` adjusts sample weights so each class contributes equally
to the Gini computation at each split node.

### 9.7 Hyperparameter Tuning (GridSearchCV)

| Parameter | Grid | Best value |
|---|---|---|
| `n_estimators` | [100, 200] | **100** |
| `max_depth` | [None, 20] | **20** |
| `min_samples_leaf` | [1, 4] | **1** |

Grid searched with 5-fold CV (8 combos × 5 folds = 40 fits), refit on best
precision.  Training time: ~7.2 s.

### 9.8 Results

| Metric | Baseline | Default RF (CV) | Tuned RF (CV) | Test (once) |
|---|---|---|---|---|
| Precision (ad) | 0.1347 ± 0.0666 | 0.9101 ± 0.0364 | **0.9559 ± 0.0162** | **0.9828** |
| Recall (ad) | 0.1417 ± 0.0692 | 0.8493 ± 0.0659 | 0.8318 ± 0.0599 | 0.7403 |

| Test metric | Value |
|---|---|
| Threshold (validation-selected) | 0.5816 |
| Validation FPR | 0.245 % |
| Validation precision | 0.9844 |
| Validation recall | 0.8182 |
| **Test FPR** | **0.245 %** (1 / 408 non-ads blocked) |
| **Test precision** | **0.9828** |
| **Test recall** | **0.7403** |
| **Test F1** | **0.8444** |
| Test accuracy | 95.7 % |

---

## 10. Algorithm 4 — Support Vector Machine (SVM)

**Script:** `ml/svm.py`  **Artifact:** `ml/artifacts/metrics.json`

### 10.1 Background

A Support Vector Machine finds the **maximum-margin hyperplane** that separates
classes in feature space (or a kernel-transformed version of it).

### 10.2 Hard-Margin Primal (linearly separable case)

For a linearly separable dataset the primal optimisation problem is:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 \qquad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i$$

The **decision boundary** is $\mathbf{w}^\top \mathbf{x} + b = 0$; the **margin**
is $\dfrac{2}{\|\mathbf{w}\|}$.  Maximising the margin is equivalent to minimising
$\|\mathbf{w}\|^2$.

### 10.3 Soft-Margin (non-separable case)

Real data is not linearly separable, so **slack variables** $\xi_i \geq 0$ allow
misclassifications at a cost:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i$$
$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall i$$

$C > 0$ controls the trade-off: large $C$ penalises misclassifications heavily
(low bias, high variance); small $C$ allows more margin violations (higher bias,
lower variance).

### 10.4 Dual Formulation

The Lagrangian dual turns this into a quadratic programme over the dual variables
$\alpha_i \geq 0$:

$$\max_{\boldsymbol{\alpha}} \sum_{i} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$
$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

Samples with $\alpha_i > 0$ are the **support vectors**.

### 10.5 RBF Kernel

This project uses the **Radial Basis Function (Gaussian) kernel**, which implicitly
maps inputs to an infinite-dimensional space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

The dual problem becomes:

$$\max_{\boldsymbol{\alpha}} \sum_{i} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$\gamma$ controls the width of the Gaussian: large $\gamma$ → narrow kernel →
complex boundary (high variance); small $\gamma$ → wide kernel → smooth boundary.

### 10.6 Decision Function and Threshold

The SVM produces a real-valued **decision function** (signed distance from the
hyperplane) rather than a probability:

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

A threshold $\tau$ is applied to this score (selected on the validation set):

$$\hat{y} = \begin{cases} 1 & \text{if } f(\mathbf{x}) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

### 10.7 Class Imbalance

`class_weight="balanced"` sets $C_k = C \cdot \dfrac{n}{2 n_k}$ per class,
giving the minority ad class a higher misclassification cost.

### 10.8 Hyperparameters (svm.py — fixed, no GridSearch yet)

| Parameter | Value |
|---|---|
| `kernel` | `"rbf"` |
| `class_weight` | `"balanced"` |
| `C` | 1.0 (default) |
| `gamma` | `"scale"` (default: $1 / (d \cdot \text{Var}(X))$) |

> **Note:** `svm.py` uses a 60/20/20 split but selects threshold to **maximise
> precision** under a strict FPR < 5 % cap with a 0.50 recall floor.  It does not
> yet include 5-fold CV or GridSearchCV; these are recommended additions.

### 10.9 Results (from ml/artifacts/metrics.json)

| Test metric | Value |
|---|---|
| Threshold (validation-selected) | 0.7978 |
| Validation FPR (strict) | 0.0 % |
| Validation precision | 1.000 |
| Validation recall | 0.652 |
| **Test FPR** | **0.53 %** (3 / 564 non-ads) |
| **Test precision** | **0.9545** |
| **Test recall** | **0.6848** |
| **Test F1** | **0.7975** |
| Test accuracy | 95.1 % |
| Test ROC-AUC | 0.9666 |

> The SVM uses the 3,279-row (pre-dedup) dataset and a slightly different split
> (1,967 / 656 / 656), which is why the support counts differ from the other models.

---

## 11. Results Comparison

All models use the **same FPR ≤ 5 % cap** and **precision-focused threshold
selection** (except Logistic Regression, which maximises recall under the same cap).

| Model | CV Precision | CV Recall | Test Precision | Test Recall | Test F1 | Test FPR |
|---|---|---|---|---|---|---|
| DummyClassifier (baseline) | 0.135 ± 0.067 | 0.142 ± 0.069 | — | — | — | — |
| Logistic Regression† | — | — | 0.9041 | 0.8571 | 0.8800 | 1.72 % |
| Bernoulli Naive Bayes | 0.939 ± 0.039 | 0.716 ± 0.072 | **0.9800** | 0.6364 | 0.7717 | 0.245 % |
| Random Forest | 0.956 ± 0.016 | 0.832 ± 0.060 | **0.9828** | 0.7403 | 0.8444 | 0.245 % |
| SVM (RBF)‡ | — | — | 0.9545 | 0.6848 | 0.7975 | 0.53 % |

†Logistic Regression maximises recall (not precision) under the FPR cap; CV not yet
reported with this codebase version.  
‡SVM is trained on the full 3,279-row dataset; CV and GridSearch not yet added.

**Key takeaways:**

- All four models massively beat the baseline (~0.13 precision).
- **Random Forest** achieves the best CV precision (0.956) and best test precision
  (0.983) with the lowest FPR (0.245 %).
- **Logistic Regression** achieves the best recall (0.857) — it was tuned to
  maximise recall; the trade-off is a higher FPR (1.72 %).
- **Bernoulli Naive Bayes** matches Random Forest's precision (0.980) at the cost of
  the lowest recall (0.636), due to the over-confident probabilities discussed above.
- **SVM** sits in the middle on both precision and recall; its advantage is a
  naturally high decision boundary (threshold 0.798) that keeps FPR very low.

---

## 12. Feature Importance and Interpretation

### 12.1 Random Forest — Gini Importance (top 10)

| Rank | Feature (from ad.names) | Importance |
|---|---|---|
| 1 | `width` | 0.0865 |
| 2 | `aratio` | 0.0609 |
| 3 | `ancurl*com` | 0.0564 |
| 4 | `ancurl*click` | 0.0291 |
| 5 | `alt*click` | 0.0285 |
| 6 | `ancurl*http+www` | 0.0272 |
| 7 | `url*ads` | 0.0253 |
| 8 | `height` | 0.0251 |
| 9 | `alt*click+here` | 0.0229 |
| 10 | `alt*here` | 0.0217 |

The top features split into two groups: **image geometry** (`width`, `aratio`,
`height`) and **text tokens from anchor/alt/URL fields** (`ancurl*click`,
`alt*click`, `url*ads`).  This is consistent with the domain: 1998 banner ads had
standardised narrow-and-wide or tall-and-narrow dimensions, and their click-through
URLs systematically contained tracking parameters.

### 12.2 Bernoulli Naive Bayes — Log-Likelihood Ratios (top 10 toward ad)

| Rank | Feature | LLR |
|---|---|---|
| 1 | `ancurl*http+www` | 5.44 |
| 2 | `ancurl*familyid` | 4.95 |
| 3 | `ancurl*tagvalues` | 4.95 |
| 4 | `ancurl*groupid` | 4.95 |
| 5 | `ancurl*groupid+1` | 4.70 |
| 6 | `ancurl*1+familyid` | 4.70 |
| 7 | `ancurl*cid` | 4.60 |
| 8 | `ancurl*url` | 4.54 |
| 9 | `ancurl*adclick` | 4.54 |
| 10 | `url*ads+media` | 4.54 |

These are the click-tracking and ad-serving URL parameters used by 1998 ad
networks.  Tokens like `familyid`, `groupid`, `tagvalues` and `cid` are campaign
and targeting identifiers; `adclick` and `http+www` indicate a redirect to a
click-tracking server.

**Top 5 features pushing toward nonad (negative LLR):**

| Feature | LLR |
|---|---|
| `url*geoguideii` | −3.06 |
| `url*b+gif` | −2.75 |
| `ancurl*ring` | −2.32 |
| `ancurl*webring.org` | −2.28 |
| `url*aol.com` | −2.20 |

These are tokens from 1990s personal homepages and web-ring navigation buttons —
normal page images that are definitely not ads.

### 12.3 Logistic Regression — Coefficients

Logistic Regression produces signed weights $w_j$.  Positive $w_j$ increases the
log-odds of the ad class when feature $j$ is active.  Top features align with the
same ad-network tokens seen in the NB and RF analysis (see notebook for the full
sorted coefficient table).

---

## 13. Limitations

### 13.1 Data is from 1998

This is the most critical limitation.  The features were engineered for the 1998
web:
- Ads were static GIF/JPEG banners with predictable sizes (468×60 "full banner",
  120×90 "button", etc.).  Modern ads are responsive, often non-rectangular, and
  embedded in video/iframe contexts.
- Ad URLs passed campaign identifiers as query-string parameters (`familyid`,
  `groupid`, `cid`).  Modern programmatic ads use opaque CDN URLs with no readable
  tokens.
- Text features come from alt attributes and anchor text; modern ads often have
  empty alt text and are loaded via JavaScript, so these signals don't exist.

A model trained here **should not be deployed on today's web without retraining on
current data**.

### 13.2 Naive Independence Assumption (NB)

BernoulliNB assumes features are independent given the class.  In this dataset URL
tokens are strongly correlated (the same ad network sets `familyid`, `groupid`,
`tagvalues` together).  NB double-counts this correlated evidence, making its
probabilities over-confident — the threshold ends up at ~0.9998 rather than 0.5.
This means the probability scores are not reliable for ranking or calibration, only
for binary classification after threshold correction.

### 13.3 Imputation Hides Missing Geometry

The ~28 % of rows with missing continuous values are imputed with the training-set
mean.  An image that genuinely has no width/height information (e.g. dynamically
sized) is forced into the "average" geometry, which may push it toward or away from
the ad class incorrectly.

### 13.4 No Context, No Network Layer

The model classifies a single image in isolation.  Real ad blockers use:
- Page context (position, surrounding text, layout)
- Network-level signals (request to a known ad server)
- Filter lists (EasyList, uBlock Origin)  
None of these signals are in this feature set.

### 13.5 Fragile Operating Point

The FPR ≤ 5 % threshold was chosen on a 485-row validation split.  On a different
web population the class distribution, feature distribution, and correlations may
all shift, changing the achieved FPR.  The threshold should be re-calibrated before
deployment on a new domain or time period.

### 13.6 Class Imbalance

Even after `class_weight="balanced"`, the 6:1 class imbalance means the minority
ad class has fewer training examples and a higher chance of being under-fitted.
Techniques such as SMOTE, threshold-moving ensembles, or cost-sensitive learning
could be explored.

---

## 14. File and Notebook Index

| File / Notebook | Purpose |
|---|---|
| `ad.data` | Raw dataset (root copy) |
| `ml/dataset/ad.data` | Dataset copy used by notebooks |
| `ml/dataset/ad.names` | Attribute name definitions (used for feature labelling) |
| `ml/dataset/ad.DOCUMENTATION` | UCI dataset documentation |
| `ml/eda.ipynb` | Exploratory Data Analysis — class balance, missing values, duplicate analysis, feature-class association |
| `ml/LogisticRegression.ipynb` | Logistic Regression — preprocessing pipeline, recall-maximising threshold, test evaluation |
| `naivevayesclassifier.ipynb` | Bernoulli Naive Bayes — ColumnTransformer preprocessing, baseline, 5-fold CV, GridSearchCV, precision-maximising threshold, log-LR interpretation |
| `ml/randomforest.ipynb` | Random Forest — ColumnTransformer preprocessing, baseline, 5-fold CV, GridSearchCV, precision-maximising threshold, Gini importance interpretation |
| `ml/svm.py` | SVM (RBF) — ColumnTransformer preprocessing, precision-maximising threshold, saves model and metrics to `ml/artifacts/` |
| `ml/artifacts/svm_model.joblib` | Serialised SVM pipeline + threshold |
| `ml/artifacts/metrics.json` | SVM test metrics |
| `ml/artifacts/confusion_matrix.png` | SVM confusion matrix plot |
| `PROJECT_DOCUMENTATION.md` | This file |

---

*Generated for G13 — Web Advertisement Blocking.  
Dataset: UCI Internet Advertisements (N. Kushmerick, 1998).  
Tools: Python 3.13, scikit-learn 1.8, pandas, NumPy, Matplotlib.*
