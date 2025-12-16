# Basic Classification Part B: Model Evaluation

## Overview

This module covers three main areas:
1. **Metrics for Performance Evaluation** - How to evaluate the performance of a model?
2. **Methods for Performance Evaluation** - How to obtain reliable estimates?
3. **Methods for Model Comparison** - How to compare relative performance among competing models?

---

## 1. Metrics for Performance Evaluation

Focus is on the **predictive capability** of a model (not speed, scalability, etc.)

### 1.1 Confusion Matrix

The confusion matrix is a fundamental tool for evaluating classification models.

**Binary Classification Confusion Matrix:**

|                | Predicted: Yes | Predicted: No |
|----------------|----------------|---------------|
| **Actual: Yes** | a (TP)         | b (FN)        |
| **Actual: No**  | c (FP)         | d (TN)        |

**Terminology:**
- **TP (True Positive)** = a: Correctly predicted positive
- **FN (False Negative)** = b: Actual positive, predicted negative (Type II error)
- **FP (False Positive)** = c: Actual negative, predicted positive (Type I error)
- **TN (True Negative)** = d: Correctly predicted negative

**Example with Cat/Dog Classification:**

|              | Predicted: Cat | Predicted: Dog |
|--------------|----------------|----------------|
| **Actual: Cat** | a (TP)        | b (FN)         |
| **Actual: Dog** | c (FP)        | d (TN)         |

---

### 1.2 Key Metrics

#### Recall (Sensitivity, True Positive Rate)

**Definition:** Of all actual positive samples, how many did we correctly identify?

```
Recall (r) = a / (a + b) = TP / (TP + FN)
```

- Focuses on the **row** of actual positive class
- Answers: "Of all the actual YES cases, how many did we catch?"

#### Precision

**Definition:** Of all samples we predicted as positive, how many were actually positive?

```
Precision (p) = a / (a + c) = TP / (TP + FP)
```

- Focuses on the **column** of predicted positive class
- Answers: "Of all our YES predictions, how many were correct?"

#### F-Measure (F1-Score)

**Definition:** Harmonic mean of precision and recall - balances both metrics.

```
F-measure (F) = 2rp / (r + p) = 2a / (2a + b + c)
```

- Useful when you need a single metric that considers both precision and recall
- F1 = 1 is perfect, F1 = 0 is worst

#### Accuracy

**Definition:** Overall correctness of the model.

```
Accuracy = (a + d) / (a + b + c + d) = (TP + TN) / (TP + TN + FP + FN)
```

- Most widely-used metric
- Proportion of correct predictions out of all predictions

---

### 1.3 Limitation of Accuracy

**Problem:** Accuracy can be misleading with imbalanced datasets!

**Example:**
- Class 0 examples = 9990
- Class 1 examples = 10
- Total = 10,000

If a model predicts **everything as Class 0**:
- Accuracy = 9990/10000 = **99.9%**
- But the model fails to detect ANY Class 1 examples!

**Key Insight:** High accuracy doesn't mean the model is useful, especially when:
- Classes are imbalanced
- The cost of different errors varies
- The minority class is the important one (e.g., fraud detection, disease diagnosis)

---

### 1.4 Cost Matrix

Different misclassifications may have different costs.

**Cost Matrix Structure:**

|              | Predicted: Yes | Predicted: No |
|--------------|----------------|---------------|
| **Actual: Yes** | C(Yes\|Yes)    | C(No\|Yes)    |
| **Actual: No**  | C(Yes\|No)     | C(No\|No)     |

**C(i|j)** = Cost of misclassifying class j example as class i

---

### 1.5 Computing Cost of Classification

#### Example Problem:

**Cost Matrix:**
|        | Pred + | Pred - |
|--------|--------|--------|
| Act +  | -1     | 100    |
| Act -  | 1      | 0      |

**Model M1 Confusion Matrix:**
|        | Pred + | Pred - |
|--------|--------|--------|
| Act +  | 150    | 40     |
| Act -  | 60     | 250    |

**Model M2 Confusion Matrix:**
|        | Pred + | Pred - |
|--------|--------|--------|
| Act +  | 250    | 45     |
| Act -  | 5      | 200    |

**Calculations:**

**Model M1:**
- Accuracy = (150 + 250) / 500 = **80%**
- Cost = 150(-1) + 40(100) + 60(1) + 250(0) = -150 + 4000 + 60 + 0 = **3910**

**Model M2:**
- Accuracy = (250 + 200) / 500 = **90%**
- Cost = 250(-1) + 45(100) + 5(1) + 200(0) = -250 + 4500 + 5 + 0 = **4255**

**Key Insight:** M2 has higher accuracy (90% > 80%) but higher cost (4255 > 3910)!
- Sometimes the model with **lower accuracy is better** when costs are considered
- This demonstrates why accuracy alone isn't always sufficient

---

### 1.6 Weighted Accuracy

Assigns different weights to different outcomes:

```
Weighted Accuracy = (w1*a + w4*d) / (w1*a + w2*b + w3*c + w4*d)
```

Where:
- w1 = weight for True Positives
- w2 = weight for False Negatives
- w3 = weight for False Positives
- w4 = weight for True Negatives

---

## Quiz Problem 1: Evaluation Metrics

**Problem:** Spam email classification using Random Decision algorithm

**Data:** 100 spam emails, 200 non-spam emails

**Results (Confusion Matrix):**

|              | Pred: Spam | Pred: Non-spam |
|--------------|------------|----------------|
| **Spam**     | 70         | 30             |
| **Non-spam** | 20         | 180            |

**Calculate:** i) Recall, ii) Precision, iii) F-measure, iv) Accuracy

**Solution:**

Given: TP = 70, FN = 30, FP = 20, TN = 180

**i) Recall:**
```
Recall = TP / (TP + FN) = 70 / (70 + 30) = 70/100 = 0.70 or 70%
```

**ii) Precision:**
```
Precision = TP / (TP + FP) = 70 / (70 + 20) = 70/90 = 0.778 or 77.8%
```

**iii) F-measure:**
```
F = 2rp / (r + p) = 2(0.70)(0.778) / (0.70 + 0.778) = 1.089 / 1.478 = 0.737 or 73.7%

OR using the alternative formula:
F = 2a / (2a + b + c) = 2(70) / (2*70 + 30 + 20) = 140/190 = 0.737 or 73.7%
```

**iv) Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN) = (70 + 180) / (70 + 180 + 20 + 30)
         = 250/300 = 0.833 or 83.3%
```

---

## Quiz Problem 2: Multi-class Accuracy

**Problem:** Animal classification (3-class problem)

**Confusion Matrix:**

|          | Pred: Dog | Pred: Cat | Pred: Monkey |
|----------|-----------|-----------|--------------|
| **Dog**    | 80        | 10        | 5            |
| **Cat**    | 30        | 60        | 10           |
| **Monkey** | 10        | 8         | 82           |

**Calculate:** Accuracy

**Solution:**

For multi-class problems:
```
Accuracy = Sum of diagonal elements / Total samples
```

- Correct predictions (diagonal): 80 + 60 + 82 = 222
- Total samples: 80+10+5 + 30+60+10 + 10+8+82 = 95 + 100 + 100 = 295

```
Accuracy = 222/295 = 0.7525 or 75.25%
```

---

## 2. Methods for Performance Evaluation

**Goal:** Obtain reliable estimates of model performance

**Factors affecting performance:**
- Class distribution
- Cost of misclassification
- Size of training and test sets

---

### 2.1 Learning Curve

**Definition:** Shows how accuracy changes with varying sample size

**Characteristics:**
- X-axis: Sample size (often log scale)
- Y-axis: Accuracy
- Shows the effect of training data size on model performance

**Sampling Schedules:**
1. **Arithmetic sampling** (Langley et al.) - equal increments
2. **Geometric sampling** (Provost et al.) - multiplicative increments

**Effects of Small Sample Size:**
- **Bias** in the estimate
- **High variance** of estimate

**Interpretation:**
- Curve typically rises quickly then plateaus
- Large variance (error bars) at small sample sizes
- Converges to true performance as sample size increases

---

### 2.2 Holdout Method

**Process:** Split data into training and testing sets

**Common Splits:**
1. **2/3 - 1/3 split:** 2/3 for training, 1/3 for testing
2. **1/2 - 1/2 split:** Half for training, half for testing

```
[    Training (2/3)    |  Testing (1/3)  ]
```

**Advantages:**
- Simple and fast
- Good for large datasets

**Disadvantages:**
- Wastes data (test set not used for training)
- Results depend on the particular split
- High variance with small datasets

---

### 2.3 Cross-Validation

**Process:** Partition data into k disjoint subsets

**k-Fold Cross-Validation:**
1. Divide data into k equal-sized folds
2. For each fold i (1 to k):
   - Train on k-1 folds
   - Test on fold i
3. Average the k performance estimates

```
Fold 1: [ Test  ][Train][Train][Train]
Fold 2: [Train][ Test ][Train][Train]
Fold 3: [Train][Train][ Test ][Train]
Fold 4: [Train][Train][Train][ Test ]
```

**Common choices:** k = 5 or k = 10

**Leave-One-Out Cross-Validation (LOOCV):**
- Special case where k = n (number of samples)
- Each sample is used once as test set
- Most expensive but lowest bias

**Advantages:**
- All data used for both training and testing
- Lower variance than holdout
- Good estimate of true performance

---

### 2.4 Bootstrap Sampling

**Process:** Sampling **with replacement**

**Procedure:**
1. From dataset of n samples, randomly draw n samples with replacement
2. Some samples appear multiple times, others don't appear
3. Use selected samples for training
4. Use non-selected samples (out-of-bag) for testing

```
Original Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Bootstrap Sample: [2, 5, 5, 3, 8, 1, 1, 9, 3, 7]  (Training)
Out-of-bag: [4, 6, 10]  (Testing)
```

**Probability of not being selected:**
- P(not selected in one draw) = (n-1)/n
- P(not selected in n draws) = ((n-1)/n)^n ≈ 1/e ≈ 0.368 for large n

So approximately 63.2% of samples appear in training set.

---

### 2.5 Stratified Sampling

**Purpose:** Maintain class distribution in train/test splits

**Process:**
- Sample from each class separately
- Ensure same proportion of each class in training and testing

**Techniques for Imbalanced Data:**
1. **Oversampling:** Duplicate minority class samples
2. **Undersampling:** Remove majority class samples

```
Original: [Class A: 900][Class B: 100]
                    ↓ Stratified split
Training:  [Class A: 600][Class B: 67]  (same ratio maintained)
Testing:   [Class A: 300][Class B: 33]
```

---

### Summary: Evaluation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Learning Curve | Vary sample size | Understanding data needs |
| Hold-out | Single train/test split | Large datasets |
| Cross-validation | k-fold rotation | Medium datasets |
| Bootstrap | Sampling with replacement | Small datasets |
| Stratified sampling | Preserve class ratios | Imbalanced data |

---

## 3. Methods for Model Comparison

### 3.1 ROC Curve (Receiver Operating Characteristic)

**History:** Developed in 1950s for signal detection theory to analyze noisy signals

**Purpose:** Characterize the trade-off between positive hits and false alarms

**Definition:** ROC curve plots **True Positive Rate (TPR)** on y-axis against **False Positive Rate (FPR)** on x-axis

**Rates:**
```
TPR (True Positive Rate) = TP / (TP + FN) = Recall = Sensitivity
FPR (False Positive Rate) = FP / (FP + TN) = 1 - Specificity
```

---

### 3.2 How to Construct an ROC Curve

**Process:**
1. Use a classifier that produces posterior probability P(+|A) for each instance
2. Sort instances by P(+|A) in decreasing order
3. For each unique probability value as threshold:
   - Count TP, FP, TN, FN
   - Calculate TPR and FPR
4. Plot (FPR, TPR) points

#### Detailed Example:

**Given Data (sorted by P(+|A)):**

| Instance | P(+\|A) | True Class |
|----------|---------|------------|
| 1        | 0.95    | +          |
| 2        | 0.93    | +          |
| 3        | 0.87    | -          |
| 4        | 0.85    | -          |
| 5        | 0.85    | -          |
| 6        | 0.85    | +          |
| 7        | 0.76    | -          |
| 8        | 0.53    | +          |
| 9        | 0.43    | -          |
| 10       | 0.25    | +          |

**Total:** 5 positive (+), 5 negative (-)

**ROC Table (for each threshold):**

| Threshold >= | TP | FP | TN | FN | TPR | FPR |
|--------------|----|----|----|----|-----|-----|
| 1.00         | 0  | 0  | 5  | 5  | 0   | 0   |
| 0.95         | 1  | 0  | 5  | 4  | 0.2 | 0   |
| 0.93         | 2  | 0  | 5  | 3  | 0.4 | 0   |
| 0.87         | 2  | 1  | 4  | 3  | 0.4 | 0.2 |
| 0.85         | 3  | 3  | 2  | 2  | 0.6 | 0.6 |
| 0.76         | 3  | 4  | 1  | 2  | 0.6 | 0.8 |
| 0.53         | 4  | 4  | 1  | 1  | 0.8 | 0.8 |
| 0.43         | 4  | 5  | 0  | 1  | 0.8 | 1   |
| 0.25         | 5  | 5  | 0  | 0  | 1   | 1   |

**Plot points:** (0,0), (0,0.2), (0,0.4), (0.2,0.4), (0.6,0.6), (0.8,0.6), (0.8,0.8), (1,0.8), (1,1)

---

### 3.3 Understanding the ROC Curve

**Key Points on ROC Space:**

| Point | (FPR, TPR) | Meaning |
|-------|------------|---------|
| (0, 0) | Origin | Declare everything negative |
| (1, 1) | Top-right | Declare everything positive |
| (0, 1) | Top-left | **IDEAL** - Perfect classifier |
| Diagonal | y = x | Random guessing |

**Interpretation:**
- **Above diagonal:** Better than random
- **On diagonal:** Random guessing (useless classifier)
- **Below diagonal:** Worse than random (flip predictions to improve!)
- **Closer to top-left:** Better classifier

**What changes the position on ROC curve:**
- Changing the threshold
- Changing sample distribution
- Changing cost matrix

---

### 3.4 Using ROC for Model Comparison

**Comparing Two Models:**
- If one curve is always above another, that model is better
- If curves cross, performance depends on the operating point (threshold)

**Example Interpretation:**
- Model M1 might be better for small FPR (high precision needed)
- Model M2 might be better for large FPR (high recall needed)

---

### 3.5 Area Under the ROC Curve (AUC)

**Definition:** Single scalar value summarizing ROC curve performance

**Interpretation:**
| AUC Value | Meaning |
|-----------|---------|
| 1.0 | Perfect classifier |
| 0.5 | Random guessing |
| < 0.5 | Worse than random |
| 0.7-0.8 | Acceptable |
| 0.8-0.9 | Excellent |
| > 0.9 | Outstanding |

**Advantages of AUC:**
- Threshold-independent
- Scale-invariant
- Good for comparing models across different datasets

---

## Quiz Problem 3: ROC Comparison

**Problem:** For spam email classification, three models have different ROC curves. Which is best?

**Answer:** The model with the **highest AUC** (Area Under Curve) is generally best. If curves cross:
- Choose based on your operating requirements
- High precision needed (low FPR) → choose model that's higher on the left
- High recall needed (high TPR) → choose model that's higher overall

---

## Summary

### Metrics for Performance Evaluation
- **Confusion Matrix:** Foundation for all metrics
- **Recall:** TP/(TP+FN) - sensitivity to positives
- **Precision:** TP/(TP+FP) - accuracy of positive predictions
- **F-measure:** Harmonic mean of precision and recall
- **Accuracy:** (TP+TN)/Total - can be misleading with imbalanced data
- **Cost-sensitive metrics:** When different errors have different costs

### Methods for Performance Evaluation
1. **Learning curve:** Study effect of sample size
2. **Hold-out:** Simple train/test split
3. **Cross-validation:** k-fold rotation for robust estimates
4. **Bootstrap:** Sampling with replacement
5. **Stratified sampling:** Preserve class distribution

### Methods for Model Comparison
- **ROC Curve:** Trade-off visualization between TPR and FPR
- **AUC:** Single metric for overall comparison
- **Key insight:** (0,1) is ideal, diagonal is random, above diagonal is good

---

## Key Formulas Quick Reference

| Metric | Formula |
|--------|---------|
| Recall (TPR) | TP / (TP + FN) |
| Precision | TP / (TP + FP) |
| F-measure | 2TP / (2TP + FP + FN) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| FPR | FP / (FP + TN) |
| Specificity | TN / (TN + FP) = 1 - FPR |

---

## QUICK REVIEW CHECKLIST

### Confusion Matrix & Metrics
- [ ] Can I draw a confusion matrix with TP, FP, TN, FN in correct positions?
- [ ] Can I calculate Recall = TP/(TP+FN) from a confusion matrix?
- [ ] Can I calculate Precision = TP/(TP+FP) from a confusion matrix?
- [ ] Can I calculate F-measure = 2TP/(2TP+FP+FN)?
- [ ] Do I understand why accuracy fails with imbalanced data (99.9% accuracy with 9990:10 ratio)?
- [ ] Can I identify Type I error (FP) vs Type II error (FN)?

### Cost-Sensitive Classification
- [ ] Can I apply a cost matrix to calculate expected cost?
- [ ] Do I understand how to modify predictions when FP and FN have different costs?

### Performance Evaluation Methods
- [ ] Can I explain holdout method (train/test split)?
- [ ] Can I explain k-fold cross-validation and why it's more robust?
- [ ] Do I understand stratified sampling (preserving class distribution)?
- [ ] Can I explain bootstrap sampling (with replacement)?
- [ ] Do I know what a learning curve shows (performance vs training set size)?

### ROC Curve & AUC
- [ ] Can I construct an ROC curve from probability predictions?
- [ ] Do I know what (0,0), (1,1), (0,1) represent on ROC curve?
- [ ] Do I understand that diagonal = random guessing?
- [ ] Can I interpret AUC values (0.5 = random, 1.0 = perfect)?
- [ ] Do I know how to compare models using ROC curves?

### Key Questions to Test Understanding
- [ ] Why is F-measure better than accuracy for imbalanced data?
- [ ] What does TPR (True Positive Rate) mean?
- [ ] What does FPR (False Positive Rate) mean?
- [ ] Why might we want high recall vs high precision in different scenarios?
- [ ] What happens to the ROC point when we change the classification threshold?

---

*END OF NOTES*
