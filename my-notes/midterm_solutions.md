# CS653: Data Mining - Midterm Solutions
## Fall 2025

---

## Question 1: Which systems are NOT data mining?

Answer "Yes" if NOT data mining, "No" if it IS data mining:

| System | Answer | Reasoning |
|--------|--------|-----------|
| (a) Group customers by gender | **Yes** | Simple query/filtering, not pattern discovery |
| (b) Recommend products | **No** | Recommendation systems use data mining |
| (c) Sort by student ID | **Yes** | Simple database operation |
| (d) Calculate total sales | **Yes** | Simple aggregation |
| (e) Extract sound frequencies | **Yes** | Signal processing, not data mining |
| (f) Predict stock prices | **No** | Predictive modeling |
| (g) Monitor heart rate for abnormalities | **No** | Anomaly detection |
| (h) Predict seismic waves | **No** | Predictive modeling |
| (i) Identify fraud transactions | **No** | Anomaly detection |

---

## Question 2: Box Plot Information

A box plot provides:

- **Median**: The line inside the box (50th percentile)
- **Q1 and Q3**: The box edges (25th and 75th percentiles)
- **Interquartile Range (IQR)**: Height of the box (Q3-Q1), showing spread of middle 50%
- **Whiskers**: Extend to min/max values within 1.5×IQR
- **Outliers**: Points beyond whiskers
- **Skewness**: Asymmetry in box or whisker lengths
- **Comparison**: Allows comparing distributions across features (sepal length has largest range, petal width has smallest)

---

## Question 3: Decision Tree Predictions

Following the tree: `Refund → (Yes→N) or (No→MarSt→(Married→N) or (Single/Divorced→TaxInc→(<80K→N, >80K→YES)))`

| Refund | MarSt | TaxInc | Ground-truth | **Prediction** |
|--------|-------|--------|--------------|----------------|
| Yes | Divorced | 10 | Y | **N** |
| No | Married | 82 | N | **N** |
| No | Single | 60 | N | **N** |
| No | Divorced | 100 | N | **YES** |
| No | Divorced | 70 | Y | **N** |
| No | Single | 90 | N | **YES** |
| No | Married | 89 | Y | **N** |

---

## Question 4: Confusion Matrix & Precision/Recall

### Confusion Matrix:

|  | Predicted Y | Predicted N |
|--|-------------|-------------|
| **Actual Y** | TP = 0 | FN = 3 |
| **Actual N** | FP = 2 | TN = 2 |

### Per-class metrics:

| Class | Precision | Recall |
|-------|-----------|--------|
| Y (Cheater) | 0/(0+2) = **0** | 0/(0+3) = **0** |
| N (Non-cheater) | 2/(2+3) = **0.4** | 2/(2+2) = **0.5** |

---

## Question 5: Gini Index for Overall Collection

Total samples: 20 (C0: 10, C1: 10)

**GINI = 1 - Σ(pᵢ)²**

```
= 1 - (10/20)² - (10/20)²
= 1 - 0.25 - 0.25
= 0.5
```

**Answer: GINI = 0.5**

---

## Question 6: GINI Split Calculations

### (a) Split by Gender:

**M**: 10 samples (C0: 7, C1: 3)
```
GINI(M) = 1 - (7/10)² - (3/10)² = 1 - 0.49 - 0.09 = 0.42
```

**F**: 10 samples (C0: 3, C1: 7)
```
GINI(F) = 1 - (3/10)² - (7/10)² = 0.42
```

**GINI_split(a) = (10/20)×0.42 + (10/20)×0.42 = 0.42**

### (b) Split by Car Type:

**Luxury**: 8 samples (C0: 1, C1: 7)
```
GINI(Luxury) = 1 - (1/8)² - (7/8)² = 1 - 0.0156 - 0.7656 = 0.2188
```

**Family+Sports**: 12 samples (C0: 9, C1: 3)
```
GINI(F+S) = 1 - (9/12)² - (3/12)² = 1 - 0.5625 - 0.0625 = 0.375
```

**GINI_split(b) = (8/20)×0.2188 + (12/20)×0.375 = 0.0875 + 0.225 = 0.3125**

### Conclusion:
**(b) is the better split** (lower GINI: 0.3125 < 0.42)

---

## Question 7: K-NN Classification

### Distances from x=5.0 (sorted):

| x | Distance | y |
|---|----------|---|
| 4.9 | 0.1 | + |
| 5.2 | 0.2 | - |
| 5.3 | 0.3 | - |
| 4.6 | 0.4 | + |
| 5.5 | 0.5 | + |
| 4.2 | 0.8 | + |
| 3.0 | 2.0 | - |
| 7.0 | 2.0 | - |
| 0.5 | 4.5 | - |
| 9.5 | 4.5 | - |

### Classifications:

- **1-NN**: Nearest is 4.9 (+) → Predict **+**
- **3-NN**: +, -, - → 2 negative → Predict **-**
- **5-NN**: +, -, -, +, + → 3 positive → Predict **+**

### Discussion of K value:

**Small K:**
- More sensitive to noise and outliers
- Creates complex, irregular decision boundaries
- Risk of overfitting
- High variance, low bias

**Large K:**
- Smoother, more stable decision boundaries
- More robust to noise
- May include irrelevant distant points
- Risk of underfitting
- Low variance, high bias

---

## Question 8: Naïve Bayes

### (a) Conditional Probabilities:

**Class distribution:**
- Class +: Records 1, 5, 6, 9, 10 (5 samples)
- Class -: Records 2, 3, 4, 7, 8 (5 samples)

| Probability | Value |
|-------------|-------|
| P(A=1\|+) | 3/5 = 0.6 |
| P(A=0\|+) | 2/5 = 0.4 |
| P(B=1\|+) | 2/5 = 0.4 |
| P(B=0\|+) | 3/5 = 0.6 |
| P(C=1\|+) | 3/5 = 0.6 |
| P(C=0\|+) | 2/5 = 0.4 |
| P(A=1\|-) | 2/5 = 0.4 |
| P(A=0\|-) | 3/5 = 0.6 |
| P(B=1\|-) | 2/5 = 0.4 |
| P(B=0\|-) | 3/5 = 0.6 |
| P(C=1\|-) | 5/5 = 1.0 |
| P(C=0\|-) | 0/5 = 0.0 |

### (b) Predict class for (A=0, B=1, C=0):

**For class +:**
```
P(+) × P(A=0|+) × P(B=1|+) × P(C=0|+)
= (1/2) × (2/5) × (2/5) × (2/5)
= (1/2) × (8/125)
= 8/250 = 0.032
```

**For class -:**
```
P(-) × P(A=0|-) × P(B=1|-) × P(C=0|-)
= (1/2) × (3/5) × (2/5) × (0/5)
= (1/2) × (3/5) × (2/5) × 0
= 0
```

**Prediction: + (positive class)**

> **Note:** P(C=0|-) = 0 makes the entire negative class probability zero. In practice, Laplace smoothing would be applied to avoid this zero-probability issue.
