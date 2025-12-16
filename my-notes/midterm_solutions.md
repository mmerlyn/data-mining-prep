# CS653: Data Mining - Midterm Solutions
## Fall 2025

---

## Question 1: Which systems are NOT data mining?

> **Question:** Please read the following statements each describing a computer system, and identify which system(s) is (are) not addressed by data mining techniques. Put 'Yes' or 'No' after each activity's number in your answers (e.g., a: No).
>
> (a) A system that can group the customers of a company according to their genders
> (b) A system that can intelligently recommend a new product to existing customers who are likely to buy it
> (c) A system that can sort a student database based on student identification numbers
> (d) A system that can calculate the total sales of a company given the sale records
> (e) A system that can extract the frequencies of a sound wave
> (f) A system that can predict the future stock price of a company using its historical records
> (g) A system that can monitor the heart rate of a patient for abnormalities
> (h) A system that can predict seismic waves for earthquake activities
> (i) A system that can identify fraud transactions which are substantially different from other transactions

### Answer:
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

> **Question:** The following figure shows the box-plot of the Iris flower dataset, where each flower includes 4 attributes: sepal length, sepal width, petal length, petal width. Describe how a box plot can give information about each feature.

### Answer:
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

> **Question:** Suppose you have trained a decision tree to predict if a user has cheated on tax claim. Please apply the tree model to the following test samples and fill in the predictions (Y or N) in the last column.

### Answer:

**Decision Tree Diagram:**
```
                    ┌─────────┐
                    │ Refund  │
                    └────┬────┘
              ┌──────────┴──────────┐
            Yes                     No
              │                      │
           ┌──┴──┐              ┌────┴────┐
           │  N  │              │  MarSt  │
           └─────┘              └────┬────┘
                          ┌──────────┴──────────┐
                    Single,                  Married
                    Divorced                    │
                         │                   ┌──┴──┐
                    ┌────┴────┐              │  N  │
                    │ TaxInc  │              └─────┘
                    └────┬────┘
              ┌──────────┴──────────┐
            <80K                  >80K
              │                      │
           ┌──┴──┐              ┌────┴────┐
           │  N  │              │   YES   │
           └─────┘              └─────────┘
```

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

> **Question:** Based on your predictions in the last question, please calculate the confusion matrix, and per-class precision and recall.

### Answer:

**Confusion Matrix:**

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

> **Question:** Consider the training samples shown in the table for a binary classification problem (20 customers with Gender, Car Type, Shirt Size attributes, and Class C0/C1). Compute the Gini Index (i.e., GINI) for the overall collection of training examples.

**Training Data Summary:**
| Customer ID | Gender | Car Type | Shirt Size | Class |
|-------------|--------|----------|------------|-------|
| 1-6 | M | Family/Sports | Various | C0 |
| 7-9 | F | Sports | Small/Medium | C0 |
| 10 | M | Luxury | Large | C0 |
| 11-13 | M | Family | Various | C1 |
| 14-20 | F | Luxury | Various | C1 |

### Answer:
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

> **Question:** When a set of samples is split into k partitions (children), the quality of this split is computed as:
>
> **GINI_split = Σ (nᵢ/n) × GINI(i)**
>
> where nᵢ is the number of samples at children i, n is the total number of samples.
>
> Please calculate the qualities of the following two ways of splitting:
> - (a) Split by Gender: M vs F
> - (b) Split by Car Type: Luxury vs {Family, Sports}

### Answer:

**(a) Split by Gender:**

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

> **Question:** Consider the one-dimensional data set shown in the following table. Classify the data point x = 5.0 according to its 1-, 3-, and 5- nearest neighbors (using majority vote). Please briefly discuss the consequences of using small or large K for K-NN (K-nearest neighbor method).
>
> | x | 0.5 | 3.0 | 4.2 | 4.6 | 4.9 | 5.2 | 5.3 | 5.5 | 7.0 | 9.5 |
> |---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
> | y | - | - | + | + | + | - | - | + | - | - |

### Answer:

**Distances from x=5.0 (sorted):**

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

> **Question:** Consider the data set shown in the following table. We will apply the Naïve Bayes method to predict the class of an unseen record. Please accomplish the following two steps:
>
> (a) Estimate the conditional probabilities: P(A|+), P(B|+), P(C|+), P(A|-), P(B|-), and P(C|-).
>
> (b) Use the estimate of conditional probabilities to predict the class label for a test sample (A=0, B=1, C=0) using the Naïve Bayes method.

### (a) Conditional Probabilities:

**Class distribution:**
- Class +: Records 1, 5, 6, 9, 10 (5 samples)
- Class -: Records 2, 3, 4, 7, 8 (5 samples)

**From the data table:**
| Record | A | B | C | Class |
|--------|---|---|---|-------|
| 1 | 0 | 0 | 0 | + |
| 2 | 0 | 0 | 1 | - |
| 3 | 0 | 1 | 1 | - |
| 4 | 0 | 1 | 1 | - |
| 5 | 0 | 0 | 1 | + |
| 6 | 1 | 0 | 1 | + |
| 7 | 1 | 0 | 1 | - |
| 8 | 1 | 0 | 1 | - |
| 9 | 1 | 1 | 1 | + |
| 10 | 1 | 0 | 1 | + |

| Probability | Value |
|-------------|-------|
| P(A=1\|+) | 3/5 = 0.6 |
| P(A=0\|+) | 2/5 = 0.4 |
| P(B=1\|+) | 1/5 = 0.2 |
| P(B=0\|+) | 4/5 = 0.8 |
| P(C=1\|+) | 4/5 = 0.8 |
| P(C=0\|+) | 1/5 = 0.2 |
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
= (1/2) × (2/5) × (1/5) × (1/5)
= (1/2) × (2/125)
= 2/250 = 1/125 = 0.008
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
