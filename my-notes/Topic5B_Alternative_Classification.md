# TOPIC 5B: ALTERNATIVE CLASSIFICATION TECHNIQUES
### CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. Artificial Neural Networks (ANN)
2. Support Vector Machines (SVM)
3. Naive Bayes Classifier
4. Ensemble Methods & Boosting
5. Worked Examples & Problems
6. Python Code Examples
7. Key Takeaways for Exam


---

## 1. ARTIFICIAL NEURAL NETWORKS (ANN)

### OVERVIEW
ANN is inspired by biological neural networks. It's a "black box" model that:
- Takes multiple inputs
- Processes them through interconnected nodes
- Produces output(s)

### EXAMPLE PROBLEM (from slides):

Given inputs X1, X2, X3 and output Y:

| X1 | X2 | X3 | Y |
|----|----|----|---|
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 0 | 0 | 0 | 0 |

Rule: Output Y is 1 if AT LEAST TWO of the three inputs equal 1.


### PERCEPTRON MODEL (Single Neuron)

The simplest neural network - a single neuron.

**Components:**
- Input nodes: X1, X2, ..., Xn
- Weights: w1, w2, ..., wn
- Threshold: t
- Output: Y

**Mathematical Model:**
```
    Y = I(sum(wi * Xi) - t > 0)

    where I(z) = { 1  if z is true
                 { 0  otherwise
```

OR equivalently:
```
    Y = sign(sum(wi * Xi) - t)
```


### EXAMPLE CALCULATION:

With weights w1=0.3, w2=0.3, w3=0.3 and threshold t=0.4:
```
    Y = I(0.3*X1 + 0.3*X2 + 0.3*X3 - 0.4 > 0)
```

For input (1, 1, 0):
```
    = I(0.3*1 + 0.3*1 + 0.3*0 - 0.4 > 0)
    = I(0.6 - 0.4 > 0)
    = I(0.2 > 0)
    = 1 (TRUE)
```


### GENERAL STRUCTURE OF ANN

Multi-layer neural network consists of:

1. **INPUT LAYER:** Receives the feature values (x1, x2, ..., xn)
2. **HIDDEN LAYER(S):** One or more layers of neurons
3. **OUTPUT LAYER:** Produces final classification (y)

Each neuron i:
- Receives inputs I1, I2, I3, ... with weights wi1, wi2, wi3, ...
- Computes weighted sum: Si = sum(Ij * wij)
- Applies ACTIVATION FUNCTION: Oi = g(Si)
- Compares against threshold t

**ACTIVATION FUNCTIONS:**
- Step function (perceptron)
- Sigmoid function
- ReLU (Rectified Linear Unit)
- Tanh


### TRAINING ANN - BACKPROPAGATION

Training means LEARNING THE WEIGHTS of neurons.

**Algorithm:**
1. Initialize weights (w0, w1, ..., wk) - often randomly

2. Adjust weights so output is consistent with training labels

3. Objective Function (Error/Loss):
```
   E = sum_i [Yi - f(wi, Xi)]^2
```
   where Yi = true label, f(wi, Xi) = predicted output

4. Find weights that MINIMIZE the objective function
   - Use backpropagation algorithm
   - Gradient descent to update weights


### KEY CHARACTERISTICS OF ANN:
- Non-linear decision boundaries
- Can model complex relationships
- "Black box" - difficult to interpret
- Requires careful tuning (learning rate, architecture)
- Can overfit with too many hidden nodes


---

## 2. SUPPORT VECTOR MACHINES (SVM)

### CORE CONCEPT
Find a linear HYPERPLANE (decision boundary) that separates the data.

Key Question: Which hyperplane is "best"?
Answer: The one that MAXIMIZES THE MARGIN between classes.


### MARGIN DEFINITION
The margin is the distance between the decision boundary and the
NEAREST data points from each class.

These nearest points are called SUPPORT VECTORS.


### MATHEMATICAL FORMULATION

Decision boundary (hyperplane):  w . x + b = 0

Where:
- w = weight vector (normal to hyperplane)
- x = input vector
- b = bias term

**Classification Rule:**
```
    f(x) = { +1  if w . x + b >= 1
           { -1  if w . x + b <= -1
```

**Margin boundaries:**
- w . x + b = +1  (for positive class)
- w . x + b = -1  (for negative class)

**MARGIN FORMULA:**
```
    Margin = 2 / ||w||^2
```


### OPTIMIZATION PROBLEM

To MAXIMIZE margin, we need to MINIMIZE:
```
    L(w) = ||w||^2 / 2
```

Subject to constraints:
- f(xi) = +1  if w . xi + b >= 1  (positive class)
- f(xi) = -1  if w . xi + b <= -1 (negative class)

This is a CONSTRAINED OPTIMIZATION PROBLEM
Solved using: Quadratic Programming


### SOFT MARGIN SVM (Non-linearly Separable Data)

What if data is NOT perfectly linearly separable?

**Solution:** Introduce SLACK VARIABLES (ξi)

**Modified Objective Function:**
```
    L(w) = ||w||^2 / 2 + C * sum(ξi^k)
```

Where:
- C = penalty parameter (trade-off between margin and misclassification)
- ξi = slack variable for point i
- k = typically 1 or 2

**Modified Constraints:**
- f(xi) = +1  if w . xi + b >= 1 - ξi
- f(xi) = -1  if w . xi + b <= -1 + ξi


### NONLINEAR SVM (Kernel Trick)

What if decision boundary is NOT linear?

**Solution:** Transform data into HIGHER DIMENSIONAL SPACE where
it becomes linearly separable.

Example from slides:
- Original space: 2D with (x1, x2)
- Transformed space: Add feature (x1 + x2)^4
- In new space, data becomes linearly separable

**Common Kernel Functions:**
- Linear: K(x, y) = x . y
- Polynomial: K(x, y) = (x . y + c)^d
- RBF (Gaussian): K(x, y) = exp(-||x - y||^2 / 2σ^2)
- Sigmoid: K(x, y) = tanh(α * x . y + c)


### KEY CHARACTERISTICS OF SVM:
- Effective in high-dimensional spaces
- Memory efficient (uses only support vectors)
- Versatile through kernel functions
- Works well with clear margin of separation
- Not suitable for large datasets (training can be slow)
- Sensitive to feature scaling


---

## 3. NAIVE BAYES CLASSIFIER

### BAYES THEOREM REVIEW

**Conditional Probability:**
```
    P(C|A) = P(A,C) / P(A)
    P(A|C) = P(A,C) / P(C)
```

**Bayes Theorem:**
```
    P(C|A) = P(A|C) * P(C) / P(A)
```

Where:
- P(C|A) = Posterior probability (probability of class given attributes)
- P(A|C) = Likelihood (probability of attributes given class)
- P(C) = Prior probability (probability of class)
- P(A) = Evidence (probability of attributes)


### QUIZ EXAMPLE (Meningitis - from slides):

Given:
- P(Stiff Neck | Meningitis) = 0.5 (50% of meningitis patients have stiff neck)
- P(Meningitis) = 1/50,000 (prior probability)
- P(Stiff Neck) = 1/20 (prior probability)

Question: If patient has stiff neck, what's P(Meningitis)?

**Solution:**
```
    P(M|S) = P(S|M) * P(M) / P(S)
           = (0.5 * 1/50000) / (1/20)
           = (0.5 / 50000) / 0.05
           = 0.00001 / 0.05
           = 0.0002 = 0.02%
```


### BAYESIAN CLASSIFIER

Given record with attributes (A1, A2, ..., An):
- Goal: Predict class C
- Find value of C that MAXIMIZES P(C | A1, A2, ..., An)

Using Bayes Theorem:
```
    P(C | A1, A2, ..., An) = P(A1, A2, ..., An | C) * P(C) / P(A1, A2, ..., An)
```

Since P(A1, A2, ..., An) is constant for all classes:

**CHOOSE C that MAXIMIZES:**  P(A1, A2, ..., An | C) * P(C)


### THE "NAIVE" ASSUMPTION

**Problem:** Estimating P(A1, A2, ..., An | C) requires huge amounts of data.

**Naive Assumption:** Attributes are CONDITIONALLY INDEPENDENT given the class.
```
    P(A1, A2, ..., An | C) = P(A1|C) * P(A2|C) * ... * P(An|C)
                          = Product of P(Ai|C) for all i
```

This is "naive" because attributes are rarely truly independent.
But it works surprisingly well in practice!


### ESTIMATING PROBABILITIES FROM DATA

**For Class Prior:**
```
    P(C) = Nc / N
```
    where Nc = number of records with class C, N = total records

**For Discrete Attributes:**
```
    P(Ai = v | Ck) = |Aik| / Nck
```
    where |Aik| = records with attribute Ai=v AND class Ck

**For Continuous Attributes:**
Three approaches:
1. DISCRETIZE: Convert to bins (may violate independence)
2. TWO-WAY SPLIT: (A < v) or (A > v)
3. PROBABILITY DENSITY: Assume normal distribution

**For Normal Distribution:**
```
    P(Ai | Cj) = (1 / sqrt(2π * σij^2)) * exp(-(Ai - μij)^2 / (2σij^2))
```
Where μij = mean, σij = standard deviation for attribute Ai in class Cj


### COMPLETE EXAMPLE (Tax Evader - from slides):

**Dataset:**

| Tid | Refund | Marital Status | Taxable Income | Evade |
|-----|--------|----------------|----------------|-------|
| 1 | Yes | Single | 125K | No |
| 2 | No | Married | 100K | No |
| 3 | No | Single | 70K | No |
| 4 | Yes | Married | 120K | No |
| 5 | No | Divorced | 95K | Yes |
| 6 | No | Married | 60K | No |
| 7 | Yes | Divorced | 220K | No |
| 8 | No | Single | 85K | Yes |
| 9 | No | Married | 75K | No |
| 10 | No | Single | 90K | Yes |

**Calculated Probabilities:**

Class Priors:
- P(No) = 7/10 = 0.7
- P(Yes) = 3/10 = 0.3

Refund:
- P(Refund=Yes|No) = 3/7
- P(Refund=No|No) = 4/7
- P(Refund=Yes|Yes) = 0
- P(Refund=No|Yes) = 1

Marital Status:
- P(Single|No) = 2/7
- P(Divorced|No) = 1/7
- P(Married|No) = 4/7
- P(Single|Yes) = 2/3
- P(Divorced|Yes) = 1/3
- P(Married|Yes) = 0

Taxable Income (continuous - assume normal):
- If Class=No: mean=110, variance=2975
- If Class=Yes: mean=90, variance=25

### CLASSIFYING NEW RECORD:

Test Record X = (Refund=No, Married, Income=120K)

**Calculate P(X|No):**
```
    = P(Refund=No|No) * P(Married|No) * P(Income=120K|No)
    = (4/7) * (4/7) * P(120K|No)
```

For Income=120K given No:
```
    P(120|No) = (1/sqrt(2π*2975)) * exp(-(120-110)^2 / (2*2975))
              = (1/sqrt(2π*54.54)) * exp(-100/5950)
              = 0.0072
```

So: P(X|No) = (4/7) * (4/7) * 0.0072 = 0.0024

**Calculate P(X|Yes):**
```
    = P(Refund=No|Yes) * P(Married|Yes) * P(Income=120K|Yes)
    = 1 * 0 * (some value)
    = 0
```

Since P(X|No)*P(No) > P(X|Yes)*P(Yes):
    0.0024 * 0.7 > 0 * 0.3

Therefore: P(No|X) > P(Yes|X) => **Class = NO**


### MAMMAL CLASSIFICATION EXAMPLE (from slides):

Dataset with attributes: Give Birth, Can Fly, Live in Water, Have Legs

Test Record: (Give Birth=yes, Can Fly=no, Live in Water=yes, Have Legs=no)

Calculated:
```
    P(A|Mammals) = (6/7) * (6/7) * (2/7) * (2/7) = 0.06
    P(A|Non-mammals) = (1/13) * (10/13) * (3/13) * (4/13) = 0.0042

    P(A|M)*P(M) = 0.06 * (7/20) = 0.021
    P(A|N)*P(N) = 0.004 * (13/20) = 0.0027
```

Since 0.021 > 0.0027 => **Class = MAMMALS**


### NAIVE BAYES SUMMARY:

**Advantages:**
- Robust to isolated noise points
- Handles missing values (ignore during probability calculation)
- Robust to irrelevant attributes
- Fast training and prediction
- Works well with high-dimensional data

**Disadvantages:**
- Independence assumption may not hold
- Zero probability problem (if P(Ai|C) = 0, entire product = 0)
  - Solution: Laplace smoothing

Alternative: Bayesian Belief Networks (BBN) for correlated attributes


---

## 4. ENSEMBLE METHODS & BOOSTING

### ENSEMBLE METHODS - CONCEPT

Instead of using ONE classifier, use MULTIPLE classifiers and combine predictions.

**General Idea:**
1. Create multiple data sets (D1, D2, ..., Dt) from original data D
2. Build multiple classifiers (C1, C2, ..., Ct)
3. Combine classifiers into final classifier C*

**Why it works:**
- If we have 25 independent classifiers, each with error rate ε = 0.35
- Probability of ensemble making wrong prediction:
```
  P(>12 wrong) = sum(i=13 to 25) C(25,i) * ε^i * (1-ε)^(25-i) = 0.06
```

Much better than individual error of 0.35!


### BAGGING (Bootstrap Aggregating)

**Method:** Sampling WITH REPLACEMENT

Example:

| Original Data | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---------------|---|---|---|---|---|---|---|---|---|-----|
| Bagging (Round 1) | 7 | 8 |10 | 8 | 2 | 5 |10 |10 | 5 | 9 |
| Bagging (Round 2) | 1 | 4 | 9 | 1 | 2 | 3 | 2 | 7 | 3 | 2 |
| Bagging (Round 3) | 1 | 8 | 5 |10 | 5 | 5 | 9 | 6 | 3 | 7 |

- Build classifier on each bootstrap sample
- Each sample has probability (1 - 1/n)^n ≈ 0.368 of being selected
- About 63.2% of original data appears in each sample


### BOOSTING (AdaBoost)

**Key Idea:** Focus on MISCLASSIFIED records in subsequent rounds.

**Procedure:**
1. Initially, all N records have EQUAL weights: wi = 1/N
2. Build classifier on weighted data
3. Increase weights of MISCLASSIFIED records
4. Decrease weights of CORRECTLY CLASSIFIED records
5. Repeat for T rounds


### ADABOOST ALGORITHM - DETAILED EXAMPLE

**Setup (from slides):**
- 2 dimensions
- 10 training samples (+5 positive, -5 negative)
- Weak classifiers: horizontal or vertical lines
- Initial weight: D(i) = 1/m = 1/10 = 0.1

**ITERATION 1:**

1. Select weak classifier h1 that minimizes weighted error
2. Classifier h1 misclassifies 3 samples
3. Calculate weighted error rate:
```
   ε1 = (sum of D(i) for misclassified) / (sum of all D(i))
      = (3 * 0.1) / 1.0
      = 0.3
```

4. Calculate classifier importance (weight):
```
   α1 = (1/2) * ln((1 - ε1) / ε1)
      = (1/2) * ln(0.7 / 0.3)
      = (1/2) * ln(2.33)
      = 0.42
```

5. Update sample weights:
   - Misclassified: w_new = w_old * exp(α1)  [INCREASE]
   - Correct: w_new = w_old * exp(-α1)       [DECREASE]
   - Normalize so weights sum to 1

**ITERATION 2:**

1. Select weak classifier h2 (different from h1)
2. h2 misclassifies 3 samples (but they have smaller weights now)
3. ε2 = 0.21 (weighted error rate)
4. α2 = 0.65 (higher than α1 because lower error)
5. Update weights again

**ITERATION 3:**

1. Select weak classifier h3
2. ε3 = 0.14
3. α3 = 0.92 (highest because lowest error)

**STOPPING:** Error rate 0.14 is "good enough"

**FINAL CLASSIFIER (Strong Classifier):**
```
    H_final(x) = sign(α1*h1(x) + α2*h2(x) + α3*h3(x))
               = sign(0.42*h1(x) + 0.65*h2(x) + 0.92*h3(x))
```


### ADABOOST FORMULAS SUMMARY

**Error Rate:**
```
    εi = (1/N) * sum(wj * δ(Ci(xj) ≠ yj))
```
    where δ = 1 if misclassified, 0 otherwise

**Classifier Importance:**
```
    αi = (1/2) * ln((1 - εi) / εi)
```

**Weight Update:**
```
    w_i^(j+1) = (w_i^(j) / Zj) * { exp(-αj)  if Cj(xi) = yi (correct)
                                 { exp(αj)   if Cj(xi) ≠ yi (wrong)
```
    where Zj = normalization factor

**Final Classification:**
```
    C*(x) = argmax_y sum(αj * δ(Cj(x) = y))
```

**SPECIAL RULE:** If any round produces error > 50%, reset weights to 1/n and resample.


### BAGGING vs BOOSTING COMPARISON

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Sampling | With replacement (random) | Weighted by errors |
| Weights | Equal for all samples | Adaptive (change each round) |
| Classifiers | Independent | Sequential (depends on prev) |
| Focus | Reduce variance | Reduce bias |
| Overfitting | Less prone | Can overfit if too many rounds |
| Parallelizable | Yes | No (sequential) |


---

## 5. WORKED EXAMPLES & PROBLEMS

### EXAMPLE 1: SVM Margin Calculation

Q: Given two possible hyperplanes B1 and B2, which is better?

A: The one with LARGER MARGIN.
   - Margin = 2 / ||w||^2
   - Larger margin = better generalization
   - B1 typically better if it has more distance to nearest points


### EXAMPLE 2: Naive Bayes Zero Probability

Q: What happens when P(Married|Yes) = 0?

A: The entire product becomes 0!
   P(X|Yes) = 1 * 0 * ... = 0

**Solution:** Use LAPLACE SMOOTHING
```
   P(Ai|C) = (count(Ai, C) + 1) / (count(C) + k)
```
   where k = number of possible values for Ai


### EXAMPLE 3: AdaBoost Weight Update

Q: If ε = 0.3, what is α?

A: α = (1/2) * ln((1 - 0.3) / 0.3)
     = (1/2) * ln(0.7 / 0.3)
     = (1/2) * ln(2.33)
     = (1/2) * 0.847
     = 0.424


### EXAMPLE 4: Neural Network Output

Q: With weights w = (0.3, 0.3, 0.3), threshold t = 0.4,
   and input X = (1, 0, 1), what is output?

A: Y = I(0.3*1 + 0.3*0 + 0.3*1 - 0.4 > 0)
     = I(0.3 + 0 + 0.3 - 0.4 > 0)
     = I(0.2 > 0)
     = 1


---

## 6. PYTHON CODE EXAMPLES

### SVM IN PYTHON (sklearn):
```python
from sklearn import svm

# Training data
X = [[0, 0], [1, 1]]
y = [0, 1]

# Create and train SVM classifier
clf = svm.SVC()
clf.fit(X, y)

# Predict new data
clf.predict([[2., 2.]])

# Get support vectors
clf.support_vectors_

# Get indices of support vectors
clf.support_

# Get number of support vectors per class
clf.n_support_
```


### ADABOOST IN PYTHON (sklearn):
```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

# Load data
X, y = load_iris(return_X_y=True)

# Create AdaBoost classifier with 100 weak learners
clf = AdaBoostClassifier(n_estimators=100)

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())  # Output: 0.9466...
```


### NAIVE BAYES IN PYTHON (sklearn):
```python
from sklearn.naive_bayes import GaussianNB

# Create classifier
clf = GaussianNB()

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
```


---

## 7. KEY TAKEAWAYS FOR EXAM

### NEURAL NETWORKS:
- Know the perceptron model formula: Y = I(sum(wi*Xi) - t > 0)
- Understand training = learning weights via backpropagation
- Objective: minimize E = sum[Yi - f(wi, Xi)]^2

### SUPPORT VECTOR MACHINES:
- Goal: Maximize margin = 2/||w||^2
- Decision boundary: w.x + b = 0
- Support vectors = points on margin boundaries
- Soft margin: introduce slack variables for non-separable data
- Kernel trick: transform to higher dimensions for nonlinear boundaries

### NAIVE BAYES:
- Bayes theorem: P(C|A) = P(A|C)*P(C) / P(A)
- "Naive" = assumes conditional independence of attributes
- P(A1,...,An|C) = product of P(Ai|C)
- For continuous: use normal distribution
- Know how to calculate from a table (very likely exam question)

### ENSEMBLE/BOOSTING:
- Combine multiple weak classifiers into strong classifier
- Bagging: sample with replacement, equal weights
- Boosting: adaptive weights, focus on misclassified
- AdaBoost formulas:
  - Error: εi = weighted misclassification rate
  - Importance: αi = (1/2) * ln((1-εi)/εi)
  - Final: C*(x) = sign(sum(αi * hi(x)))


### COMMON EXAM QUESTION TYPES:
1. Calculate Naive Bayes probabilities from a table
2. Determine class using Naive Bayes
3. Calculate SVM margin given weight vector
4. Explain why SVM chooses one hyperplane over another
5. Calculate AdaBoost classifier weight given error rate
6. Trace through one iteration of AdaBoost weight updates
7. Compare/contrast methods (ANN vs SVM vs Naive Bayes)


---

## QUICK REVIEW CHECKLIST

### Artificial Neural Networks (ANN)
- [ ] Can I write the perceptron model formula: Y = I(Σwi·Xi - t > 0)?
- [ ] Can I calculate output for given inputs, weights, and threshold?
- [ ] Do I understand input layer → hidden layer(s) → output layer structure?
- [ ] Can I explain backpropagation as the method for learning weights?
- [ ] Do I know the objective function: minimize E = Σ[Yi - f(wi, Xi)]²?

### Support Vector Machines (SVM)
- [ ] Do I understand the goal: maximize margin = 2/||w||²?
- [ ] Can I identify the decision boundary equation: w·x + b = 0?
- [ ] Do I know what support vectors are (points on margin boundaries)?
- [ ] Can I explain soft margin with slack variables for non-separable data?
- [ ] Do I understand kernel trick for nonlinear boundaries?

### Naive Bayes Classifier
- [ ] Can I write Bayes theorem: P(C|A) = P(A|C)·P(C) / P(A)?
- [ ] Do I understand the "naive" assumption: conditional independence of attributes?
- [ ] Can I apply: P(A1,...,An|C) = Π P(Ai|C)?
- [ ] Can I calculate Naive Bayes probabilities from a frequency table?
- [ ] Do I know how to handle continuous attributes (use normal distribution)?
- [ ] Can I determine class by comparing P(C=yes|X) vs P(C=no|X)?

### Ensemble Methods & Boosting
- [ ] Do I understand bagging: sample with replacement, equal weights?
- [ ] Do I understand boosting: adaptive weights, focus on misclassified?
- [ ] Can I calculate AdaBoost error: εi = weighted misclassification rate?
- [ ] Can I calculate AdaBoost importance: αi = (1/2) · ln((1-εi)/εi)?
- [ ] Do I know the final classifier: C*(x) = sign(Σ αi · hi(x))?
- [ ] Can I trace through one iteration of AdaBoost weight updates?

### Key Comparisons
- [ ] Do I know when to use ANN vs SVM vs Naive Bayes?
- [ ] Can I explain advantages/disadvantages of each method?

---

*END OF NOTES*
