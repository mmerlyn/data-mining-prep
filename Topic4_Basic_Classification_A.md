# TOPIC 4A: CLASSIFICATION - BASIC CONCEPTS & DECISION TREES
## CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS

1. Classification Definition
2. Classification Techniques Overview
3. Decision Trees - Introduction
4. Hunt's Algorithm
5. Tree Induction Issues
6. Splitting Based on Attribute Types
7. Impurity Measures (CRITICAL FOR EXAM!)
8. GINI Index - Detailed Examples
9. Classification Error
10. Entropy and Information Gain
11. Gain Ratio
12. Stopping Criteria
13. Decision Tree Advantages
14. C4.5 Algorithm
15. Python Implementation
16. Quiz Questions & Worked Examples


---

## 1. CLASSIFICATION DEFINITION

### DEFINITION:

Given a collection of records (TRAINING SET):
- Each record contains a set of ATTRIBUTES
- One attribute is the CLASS (target variable)

Goal: Find a MODEL for the class attribute as a function of other attributes
- Previously UNSEEN records should be assigned a class as ACCURATELY as possible
- A TEST SET is used to determine the accuracy of the model

### CLASSIFICATION PROCESS:

```
Training Set --> Learning Algorithm --> Model --> Apply to Test Set --> Predictions
```

Two Phases:
1. INDUCTION: Learning algorithm builds model from training data
2. DEDUCTION: Model applied to test data to make predictions


### CLASSIFICATION EXAMPLES:

- Predicting tumor cells as BENIGN or MALIGNANT
- Classifying credit card transactions as LEGITIMATE or FRAUDULENT
- Classifying protein structures as alpha-helix, beta-sheet, or random coil
- Categorizing news stories as finance, weather, entertainment, sports


### EXAMPLE: Tax Fraud Detection Dataset

| Tid | Refund | Marital Status | Income | Cheat |
|-----|--------|----------------|--------|-------|
| 1   | Yes    | Single         | 125K   | No    |
| 2   | No     | Married        | 100K   | No    |
| 3   | No     | Single         | 70K    | No    |
| 4   | Yes    | Married        | 120K   | No    |
| 5   | No     | Divorced       | 95K    | Yes   |
| 6   | No     | Married        | 60K    | No    |
| 7   | Yes    | Divorced       | 220K   | No    |
| 8   | No     | Single         | 85K    | Yes   |
| 9   | No     | Married        | 75K    | No    |
| 10  | No     | Single         | 90K    | Yes   |

- Refund, Marital Status: CATEGORICAL attributes
- Income: CONTINUOUS attribute
- Cheat: CLASS label (what we want to predict)


---

## 2. CLASSIFICATION TECHNIQUES OVERVIEW

### CLASSIFICATION TOOLBOX:

1. Decision Tree based Methods  <-- Focus of this lecture
2. Rule-based Methods
3. Memory based reasoning
4. Neural Networks
5. Naive Bayes and Bayesian Belief Networks
6. Support Vector Machines


---

## 3. DECISION TREES - INTRODUCTION

### DECISION TREE STRUCTURE:

- ROOT NODE: Top of tree, contains all data
- INTERNAL NODES: Test on an attribute (splitting nodes)
- BRANCHES: Outcomes of the test
- LEAF NODES: Class labels (predictions)

### EXAMPLE DECISION TREE (Tax Fraud):

```
                    [Refund]
                   /        \
                Yes          No
                 |            |
               [NO]       [MarSt]
                         /       \
              Single,Divorced   Married
                    |              |
                [TaxInc]         [NO]
                /      \
            <80K      >80K
              |          |
            [NO]       [YES]
```

Reading the tree:
- If Refund = Yes --> Don't Cheat (No)
- If Refund = No AND MarSt = Married --> Don't Cheat (No)
- If Refund = No AND MarSt = Single/Divorced AND TaxInc < 80K --> Don't Cheat
- If Refund = No AND MarSt = Single/Divorced AND TaxInc >= 80K --> Cheat (Yes)

IMPORTANT: There can be MORE THAN ONE TREE that fits the same data!


---

## 4. HUNT'S ALGORITHM

### DEFINITION:

Hunt's Algorithm is one of the EARLIEST decision tree algorithms.
Many modern algorithms (CART, ID3, C4.5) are based on it.

### GENERAL PROCEDURE:

Let Dt be the set of training records that reach node t

1. If Dt contains records that belong to MORE THAN ONE CLASS:
   - Use an attribute test to split data into smaller subsets
   - RECURSIVELY apply the procedure to each subset

2. If Dt contains records that belong to the SAME CLASS yt:
   - Node t is a LEAF NODE labeled as yt

3. If Dt is an EMPTY SET:
   - Node t is a LEAF NODE labeled by the DEFAULT CLASS yd


### HUNT'S ALGORITHM EXAMPLE (Step by Step):

Step 1: Start with all 10 records
        - 7 records: Cheat = No
        - 3 records: Cheat = Yes
        - NOT pure, need to split

Step 2: Split on "Refund"
        - Refund = Yes: 3 records, all Cheat = No --> LEAF (No)
        - Refund = No: 7 records, mixed classes --> Need more splits

Step 3: For Refund = No subset, split on "Marital Status"
        - Married: 4 records, all Cheat = No --> LEAF (No)
        - Single/Divorced: 3 records, mixed --> Need more splits

Step 4: For Single/Divorced subset, split on "Taxable Income"
        - Income < 80K: 1 record, Cheat = No --> LEAF (No)
        - Income >= 80K: 2 records, Cheat = Yes --> LEAF (Yes)


---

## 5. TREE INDUCTION - THREE KEY ISSUES

### THREE CRITICAL QUESTIONS:

1. How to specify the ATTRIBUTE TEST CONDITION?
2. How to determine the BEST SPLIT?
3. When to STOP SPLITTING?


---

## 6. SPLITTING BASED ON ATTRIBUTE TYPES

### 6.1 SPLITTING ON NOMINAL ATTRIBUTES

**MULTI-WAY SPLIT:**
- Use as many partitions as distinct values
- Example: CarType has 3 values --> 3 branches

```
            [CarType]
           /    |    \
      Family  Sports  Luxury
```

**BINARY SPLIT:**
- Divide values into TWO subsets
- Need to find OPTIMAL partitioning

```
       [CarType]              [CarType]
       /       \       OR     /       \
{Sports,     {Family}    {Family,    {Sports}
 Luxury}                  Luxury}
```


### 6.2 SPLITTING ON ORDINAL ATTRIBUTES

**MULTI-WAY SPLIT:**
- Use as many partitions as distinct values

```
            [Size]
           /   |   \
        Small Medium Large
```

**BINARY SPLIT:**
- Must PRESERVE ORDER!
- Valid: {Small, Medium} vs {Large}
- Valid: {Small} vs {Medium, Large}
- INVALID: {Small, Large} vs {Medium}  <-- Violates order!


### 6.3 SPLITTING ON CONTINUOUS ATTRIBUTES

**TWO APPROACHES:**

1. DISCRETIZATION:
   - Convert continuous to ordinal categorical
   - Static: Discretize once at the beginning
   - Dynamic: Equal interval bucketing, equal frequency, clustering

2. BINARY DECISION: (A < v) or (A >= v)
   - Consider all possible splits
   - Find the best cut point
   - More compute intensive

**EXAMPLE:**

```
   [Taxable Income]           [Taxable Income]
        > 80K?                      ?
       /     \                /   /    \   \   \
     Yes     No          <10K [10K,25K) [25K,50K) [50K,80K) >80K

   (Binary split)            (Multi-way split)
```


---

## 7. IMPURITY MEASURES (CRITICAL FOR EXAM!)

### PURPOSE:

To determine the BEST SPLIT, we need to measure NODE IMPURITY

GOAL: Prefer nodes with HOMOGENEOUS class distribution

```
+-------------+    +-------------+
| C0: 5       |    | C0: 9       |
| C1: 5       |    | C1: 1       |
+-------------+    +-------------+
Non-homogeneous    Homogeneous
HIGH impurity      LOW impurity
```


### THREE IMPURITY MEASURES:

1. GINI Index
2. Misclassification Error (Classification Error)
3. Entropy


---

## 8. GINI INDEX - DETAILED EXAMPLES

### FORMULA:

```
                    c
    GINI(t) = 1 - SUM [p(j|t)]^2
                   j=1
```

Where p(j|t) = relative frequency of class j at node t


### GINI PROPERTIES:

- MINIMUM (0.0): When all records belong to ONE class (pure node)
- MAXIMUM (1 - 1/nc): When records equally distributed among all classes

For BINARY classification (2 classes):
- MINIMUM = 0 (all one class)
- MAXIMUM = 0.5 (50% each class)


### EXAMPLE CALCULATIONS:

**Example 1: Node with C1=0, C2=6**
```
+----+---+
| C1 | 0 |
| C2 | 6 |
+----+---+
P(C1) = 0/6 = 0
P(C2) = 6/6 = 1
GINI = 1 - (0)^2 - (1)^2 = 1 - 0 - 1 = 0  (PURE!)
```


**Example 2: Node with C1=1, C2=5**
```
+----+---+
| C1 | 1 |
| C2 | 5 |
+----+---+
P(C1) = 1/6
P(C2) = 5/6
GINI = 1 - (1/6)^2 - (5/6)^2
     = 1 - 1/36 - 25/36
     = 1 - 26/36
     = 10/36 = 0.278
```


**Example 3: Node with C1=2, C2=4**
```
+----+---+
| C1 | 2 |
| C2 | 4 |
+----+---+
P(C1) = 2/6 = 1/3
P(C2) = 4/6 = 2/3
GINI = 1 - (1/3)^2 - (2/3)^2
     = 1 - 1/9 - 4/9
     = 1 - 5/9
     = 4/9 = 0.444
```


**Example 4: Node with C1=3, C2=3 (Maximum impurity for 2 classes)**
```
+----+---+
| C1 | 3 |
| C2 | 3 |
+----+---+
P(C1) = 3/6 = 0.5
P(C2) = 3/6 = 0.5
GINI = 1 - (0.5)^2 - (0.5)^2
     = 1 - 0.25 - 0.25
     = 0.5  (MAXIMUM!)
```


### GINI FOR SPLITS (Weighted Average)

When node p is split into k children:

```
                k   n_i
    GINI_split = SUM --- * GINI(i)
               i=1   n
```

Where:
- n_i = number of records at child i
- n = number of records at parent node p


### WORKED EXAMPLE: Binary Attribute Split

Parent Node: C1=6, C2=6 (12 total)
Parent GINI = 1 - (6/12)^2 - (6/12)^2 = 0.5

Split on attribute B:

```
             [B?]
           /      \
         Yes       No
          |         |
    +--------+  +--------+
    | C1: 5  |  | C1: 1  |
    | C2: 2  |  | C2: 4  |
    +--------+  +--------+
       N1          N2
    (7 records) (5 records)
```

```
GINI(N1) = 1 - (5/7)^2 - (2/7)^2
         = 1 - 25/49 - 4/49
         = 1 - 29/49
         = 20/49 = 0.4082

GINI(N2) = 1 - (1/5)^2 - (4/5)^2
         = 1 - 1/25 - 16/25
         = 1 - 17/25
         = 8/25 = 0.3200

GINI_split = (7/12) * 0.4082 + (5/12) * 0.3200
           = 0.2381 + 0.1333
           = 0.3715
```


### GINI FOR CATEGORICAL ATTRIBUTES

Example: CarType attribute with values {Family, Sports, Luxury}

**Count Matrix:**

|         | Family | Sports | Luxury |
|---------|--------|--------|--------|
| C1      |   1    |   2    |   1    |
| C2      |   4    |   1    |   1    |

**MULTI-WAY SPLIT (3 branches):**
- Family: GINI = 1 - (1/5)^2 - (4/5)^2 = 0.32
- Sports: GINI = 1 - (2/3)^2 - (1/3)^2 = 0.444
- Luxury: GINI = 1 - (1/2)^2 - (1/2)^2 = 0.5
- GINI_split = (5/10)*0.32 + (3/10)*0.444 + (2/10)*0.5 = 0.393

**TWO-WAY SPLIT {Sports,Luxury} vs {Family}:**
- {Sports,Luxury}: 5 records, C1=3, C2=2
  GINI = 1 - (3/5)^2 - (2/5)^2 = 0.48
- {Family}: 5 records, C1=1, C2=4
  GINI = 1 - (1/5)^2 - (4/5)^2 = 0.32
- GINI_split = (5/10)*0.48 + (5/10)*0.32 = 0.400


### GINI FOR CONTINUOUS ATTRIBUTES

**EFFICIENT COMPUTATION METHOD:**
1. SORT the attribute values
2. LINEARLY SCAN values, updating count matrix at each position
3. Compute GINI for each possible split point
4. Choose split with LOWEST GINI

### WORKED EXAMPLE: Taxable Income

Sorted by Income:

| Cheat  | No | No | No | Yes | Yes| Yes | No  | No  | No  | No  |
|--------|----|----|----|----|----|----|-----|-----|-----|-----|
| Income | 60 | 70 | 75 | 85  | 90 | 95  | 100 | 120 | 125 | 220 |
| Split Points | 55 | 65 | 72 | 80 | 87 | 92 | 97 | 110 | 122 | 172 | 230 |

For each split point, compute count matrix and GINI:

**Split at 97 (Income <= 97 vs > 97):**

|     | <= 97   | > 97    |
|-----|---------|---------|
| Yes |    3    |    0    |
| No  |    3    |    4    |

```
Left child GINI = 1 - (3/6)^2 - (3/6)^2 = 0.5
Right child GINI = 1 - (0/4)^2 - (4/4)^2 = 0

GINI_split = (6/10)*0.5 + (4/10)*0 = 0.300  <-- LOWEST!
```

Best split: Taxable Income <= 97


---

## 9. CLASSIFICATION ERROR

### FORMULA:

```
    Error(t) = 1 - max P(i|t)
                    i
```

Where P(i|t) = relative frequency of class i at node t


### EXAMPLE CALCULATIONS:

**Example 1: C1=0, C2=6**
```
P(C1) = 0, P(C2) = 1
Error = 1 - max(0, 1) = 1 - 1 = 0
```


**Example 2: C1=1, C2=5**
```
P(C1) = 1/6, P(C2) = 5/6
Error = 1 - max(1/6, 5/6) = 1 - 5/6 = 1/6 = 0.167
```


**Example 3: C1=2, C2=4**
```
P(C1) = 2/6, P(C2) = 4/6
Error = 1 - max(2/6, 4/6) = 1 - 4/6 = 2/6 = 1/3 = 0.333
```


---

## 10. ENTROPY AND INFORMATION GAIN

### ENTROPY FORMULA:

```
                    c
    Entropy(t) = - SUM p(j|t) * log2[p(j|t)]
                   j=1
```

Where p(j|t) = relative frequency of class j at node t

NOTE: By convention, 0 * log(0) = 0


### ENTROPY PROPERTIES:

- MINIMUM (0.0): All records belong to ONE class (pure, most information)
- MAXIMUM (log2 nc): Records equally distributed (least information)

For BINARY classification:
- MINIMUM = 0
- MAXIMUM = log2(2) = 1


### EXAMPLE CALCULATIONS:

**Example 1: C1=0, C2=6 (Pure node)**
```
P(C1) = 0, P(C2) = 1
Entropy = -0*log2(0) - 1*log2(1) = 0 - 0 = 0
```


**Example 2: C1=1, C2=5**
```
P(C1) = 1/6, P(C2) = 5/6
Entropy = -(1/6)*log2(1/6) - (5/6)*log2(5/6)
        = -(1/6)*(-2.585) - (5/6)*(-0.263)
        = 0.431 + 0.219
        = 0.65
```


**Example 3: C1=2, C2=4**
```
P(C1) = 2/6, P(C2) = 4/6
Entropy = -(2/6)*log2(2/6) - (4/6)*log2(4/6)
        = -(1/3)*(-1.585) - (2/3)*(-0.585)
        = 0.528 + 0.390
        = 0.92
```


**Example 4: C1=3, C2=3 (Maximum entropy for 2 classes)**
```
P(C1) = 0.5, P(C2) = 0.5
Entropy = -0.5*log2(0.5) - 0.5*log2(0.5)
        = -0.5*(-1) - 0.5*(-1)
        = 0.5 + 0.5
        = 1.0  (MAXIMUM!)
```


### INFORMATION GAIN

#### FORMULA:

```
                                    k   n_i
    GAIN_split = Entropy(parent) - SUM --- * Entropy(i)
                                   i=1  n
```

Where:
- Parent Node p is split into k partitions
- n_i = number of records in partition i
- n = total records at parent

**INTERPRETATION:**
- Measures REDUCTION in entropy achieved by the split
- HIGHER GAIN = BETTER SPLIT
- Used in ID3 and C4.5


**DISADVANTAGE:**
- Tends to prefer splits with LARGE NUMBER of partitions
- Each partition small but pure
- Example: Splitting on "Student ID" creates many pure nodes but overfits!


---

## 11. GAIN RATIO

### FORMULA:

```
                      GAIN_split
    GainRATIO_split = ----------
                      SplitINFO
```

Where:
```
                    k   n_i       n_i
    SplitINFO = - SUM  --- * log2 ---
                  i=1   n          n
```


### PURPOSE:

- ADJUSTS Information Gain by entropy of the partitioning
- PENALIZES splits with many small partitions
- Used in C4.5


### EXAMPLE:

Split into 2 partitions: 6 records and 4 records (total 10)

```
SplitINFO = -(6/10)*log2(6/10) - (4/10)*log2(4/10)
          = -0.6*(-0.737) - 0.4*(-1.322)
          = 0.442 + 0.529
          = 0.971

If GAIN = 0.5, then GainRATIO = 0.5/0.971 = 0.515
```


---

## 12. STOPPING CRITERIA FOR TREE INDUCTION

### WHEN TO STOP SPLITTING:

1. ALL records belong to the SAME CLASS
   - Node is pure, no need to split further

2. ALL records have SIMILAR ATTRIBUTE VALUES
   - Cannot distinguish between records

3. EARLY TERMINATION (Pruning)
   - Stop before tree becomes too complex
   - Prevents overfitting
   - (Discussed in later lectures)


---

## 13. DECISION TREE ADVANTAGES

### ADVANTAGES:

1. INEXPENSIVE to construct
2. EXTREMELY FAST at classifying unknown records
3. EASY TO INTERPRET for small-sized trees
4. ACCURACY comparable to other techniques for simple datasets


---

## 14. C4.5 ALGORITHM

### CHARACTERISTICS:

- Simple DEPTH-FIRST construction
- Uses INFORMATION GAIN (and Gain Ratio)
- SORTS continuous attributes at each node
- Needs ENTIRE DATA to fit in memory
- UNSUITABLE for very large datasets


### OTHER DECISION TREE ALGORITHMS:

| Algorithm  | Description                              |
|------------|------------------------------------------|
| Hunt's     | One of the earliest                      |
| CART       | Extension of Hunt's, uses GINI           |
| ID3        | Iterative Dichotomiser 3, uses Entropy   |
| C4.5       | Improvement of ID3, uses Gain Ratio      |
| SLIQ       | Fast scalable classifier                 |
| SPRINT     | Scalable parallel classifier             |


---

## 15. PYTHON IMPLEMENTATION

### SCIKIT-LEARN SUPPORT:

- DecisionTreeClassifier: For classification problems
- DecisionTreeRegressor: For regression problems

### CASE 1: Logic AND Operator

```python
from sklearn import tree

# Logic And
X = [[0, 0], [0,1], [1,0], [1, 1]]
Y = [0, 0, 0, 1]

# Create a classifier
clf = tree.DecisionTreeClassifier()

# Train
clf = clf.fit(X, Y)

# Test
clf.predict([[0,1]])  # Output: [0]

# Visualize the tree
tree.plot_tree(clf)
```


### CASE 2: Iris Classification

```python
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# Load data
iris = load_iris()

# Create a decision tree classifier
iris_clf = DecisionTreeClassifier()

# Training
iris_clf = iris_clf.fit(iris.data, iris.target)

# Visualizing the tree
tree.plot_tree(iris_clf)

# Visualizing the tree in text format
r = export_text(iris_clf, feature_names=iris['feature_names'])
print(r)

# Output example:
# |--- petal width (cm) <= 0.80
# |   |--- class: 0
# |--- petal width (cm) > 0.80
# |   |--- petal width (cm) <= 1.75
# |   |   |--- class: 1
# |   |--- petal width (cm) > 1.75
# |   |   |--- class: 2
```


### CASE 3: Decision Tree for Regression

```python
from sklearn import tree

X = [[1], [4], [5], [8], [9]]
Y = [10, 15, 30, 45, 55]

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, Y)

# Predict
clf.predict([[3]])  # Output: 15.0

# Plot tree
tree.plot_tree(clf)
```


---

## 16. QUIZ QUESTIONS & WORKED EXAMPLES

### QUESTION 1: GINI Calculation

Calculate GINI for a node with class distribution: C1=4, C2=6

**Solution:**
```
P(C1) = 4/10 = 0.4
P(C2) = 6/10 = 0.6
GINI = 1 - (0.4)^2 - (0.6)^2
     = 1 - 0.16 - 0.36
     = 0.48
```


### QUESTION 2: GINI for Split

Parent node has 20 records (10 each class). After split:
- Child 1: 8 records (C1=7, C2=1)
- Child 2: 12 records (C1=3, C2=9)

Calculate GINI_split.

**Solution:**
```
GINI(Child1) = 1 - (7/8)^2 - (1/8)^2
             = 1 - 49/64 - 1/64
             = 1 - 50/64
             = 14/64 = 0.219

GINI(Child2) = 1 - (3/12)^2 - (9/12)^2
             = 1 - 1/16 - 9/16
             = 1 - 10/16
             = 6/16 = 0.375

GINI_split = (8/20)*0.219 + (12/20)*0.375
           = 0.4*0.219 + 0.6*0.375
           = 0.0876 + 0.225
           = 0.3126
```


### QUESTION 3: Information Gain

Parent entropy = 1.0 (equal distribution)
After split:
- Child 1: 6 records, Entropy = 0.65
- Child 2: 4 records, Entropy = 0.81

Calculate Information Gain.

**Solution:**
```
Weighted entropy of children = (6/10)*0.65 + (4/10)*0.81
                             = 0.39 + 0.324
                             = 0.714

GAIN = Entropy(parent) - Weighted entropy of children
     = 1.0 - 0.714
     = 0.286
```


### QUESTION 4: When is GINI Maximum/Minimum?

For a binary classification problem:

**MINIMUM GINI = 0**
- When: All records belong to ONE class
- Example: C1=0, C2=10 or C1=10, C2=0
- GINI = 1 - 0^2 - 1^2 = 0

**MAXIMUM GINI = 0.5**
- When: Records equally distributed
- Example: C1=5, C2=5
- GINI = 1 - 0.5^2 - 0.5^2 = 0.5


### QUESTION 5: Ordinal Attribute Split

Which split is INVALID for ordinal attribute Size = {Small, Medium, Large}?

a) {Small} vs {Medium, Large}
b) {Small, Medium} vs {Large}
c) {Small, Large} vs {Medium}

**Answer:** (c) is INVALID because it violates the order!
You cannot group Small and Large together while separating Medium.


### QUESTION 6: Compare Impurity Measures

Node with C1=2, C2=8 (10 total)

```
GINI = 1 - (2/10)^2 - (8/10)^2
     = 1 - 0.04 - 0.64
     = 0.32

Error = 1 - max(2/10, 8/10)
      = 1 - 0.8
      = 0.2

Entropy = -(2/10)*log2(2/10) - (8/10)*log2(8/10)
        = -0.2*(-2.322) - 0.8*(-0.322)
        = 0.464 + 0.258
        = 0.722
```


### QUESTION 7: Apply Decision Tree

Given this tree:
```
                    [Refund]
                   /        \
                Yes          No
                 |            |
               [NO]       [MarSt]
                         /       \
              Single,Divorced   Married
                    |              |
                [TaxInc]         [NO]
                /      \
            <80K      >80K
              |          |
            [NO]       [YES]
```

Classify: Refund=No, MarSt=Single, TaxInc=95K

**Solution:**
1. Refund = No --> Go right to MarSt
2. MarSt = Single --> Go left to TaxInc
3. TaxInc = 95K > 80K --> Go right
4. Prediction: YES (Cheat)


---

## IMPURITY MEASURES COMPARISON TABLE

| Measure          | Formula                | Used In          |
|------------------|------------------------|------------------|
| GINI Index       | 1 - SUM[p(j|t)]^2      | CART, SLIQ, SPRINT |
| Classification Error | 1 - max P(i|t)     | General          |
| Entropy          | -SUM p(j|t)*log p(j|t) | ID3, C4.5        |
| Information Gain | Entropy(p) - weighted sum of children entropy | ID3, C4.5 |
| Gain Ratio       | GAIN / SplitINFO       | C4.5             |


---

## KEY CONCEPTS SUMMARY

**CLASSIFICATION:**
- Predictive task: predict class from attributes
- Training set to build model, test set to validate
- Decision tree is one classification technique

**DECISION TREE:**
- Root, internal nodes (tests), leaf nodes (classes)
- Hunt's Algorithm: recursive splitting

**THREE TREE INDUCTION ISSUES:**
1. Test conditions: depends on attribute type
2. Best split: use impurity measures
3. Stopping: pure nodes, similar attributes

**IMPURITY MEASURES:**
- GINI: 1 - SUM[p(j|t)]^2 (lower = purer)
- Error: 1 - max P(i|t)
- Entropy: -SUM p*log(p) (lower = purer)

**INFORMATION GAIN:**
- GAIN = Entropy(parent) - weighted entropy(children)
- Higher = better split

**GAIN RATIO:**
- Adjusts for splits with many partitions
- GainRatio = GAIN / SplitINFO


---

## QUICK REVIEW CHECKLIST

- [ ] Can define classification and its components
- [ ] Know the classification process (induction vs deduction)
- [ ] Understand decision tree structure
- [ ] Know Hunt's Algorithm and its three cases
- [ ] Understand three tree induction issues
- [ ] Know how to split on Nominal, Ordinal, Continuous attributes
- [ ] Can calculate GINI index for a single node
- [ ] Can calculate GINI for a split (weighted average)
- [ ] Know when GINI is maximum and minimum
- [ ] Can calculate Classification Error
- [ ] Can calculate Entropy
- [ ] Can calculate Information Gain
- [ ] Understand Gain Ratio and its purpose
- [ ] Know stopping criteria for tree induction
- [ ] Know advantages of decision trees
- [ ] Can trace through a decision tree to classify a record
- [ ] Know Python sklearn implementation

---

## END OF TOPIC 4A
