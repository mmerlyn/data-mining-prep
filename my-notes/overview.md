# CS653 DATA MINING - COMPLETE TOPIC OVERVIEW
## Final Exam Preparation Guide

---

## THE BIG PICTURE: WHAT IS DATA MINING?

Data Mining is the process of **DISCOVERING PATTERNS, CORRELATIONS, AND INSIGHTS** from large datasets using statistical, mathematical, and computational techniques. It sits at the intersection of databases, statistics, and machine learning.

**Definition:** Non-trivial extraction of implicit, previously unknown, potentially useful information from data.

**What IS Data Mining:**
- Prediction, classification, clustering, anomaly detection, association discovery

**What is NOT Data Mining:**
- Simple queries, sorting, aggregation (sum/count), signal processing

---

## TOPIC 1: INTRODUCTION TO DATA MINING

**Core Concept:** Understanding what data mining is (and isn't)

### KDD (Knowledge Discovery in Databases) Process:
```
Data -> Selection -> Preprocessing -> Transformation -> Data Mining -> Interpretation -> Knowledge
```

### Key Applications:
- Customer segmentation
- Fraud detection
- Recommendation systems
- Medical diagnosis
- Stock prediction

---

## TOPIC 2: DATA PROCESSING / PREPROCESSING

**Core Concept:** Garbage in = Garbage out. Data must be cleaned and prepared.

### PART A: DATA TYPES & QUALITY

**Attribute Types:**
1. **Nominal** - Categories with no order (e.g., colors, gender)
2. **Ordinal** - Categories with order (e.g., small/medium/large)
3. **Interval** - Numeric, differences meaningful, no true zero (e.g., temperature in Celsius)
4. **Ratio** - Numeric, has true zero (e.g., height, weight, income)

**Data Quality Issues:**
- Missing values
- Noise (random errors)
- Outliers
- Inconsistencies
- Duplicates

**Handling Missing Data:**
- Deletion (remove records/attributes)
- Imputation (fill with mean/median/mode)
- Prediction (use other attributes to predict missing value)

### PART B: DATA TRANSFORMATION

**Normalization:**

1. **Min-Max Scaling:** `x' = (x - min) / (max - min)`
   - Scales to [0, 1] range

2. **Z-Score Standardization:** `x' = (x - mean) / std_dev`
   - Scales to mean=0, std=1

**Discretization:**
- Equal-width binning: Divide range into equal-sized bins
- Equal-frequency binning: Each bin has same number of records

**Aggregation:**
- Combining attributes or records (e.g., daily -> monthly sales)

**Sampling:**
- Simple random sampling
- Stratified sampling (preserve class distribution)

---

## TOPIC 3: DATA EXPLORATION

**Core Concept:** Understand your data BEFORE mining it

### SUMMARY STATISTICS
- **Mean:** Average value
- **Median:** Middle value (robust to outliers)
- **Mode:** Most frequent value
- **Variance:** Average squared deviation from mean
- **Standard Deviation:** Square root of variance
- **Quartiles:** Q1 (25%), Q2 (50%/median), Q3 (75%)
- **IQR (Interquartile Range):** Q3 - Q1

### VISUALIZATION TECHNIQUES

1. **Box Plot:**
   - Shows: Min, Q1, Median, Q3, Max, Outliers
   - Outliers: Points beyond 1.5 * IQR from Q1 or Q3
   - Great for comparing distributions across groups

2. **Histogram:**
   - Shows distribution of single variable
   - Reveals skewness, modality

3. **Scatter Plot:**
   - Shows relationship between two variables
   - Reveals correlation, clusters, outliers

### CORRELATION & COVARIANCE

**Pearson Correlation Coefficient (r):**
- Range: [-1, +1]
- +1: Perfect positive correlation
- -1: Perfect negative correlation
- 0: No linear correlation

**Formula:** `r = Cov(X,Y) / (std_X * std_Y)`

**Covariance:** `Cov(X,Y) = E[(X - mean_X)(Y - mean_Y)]`

---

## TOPIC 4: BASIC CLASSIFICATION

**Core Concept:** Predict a categorical class label from training data (SUPERVISED)

### PART A: DECISION TREES

**How It Works:**
- Recursive partitioning using attribute tests
- Internal nodes: Test on attribute
- Branches: Outcome of test
- Leaf nodes: Class label

**Splitting Criteria:**

#### 1. GINI INDEX:
```
GINI(t) = 1 - SUM(p_i)^2
```
Where p_i is the probability of class i at node t

- GINI = 0: Pure node (all same class)
- GINI = 0.5: Maximum impurity (for binary classification)

For a split into k partitions:
```
GINI_split = SUM(n_i/n * GINI_i)
```

#### 2. ENTROPY / INFORMATION GAIN:
```
Entropy(t) = -SUM(p_i * log2(p_i))

Information Gain = Entropy(parent) - Weighted_Avg_Entropy(children)
```
- Entropy = 0: Pure node
- Entropy = 1: Maximum impurity (binary)

**Tree Traversal:**
- Start at root
- Follow branch based on attribute value
- Repeat until leaf node reached
- Leaf node gives prediction

**Overfitting Prevention:**
- Pre-pruning: Stop growing early (max depth, min samples)
- Post-pruning: Grow full tree, then remove nodes

### PART B: MODEL EVALUATION

**Confusion Matrix (Binary Classification):**
```
                Predicted
                Pos     Neg
Actual  Pos     TP      FN
        Neg     FP      TN
```

**Key Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| Precision | TP / (TP + FP) | Of predicted positives, how many are actually positive? |
| Recall (Sensitivity) | TP / (TP + FN) | Of actual positives, how many did we find? |
| Specificity | TN / (TN + FP) | Of actual negatives, how many did we correctly identify? |
| F1-Score | 2 * (Precision * Recall) / (Precision + Recall) | Harmonic mean of precision and recall |

**Cross-Validation:**
- K-fold: Split data into k parts, train on k-1, test on 1, rotate
- Provides more reliable estimate of model performance

---

## TOPIC 5: ALTERNATIVE CLASSIFICATION METHODS

**Core Concept:** Other approaches beyond decision trees

### PART A: K-NEAREST NEIGHBORS (K-NN)

**Algorithm:**
1. Calculate distance from test point to all training points
2. Find k nearest neighbors
3. Majority vote among neighbors = predicted class

**Distance Metrics:**
- **Euclidean:** `sqrt(SUM(x_i - y_i)^2)`
- **Manhattan:** `SUM(|x_i - y_i|)`
- **Minkowski:** `(SUM(|x_i - y_i|^p))^(1/p)`

**Choosing K:**
- Small K (e.g., 1): Sensitive to noise, complex decision boundary, may overfit
- Large K (e.g., 20): Smoother decision boundary, may miss local patterns, may underfit

### PART B: NAIVE BAYES

**Based on Bayes Theorem:**
```
P(C|X) = P(X|C) * P(C) / P(X)
```

**Naive Assumption:**
- All attributes are conditionally independent given the class
- `P(X|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)`

**Classification:**
- Calculate P(C|X) for each class
- Predict class with highest probability

**Steps:**
1. Calculate prior probabilities: `P(C) = count(C) / total`
2. Calculate conditional probabilities: `P(xi|C) = count(xi and C) / count(C)`
3. For test sample, multiply: `P(C) * P(x1|C) * P(x2|C) * ...`
4. Compare across classes, pick highest

**Laplace Smoothing (for zero probabilities):**
```
P(xi|C) = (count(xi and C) + 1) / (count(C) + number_of_values)
```

### OTHER METHODS:
- **Support Vector Machines (SVM):** Find optimal hyperplane separator
- **Neural Networks:** Layers of connected neurons with activation functions
- **Ensemble Methods:**
  - Bagging: Train multiple models on bootstrap samples, vote
  - Boosting: Sequentially train models, focus on misclassified
  - Random Forest: Ensemble of decision trees

---

## TOPIC 6: ASSOCIATION ANALYSIS

**Core Concept:** Find interesting relationships between items (UNSUPERVISED)

### PART A: FREQUENT ITEMSETS

**Terminology:**
- **Transaction:** A set of items (e.g., a shopping basket)
- **Itemset:** A collection of items
- **Support:** Frequency of itemset
  ```
  Support(A) = count(A) / total_transactions
  ```
- **Frequent Itemset:** Itemset with support >= minimum support threshold

### APRIORI ALGORITHM:
1. Generate candidate 1-itemsets
2. Scan database, count support
3. Prune itemsets below min_support
4. Generate candidate k+1 itemsets from frequent k-itemsets
5. Repeat until no more frequent itemsets

**Apriori Principle (Anti-Monotone Property):**
> "If an itemset is infrequent, all its supersets are infrequent"
- Used to prune candidate generation
- Dramatically reduces search space

**Candidate Generation (Apriori-gen):**
- Join step: Combine frequent k-itemsets sharing k-1 items
- Prune step: Remove candidates with infrequent subsets

### PART B: ASSOCIATION RULES

**Rule Format:** `{Antecedent} -> {Consequent}`

**Example:** `{Bread, Butter} -> {Milk}`

**Metrics:**

1. **Support(A -> B) = P(A ∩ B)**
   - How often the rule applies
   - `Support = count(A and B) / total_transactions`

2. **Confidence(A -> B) = P(B|A) = Support(A ∪ B) / Support(A)**
   - Reliability of the rule
   - "Given A, how likely is B?"

3. **Lift(A -> B) = Confidence(A -> B) / Support(B)**
   - Lift > 1: Positive correlation (A helps predict B)
   - Lift = 1: Independent
   - Lift < 1: Negative correlation

**Rule Generation:**
1. Find all frequent itemsets
2. For each frequent itemset, generate all non-empty subsets
3. For each subset s, create rule: s -> (itemset - s)
4. Keep rules with confidence >= min_confidence

---

## TOPIC 7: CLUSTERING

**Core Concept:** Group similar objects WITHOUT predefined labels (UNSUPERVISED)

### K-MEANS CLUSTERING

**Algorithm:**
1. Choose k initial centroids (randomly or strategically)
2. REPEAT:
   - a. Assign each point to nearest centroid
   - b. Recalculate centroids as mean of assigned points
3. UNTIL centroids don't change (convergence)

**Distance:** Usually Euclidean distance

**SSE (Sum of Squared Errors):**
```
SSE = SUM_clusters SUM_points (distance(point, centroid))^2
```
- Lower SSE = tighter clusters
- Used to evaluate cluster quality

**Issues with K-Means:**
- Must specify k in advance
- Sensitive to initial centroid selection
- Finds spherical/globular clusters only
- Sensitive to outliers
- May converge to local minimum

**Choosing K:**
- Elbow method: Plot SSE vs k, find "elbow"
- Domain knowledge

### HIERARCHICAL CLUSTERING

**Types:**

1. **Agglomerative (Bottom-Up):**
   - Start: Each point is its own cluster
   - Repeat: Merge two closest clusters
   - End: One cluster containing all points

2. **Divisive (Top-Down):**
   - Start: All points in one cluster
   - Repeat: Split clusters
   - End: Each point is its own cluster

**Linkage Methods:**

| Method | Distance Measure | Pros | Cons |
|--------|-----------------|------|------|
| Single-Link (MIN) | Minimum distance between any two points | Can find elongated clusters | Sensitive to noise (chaining) |
| Complete-Link (MAX) | Maximum distance between any two points | Compact clusters | Sensitive to outliers |
| Average-Link | Average of all pairwise distances | Compromise | - |
| Centroid | Distance between cluster centers | - | - |
| Ward's Method | Minimize increase in total SSE | - | - |

**Dendrogram:**
- Tree diagram showing merge/split history
- Height = distance at which clusters merge
- Cut at any level to get desired number of clusters

### DBSCAN (Density-Based):
- **Core point:** Has >= MinPts within Eps radius
- **Border point:** Within Eps of core point
- **Noise point:** Neither core nor border
- **Pros:** Finds arbitrary shapes, handles noise
- **Cons:** Struggles with varying densities

---

## TOPIC 8: ANOMALY DETECTION

**Core Concept:** Find unusual/abnormal data points (outliers)

### TYPES OF ANOMALIES

1. **Point Anomalies:**
   - Single instance is anomalous
   - Example: Unusual transaction amount

2. **Contextual Anomalies:**
   - Instance anomalous in specific context
   - Example: High temperature in winter (normal in summer)

3. **Collective Anomalies:**
   - Collection of related instances is anomalous
   - Example: Sequence of network packets forming an attack

### DETECTION APPROACHES

1. **Statistical Methods:**
   - **Z-Score:** `z = (x - mean) / std_dev`
     - |z| > 3 often considered anomaly
   - **IQR Method:**
     - Lower bound: Q1 - 1.5 * IQR
     - Upper bound: Q3 + 1.5 * IQR
     - Points outside bounds are outliers

2. **Distance-Based:**
   - Points far from their neighbors
   - Use k-nearest neighbor distance
   - Threshold on distance to k-th neighbor

3. **Density-Based:**
   - Points in low-density regions
   - **LOF (Local Outlier Factor):**
     - Compare local density to neighbors' density
     - LOF >> 1 indicates outlier

4. **Clustering-Based:**
   - Points not belonging to any cluster
   - Points far from cluster centroids
   - Small clusters may be anomalies

### APPLICATIONS
- Fraud detection (credit card, insurance)
- Network intrusion detection
- Medical diagnosis
- Manufacturing defect detection
- Sensor data monitoring

---

## HOW TOPICS CONNECT - THE DATA MINING PIPELINE

```
                              Raw Data
                                  |
                                  v
              [TOPIC 2] Data Preprocessing
                   Clean, transform, normalize
                                  |
                                  v
              [TOPIC 3] Data Exploration
                   Understand distributions, patterns
                                  |
                                  v
         +------------------------+------------------------+
         |                        |                        |
         v                        v                        v
    SUPERVISED              UNSUPERVISED              PATTERN
    LEARNING                LEARNING                  DISCOVERY
         |                        |                        |
         v                        v                        v
   [TOPICS 4-5]             [TOPIC 7]               [TOPIC 6]
   Classification           Clustering              Association
   - Decision Trees         - K-Means               - Apriori
   - K-NN                   - Hierarchical          - Rules
   - Naive Bayes            - DBSCAN
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                           [TOPIC 8]
                        Anomaly Detection
                    (Can use any technique above)
```

---

## QUICK FORMULA REFERENCE

### GINI Index:
```
GINI(t) = 1 - SUM(p_i)^2
GINI_split = SUM(n_i/n * GINI_i)
```

### Entropy:
```
Entropy(t) = -SUM(p_i * log2(p_i))
Information Gain = Entropy(parent) - Weighted_Avg_Entropy(children)
```

### Classification Metrics:
```
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 * Precision * Recall / (Precision + Recall)
```

### Distance Metrics:
```
Euclidean: sqrt(SUM(x_i - y_i)^2)
Manhattan: SUM(|x_i - y_i|)
```

### Association Rules:
```
Support(A)       = count(A) / total_transactions
Confidence(A->B) = Support(A ∪ B) / Support(A)
Lift(A->B)       = Confidence(A->B) / Support(B)
```

### Clustering:
```
SSE = SUM_clusters SUM_points (distance(point, centroid))^2
```

### Anomaly Detection:
```
Z-score = (x - mean) / std_dev
IQR = Q3 - Q1
Outlier bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
```

---

*END OF OVERVIEW*
