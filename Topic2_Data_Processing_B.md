# TOPIC 2: DATA PROCESSING (Part B)
## SIMILARITY AND DISSIMILARITY MEASURES
### CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. Similarity and Dissimilarity - Definitions
2. Similarity/Dissimilarity for Simple Attributes
3. Distance Metrics
   - Euclidean Distance
   - Minkowski Distance
   - Mahalanobis Distance
4. Properties of Distance (Metric Properties)
5. Similarity Measures
   - Cosine Similarity
   - Correlation
6. Combining Similarities
7. Density Concepts
8. Worked Examples and Practice Problems

---

## 1. SIMILARITY AND DISSIMILARITY - DEFINITIONS

### SIMILARITY:
- Numerical measure of how **ALIKE** two data objects are
- **HIGHER** when objects are **MORE ALIKE**
- Often falls in the range [0, 1]
- s(p,q) = 1 means p and q are identical

### DISSIMILARITY (Distance):
- Numerical measure of how **DIFFERENT** two data objects are
- **LOWER** when objects are **MORE ALIKE**
- Minimum dissimilarity is often 0
- Upper limit varies (can be infinity)
- d(p,q) = 0 means p and q are identical

### PROXIMITY:
- General term that refers to EITHER similarity or dissimilarity

### KEY RELATIONSHIP:
Similarity and Dissimilarity are inversely related:
- High similarity = Low dissimilarity
- Low similarity = High dissimilarity

**Common conversions:**
- `s = 1 - d` (if d is normalized to [0,1])
- `s = 1 / (1 + d)`
- `s = 1 - (d - min_d) / (max_d - min_d)`

---

## 2. SIMILARITY/DISSIMILARITY FOR SIMPLE ATTRIBUTES

For two data objects with attribute values p and q:

| Attribute Type | Dissimilarity | Similarity |
|---------------|---------------|------------|
| **NOMINAL** | d = 0 if p = q, d = 1 if p ≠ q | s = 1 if p = q, s = 0 if p ≠ q |
| **ORDINAL** | d = \|p - q\| / (n - 1) | s = 1 - \|p - q\| / (n - 1) |
| **INTERVAL/RATIO** | d = \|p - q\| | s = -d, or s = 1/(1+d), or s = 1 - (d-min_d)/(max_d-min_d) |

*For ordinal: values mapped to integers 0 to n-1, n = number of values*

### EXAMPLES:

**Nominal Example:**
- Color: p = "red", q = "blue"
- d = 1 (different), s = 0

**Ordinal Example:**
- Size values: {Small, Medium, Large} mapped to {0, 1, 2}
- p = Small (0), q = Large (2)
- n = 3
- d = |0 - 2| / (3 - 1) = 2/2 = 1
- s = 1 - 1 = 0

**Interval/Ratio Example:**
- Temperature: p = 20°C, q = 25°C
- d = |20 - 25| = 5

---

## 3. DISTANCE METRICS

### A. EUCLIDEAN DISTANCE (L2 Norm)

**FORMULA:**
```
dist = √(Σ(p_k - q_k)²) for k=1 to n
```

Where:
- n = number of dimensions (attributes)
- p_k = k-th attribute of object p
- q_k = k-th attribute of object q

> **IMPORTANT:** Standardization is necessary if scales differ!

### QUIZ EXAMPLE (From Lecture):

Given 4 points:

| point | x | y |
|-------|---|---|
| p1 | 0 | 2 |
| p2 | 2 | 0 |
| p3 | 3 | 1 |
| p4 | 5 | 1 |

**Step-by-step for d(p1, p2):**
```
d(p1,p2) = √[(0-2)² + (2-0)²] = √[4 + 4] = √8 = 2.828
```

**COMPLETE DISTANCE MATRIX (L2/Euclidean):**

| | p1 | p2 | p3 | p4 |
|---|-------|-------|-------|-------|
| p1 | 0 | 2.828 | 3.162 | 5.099 |
| p2 | 2.828 | 0 | 1.414 | 3.162 |
| p3 | 3.162 | 1.414 | 0 | 2 |
| p4 | 5.099 | 3.162 | 2 | 0 |

---

### B. MINKOWSKI DISTANCE (Generalized)

**FORMULA:**
```
dist = (Σ |p_k - q_k|^r)^(1/r) for k=1 to n
```

Where r is a parameter that determines the type of distance.

### SPECIAL CASES:

**1. r = 1: MANHATTAN DISTANCE (City Block, Taxicab, L1 Norm)**
```
dist = Σ |p_k - q_k|
```
- Sum of absolute differences
- Called "city block" because it's like walking on a grid
- HAMMING DISTANCE is a special case for binary vectors (counts number of bits that differ)

**2. r = 2: EUCLIDEAN DISTANCE (L2 Norm)**
- Standard straight-line distance

**3. r → ∞: SUPREMUM DISTANCE (L_max, L_∞, Chebyshev Distance)**
```
dist = max |p_k - q_k| over all k
```
- Maximum difference between any component
- Only considers the largest difference

> **IMPORTANT:** Don't confuse r (the parameter) with n (number of dimensions)!
> All these distances work for ANY number of dimensions.

### MINKOWSKI DISTANCE EXAMPLE (Same 4 points):

| point | x | y |
|-------|---|---|
| p1 | 0 | 2 |
| p2 | 2 | 0 |
| p3 | 3 | 1 |
| p4 | 5 | 1 |

**L1 (Manhattan) Distance Matrix:**

| | p1 | p2 | p3 | p4 |
|---|----|----|----|----|
| p1 | 0 | 4 | 4 | 6 |
| p2 | 4 | 0 | 2 | 4 |
| p3 | 4 | 2 | 0 | 2 |
| p4 | 6 | 4 | 2 | 0 |

Example: d_L1(p1, p2) = |0-2| + |2-0| = 2 + 2 = 4

**L∞ (Supremum) Distance Matrix:**

| | p1 | p2 | p3 | p4 |
|---|----|----|----|----|
| p1 | 0 | 2 | 3 | 5 |
| p2 | 2 | 0 | 1 | 3 |
| p3 | 3 | 1 | 0 | 2 |
| p4 | 5 | 3 | 2 | 0 |

Example: d_L∞(p1, p4) = max(|0-5|, |2-1|) = max(5, 1) = 5

---

### C. MAHALANOBIS DISTANCE

**PURPOSE:**
- Accounts for **CORRELATIONS** between variables
- Considers the **DISTRIBUTION/SPREAD** of the data
- Useful when data has different scales AND correlations

**FORMULA:**
```
mahalanobis(p, q) = √[(p - q) Σ^(-1) (p - q)^T]
```

Where:
- Σ (Sigma) is the **COVARIANCE MATRIX** of the data
- Σ^(-1) is the inverse of the covariance matrix
- T denotes transpose

**COVARIANCE MATRIX FORMULA:**
```
Σ_j,k = (1/(n-1)) Σ (X_ij - X̄_j)(X_ik - X̄_k) for i=1 to n
```

### WHY MAHALANOBIS?

Consider a dataset with correlated variables:
```
        *
      * * *
    * * * * *        Points A and B have the SAME Euclidean
  A * * * * * * B    distance from the center, BUT:
    * * * * *        - A is an outlier (perpendicular to data spread)
      * * *          - B follows the data pattern
        *
```

Mahalanobis distance recognizes that B is "closer" to the distribution than A!

**EXAMPLE FROM LECTURE:**
- Two red points at (-6, -3) and (7, 3)
- Euclidean distance between them: 14.7
- Mahalanobis distance between them: 6

The Mahalanobis distance is smaller because it accounts for the spread of the data along its principal axis.

**ANOTHER EXAMPLE:**

Given covariance matrix:
```
        | 0.3  0.2 |
    Σ = |          |
        | 0.2  0.3 |
```

Points: A = (0.5, 0.5), B = (0, 1), C = (1.5, 1.5)

Results:
- Mahal(A, B) = 5
- Mahal(A, C) = 4

Even though C is farther in Euclidean distance, it has a **SMALLER Mahalanobis distance** because it lies along the direction of data spread.

---

## 4. PROPERTIES OF DISTANCE (METRIC PROPERTIES)

A distance function d(p, q) is a **METRIC** if it satisfies:

### 1. POSITIVE DEFINITENESS:
- d(p, q) ≥ 0 for all p and q
- d(p, q) = 0 **ONLY IF** p = q

### 2. SYMMETRY:
- d(p, q) = d(q, p) for all p and q

### 3. TRIANGLE INEQUALITY:
- d(p, r) ≤ d(p, q) + d(q, r) for all p, q, r
- (Direct path is never longer than going through an intermediate point)

### EXAMPLES OF METRICS:
- Euclidean distance: YES (is a metric)
- Manhattan distance: YES (is a metric)
- Minkowski distance (r ≥ 1): YES (is a metric)
- Mahalanobis distance: YES (is a metric)

---

## 5. PROPERTIES OF SIMILARITY

Similarity function s(p, q) typically satisfies:

1. **s(p, q) = 1** (or maximum) ONLY IF p = q
2. **SYMMETRY:** s(p, q) = s(q, p) for all p and q

> **NOTE:** Triangle inequality does NOT apply to similarity in the same way as distance.

---

## 6. SIMILARITY MEASURES

### A. COSINE SIMILARITY

**DEFINITION:** Measures the cosine of the angle between two vectors.

**FORMULA:**
```
cos(d1, d2) = (d1 · d2) / (||d1|| × ||d2||)
```

Where:
- d1 · d2 = dot product of vectors
- ||d|| = length (magnitude) of vector d = √(Σ d_k²)

### PROPERTIES:
- Range: [-1, 1] for general vectors, [0, 1] for non-negative vectors
- cos = 1: vectors point in same direction (identical)
- cos = 0: vectors are perpendicular (orthogonal)
- cos = -1: vectors point in opposite directions

### COMMONLY USED FOR:
- Document similarity (text mining)
- Recommendation systems
- Any sparse, high-dimensional data

### WORKED EXAMPLE (From Lecture):

```
d1 = [3, 2, 0, 5, 0, 0, 0, 2, 0, 0]
d2 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 2]
```

**Step 1: Calculate dot product (d1 · d2)**
```
d1 · d2 = 3×1 + 2×0 + 0×0 + 5×0 + 0×0 + 0×0 + 0×0 + 2×1 + 0×0 + 0×2
       = 3 + 0 + 0 + 0 + 0 + 0 + 0 + 2 + 0 + 0
       = 5
```

**Step 2: Calculate ||d1||**
```
||d1|| = √(3² + 2² + 0² + 5² + 0² + 0² + 0² + 2² + 0² + 0²)
      = √(9 + 4 + 0 + 25 + 0 + 0 + 0 + 4 + 0 + 0)
      = √42
      = 6.481
```

**Step 3: Calculate ||d2||**
```
||d2|| = √(1² + 0² + 0² + 0² + 0² + 0² + 0² + 1² + 0² + 2²)
      = √(1 + 0 + 0 + 0 + 0 + 0 + 0 + 1 + 0 + 4)
      = √6
      = 2.449
```

**Step 4: Calculate cosine similarity**
```
cos(d1, d2) = 5 / (6.481 × 2.449)
            = 5 / 15.876
            = 0.315
```

**INTERPRETATION:** The documents have moderate similarity (0.315).

---

### B. CORRELATION (Pearson Correlation Coefficient)

**DEFINITION:** Measures the LINEAR RELATIONSHIP between two objects.

**FORMULA:**

**Step 1:** Standardize each object
```
p'_k = (p_k - mean(p)) / std(p)
q'_k = (q_k - mean(q)) / std(q)
```

**Step 2:** Compute correlation as dot product
```
correlation(p, q) = p' · q'
```

**ALTERNATIVE FORMULA:**
```
r = Σ(p_k - p̄)(q_k - q̄) / [√(Σ(p_k - p̄)²) × √(Σ(q_k - q̄)²)]
```

### PROPERTIES:
- Range: [-1, 1]
- r = 1: Perfect positive correlation
- r = 0: No linear correlation
- r = -1: Perfect negative correlation

### CORRELATION vs. COSINE:
- **Correlation:** Centers data before computing (subtracts mean)
- **Cosine:** Does not center data
- Correlation measures linear relationship
- Cosine measures directional similarity

---

## 7. COMBINING SIMILARITIES

When dealing with **MIXED ATTRIBUTE TYPES:**

### GENERAL APPROACH:
1. For each attribute k, compute similarity s_k in range [0, 1]

2. Define indicator variable δ_k:
   - δ_k = 0 if k-th attribute is binary asymmetric AND both objects have value 0, OR if one object has missing value
   - δ_k = 1 otherwise

3. Compute overall similarity:
```
similarity(p,q) = Σ(δ_k × s_k) / Σ(δ_k) for k=1 to n
```

### USING WEIGHTS:
When some attributes are more important:
```
similarity(p,q) = Σ(w_k × δ_k × s_k) / Σ(δ_k) for k=1 to n
```

Where w_k are weights between 0 and 1 that sum to 1.

### WEIGHTED DISTANCE:
```
distance(p,q) = (Σ w_k |p_k - q_k|^r)^(1/r) for k=1 to n
```

---

## 8. DENSITY CONCEPTS

Density is important for:
- Density-based clustering (DBSCAN)
- Outlier detection

### TYPES OF DENSITY:

### A. EUCLIDEAN DENSITY (Cell-based):
- Divide space into rectangular cells of equal volume
- Density = number of points in each cell

Example Grid (7×7 region):
```
|  0 |  0 |  0 |  0 |  0 |  0 |  0 |
|  0 |  0 |  0 |  0 |  0 |  0 |  0 |
|  4 | 17 | 18 |  6 |  0 |  0 |  0 |
| 14 | 14 | 13 | 13 |  0 | 18 | 27 |
| 11 | 18 | 10 | 21 |  0 | 24 | 31 |
|  3 | 20 | 14 |  4 |  0 |  0 |  0 |
|  0 |  0 |  0 |  0 |  0 |  0 |  0 |
```

High-density cells have many points (e.g., 31, 27)
Low-density cells have few or zero points

### B. EUCLIDEAN DENSITY (Center-based):
- For each point, count points within radius r
- Density of point p = # points within distance r of p

### C. PROBABILITY DENSITY:
- Based on probability distributions
- Estimates likelihood at each point

### D. GRAPH-BASED DENSITY:
- Based on connectivity in a graph
- Number of edges or neighbors

---

## 9. PRACTICE PROBLEMS

### PROBLEM 1: Euclidean Distance
Given points: A = (1, 2), B = (4, 6)
Calculate Euclidean distance.

**Solution:**
```
d(A,B) = √[(4-1)² + (6-2)²]
       = √[9 + 16]
       = √25
       = 5
```

### PROBLEM 2: Manhattan Distance
Given points: A = (1, 2), B = (4, 6)
Calculate Manhattan distance.

**Solution:**
```
d(A,B) = |4-1| + |6-2|
       = 3 + 4
       = 7
```

### PROBLEM 3: Supremum Distance
Given points: A = (1, 2, 3), B = (5, 3, 1)
Calculate L∞ distance.

**Solution:**
```
d(A,B) = max(|5-1|, |3-2|, |1-3|)
       = max(4, 1, 2)
       = 4
```

### PROBLEM 4: Cosine Similarity
Given vectors: x = [1, 0, 1], y = [1, 1, 0]
Calculate cosine similarity.

**Solution:**
```
x · y = 1×1 + 0×1 + 1×0 = 1
||x|| = √(1 + 0 + 1) = √2
||y|| = √(1 + 1 + 0) = √2

cos(x,y) = 1 / (√2 × √2) = 1/2 = 0.5
```

### PROBLEM 5: Which Distance Metric?

| Scenario | Best Metric | Reason |
|----------|-------------|--------|
| Comparing two documents by word frequency | COSINE SIMILARITY | Documents may have different lengths; cosine normalizes by length |
| Finding outliers considering variable correlations | MAHALANOBIS DISTANCE | Accounts for covariance structure of data |
| Grid-based city navigation | MANHATTAN (L1) DISTANCE | Can only travel along grid lines |

### PROBLEM 6: Nominal Similarity

Two objects with attribute values:
- Object 1: (Red, Large, Yes)
- Object 2: (Red, Small, Yes)

Calculate simple matching similarity.

**Solution:**
```
Matches: Red=Red (1), Large≠Small (0), Yes=Yes (1)
Similarity = 2/3 = 0.667
```

---

## KEY FORMULAS SUMMARY

| Distance/Similarity | Formula |
|---------------------|---------|
| EUCLIDEAN (L2) | d = √[Σ(p_k - q_k)²] |
| MANHATTAN (L1) | d = Σ\|p_k - q_k\| |
| MINKOWSKI (Lr) | d = (Σ\|p_k - q_k\|^r)^(1/r) |
| SUPREMUM (L∞) | d = max\|p_k - q_k\| |
| MAHALANOBIS | d = √[(p-q)Σ^(-1)(p-q)^T] |
| COSINE | s = (p·q) / (\|\|p\|\| × \|\|q\|\|) |
| CORRELATION | r = Σ(p'_k × q'_k) where p', q' are standardized |

---

## QUICK REVIEW CHECKLIST

- [ ] Understand difference between similarity and dissimilarity
- [ ] Know formulas for simple attribute similarity/dissimilarity
- [ ] Can calculate Euclidean distance
- [ ] Can calculate Manhattan distance
- [ ] Can calculate Supremum (L∞) distance
- [ ] Understand Minkowski distance and its special cases
- [ ] Know when to use Mahalanobis distance
- [ ] Know the three metric properties (positive definiteness, symmetry, triangle inequality)
- [ ] Can calculate cosine similarity step-by-step
- [ ] Understand correlation and how it differs from cosine
- [ ] Know how to combine similarities for mixed attribute types
- [ ] Understand cell-based vs. center-based density

---

*END OF TOPIC 2B*
