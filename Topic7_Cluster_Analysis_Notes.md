# CLUSTER ANALYSIS (BASIC CLUSTERING)
## CS653: DATA MINING
### Detailed Study Notes - Lecture 07-A

---

## TABLE OF CONTENTS
1. Introduction to Cluster Analysis
2. Applications of Cluster Analysis
3. What is NOT Cluster Analysis
4. Types of Clusterings
5. Types of Clusters
6. K-means Clustering Algorithm
7. K-means Evaluation and SSE
8. Problems with Initial Centroids
9. Solutions and Improvements
10. Limitations of K-means
11. Hierarchical Clustering
12. Agglomerative Clustering Algorithm
13. Examples and Problems
14. Summary

---

## 1. INTRODUCTION TO CLUSTER ANALYSIS

### WHAT IS CLUSTER ANALYSIS?

**Definition:** Finding groups of objects such that:
- Objects in a group are **SIMILAR** (or related) to one another
- Objects in different groups are **DIFFERENT** (or unrelated) from each other

### KEY PRINCIPLES

**Two Main Goals:**
1. **Intra-cluster distances are MINIMIZED** - Points within the same cluster should be close together
2. **Inter-cluster distances are MAXIMIZED** - Points in different clusters should be far apart

```
    Cluster A          Cluster B
     (o o)              (x x)
    (o o o)     <-->   (x x x)
     (o o)              (x x)
       ^                  ^
       |                  |
  Small distances    Large distance
  within cluster     between clusters
```

### CLUSTERING vs CLASSIFICATION

| Clustering | Classification |
|------------|----------------|
| UNSUPERVISED learning | SUPERVISED learning |
| No class labels | Has class labels |
| Discovers hidden structure | Predicts known categories |
| Groups based on similarity | Assigns to predefined classes |


---

## 2. APPLICATIONS OF CLUSTER ANALYSIS

### UNDERSTANDING
- **Document grouping:** Group related documents for browsing
- **Gene/Protein analysis:** Group genes and proteins with similar functionality
- **Stock market:** Group stocks with similar price fluctuations

**Example: Stock Market Clustering**

| Cluster | Discovered Clusters | Industry Group |
|---------|---------------------|----------------|
| 1 | Applied-Matl-DOWN, Bay-Network-Down, CISCO-DOWN, HP-DOWN, INTEL-DOWN, etc. | Technology1-DOWN |
| 2 | Apple-Comp-DOWN, Autodesk-DOWN, Microsoft-DOWN, etc. | Technology2-DOWN |
| 3 | Fannie-Mae-DOWN, Fed-Home-Loan-DOWN, Morgan-Stanley-DOWN | Financial-DOWN |
| 4 | Baker-Hughes-UP, Halliburton-HLD-UP, Phillips-Petro-UP, etc. | Oil-UP |

### SUMMARIZATION
- **Data reduction:** Reduce the size of large data sets
- **Example:** Clustering precipitation data in Australia into distinct climate regions


---

## 3. WHAT IS NOT CLUSTER ANALYSIS

### SUPERVISED CLASSIFICATION
- Has class label information beforehand
- Uses training data to build models
- **NOT clustering** - clustering has no labels

### SIMPLE SEGMENTATION
- Dividing data based on arbitrary rules
- Example: Dividing students into registration groups alphabetically by last name
- **NOT clustering** - no similarity consideration

### RESULTS OF A QUERY
- Groupings from external specification (SQL WHERE clause, filters)
- **NOT clustering** - groupings are predefined, not discovered

### GRAPH PARTITIONING
- Some overlap with clustering concepts
- But areas are not identical
- Different objectives and constraints


---

## 4. TYPES OF CLUSTERINGS

### IMPORTANT DISTINCTION: Hierarchical vs Partitional

### PARTITIONAL CLUSTERING
- **Definition:** Division of data objects into **non-overlapping subsets** (clusters)
- Each data object is in **exactly one subset**
- Produces a flat structure of K clusters

```
Original Points          Partitional Clustering
    o o                     [Cluster 1]
   o o o    -->           o o o o o
    o o
                          [Cluster 2]
  x x x                   x x x x x
 x x x x
```

### HIERARCHICAL CLUSTERING
- **Definition:** A set of **nested clusters** organized as a hierarchical tree
- Can be visualized as a **dendrogram**
- Shows sequences of merges or splits

```
Traditional Hierarchical:        Traditional Dendrogram:
    _______________                    |
   /               \                 /   \
  /     _____       \              /       \
 /     /     \       \            |         |
p1    p2    p3  p4              p1  p2   p3  p4
```

**Non-traditional Hierarchical Clustering:**
- Allows overlapping clusters at same level
- More flexible representation

### OTHER DISTINCTIONS BETWEEN SETS OF CLUSTERS

| Distinction | Type 1 | Type 2 |
|-------------|--------|--------|
| **Membership** | Exclusive (point in ONE cluster) | Non-exclusive (point in MULTIPLE clusters) |
| **Assignment** | Non-fuzzy (hard assignment) | Fuzzy (weighted membership 0-1) |
| **Coverage** | Complete (all data clustered) | Partial (some data excluded) |
| **Uniformity** | Homogeneous (similar sizes/shapes) | Heterogeneous (different sizes/shapes/densities) |

### FUZZY CLUSTERING
- A point belongs to **every cluster** with some weight between 0 and 1
- **Weights must sum to 1**
- Probabilistic clustering has similar characteristics

**Example:**
```
Point X membership:
- Cluster A: 0.7
- Cluster B: 0.2
- Cluster C: 0.1
Total: 0.7 + 0.2 + 0.1 = 1.0
```


---

## 5. TYPES OF CLUSTERS

### 1. WELL-SEPARATED CLUSTERS
- Each point is closer to all points in its cluster than to any point in another cluster
- Clear boundaries between clusters

### 2. CENTER-BASED CLUSTERS
- Each point is closer to the **center (centroid)** of its cluster than to any other center
- K-means produces this type

### 3. CONTIGUOUS CLUSTERS (Nearest Neighbor)
- Each point is closer to **at least one point** in its cluster than to any point outside
- Can form chains of similar points

### 4. DENSITY-BASED CLUSTERS
- Cluster is a **dense region** of points separated by low-density regions
- Good for arbitrary shapes
- Examples: DBSCAN, OPTICS

### 5. PROPERTY OR CONCEPTUAL CLUSTERS
- Points share some common property
- Defined by conceptual description

### 6. DESCRIBED BY OBJECTIVE FUNCTION
- Clusters minimize or maximize an objective
- Example: Minimize Sum of Squared Errors (SSE)


---

## 6. K-MEANS CLUSTERING ALGORITHM

### OVERVIEW
- **Type:** Partitional clustering approach
- **Key feature:** Each cluster is associated with a **centroid** (center point)
- **Assignment:** Each point assigned to cluster with **closest centroid**
- **Requirement:** Number of clusters K must be **specified in advance**

### BASIC K-MEANS ALGORITHM

```
ALGORITHM: K-means
INPUT: K (number of clusters), Dataset D
OUTPUT: K clusters

1: Select K points as the initial centroids
2: REPEAT
3:     Form K clusters by assigning all points to the closest centroid
4:     Recompute the centroid of each cluster
5: UNTIL The centroids don't change (convergence)
```

### VISUAL DEMONSTRATION

```
Step 1: Data points         Step 2: Initial centroids (X marks)
    o o                         o o
   o o o                       o o o
    o o                X        o o
                                        X
  o o o                       o o o
 o o o o                     o o o o

Step 3: Assign points       Step 4: Recalculate centroids
    . .                         . .
   . . .     (red)             . . .      X (new)
    . .                         . .

  x x x      (blue)           x x x
 x x x x                     x x x x     X (new)

Step 5: Repeat until convergence
```

### K-MEANS CLUSTERING DETAILS

**Initial Centroids:**
- Often chosen **randomly**
- Clusters produced **vary from one run to another**

**Centroid Calculation:**
- Centroid is the **mean** of all points in the cluster
- For cluster C with points x1, x2, ..., xn:
```
centroid = (1/n) * Σ xi
```

**Closeness Measures:**
- Euclidean distance (most common)
- Cosine similarity
- Correlation

**Convergence:**
- K-means will converge for common similarity measures
- Most convergence happens in **first few iterations**
- Alternative stopping: "Until relatively few points change clusters"

**Time Complexity:**
```
O(n * K * I * d)

Where:
- n = number of points
- K = number of clusters
- I = number of iterations
- d = number of attributes (dimensions)
```

### QUIZ: K-MEANS vs K-NEAREST NEIGHBOR

| K-means | K-Nearest Neighbor |
|---------|-------------------|
| Unsupervised | Supervised |
| Clustering algorithm | Classification algorithm |
| K = number of clusters | K = number of neighbors |
| Creates groups | Assigns to existing classes |
| Uses centroids | Uses nearest data points |

**Similarities:**
- Both use distance/similarity measures
- Both have parameter K
- Both partition the feature space


---

## 7. K-MEANS EVALUATION AND SSE

### SUM OF SQUARED ERROR (SSE)

**Definition:** Most common measure for evaluating K-means clustering quality

**Formula:**
```
        K
SSE = Σ    Σ    dist²(mi, x)
      i=1  x∈Ci

Where:
- K = number of clusters
- Ci = cluster i
- x = data point in cluster Ci
- mi = centroid (representative point) of cluster Ci
- dist² = squared distance
```

### PROPERTIES OF SSE

1. **Lower SSE = Better clustering**
   - For each point, error = distance to nearest centroid
   - SSE sums the squared errors

2. **Centroid minimizes SSE**
   - Can prove that mi = mean of cluster minimizes SSE
   - This is why we use the mean as centroid

3. **Comparing clusterings**
   - Given two clusterings, choose one with smaller SSE

4. **Effect of K on SSE**
   - Increasing K reduces SSE (more clusters = lower error)
   - But more clusters isn't always better!
   - A good clustering with smaller K can have lower SSE than poor clustering with higher K

### EXAMPLE: SSE Calculation

```
Cluster 1 points: (1,1), (2,1), (1,2)
Centroid m1 = ((1+2+1)/3, (1+1+2)/3) = (1.33, 1.33)

SSE for Cluster 1:
= dist²(m1, (1,1)) + dist²(m1, (2,1)) + dist²(m1, (1,2))
= (0.33² + 0.33²) + (0.67² + 0.33²) + (0.33² + 0.67²)
= 0.22 + 0.56 + 0.56
= 1.34
```


---

## 8. PROBLEMS WITH INITIAL CENTROIDS

### THE INITIAL CENTROID PROBLEM

**Issue:** Random initial centroids can lead to:
- Sub-optimal clustering results
- Different results on different runs
- Convergence to local minimum instead of global minimum

### PROBABILITY OF GOOD INITIALIZATION

If there are K 'real' clusters with same size n:

```
P = (number of ways to select one centroid from each cluster) / (number of ways to select K centroids)

P = K! × n^K / (Kn)^K = K! / K^K
```

**Example:** If K = 10:
```
P = 10! / 10^10 = 3,628,800 / 10,000,000,000 = 0.00036 = 0.036%
```

**Very low probability** of selecting one centroid from each real cluster!

### VISUAL EXAMPLE: 10 CLUSTERS

**Scenario:** 5 pairs of clusters (10 total), but we start with 2 centroids in ONE cluster of each pair

**Result:**
- Algorithm converges
- But finds **sub-optimal** clustering
- Pairs of clusters merged incorrectly
- Cannot recover correct structure

### IMPORTANCE OF INITIAL CENTROIDS - DEMONSTRATION

**Good Initialization:**
```
Iteration 1: Centroids placed well
Iteration 2-6: Minor adjustments
Result: Optimal clustering found
```

**Bad Initialization:**
```
Iteration 1: Both centroids in top cluster
Iteration 2-6: Algorithm converges but...
Result: Sub-optimal clustering - one cluster split, two clusters merged
```

### TWO DIFFERENT K-MEANS RESULTS

Same data, different initial centroids:

```
Original Points:        Optimal (SSE=low):      Sub-optimal (SSE=high):
  * * *                   [Cluster 1]             [Mixed clusters]
 * * * *                  * * * * *               * * * x x

 o o o o                  [Cluster 2]             [Mixed clusters]
  o o o                   o o o o o               o o x x x

 x x x x                  [Cluster 3]             [Different grouping]
  x x x                   x x x x x               x x o o o
```


---

## 9. SOLUTIONS AND IMPROVEMENTS

### SOLUTIONS TO INITIAL CENTROID PROBLEM

#### 1. MULTIPLE RUNS
- Run K-means multiple times with different random initializations
- Keep the result with lowest SSE
- **Limitation:** Probability still not favorable for large K

#### 2. HIERARCHICAL CLUSTERING INITIALIZATION
- First, sample data and apply hierarchical clustering
- Use hierarchical clustering results to determine initial centroids
- More likely to get good starting points

#### 3. SELECT MORE THAN K INITIAL CENTROIDS
- Select more than K candidates
- Then choose K that are **most widely separated**
- Reduces chance of starting with close centroids

#### 4. POSTPROCESSING
- After K-means converges, refine the results
- Split, merge, or relocate clusters

#### 5. BISECTING K-MEANS
- Start with all points in one cluster
- Repeatedly split clusters using 2-means
- **Less susceptible** to initialization issues

```
ALGORITHM: Bisecting K-means
1: Start with all points in one cluster
2: REPEAT
3:     Select a cluster to split (e.g., largest or highest SSE)
4:     Apply 2-means to split selected cluster
5: UNTIL K clusters are obtained
```

### HANDLING EMPTY CLUSTERS

**Problem:** Basic K-means can yield empty clusters when all points are closer to other centroids

**Strategies:**
1. **Choose point with highest SSE contribution** as new centroid
2. **Choose point from cluster with highest SSE** as new centroid
3. **Repeat** if there are several empty clusters

### UPDATING CENTERS INCREMENTALLY

**Basic K-means:** Update all centroids AFTER assigning all points

**Incremental K-means:** Update centroid AFTER each point assignment

| Basic K-means | Incremental K-means |
|---------------|---------------------|
| Batch updates | Online updates |
| Less expensive per iteration | More expensive |
| No order dependency | Order dependent |
| Can have empty clusters | Never get empty cluster |
| Standard approach | Can use weights to change impact |

### PRE-PROCESSING AND POST-PROCESSING

#### PRE-PROCESSING
- **Normalize the data** - Ensure all features on same scale
- **Eliminate outliers** - Remove extreme points that can distort centroids

#### POST-PROCESSING
- **Eliminate small clusters** - May represent outliers
- **Split 'loose' clusters** - Clusters with relatively high SSE
- **Merge 'close' clusters** - Clusters that are nearby with low SSE
- **ISODATA algorithm** - Uses these steps during clustering process


---

## 10. LIMITATIONS OF K-MEANS

### K-MEANS STRUGGLES WITH:

#### 1. DIFFERING CLUSTER SIZES

```
Original:                    K-means Result (Wrong):
  Small cluster              Boundary drawn incorrectly!
    o o                        o|o
   o o o                      o o|o
                      +         |
  Large cluster               x x|x
 x x x x x x                x x x|x x
x x x x x x x              x x x x|x x
```

K-means tends to create **equal-sized clusters**, splitting large clusters and merging small ones.

#### 2. DIFFERING CLUSTER DENSITIES

```
Original:                    K-means Result (Wrong):
  Sparse cluster               Points from sparse cluster
    o   o                      incorrectly assigned to
  o   o   o                    dense clusters!

  Dense cluster 1     Dense cluster 2
   ****              ^^^^
  ******            ^^^^^^
```

Dense clusters "pull" points from sparse clusters.

#### 3. NON-GLOBULAR SHAPES

```
Original (two crescents):    K-means Result (Wrong):
    ****                       ***|***
   *    *                     *   |  *
  *      ****                *    |   ****
 *          *               *     |      *
  ****   ***                 ****|   ***
```

K-means assumes **spherical/globular clusters** due to Euclidean distance.

#### 4. OUTLIERS

- Outliers can significantly affect centroid positions
- Can create their own clusters or distort existing ones

### OVERCOMING K-MEANS LIMITATIONS

**Solution: Use MORE clusters than expected, then merge**

```
Original (3 natural clusters):   Use K=6-8 clusters:
    ooo                            o|o|o
   ooooo                          oo|ooo
  ooooooo                        ooo|oooo
                                    |
                                 Find sub-clusters
                                 Then merge related ones
```

**Result:**
- Finds "parts" of clusters correctly
- Post-processing merges them into correct final clusters
- Works for different sizes, densities, and shapes


---

## 11. HIERARCHICAL CLUSTERING

### OVERVIEW

**Definition:** Produces a set of **nested clusters** organized as a hierarchical tree

**Dendrogram:** A tree-like diagram that records sequences of merges or splits

```
Points:                      Dendrogram:
  o1                              |
     o3  o4                    ___|___
  o2                          |       |
                            __|__    _|_
     o5  o6                |     |  |   |
                           1  3  2  5   4  6

Height represents distance at which clusters merge
```

### STRENGTHS OF HIERARCHICAL CLUSTERING

1. **No need to specify K in advance**
   - Any desired number of clusters obtained by "cutting" dendrogram at proper level

2. **Meaningful taxonomies**
   - Natural representation for biological sciences
   - Example: Animal kingdom, phylogeny reconstruction

3. **Complete clustering history**
   - Can examine clustering at any level
   - Understand how clusters form

### TWO MAIN TYPES

#### AGGLOMERATIVE (Bottom-up)
- **Start:** Each point is its own cluster
- **Process:** At each step, merge the two **closest** clusters
- **End:** Until only one cluster (or K clusters) remain
- **More popular** approach

#### DIVISIVE (Top-down)
- **Start:** One cluster containing all points
- **Process:** At each step, **split** a cluster
- **End:** Until each cluster contains one point (or K clusters)

```
Agglomerative:              Divisive:
o o o o o                   [  all points  ]
  \ /   |                          |
   o    o  o                 _____|_____
    \  / \ /                |           |
     o    o                [left]    [right]
      \  /                    |          |
       o                   further    splits
```


---

## 12. AGGLOMERATIVE CLUSTERING ALGORITHM

### BASIC ALGORITHM

```
ALGORITHM: Agglomerative Hierarchical Clustering
INPUT: Dataset D, Proximity function
OUTPUT: Dendrogram (hierarchical tree of clusters)

1: Compute the proximity matrix
2: Let each data point be a cluster
3: REPEAT
4:     Merge the two closest clusters
5:     Update the proximity matrix
6: UNTIL only a single cluster remains
```

### KEY OPERATION: Computing Proximity Between Clusters

Different approaches define different algorithms:

| Method | Distance Between Clusters |
|--------|---------------------------|
| **Single Linkage** (MIN) | Distance between closest points |
| **Complete Linkage** (MAX) | Distance between farthest points |
| **Average Linkage** | Average distance between all pairs |
| **Centroid** | Distance between centroids |
| **Ward's Method** | Increase in SSE if merged |

### STARTING SITUATION

```
Points:                    Proximity Matrix:
  o p1                        p1  p2  p3  p4  p5
      o p3  o p4          p1  0
  o p2                    p2  .   0
            o p5          p3  .   .   0
                          p4  .   .   .   0
                          p5  .   .   .   .   0

Each point is its own cluster
Proximity matrix shows distances between all pairs
```

### INTERMEDIATE SITUATION

```
After some merges:          Updated Proximity Matrix:
                               C1   C2   C3   C4   C5
  [C3]                    C1   0
            [C4]          C2   .    0
                          C3   .    .    0
[C1]                      C4   .    .    .    0
                          C5   .    .    .    .    0
    [C2]      [C5]
                          Matrix now shows distances
                          between CLUSTERS
```

### DENDROGRAM CONSTRUCTION

```
As algorithm progresses:

Step 1:  o   o   o   o   o   o     (6 clusters)
         p1  p2  p3  p4  p5  p6

Step 2:  o   o   o___o   o   o     (5 clusters - p3,p4 merged)
         p1  p2  p3  p4  p5  p6

Step 3:  o___o   o___o   o   o     (4 clusters - p1,p2 merged)
         p1  p2  p3  p4  p5  p6

Step 4:  o___o   o___o   o___o     (3 clusters - p5,p6 merged)
         p1  p2  p3  p4  p5  p6

...continue until one cluster...
```


---

## 13. EXAMPLES AND PROBLEMS

### PROBLEM 1: K-means Calculation

**Given data points:**
```
P1 = (2, 10)
P2 = (2, 5)
P3 = (8, 4)
P4 = (5, 8)
P5 = (7, 5)
P6 = (6, 4)
P7 = (1, 2)
P8 = (4, 9)
```

**Initial centroids:** C1 = (2, 10), C2 = (5, 8), C3 = (1, 2)

**First iteration - Assign points to nearest centroid:**

| Point | Distance to C1 | Distance to C2 | Distance to C3 | Assigned |
|-------|---------------|----------------|----------------|----------|
| P1(2,10) | 0 | 3.6 | 8.1 | C1 |
| P2(2,5) | 5 | 4.2 | 3.2 | C3 |
| P3(8,4) | 8.5 | 5 | 7.3 | C2 |
| P4(5,8) | 3.6 | 0 | 7.2 | C2 |
| P5(7,5) | 7.1 | 3.6 | 6.7 | C2 |
| P6(6,4) | 7.2 | 4.1 | 5.4 | C2 |
| P7(1,2) | 8.1 | 7.2 | 0 | C3 |
| P8(4,9) | 2.2 | 1.4 | 7.6 | C2 |

**Clusters after first iteration:**
- C1: {P1}
- C2: {P3, P4, P5, P6, P8}
- C3: {P2, P7}

**Recalculate centroids:**
- New C1 = (2, 10)
- New C2 = ((8+5+7+6+4)/5, (4+8+5+4+9)/5) = (6, 6)
- New C3 = ((2+1)/2, (5+2)/2) = (1.5, 3.5)

Continue until convergence...

---

### PROBLEM 2: SSE Calculation

**Given clustering result:**
- Cluster 1: {(1,1), (2,2), (3,1)} with centroid (2, 1.33)
- Cluster 2: {(8,8), (9,9), (10,8)} with centroid (9, 8.33)

**Calculate SSE:**

**Cluster 1:**
```
dist²((2,1.33), (1,1)) = (2-1)² + (1.33-1)² = 1 + 0.11 = 1.11
dist²((2,1.33), (2,2)) = (2-2)² + (1.33-2)² = 0 + 0.45 = 0.45
dist²((2,1.33), (3,1)) = (2-3)² + (1.33-1)² = 1 + 0.11 = 1.11
SSE_C1 = 1.11 + 0.45 + 1.11 = 2.67
```

**Cluster 2:**
```
dist²((9,8.33), (8,8)) = (9-8)² + (8.33-8)² = 1 + 0.11 = 1.11
dist²((9,8.33), (9,9)) = (9-9)² + (8.33-9)² = 0 + 0.45 = 0.45
dist²((9,8.33), (10,8)) = (9-10)² + (8.33-8)² = 1 + 0.11 = 1.11
SSE_C2 = 1.11 + 0.45 + 1.11 = 2.67
```

**Total SSE = SSE_C1 + SSE_C2 = 2.67 + 2.67 = 5.34**

---

### PROBLEM 3: Probability of Good Initialization

**Question:** With K=5 clusters of equal size, what is the probability of randomly selecting one initial centroid from each cluster?

**Solution:**
```
P = K! / K^K = 5! / 5^5 = 120 / 3125 = 0.0384 = 3.84%
```

Only about 4% chance of good initialization!

---

### PROBLEM 4: When K-means Fails

**Given:** Two natural clusters - one large circle (100 points) and one small cluster (20 points) nearby

**Why K-means with K=2 fails:**
1. K-means tries to minimize total SSE
2. Splitting large cluster creates lower SSE than separating small cluster
3. Result: Large cluster split, small cluster absorbed

**Solution:** Use K=3 or more, then merge.

---

### PROBLEM 5: Hierarchical Clustering

**Given distance matrix:**
```
     A    B    C    D    E
A    0
B    2    0
C    6    5    0
D    10   9    4    0
E    9    8    5    3    0
```

**Single Linkage Agglomerative Clustering:**

**Step 1:** Find minimum distance = 2 (A-B)
- Merge A and B into cluster {A,B}
- Update distances using MIN

**Step 2:** Updated matrix:
```
      {A,B}  C    D    E
{A,B}   0
C       5    0
D       9    4    0
E       8    5    3    0
```
Minimum = 3 (D-E), merge into {D,E}

**Step 3:** Updated matrix:
```
      {A,B}  C    {D,E}
{A,B}   0
C       5    0
{D,E}   8    4    0
```
Minimum = 4 (C-{D,E}), merge into {C,D,E}

**Step 4:** Updated matrix:
```
       {A,B}  {C,D,E}
{A,B}    0
{C,D,E}  5      0
```
Merge into {A,B,C,D,E}

**Dendrogram:**
```
Height
  |
5 |     _________
  |    |         |
4 |    |     ____|____
  |    |    |         |
3 |    |    |    _____|_____
  |    |    |   |           |
2 |  __|__  |   |           |
  | |     | |   |           |
0 | A     B C   D           E
```


---

## 14. SUMMARY

### KEY CONCEPTS

1. **CLUSTER ANALYSIS:** Unsupervised grouping of similar objects
   - Minimize intra-cluster distance
   - Maximize inter-cluster distance

2. **TYPES OF CLUSTERING:**
   - Partitional: Non-overlapping flat clusters
   - Hierarchical: Nested tree structure

3. **K-MEANS ALGORITHM:**
   - Initialize K centroids
   - Assign points to nearest centroid
   - Update centroids as cluster means
   - Repeat until convergence

4. **SSE (Sum of Squared Error):**
   - Quality measure for K-means
   - Lower SSE = better clustering
   - SSE = Σ Σ dist²(centroid, point)

5. **K-MEANS LIMITATIONS:**
   - Sensitive to initial centroids
   - Struggles with different sizes/densities
   - Assumes globular clusters
   - Must specify K in advance

6. **HIERARCHICAL CLUSTERING:**
   - No need to specify K
   - Produces dendrogram
   - Agglomerative (bottom-up) vs Divisive (top-down)

### FORMULAS TO REMEMBER

```
SSE = Σ(i=1 to K) Σ(x∈Ci) dist²(mi, x)

Centroid: mi = (1/|Ci|) × Σ(x∈Ci) x

K-means Complexity: O(n × K × I × d)

Probability of good initialization: P = K! / K^K
```

### ALGORITHM COMPARISON

| Aspect | K-means | Hierarchical |
|--------|---------|--------------|
| Type | Partitional | Nested |
| K specification | Required | Not required |
| Scalability | O(nKId) - Good | O(n²) to O(n³) - Limited |
| Shape | Globular | Flexible |
| Result | K clusters | Dendrogram |
| Deterministic | No (random init) | Yes |

### COMMON PITFALLS

1. **Using K-means for non-globular clusters** - Use density-based instead
2. **Not normalizing data** - Features with large ranges dominate
3. **Ignoring initial centroid selection** - Use multiple runs or smart initialization
4. **Choosing K arbitrarily** - Use elbow method or silhouette analysis
5. **Not handling outliers** - Preprocess to remove or use robust methods

### WHEN TO USE WHAT

| Scenario | Recommended Method |
|----------|-------------------|
| Known K, large data, spherical clusters | K-means |
| Unknown K, need hierarchy | Hierarchical |
| Arbitrary shapes, varying density | Density-based (DBSCAN) |
| Need soft assignments | Fuzzy C-means |
| Very large data | Mini-batch K-means |

---

## EXAMPLES/PROBLEMS FROM LECTURE SLIDES

### EXAMPLE 1: K-means Demo (Slide 12)
Visual demonstration showing:
- Data points (scattered green dots)
- Initial centroids placement (two X marks)
- Assign points to two centroids (red and blue coloring)
- Calculate new centroids
- Repeat assignment and calculation until convergence

### EXAMPLE 2: Two Different K-means Clusterings (Slide 16)
Shows same original data with three natural clusters producing:
- **Optimal Clustering:** Three distinct clusters correctly identified (red, green, blue)
- **Sub-optimal Clustering:** Incorrect grouping due to bad initialization
- Demonstrates how initial centroid choice affects final result

### EXAMPLE 3: Importance of Initial Centroids (Slides 17-18)
**Good Initialization (6 iterations):**
- Iteration 1-6 showing gradual convergence
- Centroids start near cluster centers
- Converges to correct 3-cluster solution

**Bad Initialization:**
- All initial centroids in upper region
- Algorithm still converges but to WRONG solution
- One natural cluster split, others merged incorrectly

### EXAMPLE 4: 10 Clusters Example (Slides 21-22)
- 5 pairs of clusters (10 total clusters)
- Starting with two initial centroids in ONE cluster of each pair
- Shows how algorithm fails to find correct structure
- Demonstrates probability problem: only 0.036% chance of good initialization for K=10

### EXAMPLE 5: Limitations - Differing Sizes (Slide 28)
- Original: One large circular cluster + two small clusters
- K-means Result: Large cluster gets split, small clusters partially absorbed
- Problem: K-means creates roughly equal-sized clusters

### EXAMPLE 6: Limitations - Differing Density (Slide 29)
- Original: One sparse cluster + two dense clusters
- K-means Result: Points from sparse cluster incorrectly assigned to dense clusters
- Problem: Dense clusters "pull" points from sparse regions

### EXAMPLE 7: Limitations - Non-globular Shapes (Slide 30)
- Original: Two crescent/moon-shaped clusters interleaved
- K-means Result: Horizontal split instead of following crescents
- Problem: K-means assumes spherical clusters

### EXAMPLE 8: Overcoming K-means Limitations (Slides 31-33)
Shows using MORE clusters than natural groups:
- Use K=6 or K=8 instead of K=3
- Find "sub-clusters" within natural clusters
- Post-process to merge related sub-clusters
- Works for different sizes, densities, and shapes

### EXAMPLE 9: Hierarchical Clustering Visualization (Slide 35)
- Points: 6 data points labeled 1-6
- Shows nested cluster structure with circles
- Corresponding dendrogram showing merge sequence
- Height represents distance at merge

### EXAMPLE 10: Starting Situation - Agglomerative (Slide 39)
- 12 individual points (p1 through p12)
- Initial proximity matrix (12x12)
- Each point is its own cluster
- Dendrogram shows 12 leaf nodes

### EXAMPLE 11: Intermediate Situation - Agglomerative (Slide 40)
- After several merges: 5 clusters (C1-C5)
- Updated proximity matrix (5x5)
- Partial dendrogram showing merges completed
- Shows how clusters grow through merging

### QUIZ FROM SLIDES: K-means vs K-Nearest Neighbor (Slide 14)

**Question:** What are the differences and similarities?

**Differences:**
| K-means | K-Nearest Neighbor |
|---------|-------------------|
| Unsupervised learning | Supervised learning |
| Clustering algorithm | Classification algorithm |
| K = number of clusters to create | K = number of neighbors to consider |
| No labels needed | Requires labeled training data |
| Creates cluster centers | Uses actual data points |
| Single result for all data | Prediction per query point |

**Similarities:**
- Both use distance/similarity measures (often Euclidean)
- Both have a parameter called K
- Both partition the feature space in some way
- Both sensitive to feature scaling

---

## QUICK REVIEW CHECKLIST

### Fundamentals
- [ ] Define cluster analysis (similar within, different between)
- [ ] Explain intra-cluster vs inter-cluster distance goals
- [ ] Distinguish clustering from classification (unsupervised vs supervised)
- [ ] Know what is NOT cluster analysis (supervised classification, simple segmentation, query results)

### Types of Clustering
- [ ] Explain partitional vs hierarchical clustering
- [ ] Understand exclusive vs non-exclusive clustering
- [ ] Explain fuzzy clustering (weights sum to 1)
- [ ] Know different types of clusters (well-separated, center-based, density-based, etc.)

### K-means Algorithm
- [ ] Write the basic K-means algorithm (5 steps)
- [ ] Calculate centroid as mean of cluster points
- [ ] Know complexity: O(n × K × I × d)
- [ ] Explain convergence criteria

### K-means Evaluation
- [ ] Write SSE formula: SSE = Σ Σ dist²(mi, x)
- [ ] Calculate SSE for given clusters
- [ ] Explain why lower SSE is better
- [ ] Know that increasing K always reduces SSE

### Initial Centroid Problem
- [ ] Explain why random initialization is problematic
- [ ] Calculate probability of good initialization: P = K!/K^K
- [ ] Describe how bad initialization leads to sub-optimal results

### Solutions to K-means Problems
- [ ] List solutions: multiple runs, hierarchical init, bisecting K-means
- [ ] Explain how to handle empty clusters
- [ ] Know pre-processing steps (normalize, remove outliers)
- [ ] Know post-processing steps (eliminate small clusters, split/merge)

### K-means Limitations
- [ ] Explain problems with differing cluster sizes
- [ ] Explain problems with differing densities
- [ ] Explain problems with non-globular shapes
- [ ] Know solution: use more clusters then merge

### Hierarchical Clustering
- [ ] Define hierarchical clustering and dendrogram
- [ ] Explain agglomerative (bottom-up) approach
- [ ] Explain divisive (top-down) approach
- [ ] List strengths: no K needed, meaningful taxonomies

### Agglomerative Algorithm
- [ ] Write agglomerative clustering algorithm
- [ ] Explain proximity matrix and how it's updated
- [ ] Know different linkage methods (single, complete, average, centroid, Ward's)

### Comparisons
- [ ] Compare K-means vs K-Nearest Neighbor (the quiz!)
- [ ] Compare K-means vs Hierarchical clustering
- [ ] Know when to use which algorithm

### Examples to Practice
- [ ] Work through K-means iteration by hand
- [ ] Calculate SSE for given clustering
- [ ] Build dendrogram from distance matrix
- [ ] Identify which algorithm fails for given data shape

---

*END OF NOTES*
