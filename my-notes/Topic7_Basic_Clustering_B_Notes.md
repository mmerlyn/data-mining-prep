# BASIC CLUSTERING PART B (HIERARCHICAL, DBSCAN & VALIDITY)
## CS653: DATA MINING
### Detailed Study Notes - Lecture 07-B

---

## TABLE OF CONTENTS
1. Hierarchical Clustering (Continued)
   - 1.1 Intermediate Situation & Proximity Matrix Update
   - 1.2 Inter-Cluster Similarity Methods
   - 1.3 MIN (Single Link) Clustering
   - 1.4 MAX (Complete Link) Clustering
   - 1.5 Group Average Clustering
   - 1.6 Ward's Method
   - 1.7 Comparison of Methods
   - 1.8 Time and Space Requirements
   - 1.9 Problems and Limitations
2. MST-Based Divisive Hierarchical Clustering
3. DBSCAN (Density-Based Clustering)
   - 3.1 Core, Border, and Noise Points
   - 3.2 DBSCAN Algorithm
   - 3.3 When DBSCAN Works Well
   - 3.4 When DBSCAN Does NOT Work Well
   - 3.5 Determining Eps and MinPts
4. Cluster Validity (Measurements)
   - 4.1 Why Evaluate Clusters?
   - 4.2 Different Aspects of Validation
   - 4.3 Types of Validity Measures
   - 4.4 Using Similarity Matrix for Validation
5. Examples and Problems
6. Summary

---

## 1. HIERARCHICAL CLUSTERING (CONTINUED)

### 1.1 INTERMEDIATE SITUATION & PROXIMITY MATRIX UPDATE

After several merging steps in agglomerative hierarchical clustering, we have some clusters (C1, C2, C3, C4, C5).

**The Process:**
1. Identify the two closest clusters (e.g., C2 and C5)
2. Merge them into a single cluster (C2 U C5)
3. Update the proximity matrix

**Key Question:** How do we update the proximity matrix after merging?

When C2 and C5 merge:
- Remove rows/columns for C2 and C5
- Add row/column for new cluster (C2 U C5)
- Calculate distances from (C2 U C5) to all other clusters

**The critical decision:** How do we define the distance between the NEW merged cluster and other clusters?

---

### 1.2 INTER-CLUSTER SIMILARITY METHODS

**Five Main Approaches:**

| Method | Description | Formula Concept |
|--------|-------------|-----------------|
| **MIN (Single Link)** | Distance = closest pair of points | MIN of all pairwise distances |
| **MAX (Complete Link)** | Distance = farthest pair of points | MAX of all pairwise distances |
| **Group Average** | Distance = average of all pairwise distances | Mean of all pairwise distances |
| **Centroid** | Distance between cluster centers | Distance between mean points |
| **Ward's Method** | Increase in squared error when merging | Minimize SSE increase |

---

### 1.3 MIN (SINGLE LINK) CLUSTERING

**Definition:**
Similarity of two clusters is based on the two MOST SIMILAR (closest) points in different clusters.

**Key Characteristics:**
- Determined by ONE pair of points
- Uses one link in the proximity graph
- Also called "Nearest Neighbor" clustering

**Example with Similarity Matrix:**

```
     I1    I2    I3    I4    I5
I1  1.00  0.90  0.10  0.65  0.20
I2  0.90  1.00  0.70  0.60  0.50
I3  0.10  0.70  1.00  0.40  0.30
I4  0.65  0.60  0.40  1.00  0.80
I5  0.20  0.50  0.30  0.80  1.00
```

**Merging Order (using MIN):**
1. First merge: I1 and I2 (similarity = 0.90, highest)
2. Continue finding highest similarity between clusters

**Visual Example:**
```
Points:       Nested Clusters:           Dendrogram:
                                              |
  1•    5      5────────────────────────────┐ |
    3     •1   3──────────┐                 │ │0.2
  •5  •2         1────────┼─────────────────┼─┤0.15
   2     •3  •6     2───┐ │                 │ │
         1       2──────┼─┤                 │ │0.1
  •4         4    ┌─────┘ │                 │ │0.05
   4         3──6─┘       │                 │ │0
                 │        │                 │ │
              3  6  2  5  4  1
```

**Strengths of MIN:**
- **Can handle NON-ELLIPTICAL shapes**
- Good for elongated or irregular clusters
- Can find clusters of arbitrary shape

**Limitations of MIN:**
- **SENSITIVE to noise and outliers**
- Can create "chaining" effect
- Single noise point can incorrectly connect two clusters

**Chaining Effect Example:**
```
Cluster A         Cluster B
  • •              • •
   • •    •  •    • •
  • •      noise   • •
```
Noise points can form a "chain" connecting otherwise separate clusters.

---

### 1.4 MAX (COMPLETE LINK) CLUSTERING

**Definition:**
Similarity of two clusters is based on the two LEAST SIMILAR (most distant) points in different clusters.

**Key Characteristics:**
- Determined by ALL pairs of points (must check all)
- Uses the maximum distance in proximity graph
- Also called "Farthest Neighbor" clustering

**Using Same Similarity Matrix:**
```
     I1    I2    I3    I4    I5
I1  1.00  0.90  0.10  0.65  0.20
I2  0.90  1.00  0.70  0.60  0.50
I3  0.10  0.70  1.00  0.40  0.30
I4  0.65  0.60  0.40  1.00  0.80
I5  0.20  0.50  0.30  0.80  1.00
```

**Dendrogram for MAX:**
```
                                    0.4
                              ┌─────┴─────┐
                        0.35  │           │
                    ┌───┴───┐ │           │
              0.3   │       │ │           │
          ┌───┴───┐ │       │ │           │0.15
    0.2   │       │ │       │ │           │
    ─┴─   │       │ │       │ │           │0.1
          │       │ │       │ │           │
    3     6       4 1       2             5
```

**Strengths of MAX:**
- **LESS SUSCEPTIBLE to noise and outliers**
- Single noise point won't incorrectly merge clusters
- More robust clustering

**Limitations of MAX:**
- **Tends to BREAK LARGE clusters**
- **BIASED towards GLOBULAR clusters**
- Cannot handle non-elliptical shapes well

**Breaking Large Clusters Example:**
```
Original: One large elongated cluster
  • • • • • • • • • •
  • • • • • • • • • •

MAX Result: Incorrectly split into two clusters
  [• • • • •] [• • • • •]
```

---

### 1.5 GROUP AVERAGE CLUSTERING

**Definition:**
Proximity of two clusters is the AVERAGE of pairwise proximity between ALL points in the two clusters.

**Formula:**
```
                    Σ proximity(pi, pj)
                   pi∈Clusteri
                   pj∈Clusterj
proximity(Ci, Cj) = ─────────────────────
                    |Clusteri| × |Clusterj|
```

**Why Average (not Total)?**
- Total proximity favors large clusters
- Average connectivity provides scalability
- Fair comparison regardless of cluster size

**Example Calculation:**
If Cluster A has 3 points and Cluster B has 2 points:
- Calculate all 3×2 = 6 pairwise distances
- Take the average of these 6 distances

**Dendrogram for Group Average:**
```
                        0.25
                  ┌─────┴─────┐
            0.2   │           │
        ┌───┴───┐ │           │
  0.15  │       │ │           │
    ────┤       │ │           │0.1
        │       │ │           │
  0.05  │       │ │           │
        │       │ │           │
    3   6       4 1           2   5
```

**Characteristics:**
- **Compromise between Single and Complete Link**
- Less susceptible to noise and outliers
- **Limitation:** Still biased towards globular clusters

---

### 1.6 WARD'S METHOD

**Definition:**
Similarity of two clusters is based on the INCREASE IN SQUARED ERROR when two clusters are merged.

**Key Concept:**
- Measures how much merging increases within-cluster variance
- Similar to group average if distance = distance²
- Merges clusters that result in MINIMUM SSE increase

**Formula:**
```
SSE = Σ   Σ  (x - mi)²
      i  x∈Ci

where mi = centroid of cluster Ci
```

**Properties:**
- Less susceptible to noise and outliers
- Biased towards globular clusters
- **HIERARCHICAL ANALOGUE OF K-MEANS**
- Can be used to INITIALIZE K-means (run hierarchical first, then use result as K-means starting point)

---

### 1.7 COMPARISON OF METHODS

**Visual Comparison (6 points example):**

```
Point Layout:
        4   •1
    2
  •5  •2        5
        •3  •6
          1
    •4
      3
```

**Clustering Results:**

| Method | Cluster Structure |
|--------|-------------------|
| **MIN** | {3,6} → {3,6,2,5} → {3,6,2,5,4} → {all} |
| **MAX** | {3,6} → {4,1} → {3,6,4,1} → {2,5} → {all} |
| **Group Average** | {3,6} → {3,6,4} → {2,5} → {all} |
| **Ward's** | {3,6} → {3,6,4} → {all} (similar to Group Avg) |

**Summary Table:**

| Method | Handles Non-Elliptical | Noise Resistant | Breaks Large Clusters |
|--------|----------------------|-----------------|----------------------|
| MIN | YES | NO | NO |
| MAX | NO | YES | YES |
| Group Average | NO | YES | NO |
| Ward's | NO | YES | NO |

---

### 1.8 TIME AND SPACE REQUIREMENTS

**Space Complexity: O(N²)**
- Must store the proximity matrix
- N = number of points
- Matrix is N × N

**Time Complexity: O(N³)** in many cases
- N steps (one merge per step until one cluster remains)
- At each step: update and search N² proximity matrix

**Optimized Approaches:**
- Can be reduced to **O(N² log N)** for some approaches
- Using priority queues for finding minimum distances

---

### 1.9 PROBLEMS AND LIMITATIONS

**1. Decisions are IRREVERSIBLE**
- Once two clusters are combined, they cannot be undone
- No backtracking possible
- Early bad decisions propagate through entire hierarchy

**2. No Direct Objective Function**
- Unlike K-means (minimizes SSE), hierarchical doesn't optimize a global criterion
- Each step makes locally optimal decision

**3. Different Schemes Have Different Problems:**

| Problem | MIN | MAX | Group Avg | Ward's |
|---------|-----|-----|-----------|--------|
| Noise sensitivity | HIGH | LOW | LOW | LOW |
| Different sized clusters | OK | POOR | POOR | POOR |
| Non-convex shapes | GOOD | POOR | POOR | POOR |
| Breaking large clusters | NO | YES | NO | NO |

---

## 2. MST-BASED DIVISIVE HIERARCHICAL CLUSTERING

### Building the Minimum Spanning Tree (MST)

**Algorithm:**
1. Start with a tree consisting of any single point
2. In successive steps:
   - Find closest pair (p, q) where p is IN tree, q is NOT in tree
   - Add q to tree
   - Add edge between p and q
3. Repeat until all points are in the tree

**Visual Example:**
```
Initial Points:                MST Result:
  0.6  •1                        0.6  •1
  0.5     5                      0.5   | 5
  0.4  •5  •2                    0.4  •5--•2
  0.3       •3  •6               0.3    |  •3--•6
  0.2     •4                     0.2    •4
```

### MST Divisive Algorithm

```
Algorithm 7.5: MST Divisive Hierarchical Clustering

1: Compute minimum spanning tree for the proximity graph
2: REPEAT
3:    Create a new cluster by BREAKING the link corresponding
      to the LARGEST DISTANCE (smallest similarity)
4: UNTIL only singleton clusters remain
```

**Key Idea:**
- Build MST first (connects all points with minimum total edge weight)
- Repeatedly cut the longest edge
- Each cut creates two new clusters
- Results in divisive (top-down) hierarchy

---

## 3. DBSCAN (DENSITY-BASED SPATIAL CLUSTERING)

### 3.1 CORE, BORDER, AND NOISE POINTS

**DBSCAN is a density-based algorithm.**

**Key Parameters:**
- **Eps (ε):** Radius of neighborhood
- **MinPts:** Minimum number of points required in neighborhood

**Density Definition:**
```
Density at point p = number of points within distance Eps from p
```

**Three Types of Points:**

**1. CORE POINT:**
- Has MORE than MinPts points within Eps
- These points are at the INTERIOR of a cluster
- Can "expand" the cluster from core points

**2. BORDER POINT:**
- Has FEWER than MinPts within Eps
- BUT is in the neighborhood of a core point
- On the edge of a cluster

**3. NOISE POINT:**
- NOT a core point
- NOT in neighborhood of any core point
- Outliers that don't belong to any cluster

**Visual Example (Eps=1, MinPts=4):**
```
                    2 ┌───────────────────┐
                      │          •        │
                   1.5│    ┌─────────┐    │ Noise Point
                      │    │ Eps=1   │ ✕  │
                    1 │    └────•────┘    │
                      │       ╱ ╲         │
                   0.5│      •   •        │
                      │     Core Point    │
                    0 │  •               •│
                      │                   │ Core Point
                  -0.5│ • Border    ✕    •│
                      │   Point         / │
                   -1 │              ┌──┴─┐
                      │              │•   │
                  -1.5│              └────┘ MinPts=4
                      └───────────────────┘
                     -2  -1   0   1   2
```

---

### 3.2 DBSCAN ALGORITHM

**Two Main Steps:**

**Step 1: Eliminate noise points**
- Identify all core, border, and noise points
- Remove noise points from consideration

**Step 2: Perform clustering on remaining points**

```
DBSCAN Algorithm:
─────────────────
current_cluster_label ← 1

FOR all core points DO
    IF the core point has no cluster label THEN
        current_cluster_label ← current_cluster_label + 1
        Label the current core point with current_cluster_label
    END IF

    FOR all points in Eps-neighborhood, except the point itself DO
        IF the point does not have a cluster label THEN
            Label the point with current_cluster_label
        END IF
    END FOR
END FOR
```

**Key Properties:**
- Core points in same cluster are DENSITY-CONNECTED
- Border points assigned to their neighboring core point's cluster
- Clusters form connected components of core points

**Example Result (Eps=10, MinPts=4):**
```
Original Points          →    Point Types (Core=Green, Border=Blue, Noise=Red)

  •  •  • •  • •               ○  ●  ● ●  ● ●
 • ••• •  ••• ••              ● ●●● ●  ●●● ●●
• •••• •  ••••  •   →        ● ●●●● ●  ●●●●  ○
 • ••• •  ••• ••              ● ●●● ●  ●●● ●●
  •  •  • •  • •               ○  ●  ● ●  ● ●
```

---

### 3.3 WHEN DBSCAN WORKS WELL

**Strengths:**

**1. Resistant to Noise**
- Noise points are explicitly identified and excluded
- Doesn't force noise into clusters

**2. Can Handle Clusters of Different Shapes and Sizes**
- Not limited to spherical/globular clusters
- Can find arbitrarily shaped clusters

**Example - Complex Shapes:**
```
Original Points                     DBSCAN Clusters (7 clusters found)

  1   •    2    •    6               1   •    2    •    6
      •    •    •    ••                  •    •    •    ••
   •••• •••• ••••   •••           [Red] [Blue][Cyan] [Orange]
  •  •  •  •  •  •  •• •          Different colors = different clusters
   3     4    5
  ••••••••••••••••                    [Yellow cluster]
    7                                 [Dark blue]
```

---

### 3.4 WHEN DBSCAN DOES NOT WORK WELL

**Limitations:**

**1. Varying Densities**
- Single Eps value cannot capture clusters of different densities
- Dense cluster needs small Eps, sparse cluster needs large Eps

**Example:**
```
Original Points:              Problem with fixed Eps:

  ○ ○                         Sparse cluster (left)
 ○   ○                        might be marked as noise
○  ●●●●●●                     if Eps is too small
 ○ ●●●●●●●
  ○●●●●●●●                    OR
   ●●●●●●
                              Dense cluster (right)
Sparse    Dense               might merge incorrectly
cluster   cluster             if Eps is too large
```

**Different parameter choices:**
- MinPts=4, Eps=9.75: May split incorrectly
- MinPts=4, Eps=9.92: Different (possibly wrong) result

**2. High-Dimensional Data**
- Distance measures become less meaningful
- "Curse of dimensionality" affects neighborhood calculations
- All points appear similarly far apart

---

### 3.5 DETERMINING EPS AND MINPTS

**The k-Distance Plot Method:**

**Intuition:**
- For points IN a cluster: kth nearest neighbors are at roughly SAME distance
- For NOISE points: kth nearest neighbor is at FARTHER distance

**Method:**
1. For each point, compute distance to its kth nearest neighbor
2. Sort all these distances in increasing order
3. Plot the sorted distances
4. Look for the "knee" or "elbow" in the plot

**k-Distance Plot:**
```
4th Nearest
Neighbor          50 ┤                    ╱
Distance          45 ┤                   ╱
                  40 ┤                  ╱
                  35 ┤                 ╱
                  30 ┤                ╱
                  25 ┤              ╱
                  20 ┤            ╱
                  15 ┤          ╱
                  10 ┤     ────╱  ← Knee point
                   5 ┤────                (choose Eps here)
                   0 ┼────┬────┬────┬────┬────┬────┬
                     0   500  1000 1500 2000 2500 3000
                     Points Sorted by 4th NN Distance
```

**Selecting Parameters:**
- **Eps:** Value at the "knee" of the curve
- **MinPts:** Typically set to k (commonly 4 or more)

---

## 4. CLUSTER VALIDITY (MEASUREMENTS)

### 4.1 WHY EVALUATE CLUSTERS?

**For supervised classification:** We have accuracy, precision, recall

**For clustering:** How do we evaluate "goodness" of clusters?

**"Clusters are in the eye of the beholder!"**

**Reasons to Evaluate:**
1. **Avoid finding patterns in noise** - Random data will produce clusters!
2. **Compare clustering algorithms** - Which algorithm performs better?
3. **Compare two sets of clusters** - Different parameter settings
4. **Compare two individual clusters** - Which cluster is better defined?

**Example - Clusters in Random Data:**
```
Random Points:                K-means Result:         DBSCAN Result:
  •     •                      [A]   [B]               •  [A]  •
•    •     •                 •    •     •            •  [A] [A] •
  •  •   •                     [A] [A]  [B]           [B] [B] •
•     •    •                •     [B]   •            [B]   •  [A]
  •  •   •                     [A] [B]  [B]            •  [A]  •

Both algorithms WILL find clusters even in random data!
```

---

### 4.2 DIFFERENT ASPECTS OF CLUSTER VALIDATION

**Five Key Aspects:**

| # | Aspect | Description |
|---|--------|-------------|
| 1 | **Clustering Tendency** | Does non-random structure exist in data? |
| 2 | **External Validation** | Compare to externally known class labels |
| 3 | **Internal Validation** | Evaluate fit without external information |
| 4 | **Relative Validation** | Compare two different clustering results |
| 5 | **Optimal Number** | Determine the "correct" number of clusters |

For aspects 2, 3, and 4: Can evaluate ENTIRE clustering OR individual clusters

---

### 4.3 TYPES OF VALIDITY MEASURES (INDICES)

**Three Main Types:**

**1. EXTERNAL INDEX**
- Measures extent to which cluster labels MATCH externally supplied class labels
- Requires ground truth labels
- **Example:** Entropy

```
Entropy = - Σ pi log2(pi)

where pi = proportion of class i in the cluster
Lower entropy = purer clusters = better
```

**2. INTERNAL INDEX**
- Measures goodness of clustering WITHOUT external information
- Uses ONLY the data
- **Example:** Sum of Squared Error (SSE)

```
SSE = Σ   Σ   dist(x, ci)²
      i  x∈Ci

where ci = centroid of cluster i
Lower SSE = tighter clusters = better
```

**3. RELATIVE INDEX**
- Compares two different clusterings or clusters
- Often uses external or internal index for comparison
- **Examples:** SSE comparison, Entropy comparison

---

### 4.4 USING SIMILARITY MATRIX FOR VALIDATION

**Method:**
1. Compute similarity matrix for all points
2. ORDER rows/columns by cluster labels
3. Visualize as heatmap
4. Look for clear block-diagonal structure

**Good Clustering - Clear Structure:**
```
Points ordered by cluster label:

        Cluster 1    Cluster 2    Cluster 3
      ┌───────────┬───────────┬───────────┐
C1    │ ████████  │  ░░░░░░   │  ░░░░░░   │ High similarity
      │ ████████  │  ░░░░░░   │  ░░░░░░   │ within cluster
      ├───────────┼───────────┼───────────┤
C2    │  ░░░░░░   │ ████████  │  ░░░░░░   │
      │  ░░░░░░   │ ████████  │  ░░░░░░   │ Low similarity
      ├───────────┼───────────┼───────────┤ between clusters
C3    │  ░░░░░░   │  ░░░░░░   │ ████████  │
      │  ░░░░░░   │  ░░░░░░   │ ████████  │
      └───────────┴───────────┴───────────┘

█ = High similarity (red/warm colors)
░ = Low similarity (blue/cool colors)
```

**Poor Clustering - No Clear Structure (Random Data):**
```
        Points (no clear cluster order)
      ┌─────────────────────────────────┐
      │ █░█░░█░█░█░░█░█░░█░█░█░░█░█░░█ │
      │ ░█░░█░█░░█░█░█░░█░░█░█░█░░█░░█ │
      │ █░█░█░░█░░█░░█░█░█░░█░░█░█░█░░ │
      │ ░░█░█░█░█░█░░█░░█░█░░█░█░░█░█░ │
      │ █░░█░░█░█░░█░█░█░░█░█░░█░░█░░█ │
      └─────────────────────────────────┘

No block-diagonal structure = poor/random clustering
```

**Real Example Visualization:**
```
Well-separated clusters (3 clusters):
   10─┬─███────────────────
   20─┤─███────────────────
   30─┤─███────────────────    3 clear
   40─┤────███─────────────    diagonal
   50─┤────███─────────────    blocks
   60─┤────███─────────────
   70─┤────────███─────────
   80─┤────────███─────────
   90─┤────────███─────────
  100─┴──20─40─60─80─100───
```

---

## 5. EXAMPLES AND PROBLEMS

### PROBLEM 1: Inter-Cluster Distance Calculation

**Given two clusters:**
- Cluster A: points {(1,1), (2,1), (1,2)}
- Cluster B: points {(5,5), (6,5), (5,6), (6,6)}

**Calculate the inter-cluster distance using:**

**a) MIN (Single Link):**
Find minimum distance between any pair:
- d((2,1), (5,5)) = √((5-2)² + (5-1)²) = √(9+16) = 5
- This is the smallest distance between clusters
- **MIN distance = 5**

**b) MAX (Complete Link):**
Find maximum distance between any pair:
- d((1,1), (6,6)) = √((6-1)² + (6-1)²) = √(25+25) = √50 ≈ 7.07
- **MAX distance ≈ 7.07**

**c) Centroid:**
- Centroid A = ((1+2+1)/3, (1+1+2)/3) = (1.33, 1.33)
- Centroid B = ((5+6+5+6)/4, (5+5+6+6)/4) = (5.5, 5.5)
- d(centroid A, centroid B) = √((5.5-1.33)² + (5.5-1.33)²)
- = √(17.39 + 17.39) = √34.78 ≈ 5.90
- **Centroid distance ≈ 5.90**

**d) Group Average:**
Calculate all pairwise distances and average:
- 3 × 4 = 12 pairs total
- Sum all 12 distances, divide by 12
- **Group Average ≈ 5.83** (calculation intensive)

---

### PROBLEM 2: DBSCAN Classification

**Given points and parameters Eps=2, MinPts=3:**

```
Points: A(0,0), B(1,0), C(0,1), D(5,5), E(6,5), F(5,6), G(10,0)
```

**Classify each point as Core, Border, or Noise:**

**Step 1: Find neighbors within Eps=2 for each point:**
- A: neighbors = {B, C} (distance < 2) → 2 neighbors
- B: neighbors = {A, C} → 2 neighbors
- C: neighbors = {A, B} → 2 neighbors
- D: neighbors = {E, F} → 2 neighbors
- E: neighbors = {D, F} → 2 neighbors
- F: neighbors = {D, E} → 2 neighbors
- G: neighbors = {} → 0 neighbors

**Step 2: Classify (MinPts=3, need ≥3 neighbors including self):**
- A: 3 points in neighborhood (A,B,C) → **CORE**
- B: 3 points in neighborhood → **CORE**
- C: 3 points in neighborhood → **CORE**
- D: 3 points in neighborhood (D,E,F) → **CORE**
- E: 3 points in neighborhood → **CORE**
- F: 3 points in neighborhood → **CORE**
- G: 1 point only (itself) → **NOISE**

**Result:**
- **Cluster 1:** {A, B, C}
- **Cluster 2:** {D, E, F}
- **Noise:** {G}

---

### PROBLEM 3: When Would You Choose Each Method?

**Scenario A:** Data has elongated, snake-like clusters
- **Best choice: MIN (Single Link)** or **DBSCAN**
- Why: Can handle non-elliptical shapes

**Scenario B:** Data has significant noise/outliers
- **Best choice: MAX (Complete Link)** or **DBSCAN**
- Why: Resistant to noise

**Scenario C:** Want to initialize K-means with good starting clusters
- **Best choice: Ward's Method**
- Why: Hierarchical analogue of K-means

**Scenario D:** Clusters have varying densities
- **DBSCAN will NOT work well**
- Consider: OPTICS or multi-scale clustering

---

### PROBLEM 4: Hierarchical Clustering Dendrogram Interpretation

**Given similarity matrix:**
```
     A     B     C     D     E
A   1.00  0.80  0.20  0.30  0.10
B   0.80  1.00  0.30  0.25  0.15
C   0.20  0.30  1.00  0.90  0.40
D   0.30  0.25  0.90  1.00  0.35
E   0.10  0.15  0.40  0.35  1.00
```

**Perform agglomerative clustering using Single Link:**

**Step 1:** Find highest similarity (excluding diagonal)
- A-B: 0.80, C-D: 0.90 ← highest
- Merge C and D → {C,D}

**Step 2:** Update matrix (MIN of similarities to C and D):
```
       A     B    {C,D}   E
A    1.00  0.80   0.30  0.10
B    0.80  1.00   0.30  0.15
{CD} 0.30  0.30   1.00  0.40
E    0.10  0.15   0.40  1.00
```

**Step 3:** Find highest: A-B = 0.80
- Merge A and B → {A,B}

**Step 4:** Update:
```
      {A,B}  {C,D}   E
{AB}   1.00   0.30  0.15
{CD}   0.30   1.00  0.40
E      0.15   0.40  1.00
```

**Step 5:** Find highest: {C,D}-E = 0.40
- Merge → {C,D,E}

**Step 6:** Final merge {A,B} with {C,D,E} at similarity 0.30

**Dendrogram:**
```
Similarity
   1.0 ─┐
   0.9 ─┼─ C─┬─D
   0.8 ─┼─ A─┬─B
   0.7 ─┤   │
   0.6 ─┤   │
   0.5 ─┤   │
   0.4 ─┼───┴──┬─E
   0.3 ─┼──────┴───┬──────
   0.2 ─┤          │
   0.1 ─┤          │
   0.0 ─┴──────────┴──────
       A  B  C  D  E
```

---

### PROBLEM 5: Determining DBSCAN Parameters

**Given a k-distance plot showing:**
```
Distance: 2, 2, 3, 3, 3, 4, 4, 5, 5, 8, 12, 15, 25
Points:   1  2  3  4  5  6  7  8  9  10 11  12  13
```

**Where is the "knee"?**
- Distances relatively stable: 2-5
- Sharp increase starts at point 10 (distance 8)
- **Knee is around distance 5-8**

**Recommended parameters:**
- **Eps ≈ 5-6** (just before the steep increase)
- **MinPts = k** (whatever k was used for k-distance)

---

## 6. SUMMARY

### KEY CONCEPTS BY TOPIC

**HIERARCHICAL CLUSTERING - Inter-Cluster Similarity:**

| Method | Formula | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| MIN (Single Link) | min distance | Non-elliptical shapes | Noise sensitive, chaining |
| MAX (Complete Link) | max distance | Noise resistant | Breaks large clusters, globular bias |
| Group Average | mean of all pairs | Balanced | Globular bias |
| Ward's | minimize SSE increase | K-means analogue | Globular bias |

**Complexity:**
- Space: O(N²) - proximity matrix
- Time: O(N³) basic, O(N² log N) optimized

**DBSCAN:**

| Concept | Definition |
|---------|------------|
| Core Point | ≥ MinPts neighbors within Eps |
| Border Point | < MinPts neighbors but near a core point |
| Noise Point | Neither core nor border |
| Eps | Neighborhood radius |
| MinPts | Minimum density threshold |

**Strengths:** Arbitrary shapes, noise handling, no need to specify k
**Weaknesses:** Varying densities, high dimensions, parameter selection

**CLUSTER VALIDITY:**

| Type | Uses External Info | Example |
|------|-------------------|---------|
| External Index | YES | Entropy |
| Internal Index | NO | SSE |
| Relative Index | Comparison | Any of above |

### FORMULAS TO REMEMBER

**1. Group Average:**
```
proximity(Ci, Cj) = Σ proximity(pi, pj) / (|Ci| × |Cj|)
```

**2. Ward's Method (SSE):**
```
SSE = Σi Σx∈Ci dist(x, mi)²
```

**3. Entropy (External Validity):**
```
Entropy = -Σ pi log2(pi)
```

### ALGORITHM COMPARISON

| Criterion | K-means | Hierarchical | DBSCAN |
|-----------|---------|--------------|--------|
| Cluster Shape | Spherical | Depends on linkage | Arbitrary |
| # Clusters | Must specify K | Can cut at any level | Automatic |
| Noise Handling | Poor | Poor | Excellent |
| Scalability | O(NKT) | O(N² log N) | O(N log N) with index |
| Outlier Sensitivity | High | Depends | Low |

### DECISION GUIDE

**Use MIN (Single Link) when:**
- Clusters have irregular shapes
- Data has natural chain structure

**Use MAX (Complete Link) when:**
- Noise/outliers are present
- Want compact clusters

**Use DBSCAN when:**
- Arbitrary cluster shapes needed
- Noise must be identified
- Number of clusters unknown
- **DON'T use** for varying density data

**Use Ward's Method when:**
- Want to initialize K-means
- Prefer tight, spherical clusters

---

*END OF NOTES*
