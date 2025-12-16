# Basic Clustering Part C: Cluster Validity & Anomaly Detection

## Overview

This module covers two main topics:
1. **Cluster Validity** - Measuring the goodness of clustering results (internal and external measures)
2. **Anomaly/Outlier Detection** - Identifying data points that differ significantly from the rest

---

## Part 1: Cluster Validity

---

## 1. Internal Measures: SSE (Sum of Squared Errors)

### Definition

**Internal Index:** Used to measure the goodness of a clustering structure **without respect to external information** (no ground truth labels needed).

### Key Points About SSE

- Clusters in more complicated figures aren't always well separated
- SSE is good for comparing:
  - Two different clusterings
  - Two clusters (using average SSE)
- **Can be used to estimate the number of clusters**

### The Elbow Method

When plotting SSE vs. K (number of clusters):

```
SSE
 |
 |  *
 |    *
 |      *----*----*----*----*
 |___________________________ K
    2   5   10  15  20  25  30
```

**Interpretation:**
- SSE decreases as K increases
- Look for the "elbow" - point where rate of decrease sharply changes
- The elbow suggests the optimal number of clusters

### Example: Simple vs Complicated Data

**Simple Data (well-separated clusters):**
- Clear elbow in SSE curve
- Easy to identify optimal K

**Complicated Data (overlapping clusters):**
- SSE curve may be smoother
- Elbow less pronounced
- More difficult to determine optimal K

---

## 2. Framework for Cluster Validity

### The Interpretation Problem

- If a measure of evaluation has value 10, is that good, fair, or poor?
- Need a framework to interpret any measure

### Statistical Framework

**Key Principle:** The more "atypical" a clustering result is, the more likely it represents **valid structure** in the data.

**Approach:**
1. Compare values of an index from random data/clusterings to actual clustering result
2. If the value of the index is unlikely (under random assumption), then the cluster results are valid

**Note:** These approaches are more complicated and harder to understand.

### Comparing Two Clusterings

- When comparing results of two different cluster analyses, a framework is less necessary
- However, there's still the question of whether the difference between two index values is **significant**

---

## 3. Statistical Framework for SSE - Example

### Problem Setup

Compare SSE of 0.005 against three clusters in random data.

### Method

1. Generate 500 sets of random data points
2. Each set: 100 points distributed over range 0.2-0.8 for x and y values
3. Cluster each set into 3 clusters using K-means
4. Create histogram of SSE values

### Results

```
Histogram of SSE values for random data:
                        |
                     ****
                   *******
                 **********
               ************
SSE range: 0.016 - 0.034
```

**Interpretation:**
- If our actual clustering has SSE = 0.005
- This is FAR below the SSE values seen in random data (0.016-0.034)
- Therefore, our clustering represents **real structure** (not random chance)

---

## 4. Internal Measures: Cohesion and Separation

### Definitions

**Cluster Cohesion:** Measures how closely related objects are **within** a cluster

**Cluster Separation:** Measures how distinct or well-separated a cluster is **from other clusters**

---

### 4.1 Within-Cluster Sum of Squares (WSS)

**Formula:**

```
WSS = Σ Σ (x - m_i)²
      i  x∈C_i
```

Where:
- C_i = cluster i
- m_i = centroid of cluster i
- x = data point in cluster i

**Interpretation:** Lower WSS = better cohesion (points closer to their centroid)

---

### 4.2 Between-Cluster Sum of Squares (BSS)

**Formula:**

```
BSS = Σ |C_i| × (m - m_i)²
      i
```

Where:
- |C_i| = number of points in cluster i
- m = overall mean (centroid of entire dataset)
- m_i = centroid of cluster i

**Interpretation:** Higher BSS = better separation (cluster centroids further from overall mean)

---

### 4.3 Important Property: BSS + WSS = Constant

**Total Sum of Squares (TSS):**
```
TSS = BSS + WSS = constant (for a given dataset)
```

This means:
- As WSS decreases (better cohesion), BSS increases (better separation)
- Good clustering minimizes WSS while maximizing BSS

---

### Example: BSS + WSS Calculation

**Dataset:** Points at positions 1, 2, 4, 5 on a number line

#### K=1 Cluster (all points in one cluster)

```
Points: 1    2         4    5
              ×
           m=3 (overall mean)
```

**Calculations:**
```
WSS = (1-3)² + (2-3)² + (4-3)² + (5-3)²
    = 4 + 1 + 1 + 4 = 10

BSS = 4 × (3-3)² = 0

Total = 10 + 0 = 10
```

#### K=2 Clusters

```
Cluster 1: {1, 2}     Cluster 2: {4, 5}
    ×                      ×
   m₁=1.5                m₂=4.5
```

**Calculations:**
```
WSS = (1-1.5)² + (2-1.5)² + (4-4.5)² + (5-4.5)²
    = 0.25 + 0.25 + 0.25 + 0.25 = 1

BSS = 2×(3-1.5)² + 2×(4.5-3)²
    = 2×2.25 + 2×2.25 = 9

Total = 1 + 9 = 10
```

**Key Insight:** Total (BSS + WSS) remains constant at 10, but the split changes dramatically with better clustering.

---

### 4.4 Proximity Graph-Based Approach

An alternative way to measure cohesion and separation using graphs:

**Cluster Cohesion:** Sum of weights of all links **within** a cluster

**Cluster Separation:** Sum of weights between nodes **in** the cluster and nodes **outside** the cluster

```
[Cohesion]              [Separation]
  Cluster                Two Clusters
    /\                     /\      /\
   /  \                   /  \----/  \
  /----\                 /    \  /    \
  Points                Points  \/   Points
  within                between clusters
```

---

## 5. Internal Measures: Silhouette Coefficient

### Overview

The Silhouette Coefficient combines ideas of **both cohesion and separation** for:
- Individual points
- Clusters
- Entire clusterings

### Calculation for Individual Point i

**Step 1:** Calculate **a**
```
a = average distance of i to points in its own cluster
```

**Step 2:** Calculate **b**
```
b = min(average distance of i to points in another cluster)
    (minimum over all other clusters)
```

**Step 3:** Calculate Silhouette Coefficient **s**

```
If a < b:   s = 1 - a/b
If a ≥ b:   s = b/a - 1  (not the usual case)
```

### Interpretation

| Value | Meaning |
|-------|---------|
| s ≈ 1 | Point is well-matched to its cluster (a << b) |
| s ≈ 0 | Point is on the border between clusters (a ≈ b) |
| s < 0 | Point may be in the wrong cluster (a > b) |

**Typical Range:** 0 to 1 (closer to 1 is better)

### Aggregation

**Average Silhouette Width:**
- For a cluster: Average of silhouette coefficients of all points in that cluster
- For a clustering: Average of silhouette coefficients of all points in the dataset

### Visual Representation

```
       Point i
         ○
        /|
       / |
      a  | b
     /   |
    ○    ○
 Same   Nearest other
cluster   cluster
```

---

## 6. External Measures: Entropy and Purity

### When to Use External Measures

External measures require **ground truth labels** (actual class labels for data points). Used when you have labeled data and want to evaluate how well clustering matches known classes.

---

### 6.1 Entropy

**Definition:** Measures the "impurity" of clusters with respect to class labels.

**For a single cluster j:**

```
e_j = Σ p_ij × log₂(p_ij)
      i=1 to L

Where:
- p_ij = m_ij / m_j
- m_j = total number of values in cluster j
- m_ij = number of values of class i in cluster j
- L = number of classes
```

**Total Entropy (weighted by cluster size):**

```
e = Σ (m_j / m) × e_j
    j=1 to K

Where:
- K = number of clusters
- m = total number of data points
```

**Interpretation:**
- Lower entropy = better (more pure clusters)
- Entropy = 0 means each cluster contains only one class

---

### 6.2 Purity

**Definition:** Fraction of cluster members that belong to the majority class.

**For a single cluster j:**

```
purity_j = max(p_ij)
           i
```
(The maximum class probability in cluster j)

**Overall Purity:**

```
purity = Σ (m_j / m) × purity_j
         j=1 to K
```

**Interpretation:**
- Higher purity = better
- Purity = 1 means each cluster contains only one class

---

### Example: K-means Clustering Results for LA Document Data

| Cluster | Entertainment | Financial | Foreign | Metro | National | Sports | Entropy | Purity |
|---------|---------------|-----------|---------|-------|----------|--------|---------|--------|
| 1 | 3 | 5 | 40 | **506** | 96 | 27 | 1.2270 | 0.7474 |
| 2 | 4 | **280** | 29 | 39 | 2 | 7 | 1.1472 | 0.7756 |
| 3 | 1 | 1 | 1 | 7 | 4 | **671** | 0.1813 | 0.9796 |
| 4 | 10 | **162** | 3 | 119 | 73 | 2 | 1.7487 | 0.4390 |
| 5 | **331** | 22 | 5 | 70 | 13 | 23 | 1.3976 | 0.7134 |
| 6 | 5 | **358** | 12 | 212 | 48 | 13 | 1.5523 | 0.5525 |
| **Total** | 354 | 555 | 341 | 943 | 273 | 738 | **1.1450** | **0.7203** |

**Key Observations:**
- Cluster 3 has **lowest entropy** (0.1813) and **highest purity** (0.9796) - almost all Sports articles
- Cluster 4 has **highest entropy** (1.7487) and **lowest purity** (0.4390) - very mixed

---

## 7. Final Comment on Cluster Validity

> "The validation of clustering structures is the most difficult and frustrating part of cluster analysis. Without a strong effort in this direction, cluster analysis will remain a black art accessible only to those true believers who have experience and great courage."
>
> — *Algorithms for Clustering Data*, Jain and Dubes

**Key Takeaway:** Cluster validation is challenging but essential for meaningful results.

---

---

## Part 2: Anomaly/Outlier Detection

---

## 8. What are Anomalies/Outliers?

### Definition

**Anomalies (Outliers):** Data points that are **considerably different** from the remainder of the data.

### Characteristics

- **Relatively Rare:** Natural implication is that anomalies are uncommon
  - "One in a thousand occurs often if you have lots of data"
- **Context-Dependent:** What's anomalous depends on context
  - Example: Freezing temperatures in July (unusual in most places)

### Importance

Anomalies can be either **important** or a **nuisance**:

| Type | Example | Significance |
|------|---------|--------------|
| Nuisance | 10-foot tall 2-year-old | Likely data error |
| Important | Unusually high blood pressure | Medical concern |

---

## 9. Importance of Anomaly Detection - Real World Example

### The Ozone Depletion Story (1985)

**Background:**
- Three researchers (Farman, Gardiner, Shanklin) found Antarctic ozone levels had dropped **10% below normal**
- British Antarctic Survey ground measurements detected this

**The Problem:**
- Nimbus 7 satellite had instruments for recording ozone levels
- Why didn't the satellite detect the low concentrations?

**The Answer:**
- Satellite ozone concentrations were so low they were being **treated as outliers by computer software and DISCARDED!**

**Lesson:** Anomalies may contain the most important information in your data.

---

## 10. Causes of Anomalies

### 10.1 Data from Different Classes

- Objects from a different population mixed in with main data
- **Example:** Measuring weights of oranges, but a few grapefruit are mixed in

### 10.2 Natural Variation

- Extreme but legitimate values within the same population
- **Example:** Unusually tall people

### 10.3 Data Errors

- Mistakes in data collection, entry, or processing
- **Example:** 200-pound 2-year-old (clearly an error)

---

## 11. Distinction Between Noise and Anomalies

### Noise

- Erroneous, perhaps random, values or contaminating objects
- **Examples:**
  - Weight recorded incorrectly
  - Grapefruit mixed in with oranges
- Noise doesn't necessarily produce unusual values
- **Noise is NOT interesting**

### Anomalies

- May or may not be caused by noise
- **Anomalies MAY BE interesting** (if not caused by noise)

### Key Insight

> Noise and anomalies are **related but distinct** concepts.

---

## 12. General Issues in Anomaly Detection

### 12.1 Number of Attributes

**Single Attribute Anomalies:**
- Many anomalies defined by one attribute: Height, Shape, Color

**Multi-Attribute Challenges:**
- Can be hard to find anomalies using all attributes:
  - Noisy or irrelevant attributes obscure the signal
  - Object may only be anomalous with respect to some attributes
- However, an object may **not** be anomalous in any single attribute
  - Only appears anomalous in combination

---

### 12.2 Anomaly Scoring

**Binary Categorization:**
- Object is an anomaly or it isn't
- Common in classification-based approaches

**Continuous Scoring:**
- Assigns a score measuring degree of anomalousness
- Allows ranking of objects
- More flexible

**Practical Need:**
- Often need binary decision eventually ("Should this credit card transaction be flagged?")
- Still useful to have scores for ranking

**Open Question:** How many anomalies are there?

---

### 12.3 Other Issues

| Issue | Description |
|-------|-------------|
| **Detection Strategy** | Find all at once vs. one at a time |
| **Swamping** | Normal points incorrectly identified as outliers |
| **Masking** | Outliers hiding other outliers |
| **Evaluation** | How to measure performance (supervised vs. unsupervised) |
| **Efficiency** | Computational cost |
| **Context** | What's normal depends on context (e.g., heights on basketball team) |

---

## 13. Variants of Anomaly Detection Problems

### Variant 1: Threshold-Based
Given dataset D, find all data points x ∈ D with anomaly scores **greater than threshold t**

### Variant 2: Top-N
Given dataset D, find all data points x ∈ D having the **top-n largest anomaly scores**

### Variant 3: Novel Point Detection
Given dataset D (mostly normal, unlabeled data) and test point x, **compute anomaly score of x** with respect to D

---

## 14. Model-Based Anomaly Detection

### 14.1 Unsupervised Approaches

**Build a model and identify points that:**
- Don't fit the model well, OR
- Distort the model significantly

**Types of Models:**

| Model Type | Detection Method |
|------------|-----------------|
| Statistical Distribution | Low probability |
| Clusters | Doesn't belong to any cluster |
| Regression | Large residuals |
| Geometric | Outside boundaries |
| Graph | Unusual connectivity |

### 14.2 Supervised Approaches

- Anomalies regarded as a **rare class**
- Requires labeled training data
- Faces class imbalance challenges

---

## 15. Additional Anomaly Detection Techniques

### 15.1 Proximity-Based
- Anomalies are points **far away** from other points
- Can sometimes detect visually in low dimensions

### 15.2 Density-Based
- **Low density** points are outliers
- Points in sparse regions more likely anomalous

### 15.3 Pattern Matching
- Create profiles/templates of atypical but important events
- Algorithms usually simple and efficient

---

## 16. Visual Approaches

### Methods
- **Boxplots:** Show outliers beyond whiskers
- **Scatter plots:** Visualize points that don't fit patterns

### Boxplot Structure

```
    *     <- Outlier (> Q3 + 1.5×IQR)
    |
   ---
  | Q3 |  <- 75th percentile
  |    |
  | M  |  <- Median
  |    |
  | Q1 |  <- 25th percentile
   ---
    |
    *     <- Outlier (< Q1 - 1.5×IQR)
```

### Limitations
- **Not automatic:** Requires human interpretation
- **Subjective:** Different observers may disagree

---

## 17. Statistical Approaches

### Probabilistic Definition

> An outlier is an object that has a **LOW PROBABILITY** with respect to a probability distribution model of the data.

### Process

1. Assume a parametric model (e.g., normal distribution)
2. Apply statistical test depending on:
   - Data distribution
   - Parameters (mean, variance)
   - Number of expected outliers (confidence limit)

### Issues

| Issue | Description |
|-------|-------------|
| Unknown Distribution | Hard to identify true distribution |
| Heavy-tailed Distributions | Extreme values may be natural |
| High Dimensionality | Harder to estimate distribution |
| Mixture of Distributions | Data from multiple sources |

---

### 17.1 Normal Distribution

**One-Dimensional Gaussian:**
- Bell curve with mean μ and standard deviation σ
- Points far from mean (in tails) are anomalous

**95% Confidence Limits:**
- 95% of data within ~2 standard deviations
- 2.5% on each tail may be anomalous

**Two-Dimensional Gaussian:**
- Elliptical contours of equal probability
- Points outside high-probability regions are outliers

---

### 17.2 Grubbs' Test

**Purpose:** Detect outliers in **univariate** data

**Assumptions:** Data from normal distribution

**Process:** Detects one outlier at a time, removes it, repeats

**Hypotheses:**
- H₀: There is **no** outlier in data
- Hₐ: There is **at least one** outlier

**Grubbs' Test Statistic:**

```
G = max|X - X̄| / s
```

Where:
- X̄ = sample mean
- s = sample standard deviation

**Reject H₀ if:**

```
G > ((N-1) / √N) × √(t²(α/N, N-2) / (N - 2 + t²(α/N, N-2)))
```

---

### 17.3 Likelihood Approach

**Assumption:** Dataset D contains mixture of two distributions:
- **M (Majority):** Normal distribution
- **A (Anomalous):** Outlier distribution

**Data Distribution:**
```
D = (1 - λ)M + λA
```

**Algorithm:**

```
1. Initially, assume all points belong to M
2. Let L_t(D) = log likelihood at time t
3. For each point x_t in M:
   a. Move x_t to A
   b. Compute L_{t+1}(D)
   c. Compute Δ = L_t(D) - L_{t+1}(D)
   d. If Δ > c (threshold), x_t is anomaly
   e. Move permanently from M to A
```

**Key Points:**
- M estimated from data (naive Bayes, maximum entropy, etc.)
- A initially assumed to be uniform distribution

---

### 17.4 Strengths/Weaknesses of Statistical Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Firm mathematical foundation | Distribution may not be known |
| Can be very efficient | Hard to estimate in high dimensions |
| Good results if distribution known | Anomalies can distort parameters |

---

## 18. Distance-Based Approaches

### Definition (Knorr & Ng, 1998)

An object is an outlier if a **specified fraction** of objects is more than a **specified distance** away.

**Alternative:** Outlier score = distance to kth nearest neighbor

### Visual Examples

#### One Nearest Neighbor - Single Outlier
- Isolated point D has high outlier score (~2.0)
- Clustered points have low scores (~0.4-0.6)

#### One Nearest Neighbor - Two Outliers Close Together
- Two outliers near each other have **lower scores** than expected
- **Masking effect:** Outliers hide each other

#### Five Nearest Neighbors - Small Cluster
- Small cluster (5 points) may all appear as outliers
- Their 5-NN distances are high

#### Five Nearest Neighbors - Differing Density
- Sparse cluster points have moderate scores
- Dense cluster points have low scores
- True outlier has highest score

---

### Strengths/Weaknesses of Distance-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple | Expensive - O(n²) |
| No distribution assumptions | Sensitive to parameters |
| | Sensitive to density variations |
| | Distance less meaningful in high dimensions |

---

## 19. Density-Based Approaches

### Definition

**Density-based Outlier:** Outlier score = inverse of density around the object

### Density Definitions

- Inverse of distance to kth neighbor
- Inverse of average distance to k neighbors
- DBSCAN definition (core, border, noise)

### Problem: Varying Density

If regions have different densities, absolute density approach has problems (edge of sparse cluster vs. true outlier near dense cluster).

---

### 19.1 Relative Density

**Solution:** Compare density of point to density of its neighbors

**Formula:**

```
average relative density(x, k) = density(x, k) / [Σ density(y, k) / |N(x, k)|]
                                                  y∈N(x,k)
```

**Algorithm:**

```
1. {k = number of nearest neighbors}
2. for all objects x do
3.     Determine N(x, k) - k-nearest neighbors of x
4.     Determine density(x, k) using nearest neighbors
5. end for
6. for all objects x do
7.     Set outlier_score(x, k) = average relative density(x, k)
8. end for
```

**Interpretation:**
- Score ≈ 1: Similar density to neighbors (normal)
- Score >> 1: Much lower density (outlier)

---

### 19.2 Local Outlier Factor (LOF)

**Process:**
1. For each point, compute density of local neighborhood
2. Compute LOF = average of ratios: (density of neighbors) / (density of point p)
3. Outliers have **largest LOF values**

**Advantage over Distance-Based:**

```
Sparse Cluster C₁        Dense Cluster C₂
    . . . .                ::::::::::
     . . .      p₁×        ::::::::::
      . .                  ::::::::::
                    p₂×
```

- **NN approach:** p₂ may not be detected (close to dense cluster)
- **LOF approach:** Both p₁ and p₂ detected as outliers

---

### Strengths/Weaknesses of Density-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple | Expensive - O(n²) |
| Handles varying density (LOF) | Sensitive to parameters |
| | Density less meaningful in high dimensions |

---

## 20. Clustering-Based Approaches

### Definition

**Clustering-based Outlier:** Object doesn't strongly belong to any cluster

### Detection by Cluster Type

| Cluster Type | Outlier Detection |
|--------------|------------------|
| Prototype-based | Not close enough to any centroid |
| Density-based | Density too low |
| Graph-based | Not well connected |

### Other Issues
- Outliers can **distort** clusters
- Choice of **number of clusters** affects results

---

### 20.1 Distance from Closest Centroid

**Method:**
1. Cluster data (e.g., K-means)
2. For each point, compute distance to closest centroid
3. Points far from all centroids are outliers

**Example Scores:**
- Point C (isolated): 4.6 (outlier)
- Point A (edge): 1.2 (moderate)
- Point D (inside dense cluster): 0.17 (normal)

---

### 20.2 Relative Distance from Centroid

**Improvement:** Account for cluster size/spread

**Method:** Divide distance by average distance of cluster members

**Benefit:** Normalizes for different cluster sizes

---

### Strengths/Weaknesses of Clustering-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple | Hard to choose clustering technique |
| Many techniques available | Hard to choose number of clusters |
| | Outliers can distort clusters |

---

## Summary: Comparison of All Approaches

| Approach | Key Idea | Best For | Limitations |
|----------|----------|----------|-------------|
| **Statistical** | Low probability under distribution | Known distributions | Requires distribution assumptions |
| **Distance-based** | Far from other points | Simple cases | O(n²), sensitive to density |
| **Density-based** | Low local density | Varying density | O(n²), parameter sensitive |
| **Clustering-based** | Doesn't belong to clusters | When clusters exist | Depends on clustering quality |

---

## Key Formulas Quick Reference

### Cluster Validity

| Measure | Formula |
|---------|---------|
| WSS (Cohesion) | Σᵢ Σₓ∈Cᵢ (x - mᵢ)² |
| BSS (Separation) | Σᵢ \|Cᵢ\| × (m - mᵢ)² |
| Silhouette | s = 1 - a/b (if a < b) |
| Entropy | eⱼ = Σᵢ pᵢⱼ log₂(pᵢⱼ) |
| Purity | purityⱼ = max(pᵢⱼ) |

### Anomaly Detection

| Measure | Formula |
|---------|---------|
| Grubbs' Statistic | G = max\|X - X̄\| / s |
| Relative Density | density(x) / avg(neighbor densities) |
| Distance Score | Distance to kth nearest neighbor |

---

## Example Problems

### Example 1: WSS and BSS Calculation

**Problem:** Given points {2, 4, 6, 8} with K=2 clusters {2, 4} and {6, 8}, calculate WSS, BSS, and verify TSS.

**Solution:**

```
Overall mean m = (2+4+6+8)/4 = 5
Cluster 1 mean m₁ = (2+4)/2 = 3
Cluster 2 mean m₂ = (6+8)/2 = 7

WSS = (2-3)² + (4-3)² + (6-7)² + (8-7)²
    = 1 + 1 + 1 + 1 = 4

BSS = 2×(5-3)² + 2×(5-7)²
    = 2×4 + 2×4 = 16

TSS = WSS + BSS = 4 + 16 = 20

Verification:
TSS = Σ(x-m)² = (2-5)² + (4-5)² + (6-5)² + (8-5)²
    = 9 + 1 + 1 + 9 = 20 ✓
```

---

### Example 2: Silhouette Coefficient

**Problem:** Calculate silhouette coefficient for point x in cluster A, given:
- Average distance to own cluster (a) = 2
- Average distance to nearest other cluster B (b) = 8

**Solution:**

```
Since a < b:
s = 1 - a/b = 1 - 2/8 = 1 - 0.25 = 0.75

Interpretation: s = 0.75 is good (closer to 1)
Point x is well-assigned to cluster A
```

---

### Example 3: Purity Calculation

**Problem:** A cluster contains 60 documents: 45 Sports, 10 Entertainment, 5 Business. Calculate purity.

**Solution:**

```
p_sports = 45/60 = 0.75
p_entertainment = 10/60 = 0.167
p_business = 5/60 = 0.083

purity = max(p_ij) = max(0.75, 0.167, 0.083) = 0.75
```

---

### Example 4: Distance-Based Outlier Detection

**Problem:** Given 1D data {1, 2, 3, 4, 5, 100}, identify outliers using k=2 nearest neighbor distance.

**Solution:**

| Point | 2-NN | Distances | Avg Distance |
|-------|------|-----------|--------------|
| 1 | 2, 3 | 1, 2 | 1.5 |
| 2 | 1, 3 | 1, 1 | 1.0 |
| 3 | 2, 4 | 1, 1 | 1.0 |
| 4 | 3, 5 | 1, 1 | 1.0 |
| 5 | 4, 3 | 1, 2 | 1.5 |
| 100 | 5, 4 | 95, 96 | 95.5 |

**Conclusion:** Point 100 is clearly an outlier (score 95.5 >> others ~1.0-1.5)

---

## Key Takeaways

1. **Internal measures** (SSE, Silhouette) don't need labels; **External measures** (Entropy, Purity) require labels

2. **BSS + WSS = constant** - good clustering minimizes WSS and maximizes BSS

3. **Silhouette coefficient** combines cohesion and separation; closer to 1 is better

4. **Anomalies can be valuable** - the ozone depletion discovery was initially discarded as outliers

5. **Different approaches for different situations:**
   - Statistical: when distribution is known
   - Distance-based: simple cases, uniform density
   - Density-based (LOF): varying density
   - Clustering-based: when data has natural clusters

6. **All approaches have O(n²) complexity** - efficiency is a concern for large datasets

7. **Context matters** - what's anomalous depends on the situation (basketball team heights)
