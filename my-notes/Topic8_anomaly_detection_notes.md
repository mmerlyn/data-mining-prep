# Anomaly Detection

## Overview

Anomaly detection (also called outlier detection) is the process of identifying data points that are considerably different from the remainder of the data. This module covers:

1. **What are Anomalies?** - Definition and characteristics
2. **Causes of Anomalies** - Why anomalies occur
3. **Detection Approaches** - Statistical, distance-based, density-based, and clustering-based methods
4. **Evaluation Issues** - Scoring, evaluation, and challenges

---

## 1. What are Anomalies/Outliers?

### Definition

**Anomalies (Outliers):** Data points that are considerably different from the remainder of the data.

### Key Characteristics

- **Relatively Rare:** Anomalies are uncommon by nature
  - "One in a thousand occurs often if you have lots of data"
- **Context-Dependent:** What's anomalous depends on context
  - Example: Freezing temperatures in July (Northern Hemisphere) is anomalous

### Importance of Anomalies

Anomalies can be either **important** or a **nuisance**:

| Type | Example | Significance |
|------|---------|--------------|
| Nuisance (Error) | 10-foot tall 2-year-old | Likely data entry error |
| Important | Unusually high blood pressure | Medical concern requiring attention |

---

## 2. Causes of Anomalies

### 2.1 Data from Different Classes

- Objects from a different population mixed in with the main data
- **Example:** Measuring weights of oranges, but a few grapefruit are mixed in
- The grapefruit will appear as outliers in the orange weight distribution

### 2.2 Natural Variation

- Extreme but legitimate values within the same population
- **Example:** Unusually tall people
- These are genuine data points, just rare

### 2.3 Data Errors

- Mistakes in data collection, entry, or processing
- **Example:** 200-pound 2-year-old (clearly an error)
- These should typically be corrected or removed

---

## 3. Distinction Between Noise and Anomalies

### Noise

- Erroneous, perhaps random, values or contaminating objects
- **Examples:**
  - Weight recorded incorrectly
  - Grapefruit mixed in with oranges
- Noise doesn't necessarily produce unusual values
- **Noise is NOT interesting**

### Anomalies

- May or may not be caused by noise
- **Anomalies ARE interesting** if they are NOT a result of noise
- Can reveal important patterns or problems

### Key Insight

> Noise and anomalies are **related but distinct** concepts. Not all noise creates anomalies, and not all anomalies are noise.

---

## 4. General Issues in Anomaly Detection

### 4.1 Number of Attributes

**Single Attribute Anomalies:**
- Many anomalies are defined in terms of a single attribute
- Examples: Height, Shape, Color

**Multi-Attribute Challenges:**
- Can be hard to find anomalies using all attributes due to:
  - Noisy or irrelevant attributes
  - Object may only be anomalous with respect to some attributes
- However, an object may NOT be anomalous in any single attribute
  - May only appear anomalous when considering combinations

---

### 4.2 Anomaly Scoring

**Binary Categorization:**
- Object is an anomaly or it isn't
- Common in classification-based approaches

**Continuous Scoring:**
- Assigns a score measuring the degree of anomalousness
- Allows ranking of objects by how anomalous they are
- More flexible than binary

**Practical Consideration:**
- Even with scores, you often need a binary decision eventually
- Example: "Should this credit card transaction be flagged?"
- Question remains: How many anomalies are there?

---

### 4.3 Other Issues

| Issue | Description |
|-------|-------------|
| **Finding Strategy** | Find all at once vs. one at a time |
| **Swamping** | Normal points incorrectly identified as outliers |
| **Masking** | Outliers hiding other outliers |
| **Evaluation** | How to measure performance (supervised vs. unsupervised) |
| **Efficiency** | Computational cost of detection |
| **Context** | What's normal depends on context (e.g., heights on a basketball team) |

---

## 5. Variants of Anomaly Detection Problems

### Problem Formulations

**Variant 1: Threshold-based**
- Given dataset D, find all points x in D with anomaly scores greater than threshold t
- Requires setting an appropriate threshold

**Variant 2: Top-n**
- Given dataset D, find all points x in D with the top-n largest anomaly scores
- Useful when you know approximately how many anomalies to expect

**Variant 3: Novel Point Detection**
- Given dataset D (containing mostly normal, unlabeled data) and a test point x
- Compute the anomaly score of x with respect to D
- Common in streaming/online detection scenarios

---

## 6. Model-Based Anomaly Detection

### Overview

Build a model for the data and identify points that don't fit the model well.

### 6.1 Unsupervised Approaches

**Definition:** Anomalies are points that either:
- Don't fit the model well, OR
- Distort the model significantly

**Types of Models:**

| Model Type | How Anomalies are Detected |
|------------|---------------------------|
| Statistical Distribution | Low probability under the distribution |
| Clusters | Not belonging strongly to any cluster |
| Regression | Large residuals from the fitted model |
| Geometric | Outside expected geometric boundaries |
| Graph | Unusual connectivity patterns |

### 6.2 Supervised Approaches

- Anomalies are regarded as a **rare class**
- Requires labeled training data
- Often faces class imbalance challenges

---

## 7. Additional Anomaly Detection Techniques

### 7.1 Proximity-Based

- **Idea:** Anomalies are points far away from other points
- Can sometimes detect visually in low dimensions
- Uses distance metrics

### 7.2 Density-Based

- **Idea:** Low density points are outliers
- Points in sparse regions are more likely to be anomalous

### 7.3 Pattern Matching

- Create profiles or templates of atypical but important events/objects
- Algorithms are usually simple and efficient
- Good for known types of anomalies

---

## 8. Visual Approaches

### Methods

- **Boxplots:** Show outliers as points beyond whiskers
- **Scatter plots:** Visualize points that don't fit patterns

### Boxplot Interpretation

```
    *          <- Outlier (beyond 1.5 * IQR)
    |
   ---
  | Q3 |       <- 75th percentile
  |    |
  | M  |       <- Median
  |    |
  | Q1 |       <- 25th percentile
   ---
    |
    *          <- Outlier
```

### Limitations

- **Not automatic:** Requires human interpretation
- **Subjective:** Different observers may disagree
- **Scalability:** Difficult with high-dimensional data

---

## 9. Statistical Approaches

### Probabilistic Definition

> **An outlier is an object that has a LOW PROBABILITY with respect to a probability distribution model of the data.**

### Process

1. Assume a parametric model (e.g., normal distribution)
2. Apply a statistical test that depends on:
   - Data distribution
   - Parameters (mean, variance, etc.)
   - Number of expected outliers (confidence limit)

### Issues with Statistical Approaches

| Issue | Description |
|-------|-------------|
| Distribution Unknown | Hard to identify the true distribution |
| Heavy-tailed Distributions | Extreme values may be natural |
| High Dimensionality | Harder to estimate true distribution |
| Mixture of Distributions | Data may come from multiple sources |
| Parameter Distortion | Anomalies can distort distribution parameters |

---

### 9.1 Normal Distribution

**One-Dimensional Gaussian:**
- Bell curve with mean and standard deviation
- Points far from mean (in tails) are anomalous

**Two-Dimensional Gaussian:**
- Elliptical contours of equal probability
- Points outside high-probability regions are anomalous

**95% Confidence Limits:**
- 95% of data falls within ~2 standard deviations of mean
- Remaining 2.5% on each tail may be considered anomalous

---

### 9.2 Grubbs' Test

**Purpose:** Detect outliers in univariate data

**Assumptions:**
- Data comes from a normal distribution

**Process:**
1. Detects one outlier at a time
2. Remove the outlier
3. Repeat

**Hypotheses:**
- H0: There is no outlier in the data
- HA: There is at least one outlier

**Grubbs' Test Statistic:**

```
G = max|X - X̄| / s
```

Where:
- X̄ = sample mean
- s = sample standard deviation

**Rejection Criterion:**

Reject H0 if:

```
G > ((N-1) / √N) * √(t²(α/N, N-2) / (N - 2 + t²(α/N, N-2)))
```

Where t is the critical value from t-distribution.

---

### 9.3 Likelihood Approach

**Assumption:** Dataset D contains samples from a mixture of two distributions:
- **M (Majority):** The normal distribution
- **A (Anomalous):** The outlier distribution

**Data Distribution:**
```
D = (1 - λ)M + λA
```

Where λ is the proportion of anomalies.

**Algorithm:**

1. Initially, assume all points belong to M
2. Let Lt(D) be the log likelihood of D at time t
3. For each point xt in M:
   - Move it to A
   - Compute new log likelihood Lt+1(D)
   - Compute Δ = Lt(D) - Lt+1(D)
   - If Δ > c (threshold), declare xt as anomaly
   - Move permanently from M to A

**Key Points:**
- M is estimated from data (naive Bayes, maximum entropy, etc.)
- A is initially assumed to be uniform distribution
- Process iteratively identifies anomalies

---

### 9.4 Strengths/Weaknesses of Statistical Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Firm mathematical foundation | Distribution may not be known |
| Can be very efficient | Hard to estimate distribution in high dimensions |
| Good results if distribution is known | Anomalies can distort parameters |

---

## 10. Distance-Based Approaches

### Definition

**Knorr & Ng (1998):** An object is an outlier if a specified fraction of objects is more than a specified distance away.

**Alternative:** The outlier score of an object is the distance to its kth nearest neighbor.

### Visualization Examples (with specific scores from PDF)

**Example 1: One Nearest Neighbor - Single Outlier**
```
Setup: Points A, B in dense cluster; Point D isolated
Outlier Scores (1-NN distance):
- Point A: ~0.5 (close to cluster)
- Point B: ~0.5 (close to cluster)
- Point D: ~2.0 (isolated, clearly an outlier!)
```
- Isolated point far from all others has high outlier score
- Points in clusters have low scores (close to neighbors)

**Example 2: One Nearest Neighbor - Two Outliers (Masking Effect!)**
```
Setup: Points A, B in cluster; Points C, D close to each other but far from cluster
Outlier Scores (1-NN distance):
- Point A: 0.05
- Point B: 0.05
- Point C: 0.55 (close to D, so LOW score!)
- Point D: 0.55 (close to C, so LOW score!)
```
**MASKING EFFECT:** When two outliers are near each other, they can hide each other!
- Their distance to each other is small
- Both appear "normal" because they have nearby neighbors

**Example 3: Five Nearest Neighbors - Small Cluster**
```
Setup: Main dense cluster + small 3-point cluster
Outlier Scores (5-NN distance):
- Dense cluster points: 0.5 (have many nearby neighbors)
- Small cluster points: 1.6-2.0 (must look further for 5 neighbors)
```
- Small cluster of points may all have high scores
- Need enough neighbors to define "normal"

**Example 4: Five Nearest Neighbors - Differing Density Problem**
```
Setup: Dense cluster (small radius) + Sparse cluster (large radius) + Outlier
Issue: Points on edge of sparse cluster may appear MORE anomalous
       than actual outliers near dense cluster!
```
- Points on edge of sparse cluster may appear more anomalous
- Than actual outliers near dense cluster
- **This motivates the need for DENSITY-BASED approaches!**

---

### 10.1 Strengths/Weaknesses of Distance-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple to understand and implement | Expensive - O(n²) complexity |
| No distribution assumptions | Sensitive to parameters (k, threshold) |
| Works with any distance metric | Sensitive to variations in density |
| | Distance less meaningful in high dimensions |

---

## 11. Density-Based Approaches

### Definition

**Density-based Outlier:** The outlier score of an object is the inverse of the density around the object.

### Density Definitions

- **Inverse of distance to kth neighbor**
- **Inverse of average distance to k neighbors**
- **DBSCAN definition:** Core points, border points, noise

### Problem with Absolute Density

If there are regions of different density, points on the edge of a sparse cluster may have similar density to true outliers near a dense cluster.

---

### 11.1 Relative Density

**Solution:** Compare density of a point to density of its neighbors

**Formula:**

```
average relative density(x, k) = density(x, k) / (Σy∈N(x,k) density(y, k) / |N(x, k)|)
```

**Algorithm 10.2: Computing Relative Density Outlier Score**

```
ALGORITHM 10.2 - Relative Density-based Outlier Detection
Input: Dataset D, number of neighbors k
Output: Outlier score for each object

1. {k is the number of nearest neighbors}
2. for all objects x do
3.     Determine N(x, k), the k-nearest neighbors of x
4.     Determine density(x, k) using nearest neighbors in N(x,k)
5. end for
6. for all objects x do
7.     Set outlier_score(x, k) = average relative density(x, k)
       {Using formula: density(x,k) / avg_neighbor_density}
8. end for
```

**Interpretation:**
- Score ≈ 1: Similar density to neighbors (normal)
- Score >> 1: Much lower density than neighbors (outlier)
- Score << 1: Much higher density than neighbors (unusual)

**Example: Relative Density Outlier Scores (from PDF)**
```
Setup: Dense cluster with Point A at edge, Point D inside; Point C isolated
Relative Density Outlier Scores:
- Point A: 1.33 (slightly less dense than neighbors - border point)
- Point D: 1.40 (similar - inside but on edge of dense region)
- Point C: 6.85 (MUCH less dense than neighbors - OUTLIER!)
```
**Key Insight:** Point C has score ~6.85, meaning its density is only about 1/6.85 ≈ 15% of its neighbors' average density. This clearly identifies it as an outlier despite varying densities in the data!

---

### 11.2 Local Outlier Factor (LOF)

**Key Innovation:** Compares local density to neighbors' densities

**Process:**

1. For each point, compute the density of its local neighborhood
2. Compute LOF as the **average of ratios** of:
   - Density of point p
   - Density of each of p's nearest neighbors
3. Outliers are points with largest LOF values

**Advantage over Distance-Based:**

```
    C1 (sparse cluster)
    .  .  .  .
     .    .
      .  p1 (outlier detected by LOF)
           ×

    C2 (dense cluster)
    ::::::::
    ::::::::   p2 (outlier detected by LOF)
    ::::::::    ×
```

- In distance-based (NN) approach: p2 may not be detected as outlier (close to dense cluster)
- In LOF approach: Both p1 and p2 are detected as outliers

---

### 11.3 Strengths/Weaknesses of Density-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple conceptually | Expensive - O(n²) |
| Handles varying densities (LOF) | Sensitive to parameters |
| Gives meaningful scores | Density less meaningful in high dimensions |

---

## 12. Clustering-Based Approaches

### Definition

**Clustering-based Outlier:** An object is a cluster-based outlier if it does not strongly belong to any cluster.

### Detection Methods by Cluster Type

| Cluster Type | Outlier Detection Method |
|--------------|-------------------------|
| Prototype-based | Not close enough to any cluster center |
| Density-based | Density is too low |
| Graph-based | Not well connected |

### Other Issues

- **Impact of outliers on clusters:** Outliers can distort cluster centers/boundaries
- **Number of clusters:** Choice affects what's considered outlier

---

### 12.1 Distance from Closest Centroid

**Method:**
1. Cluster the data (e.g., using k-means)
2. For each point, compute distance to closest cluster centroid
3. Points far from all centroids are outliers

**Example Scores:**
- Point C (isolated): 4.6 (high - outlier)
- Point A (edge of cluster): 1.2 (moderate)
- Point D (inside dense cluster): 0.17 (low - normal)

---

### 12.2 Relative Distance from Centroid

**Improvement:** Account for cluster size/spread

**Method:**
- Divide distance by the average distance of cluster members
- Normalizes for different cluster sizes

**Benefit:** Points at edge of large sparse cluster don't appear as anomalous as true outliers

---

### 12.3 Strengths/Weaknesses of Clustering-Based Approaches

| Strengths | Weaknesses |
|-----------|------------|
| Simple and intuitive | Difficult to choose clustering technique |
| Many clustering techniques available | Difficult to choose number of clusters |
| Can use existing clustering infrastructure | Outliers can distort the clusters |

---

## Summary: Comparison of Approaches

| Approach | Key Idea | Best For | Limitations |
|----------|----------|----------|-------------|
| **Statistical** | Low probability under distribution | Known distributions | Requires distribution assumptions |
| **Distance-based** | Far from other points | Simple cases, any distance | O(n²), sensitive to density |
| **Density-based** | Low local density | Varying density data | O(n²), parameter sensitive |
| **Clustering-based** | Doesn't belong to clusters | When clusters exist | Depends on clustering quality |

---

## Key Formulas Quick Reference

| Concept | Formula |
|---------|---------|
| Grubbs' Statistic | G = max\|X - X̄\| / s |
| Relative Density | density(x) / avg(density of neighbors) |
| LOF | Average ratio of neighbor densities to point density |
| Distance Score | Distance to kth nearest neighbor |

---

## Example Problem: Identifying Outliers

**Problem:** Given the following 1D dataset, identify potential outliers using distance-based approach with k=2:

Data: {1, 2, 3, 4, 5, 100}

**Solution:**

1. Calculate 2-NN distance for each point:

| Point | 2 Nearest Neighbors | Avg Distance |
|-------|---------------------|--------------|
| 1 | 2, 3 | (1 + 2) / 2 = 1.5 |
| 2 | 1, 3 | (1 + 1) / 2 = 1.0 |
| 3 | 2, 4 | (1 + 1) / 2 = 1.0 |
| 4 | 3, 5 | (1 + 1) / 2 = 1.0 |
| 5 | 4, 3 | (1 + 2) / 2 = 1.5 |
| 100 | 5, 4 | (95 + 96) / 2 = 95.5 |

2. Point 100 has significantly higher distance score (95.5) compared to others (~1.0-1.5)

**Conclusion:** Point 100 is clearly an outlier based on distance-based detection.

---

## Example Problem: LOF Intuition

**Problem:** Why might distance-based methods fail but LOF succeed?

**Scenario:**
- Dense cluster A with 1000 points (radius 1)
- Sparse cluster B with 100 points (radius 10)
- Outlier O at distance 5 from cluster A

**Analysis:**

**Distance-based (k=5):**
- Points at edge of cluster B: avg 5th-NN distance ≈ 3
- Outlier O: 5th-NN distance ≈ 5
- Edge points of B might appear MORE anomalous than O!

**LOF:**
- Outlier O: Much lower density than its neighbors (from cluster A)
- Edge of B: Similar density to its neighbors (from cluster B)
- LOF correctly identifies O as more anomalous

**Key Insight:** LOF's local comparison handles varying densities better than global distance approaches.

---

## Practical Considerations

### When to Use Each Method

| Situation | Recommended Approach |
|-----------|---------------------|
| Known distribution | Statistical (Grubbs', z-score) |
| Unknown distribution, uniform density | Distance-based |
| Unknown distribution, varying density | Density-based (LOF) |
| Data has natural clusters | Clustering-based |
| Need interpretability | Visual + Statistical |
| High-dimensional data | Subspace methods, careful feature selection |

### Parameter Selection Guidelines

- **k (neighbors):** Start with sqrt(n), adjust based on results
- **Threshold:** Use domain knowledge or examine score distribution
- **Number of clusters:** Use elbow method or domain knowledge
