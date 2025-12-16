# TOPIC 2: DATA PROCESSING (Part A)
## CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. What is Data?
2. Attribute Types
3. Properties of Attribute Values
4. Discrete vs. Continuous Attributes
5. Types of Data Sets
6. Data Quality Issues
7. Data Preprocessing Methods
8. Quiz Questions & Examples

---

## 1. WHAT IS DATA?

### DEFINITION:
Data is a collection of **DATA OBJECTS** and their **ATTRIBUTES**.

### KEY TERMINOLOGY:

**ATTRIBUTE** (also called: variable, field, characteristic, feature)
- A property or characteristic of an object
- Examples: eye color of a person, temperature, income

**OBJECT** (also called: record, point, case, sample, entity, instance)
- Described by a collection of attributes
- Each row in a data table represents one object

### ATTRIBUTE VALUES:
- Numbers or symbols assigned to an attribute
- **IMPORTANT DISTINCTION:** Attributes vs. Attribute Values
  - Same attribute can map to different values
    - Example: Height can be measured in feet OR meters
  - Different attributes can map to same set of values
    - Example: ID and Age are both integers
    - BUT properties differ: ID has no limit, Age has min/max

### EXAMPLE TABLE (Tax Fraud Detection):

| Tid | Refund | Marital Status | Income | Cheat |
|-----|--------|----------------|--------|-------|
| 1 | Yes | Single | 125K | No |
| 2 | No | Married | 100K | No |
| 3 | No | Single | 70K | No |
| ... | | | | |

- Tid: Object identifier
- Refund, Marital Status, Income, Cheat: Attributes
- Each row (1, 2, 3...): Objects/Records

---

## 2. ATTRIBUTE TYPES (CRITICAL FOR EXAM!)

### Four Types of Attributes:

### 1. NOMINAL
- Values are just names/labels - only **DISTINCTNESS** matters
- Operations: `=` and `≠` only
- Examples:
  - ID numbers
  - Eye color (blue, brown, green)
  - Zip codes
  - Gender

### 2. ORDINAL
- Values have a meaningful **ORDER** but differences not meaningful
- Operations: `=`, `≠`, `<`, `>`
- Examples:
  - Rankings (1st, 2nd, 3rd place)
  - Taste ratings (1-10 scale)
  - Grades (A, B, C, D, F)
  - Height categories {tall, medium, short}
  - Education level (High School, Bachelor's, Master's, PhD)

### 3. INTERVAL
- Differences between values **ARE** meaningful
- **NO true zero point** (zero is arbitrary)
- Operations: `=`, `≠`, `<`, `>`, `+`, `-`
- Examples:
  - Calendar dates (year 0 is arbitrary)
  - Temperature in Celsius or Fahrenheit
    - 20°C is NOT "twice as hot" as 10°C
  - SAT scores

### 4. RATIO
- Has a **TRUE ZERO** point
- Ratios between values are meaningful
- Operations: `=`, `≠`, `<`, `>`, `+`, `-`, `*`, `/`
- Examples:
  - Temperature in Kelvin (0K = absolute zero)
  - Length, height, weight
  - Time duration
  - Counts (number of items)
  - Age
  - Income

---

## 3. PROPERTIES OF ATTRIBUTE VALUES (EXAM QUIZ!)

### MATCHING QUESTION FORMAT:
Link each attribute type to its properties.

| Attribute Type | Properties |
|---------------|------------|
| 1. Nominal | a. Distinctness only (= ≠) |
| 2. Ordinal | b. Distinctness & Order (= ≠ < >) |
| 3. Interval | c. Distinctness, Order & Addition (= ≠ < > + -) |
| 4. Ratio | d. Distinctness, Order, Addition & Multiplication (= ≠ < > + - * /) |

### ANSWERS:
- 1 → a (Nominal: Distinctness only)
- 2 → b (Ordinal: Distinctness & Order)
- 3 → c (Interval: Distinctness, Order, Addition)
- 4 → d (Ratio: All four properties)

> **MEMORY TIP:** "NOIR" = Nominal, Ordinal, Interval, Ratio
> Properties accumulate as you go down the list!

---

## 4. DISCRETE vs. CONTINUOUS ATTRIBUTES

### DISCRETE ATTRIBUTES:
- Has only a **FINITE or COUNTABLY INFINITE** set of values
- Examples:
  - Zip codes
  - Word counts
  - Number of children
  - Binary attributes (special case: only 0/1 or Yes/No)
- Often represented as **INTEGER** variables

### CONTINUOUS ATTRIBUTES:
- Has **REAL NUMBERS** as attribute values
- Examples:
  - Temperature
  - Height
  - Weight
  - Salary
- Represented as **FLOATING-POINT** variables
- In practice: measured with finite precision

---

## 5. TYPES OF DATA SETS

### Three Main Categories:

### A. RECORD DATA

**1. Data Matrix**
- Fixed set of numeric attributes
- Objects = points in multi-dimensional space
- Represented as m × n matrix (m rows, n columns)

Example:
| Projection of x Load | Projection of y Load | Distance | Load | Thickness |
|---------------------|---------------------|----------|------|-----------|
| 10.23 | 5.27 | 15.22 | 2.7 | 1.2 |
| 12.65 | 6.25 | 16.22 | 2.2 | 1.1 |

**2. Document Data**
- Each document = "term vector"
- Each term = component (attribute)
- Value = frequency of term in document

Example:
| | team | coach | play | ball | score | game | win | lost | timeout | season |
|------------|------|-------|------|------|-------|------|-----|------|---------|--------|
| Document 1 | 3 | 0 | 5 | 0 | 2 | 6 | 0 | 2 | 0 | 2 |
| Document 2 | 0 | 7 | 0 | 2 | 1 | 0 | 0 | 3 | 0 | 0 |
| Document 3 | 0 | 1 | 0 | 0 | 1 | 2 | 2 | 0 | 3 | 0 |

**3. Transaction Data**
- Special type of record data
- Each record = set of items
- Example: Grocery store purchases

| TID | Items |
|-----|-------|
| 1 | Bread, Coke, Milk |
| 2 | Beer, Bread |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Coke, Diaper, Milk |

### B. GRAPH DATA
- Examples:
  - World Wide Web (HTML links)
  - Molecular structures
  - Social networks
- Nodes connected by edges with weights

### C. ORDERED DATA

1. **Sequential/Sequence Data**
   - Sequences of transactions/events
   - Example: (A B) (D) (C E) - sequence of itemsets

2. **Spatial Data**
   - Data with geographic coordinates

3. **Temporal Data**
   - Time series data

4. **Spatio-Temporal Data**
   - Both location and time
   - Example: Average monthly temperature map

### IMPORTANT CHARACTERISTICS OF STRUCTURED DATA:

1. **DIMENSIONALITY**
   - "Curse of Dimensionality" - data becomes sparse in high dimensions

2. **SPARSITY**
   - Many zero values; only presence counts

3. **RESOLUTION**
   - Patterns depend on the scale of observation

---

## 6. DATA QUALITY ISSUES

### THREE KEY QUESTIONS:
1. What kinds of data quality problems exist?
2. How can we detect problems with the data?
3. What can we do about these problems?

### MAJOR DATA QUALITY PROBLEMS:

### A. NOISE
- Modification/distortion of original values
- Examples:
  - Distortion of voice on poor phone connection
  - "Snow" on television screen
  - Random errors in measurements

### B. OUTLIERS
- Data objects with characteristics **CONSIDERABLY DIFFERENT** from most other data objects
- May be errors OR may be interesting anomalies
- Important for anomaly detection tasks

### C. MISSING VALUES
**Reasons for missing values:**
- Information not collected (e.g., people decline to give age/weight)
- Attributes not applicable (e.g., annual income for children)

**Handling strategies:**
1. **ELIMINATE** data objects with missing values
2. **ESTIMATE** missing values (imputation)
3. **IGNORE** missing values during analysis
4. **REPLACE** with all possible values (weighted by probabilities)

### D. DUPLICATE DATA
- Same entity appears multiple times
- Major issue when **MERGING** data from heterogeneous sources
- Example: Same person with multiple email addresses

**Solution: DATA CLEANING**
- Process of dealing with duplicate data issues

---

## 7. DATA PREPROCESSING METHODS

### Seven Major Preprocessing Techniques:
a. AGGREGATION
b. SAMPLING
c. DIMENSIONALITY REDUCTION
d. FEATURE SUBSET SELECTION
e. FEATURE CREATION
f. DISCRETIZATION AND BINARIZATION
g. ATTRIBUTE TRANSFORMATION

---

### a. AGGREGATION

**DEFINITION:** Combining two or more attributes (or objects) into a single attribute (or object)

**PURPOSE:**
1. **Data Reduction** - Reduce number of attributes or objects
2. **Change of Scale** - Cities → Regions → States → Countries
3. **More "Stable" Data** - Aggregated data has LESS VARIABILITY
   - Example: Yearly precipitation has smaller standard deviation than monthly precipitation

**EXAMPLE: Australia Precipitation**
- Monthly data: High variability (std dev ranges 0-15)
- Yearly data: Low variability (std dev mostly 0-2)

---

### b. SAMPLING

**DEFINITION:** Selecting a subset of data for analysis

**WHY SAMPLE?**
- Statisticians: Obtaining entire data is too expensive/time-consuming
- Data Mining: Processing entire data is too expensive/time-consuming

**KEY PRINCIPLE:**
> "A sample works almost as well as the entire dataset IF the sample is REPRESENTATIVE"

**Representative sample:** Has approximately the SAME PROPERTIES as the original dataset

**TYPES OF SAMPLING:**

1. **Simple Random Sampling**
   - Equal probability of selecting any item

2. **Sampling WITHOUT Replacement**
   - Selected items removed from population
   - Each item can be selected only once

3. **Sampling WITH Replacement**
   - Items NOT removed after selection
   - Same item can be picked multiple times

4. **Stratified Sampling**
   - Split data into partitions (strata)
   - Draw random samples from EACH partition
   - Ensures all groups are represented

**SAMPLE SIZE CONSIDERATIONS:**
- Too small → May miss important patterns
- Example: 8000 points clearly shows patterns, 500 points may lose structure

**SAMPLE SIZE PROBLEM (Important Example!):**
> **Question:** What sample size is necessary to get at least one object from each of 10 groups?

**Analysis:**
- As sample size increases, probability of covering all 10 groups increases
- Probability follows an S-curve (starts flat, rises steeply, then levels off)
- Key points from probability curve:
  - ~20 samples: ~20% probability of covering all groups
  - ~40 samples: ~80% probability
  - ~50 samples: ~95% probability
  - ~60 samples: ~99% probability

**Conclusion:** Need approximately **60 samples** for 99% probability of getting at least one object from each of 10 groups

---

### c. DIMENSIONALITY REDUCTION

**THE CURSE OF DIMENSIONALITY:**
- As dimensions increase, data becomes increasingly **SPARSE**
- Distance/density definitions become **LESS MEANINGFUL**
- Critical for clustering and outlier detection

**PURPOSE:**
1. Avoid curse of dimensionality
2. Reduce time and memory requirements
3. Allow easier visualization
4. Eliminate irrelevant features / reduce noise

**TECHNIQUES:**

**1. Principal Component Analysis (PCA)**
- Goal: Find projection capturing **LARGEST VARIATION** in data
- Method: Find eigenvectors of covariance matrix
- Eigenvectors define the new space
- Projects data onto directions of maximum variance

**2. Singular Value Decomposition (SVD)**
- Matrix factorization technique
- Related to PCA

**3. ISOMAP (Non-linear)**
- Construct neighborhood graph
- Compute shortest path (geodesic) distances
- Preserves curved/manifold structure

**PCA EXAMPLE - Face Compression:**
Shows progressive quality loss as dimensions decrease:
- **206 dimensions** (Original): Full quality
- **160 dimensions**: Still highly recognizable, minimal loss
- **120 dimensions**: Good quality, slight degradation
- **80 dimensions**: Acceptable quality, noticeable blur
- **40 dimensions**: Degraded but still recognizable
- **10 dimensions**: Very poor quality, barely recognizable

> This demonstrates the tradeoff between dimensionality reduction and information preservation!

---

### d. FEATURE SUBSET SELECTION

**Goal:** Remove redundant and irrelevant features

**REDUNDANT FEATURES:**
- Duplicate information from other attributes
- Example: Purchase price AND sales tax paid (one determines other)

**IRRELEVANT FEATURES:**
- No useful information for the mining task
- Example: Student ID for predicting GPA

**TECHNIQUES:**

1. **Brute-force Approach**
   - Try ALL possible feature subsets
   - Computationally expensive (2^n subsets)

2. **Embedded Approaches**
   - Feature selection happens AS PART OF the algorithm
   - Example: Decision trees naturally select features

3. **Filter Approaches**
   - Features selected BEFORE running mining algorithm
   - Based on statistical measures

4. **Wrapper Approaches**
   - Use mining algorithm as BLACK BOX
   - Try subsets, evaluate performance
   - Select best performing subset

---

### e. FEATURE CREATION

**Goal:** Create NEW attributes that capture information more efficiently

**THREE METHODOLOGIES:**

1. **Feature Extraction**
   - Domain-specific
   - Example: Extract edges from images

2. **Mapping Data to New Space**
   - Transform representation
   - Example: Fourier transform, wavelet transform

3. **Feature Construction**
   - Combine existing features
   - Example: BMI = weight / height²

---

### f. DISCRETIZATION AND BINARIZATION

**Goal:** Convert continuous attributes to discrete categories

**SUPERVISED DISCRETIZATION (Using Class Labels):**
- Entropy-based approach
- Find split points that maximize class separation

**UNSUPERVISED DISCRETIZATION (Without Class Labels):**

1. **Equal Interval Width**
   - Divide range into equal-sized bins
   - Example: 0-20, 20-40, 40-60, 60-80, 80-100

2. **Equal Frequency**
   - Each bin has same number of data points
   - Adapts to data distribution

3. **K-means Clustering**
   - Use clustering to find natural groupings
   - Bins based on cluster boundaries

---

### g. ATTRIBUTE TRANSFORMATION

**DEFINITION:** Function that maps values to new replacement values

**SIMPLE TRANSFORMATIONS:**
- x^k (power)
- log(x)
- e^x (exponential)
- |x| (absolute value)

**STANDARDIZATION (Z-score normalization):**
```
x' = (x - mean) / std_dev
```
- Result: mean = 0, std dev = 1
- Good for comparing attributes with different scales

**NORMALIZATION (Min-Max):**
```
x' = (x - min) / (max - min)
```
- Result: Values in range [0, 1]

---

## 8. QUIZ QUESTIONS & EXAMPLES

### PRACTICE QUESTION 1: Attribute Type Identification

Identify the attribute type (Nominal, Ordinal, Interval, Ratio):

| Attribute | Answer | Explanation |
|-----------|--------|-------------|
| a) Social Security Number | NOMINAL | Just an identifier, no ordering |
| b) Temperature in Fahrenheit | INTERVAL | Differences meaningful, no true zero |
| c) Customer satisfaction rating (1-5 stars) | ORDINAL | Ordered but differences not equal |
| d) Weight in kilograms | RATIO | Has true zero, ratios meaningful |
| e) ZIP code | NOMINAL | Despite being numbers, just categories |
| f) Birth year | INTERVAL | Year 0 is arbitrary |
| g) Number of children | RATIO | Discrete but has true zero |

### PRACTICE QUESTION 2: Attribute Properties

**Which operations are valid for each attribute type?**
- Nominal: `= ≠`
- Ordinal: `= ≠ < >`
- Interval: `= ≠ < > + -`
- Ratio: `= ≠ < > + - * /`

### PRACTICE QUESTION 3: Data Quality

**Match the problem to the solution:**

| Problem | Solution |
|---------|----------|
| Duplicate customer records from merged databases | Data Cleaning |
| Sensor readings with random errors | Noise filtering/smoothing |
| Some customers didn't report their age | Handle missing values (eliminate, estimate, ignore) |

### PRACTICE QUESTION 4: Sampling

**Which sampling method ensures all customer segments are represented?**
- Answer: **STRATIFIED SAMPLING**

### PRACTICE QUESTION 5: When to Use Each Preprocessing Method

| Scenario | Solution |
|----------|----------|
| Dataset has 10,000 features but many are redundant | Dimensionality Reduction (PCA) or Feature Subset Selection |
| Need to aggregate daily sales to monthly | Aggregation |
| Converting income values to Low/Medium/High categories | Discretization |
| Different attributes have vastly different scales | Attribute Transformation (Standardization/Normalization) |

---

## KEY CONCEPTS SUMMARY

### DATA FUNDAMENTALS:
- Objects have Attributes with Values
- Four attribute types: Nominal → Ordinal → Interval → Ratio
- Properties accumulate: Distinctness → Order → Addition → Multiplication

### DATA QUALITY:
- Noise: Random errors/distortions
- Outliers: Unusual data points
- Missing Values: Multiple handling strategies
- Duplicates: Data cleaning required

### PREPROCESSING:
- Aggregation: Combine for stability/scale change
- Sampling: Representative subset selection
- Dimensionality Reduction: PCA, SVD, ISOMAP
- Feature Selection: Remove redundant/irrelevant features
- Discretization: Continuous → Categorical
- Transformation: Standardization, Normalization

---

## QUICK REVIEW CHECKLIST

- [ ] Can define data objects and attributes
- [ ] Know all 4 attribute types with examples
- [ ] Know which operations apply to each type
- [ ] Understand discrete vs. continuous attributes
- [ ] Know the three types of data sets (Record, Graph, Ordered)
- [ ] Can identify data quality problems
- [ ] Know strategies for handling missing values
- [ ] Understand all 7 preprocessing methods
- [ ] Know different sampling types
- [ ] Understand curse of dimensionality
- [ ] Know PCA purpose and basic concept
- [ ] Can identify when to use each preprocessing method

---

*END OF TOPIC 2A*
