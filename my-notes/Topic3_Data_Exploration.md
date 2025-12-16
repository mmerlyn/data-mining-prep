# TOPIC 3: EXPLORING DATA
## CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. What is Data Exploration?
2. Data Exploration Methods Overview
3. Summary Statistics
   - 3.1 Frequency and Mode
   - 3.2 Mean and Median
   - 3.3 Range and Variance
   - 3.4 Robust Measures (AAD, MAD, IQR)
4. Visualization
   - 4.1 Key Factors in Visualization
   - 4.2 Scatter Plots
   - 4.3 Histograms
   - 4.4 Box Plots
   - 4.5 Contour Plots
   - 4.6 Matrix Plots
   - 4.7 Parallel Coordinates
5. Python Code Examples
6. Quiz Questions & Examples

---

## 1. WHAT IS DATA EXPLORATION?

### DEFINITION:
Data Exploration is a **PRELIMINARY EXPLORATION** of the data to better understand its characteristics.

### PURPOSE:
- Select the **RIGHT TOOL** for preprocessing or analysis
- Making use of **HUMANS' ABILITIES** to recognize patterns
- Understand data before applying complex algorithms

### RELATED FIELD:
**Exploratory Data Analysis (EDA)**
- Created by statistician **JOHN TUKEY**
- Seminal book: "Exploratory Data Analysis" by Tukey
- Reference: NIST Engineering Statistics Handbook
  - http://www.itl.nist.gov/div898/handbook/index.htm

---

## 2. DATA EXPLORATION METHODS OVERVIEW

### Three Main Methods:

| Method | Description |
|--------|-------------|
| 1. Summary Statistics | Numbers that summarize data properties |
| 2. Visualization | Visual/graphical representation of data |
| 3. Clustering & Anomaly Detection | Automated pattern/outlier detection |

### IRIS SAMPLE DATASET (Used Throughout This Lecture):
- Source: UCI Machine Learning Repository
- Three flower types (classes):
  - Setosa
  - Virginica
  - Versicolour
- Four attributes:
  - Sepal width
  - Sepal length
  - Petal width
  - Petal length
- 150 samples total (50 per class)

---

## 3. SUMMARY STATISTICS

### DEFINITION:
Summary statistics are **NUMBERS** that summarize properties of the data.

### KEY PROPERTY:
Most summary statistics can be calculated in a **SINGLE PASS** through the data.

---

### 3.1 FREQUENCY AND MODE

**FREQUENCY:**
- The percentage of time a value occurs in the data set
- Example: 'female' occurs about 50% of the time

**MODE:**
- The **MOST FREQUENT** attribute value
- Frequency and mode are typically used with CATEGORICAL data

**Example:**
If data has values: {A, A, A, B, B, C}
- Frequency of A = 3/6 = 50%
- Mode = A (most frequent)

---

### 3.2 MEAN AND MEDIAN

**MEAN (Average):**
- Most common measure of the **LOCATION** of a set of points

**FORMULA:**
```
mean(x) = x̄ = (1/m) × Σxᵢ for i=1 to m
```
Where m = number of data points

**MEDIAN:**
- The **MIDDLE** value when data is sorted

**FORMULA:**
```
median(x) = x₍ᵣ₊₁₎           if m is odd (m = 2r + 1)
median(x) = ½(x₍ᵣ₎ + x₍ᵣ₊₁₎)  if m is even (m = 2r)
```

> **IMPORTANT:** Mean is **VERY SENSITIVE TO OUTLIERS!**

**Example:**
```
Data: {1, 2, 3, 4, 100}
- Mean = (1+2+3+4+100)/5 = 22  (heavily influenced by 100)
- Median = 3                   (not affected by 100)
```

---

### 3.3 RANGE AND VARIANCE

**RANGE:**
- Difference between MAXIMUM and MINIMUM values
- Range = max - min

**VARIANCE:**
- Most common measure of the **SPREAD** of a set of points

**FORMULA:**
```
variance(x) = s²ₓ = (1/(m-1)) × Σ(xᵢ - x̄)² for i=1 to m
```

**STANDARD DEVIATION:**
```
std_dev(x) = sₓ = √variance(x)
```

> **NOTE:** Variance is also **SENSITIVE TO OUTLIERS!**

---

### 3.4 ROBUST MEASURES (Less Sensitive to Outliers)

Since mean and variance are sensitive to outliers, other measures are often used:

**1. AAD (Average Absolute Deviation):**
```
AAD(x) = (1/m) × Σ|xᵢ - x̄| for i=1 to m
```

**2. MAD (Median Absolute Deviation):**
```
MAD(x) = median({|x₁ - x̄|, ..., |xₘ - x̄|})
```

**3. IQR (Interquartile Range):**
```
IQR(x) = x₇₅% - x₂₅%
```
Where:
- x₂₅% = 25th percentile (first quartile)
- x₇₅% = 75th percentile (third quartile)

---

## QUIZ: OUTLIER SENSITIVITY

**QUESTION:** Which measure is MORE SENSITIVE to outliers? Why?

```
AAD(x) = (1/m) Σ |xᵢ - x̄|    vs.    variance(x) = (1/(m-1)) Σ (xᵢ - x̄)²
```

**ANSWER:** **VARIANCE** is more sensitive to outliers because:
- Variance uses **SQUARED** differences
- Squaring amplifies large deviations
- An outlier far from the mean contributes disproportionately more to variance
- AAD uses absolute value, which doesn't amplify large deviations

**Example:**
```
Data: {1, 2, 3, 4, 5}, mean = 3
- Deviation of 5 from mean = |5-3| = 2 (for AAD)
- Squared deviation = (5-3)² = 4 (for variance)

Data: {1, 2, 3, 4, 100}, mean = 22
- Deviation of 100 from mean = |100-22| = 78 (for AAD)
- Squared deviation = (100-22)² = 6084 (for variance) -- MUCH LARGER!
```

---

## 4. VISUALIZATION

### DEFINITION:
Conversion of data into a **VISUAL or TABULAR FORMAT** so that:
- Characteristics of the data can be analyzed
- Relationships among data items or attributes can be reported

### WHY VISUALIZATION IS POWERFUL:
- Most powerful and appealing technique in data exploration
- Humans have well-developed ability to analyze visual information
- Can detect **GENERAL PATTERNS** and **TRENDS**
- Can detect **OUTLIERS** and unusual patterns

---

### 4.1 KEY FACTORS IN VISUALIZATION

#### FACTOR 1: REPRESENTATION
- Mapping of information to visual format
- Questions: Points, lines, shapes, colors?

**How objects and attributes are represented:**
- Objects → often represented as **POINTS**
- Attribute values → represented as:
  - Position of points
  - Color of points
  - Size of points
  - Shape of points

If **POSITION** is used, relationships (groups, outliers) are easily perceived.

#### FACTOR 2: ARRANGEMENT
- Placement of visual elements within a display
- Can make a **LARGE DIFFERENCE** in understanding data

**Example (Binary Matrix):**

Original arrangement (hard to see pattern):
```
    | 1  2  3  4  5  6
  ――+――――――――――――――――――
  1 | 0  1  0  1  1  0
  2 | 1  0  1  0  0  1
  3 | 0  1  0  1  1  0
  ...
```

Rearranged (pattern visible - two groups):
```
    | 6  1  3  2  5  4
  ――+――――――――――――――――――
  4 | 1  1  1  0  0  0
  2 | 1  1  1  0  0  0
  6 | 1  1  1  0  0  0
  8 | 1  1  1  0  0  0
  5 | 0  0  0  1  1  1
  3 | 0  0  0  1  1  1
  ...
```

#### FACTOR 3: SELECTION
- Elimination of certain objects and attributes
- Choosing a subset of attributes:
  - Dimensionality reduction
  - Pairs of attributes can be considered
- Choosing a subset of objects (SAMPLING):
  - Screen can only show limited points
  - Want to preserve points in sparse areas

---

### 4.2 SCATTER PLOTS

**DESCRIPTION:**
- Attribute values determine the **POSITION** of points
- Most common: Two-dimensional scatter plots
- Can also have three-dimensional scatter plots

**ADDITIONAL INFORMATION:**
- Size of markers → additional attribute
- Shape of markers → additional attribute
- Color of markers → additional attribute (often class label)

**SCATTER PLOT ARRAY (Scatter Matrix):**
- Arrays of scatter plots compactly summarize relationships
- Shows all pairs of attributes simultaneously
- Diagonal often shows histograms of individual attributes

**Example: Iris Scatter Plot Array**
- 4x4 grid showing all attribute pairs
- Each cell shows relationship between two attributes
- Different colors/markers for each flower class
- Clearly shows Setosa is separable from other classes

---

### 4.3 HISTOGRAMS

**DESCRIPTION:**
- Usually shows **DISTRIBUTION** of values of a single variable
- Divide values into **BINS**
- Show bar plot with height = number of objects in each bin

**KEY POINTS:**
- Height of each bar indicates the **NUMBER of objects**
- Shape of histogram depends on the **NUMBER OF BINS**
  - Too few bins: lose detail
  - Too many bins: too noisy

**Example: Petal Width Histogram**
- 10 bins: Coarser view, shows general bimodal distribution
- 20 bins: Finer detail, reveals more structure

**TWO-DIMENSIONAL HISTOGRAMS:**
- Show **JOINT DISTRIBUTION** of two attributes
- Example: Petal width vs. Petal length
- Height of bar shows count of objects in each 2D bin
- Reveals correlations between attributes

---

### 4.4 BOX PLOTS (Box-and-Whisker Plots)

**PERCENTILES:**
- For continuous data, percentile is more useful than frequency
- The pth percentile is the value such that p% of all values are less than it
- 50th percentile = MEDIAN

**BOX PLOT COMPONENTS:**
```
                ←― outlier (individual point)

                ←― 10th percentile (or 90th)
                |
          ┌─────┴─────┐
          │           │ ←― 75th percentile (Q3)
          │―――――――――――│ ←― 50th percentile (MEDIAN)
          │           │ ←― 25th percentile (Q1)
          └─────┬─────┘
                |
                ←― 10th percentile
```

**INVENTED BY:** J. Tukey

**USES:**
- Display distribution of data
- Compare multiple attributes or groups
- Identify outliers (points beyond whiskers)
- Compare medians and spreads across categories

---

### 4.5 CONTOUR PLOTS

**DESCRIPTION:**
- Useful when continuous attribute is measured on a **SPATIAL GRID**
- Partition the plane into **REGIONS OF SIMILAR VALUES**
- Contour lines connect points with **EQUAL VALUES**

**COMMON EXAMPLES:**
- Elevation maps (topographic)
- Temperature maps
- Rainfall maps
- Air pressure maps
- Sea Surface Temperature (SST)

**Example: Sea Surface Temperature (SST) for December 1998**
- Color bands show temperature ranges
- Contour lines separate temperature regions
- Can see warm equatorial waters, cold polar waters

---

### 4.6 MATRIX PLOTS

**DESCRIPTION:**
- Plot the **DATA MATRIX** directly as an image
- Useful when objects are **SORTED** according to class
- Color represents attribute value

**Example: Iris Data Matrix Visualization**
- Rows: 150 flower samples (sorted by class)
- Columns: 4 attributes (sepal length, sepal width, petal length, petal width)
- Color: standardized values (standard deviations from mean)
- Can see clear differences between classes for petal attributes

**CORRELATION MATRIX VISUALIZATION:**
- Plot pairwise correlations between objects
- Shows how similar objects are to each other
- Block diagonal structure reveals classes

---

### 4.7 PARALLEL COORDINATES

**DESCRIPTION:**
- Used to plot attribute values of **HIGH-DIMENSIONAL** data
- Instead of perpendicular axes, use **PARALLEL AXES**
- Each attribute has its own vertical axis
- Each object is represented as a **LINE** connecting its values

**HOW IT WORKS:**
1. Each attribute gets a vertical axis (parallel to others)
2. For each object, plot its value on each axis
3. Connect the points with a line
4. Each object becomes a single line across all attributes

**OBSERVATIONS:**
- Lines representing same class often **GROUP TOGETHER**
- Can see which attributes separate classes
- **ORDERING** of attributes is important for seeing patterns

**Example: Iris Parallel Coordinates**
- 4 parallel axes: sepal length, sepal width, petal length, petal width
- Different colors for each flower class
- Setosa (blue) clearly separates from others on petal attributes
- Versicolor and Virginica overlap more

---

## 5. PYTHON CODE EXAMPLES

### LOADING THE IRIS DATASET:
```python
%matplotlib inline
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import datasets

iris = load_iris()
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
print(iris.data.shape)      # (150, 4)
```

### SCATTER PLOT:
```python
X = iris.data

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c="blue", cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(4, 8)
plt.ylim(1, 5)
```

### SCATTER MATRIX:
```python
import pandas as pd

iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target

X_df = pd.DataFrame(X_iris, columns=['s len', 's width', 'p len', 'p width'])
pd.plotting.scatter_matrix(X_df)
```

### HISTOGRAM:
```python
# histogram for sepal length
plt.figure(figsize = (10, 7))
x = iris.data[:,0]

plt.hist(x, bins = 20, color = "blue")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
```

### BOX PLOT (Single Feature):
```python
# BOXPLOT
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X_iris = iris.data
X_sepal = X_iris[:, 0]

plt.boxplot(X_sepal, labels=[iris.feature_names[0]])
plt.title("Boxplot Sepal Length")
plt.ylabel("cm")
plt.show
```

### BOX PLOT (Multiple Features):
```python
plt.boxplot(X_iris, labels=[iris.feature_names[0], iris.feature_names[1],
            iris.feature_names[2], iris.feature_names[3]])
plt.title("Boxplots Iris features")
plt.ylabel("cm")
plt.show
```

### CONTOUR PLOT:
```python
# matplotlib.pyplot.contour([X, Y,] Z, [levels], **kwargs)
# X, Y: coordinates of values in Z
# Z: height values over which the contour is drawn
# levels: number and positions of contour lines
```

### MATRIX PLOT:
```python
# matplotlib.pyplot.matshow(A, fignum=None, **kwargs)
# A: array-like(M, N) - the matrix to be displayed
```

### PARALLEL COORDINATES:
```python
import pandas as pd
import numpy as np

iris = load_iris()
iris_data = np.hstack((iris.data, iris.target.reshape(-1,1)))

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names + ["classes"])
iris_df.head()
pd.plotting.parallel_coordinates(iris_df, "classes", color=["red", "green", "blue"])
```

---

## 6. QUIZ QUESTIONS & EXAMPLES

### PRACTICE QUESTION 1: Summary Statistics

Given data: {2, 4, 6, 8, 10, 100}

Calculate:
- a) Mean
- b) Median
- c) Range
- d) Which is more robust to the outlier (100)?

**ANSWERS:**
- a) Mean = (2+4+6+8+10+100)/6 = 130/6 = **21.67**
- b) Median = (6+8)/2 = **7** (average of middle two values)
- c) Range = 100 - 2 = **98**
- d) **MEDIAN** is more robust - it's only 7, while mean is 21.67

### PRACTICE QUESTION 2: Choosing Visualization

**Match the scenario to the best visualization technique:**

| Scenario | Answer |
|----------|--------|
| a) Show distribution of a single continuous variable | HISTOGRAM or BOX PLOT |
| b) Show relationship between two continuous variables | SCATTER PLOT |
| c) Compare distributions across multiple categories | BOX PLOTS (multiple) |
| d) Show high-dimensional data (>3 attributes) | PARALLEL COORDINATES |
| e) Show spatial data (temperature across a map) | CONTOUR PLOT |
| f) Show pairwise relationships for multiple attributes | SCATTER PLOT ARRAY (Matrix) |

### PRACTICE QUESTION 3: Box Plot Interpretation

Given a box plot, identify:
- Median (50th percentile)
- Q1 (25th percentile)
- Q3 (75th percentile)
- IQR (Interquartile Range)
- Outliers

```
IQR = Q3 - Q1
Outliers: typically points beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
```

### PRACTICE QUESTION 4: Sensitivity to Outliers

**Rank these measures from MOST to LEAST sensitive to outliers:**
- Mean
- Median
- Variance
- MAD

**ANSWER (Most to Least sensitive):**
1. **Variance** (squared differences amplify outliers)
2. **Mean** (outliers directly affect sum)
3. **MAD** (uses median, which is robust)
4. **Median** (middle value, outliers don't affect it much)

### PRACTICE QUESTION 5: Visualization Factors

**Name the three key factors in effective visualization:**

1. **REPRESENTATION** - How data is mapped to visual elements
2. **ARRANGEMENT** - Placement of visual elements
3. **SELECTION** - Choosing which objects/attributes to display

---

## KEY CONCEPTS SUMMARY

### SUMMARY STATISTICS:
- **Frequency/Mode:** For categorical data
- **Mean:** Average, sensitive to outliers
- **Median:** Middle value, robust to outliers
- **Variance/Std Dev:** Measure of spread, sensitive to outliers
- **AAD, MAD, IQR:** Robust alternatives

### VISUALIZATION:
- **Purpose:** Use human pattern recognition
- **Key Factors:** Representation, Arrangement, Selection

### VISUALIZATION TECHNIQUES:

| Technique | Best Used For |
|-----------|---------------|
| Scatter Plot | Relationship between 2 variables |
| Histogram | Distribution of single variable |
| Box Plot | Distribution, outliers, comparisons |
| Contour Plot | Spatial/grid data with continuous values |
| Matrix Plot | Visualizing entire data matrix |
| Parallel Coordinates | High-dimensional data |

---

## QUICK REVIEW CHECKLIST

- [ ] Understand purpose of data exploration
- [ ] Know all summary statistics (mean, median, mode, variance, etc.)
- [ ] Understand which measures are sensitive to outliers
- [ ] Know formulas for mean, median, variance, AAD, MAD, IQR
- [ ] Can explain why variance is more sensitive to outliers than AAD
- [ ] Know three key factors in visualization
- [ ] Know all visualization techniques and when to use each
- [ ] Understand scatter plots and scatter plot arrays
- [ ] Understand histograms (1D and 2D)
- [ ] Know how to interpret box plots (percentiles, outliers)
- [ ] Understand parallel coordinates for high-dimensional data
- [ ] Can write basic Python code for each visualization type

---

*END OF TOPIC 3*
