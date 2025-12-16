# TOPIC 1: INTRODUCTION TO DATA MINING
## CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. What is Data Mining?
2. Data Mining Tasks (Overview)
3. Classification (Predictive)
4. Clustering (Descriptive)
5. Association Rule Discovery (Descriptive)
6. Regression (Predictive)
7. Anomaly/Deviation Detection (Predictive)
8. Challenges of Data Mining
9. Why is Data Mining Important?
10. Related Fields
11. KDD Process
12. Exam-Style Questions

---

## 1. WHAT IS DATA MINING?

### DEFINITION:
Data Mining is also called **Knowledge Discovery and Data Mining (KDD)**.

Data Mining is the **EXTRACTION OF USEFUL PATTERNS** from data sources such as:
- Databases
- Texts
- Web
- Images
- etc.

### KEY REQUIREMENT - Patterns must be:
- Valid
- Novel (previously unknown)
- Potentially useful
- Understandable

### WHAT DATA MINING IS vs. IS NOT
> This is a **CRITICAL exam topic** (see Midterm Question 1)

**IS Data Mining:**
- Prediction (forecasting future values)
- Classification (assigning categories)
- Clustering (grouping similar items)
- Association discovery (finding relationships)
- Anomaly detection (finding unusual patterns)

**IS NOT Data Mining:**
- Simple database queries
- Sorting data
- Aggregation (sum, count, average)
- Signal processing (e.g., extracting frequencies from sound waves)
- Simple grouping by known attributes (e.g., grouping by gender)

---

## 2. DATA MINING TASKS - OVERVIEW

### Two Main Categories:

**PREDICTION METHODS (Predictive):**
- Use some variables to predict unknown or future values of other variables
- Tasks: Classification, Regression, Anomaly Detection

**DESCRIPTION METHODS (Descriptive):**
- Find human-interpretable patterns that describe the data
- Tasks: Clustering, Association Rule Discovery

### Summary Table:

| Task | Type |
|------|------|
| Classification | Predictive |
| Clustering | Descriptive |
| Association Rule Discovery | Descriptive |
| Regression | Predictive |
| Anomaly Detection | Predictive |

---

## 3. CLASSIFICATION [Predictive]

### DEFINITION:
Given a collection of records (TRAINING SET):
- Each record contains a set of **ATTRIBUTES**
- One attribute is the **CLASS** (target variable)

**Goal:** Find a MODEL for the class attribute as a function of other attributes
- Previously unseen records should be assigned a class as accurately as possible
- A TEST SET is used to determine the accuracy of the model

### Process:
```
Training Set --> Learn Classifier --> Model --> Apply to Test Set --> Predictions
```

### EXAMPLE FROM LECTURE: Tax Fraud Detection

**Training Data:**

| Tid | Refund | Marital Status | Income | Cheat |
|-----|--------|----------------|--------|-------|
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

- Refund, Marital Status: Categorical attributes
- Income: Continuous attribute
- Cheat: Class label (what we want to predict)

### CLASSIFICATION APPLICATIONS:

**Application 1: Direct Marketing**
- Goal: Reduce cost of mailing by TARGETING consumers likely to buy
- Approach:
  - Use data from similar product introduced before
  - {buy, don't buy} decision forms the class attribute
  - Collect demographic, lifestyle, company-interaction info
  - Learn a classifier model

**Application 2: Fraud Detection**
- Goal: Predict fraudulent cases in credit card transactions
- Approach:
  - Use credit card transactions and account holder info as attributes
  - Label past transactions as fraud or fair (class attribute)
  - Learn a model for the class
  - Use model to detect fraud in new transactions

**Application 3: Sky Survey Cataloging**
- Goal: Predict class (star or galaxy) of sky objects
- Data: 3000 images with 23,040 x 23,040 pixels each
- Approach:
  - Segment the image
  - Measure 40 image features per object
  - Model the class based on features
- Success: Found 16 new high red-shift quasars!

---

## 4. CLUSTERING [Descriptive]

### DEFINITION:
Given a set of data points, each having a set of attributes, and a **SIMILARITY MEASURE** among them, find clusters such that:
- Data points in ONE cluster are **MORE SIMILAR** to one another
- Data points in SEPARATE clusters are **LESS SIMILAR** to one another

### Key Concept:
- Intracluster distances are **MINIMIZED**
- Intercluster distances are **MAXIMIZED**

### Similarity Measures:
- Euclidean Distance (for continuous attributes)
- Other problem-specific measures

### CLUSTERING APPLICATIONS:

**Application 1: Market Segmentation**
- Goal: Subdivide a market into distinct subsets of customers
- Approach:
  - Collect geographical and lifestyle attributes
  - Find clusters of similar customers
  - Measure quality by observing buying patterns within vs. across clusters

**Application 2: Document Clustering**
- Goal: Find groups of similar documents based on terms
- Approach:
  - Identify frequently occurring terms in each document
  - Form similarity measure based on term frequencies
  - Cluster documents
- Benefit: Information retrieval can use clusters to relate new documents

**Example: Los Angeles Times Article Clustering**
- 3204 articles clustered by common words
- Categories: Financial, Foreign, National, Metro, Sports, Entertainment
- Achieved good accuracy (e.g., 746/943 Metro articles correctly placed)

**Application 3: S&P 500 Stock Clustering**
- Observe stock movements (UP/DOWN) daily
- Similarity: Events that frequently happen together on same day
- Discovered clusters matched industry groups:
  - Technology1-DOWN
  - Technology2-DOWN
  - Financial-DOWN
  - Oil-UP

---

## 5. ASSOCIATION RULE DISCOVERY [Descriptive]

### DEFINITION:
Given a set of records each containing some number of ITEMS from a collection, produce **DEPENDENCY RULES** that predict occurrence of an item based on occurrences of other items.

### Example:

| TID | Items |
|-----|-------|
| 1 | Bread, Coke, Milk |
| 2 | Beer, Bread |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Coke, Diaper, Milk |

**Rules Discovered:**
- {Milk} --> {Coke}
- {Diaper, Milk} --> {Beer}

### ASSOCIATION RULE APPLICATIONS:

**Application 1: Marketing and Sales Promotion**

Rule: {Bagels, ...} --> {Potato Chips}
- Potato Chips as CONSEQUENT: Determine what boosts its sales
- Bagels in ANTECEDENT: See what's affected if store stops selling bagels
- Both together: See what to sell with Bagels to promote Potato Chips

**Application 2: Supermarket Shelf Management**
- Goal: Identify items bought together by many customers
- Approach: Process point-of-sale barcode data to find dependencies
- Classic Rule: "If customer buys diaper, very likely to buy milk"

**Application 3: Inventory Management**
- Goal: Anticipate repair needs, keep service vehicles equipped
- Approach: Process data on tools/parts from previous repairs
- Discover co-occurrence patterns

### SEQUENTIAL PATTERN MINING:
A sequential rule: A --> B says that event A will be **IMMEDIATELY FOLLOWED** by event B with a certain confidence

(Note: Considers temporal ordering, unlike regular association rules)

---

## 6. REGRESSION [Predictive]

### DEFINITION:
Predict a value of a given **CONTINUOUS valued variable** based on the values of other variables, assuming a linear or nonlinear model of dependency.

### Key Difference from Classification:
- **Classification:** Predicts CATEGORICAL (discrete) values
- **Regression:** Predicts CONTINUOUS (numeric) values

Greatly studied in statistics and neural network fields.

### EXAMPLES:
- Predicting **SALES AMOUNTS** of new product based on advertising expenditure
- Predicting **WIND VELOCITIES** as a function of temperature, humidity, air pressure
- **TIME SERIES PREDICTION** of stock market indices

---

## 7. DEVIATION/ANOMALY DETECTION [Predictive]

### DEFINITION:
Detect **SIGNIFICANT DEVIATIONS** from normal behavior.

### APPLICATIONS:

**1. Credit Card Fraud Detection**
- Identify transactions that deviate from normal spending patterns
- Example: Unusual purchase location, amount, or timing

**2. Network Intrusion Detection**
- Typical university network: Over 100 million connections per day
- Detect abnormal traffic patterns that may indicate attacks

---

## 8. CHALLENGES OF DATA MINING

1. **SCALABILITY**
   - Handling very large datasets efficiently

2. **DIMENSIONALITY**
   - High number of attributes (curse of dimensionality)

3. **COMPLEX AND HETEROGENEOUS DATA**
   - Different data types, formats, sources

4. **DATA QUALITY**
   - Missing values, noise, inconsistencies

5. **DATA OWNERSHIP AND DISTRIBUTION**
   - Data spread across multiple locations/organizations

6. **PRIVACY PRESERVATION**
   - Mining while protecting sensitive information

7. **STREAMING DATA**
   - Real-time data that arrives continuously

---

## 9. WHY IS DATA MINING IMPORTANT?

### WHY KDD IS IMPORTANT:
- Computerization of businesses produces **HUGE amounts of data**
- Knowledge discovered can be used for **COMPETITIVE ADVANTAGE**
- Online e-businesses generate even larger datasets
- Online retailers (e.g., Amazon) are largely driven by data mining
- Web search engines are essentially data mining companies

### WHY DATA MINING IS NECESSARY:
- Make use of your **DATA ASSETS**
- Big gap from stored data to knowledge (won't occur automatically)
- Many interesting things **CANNOT be found using database queries:**
  - "Find people likely to buy my products"
  - "Who are likely to respond to my promotion?"
  - "Which movies should be recommended to each customer?"

### WHY NOW:
- Data is **ABUNDANT**
- Computing power is **NOT an issue**
- Data mining tools are **AVAILABLE**
- Competitive pressure is very strong (every company is doing it)

---

## 10. RELATED FIELDS

Data Mining is a **MULTI-DISCIPLINARY** field:
- Machine Learning
- Statistics
- Databases
- Information Retrieval
- Visualization
- Natural Language Processing
- Pattern Recognition
- Probability Theory

### DATA MINING APPLICATIONS:
- Marketing, customer profiling/retention, market segmentation
- Engineering: identify causes of problems in products
- Scientific data analysis (e.g., bioinformatics)
- Fraud detection: credit card fraud, intrusion detection
- Text and web: huge number of applications
- Any application involving large amounts of data

---

## 11. KDD (KNOWLEDGE DISCOVERY IN DATABASES) PROCESS

### Step-by-Step Process:
1. **UNDERSTAND** the application domain
2. **IDENTIFY** data sources and **SELECT** target data
3. **PRE-PROCESSING:** cleaning, attribute selection, etc.
4. **DATA MINING** to extract patterns or models
5. **POST-PROCESSING:** identifying interesting/useful patterns
6. **INCORPORATE** patterns/knowledge in real-world tasks

### Visual Flow:
```
Data --> Selection --> Preprocessing --> Transformation --> Data Mining --> Interpretation --> Knowledge
```

---

## 12. EXAM-STYLE QUESTIONS (Based on Midterm)

### QUESTION TYPE 1: Identify Data Mining vs. Non-Data Mining Systems

**Instructions:** Put 'Yes' (is data mining) or 'No' (not data mining)

### Practice Examples:

**(a) A system that groups customers by their genders**
- Answer: **NO** - Simple grouping by known attribute, no pattern discovery

**(b) A system that recommends products to customers likely to buy**
- Answer: **YES** - Prediction/recommendation based on learned patterns

**(c) A system that sorts a database by ID numbers**
- Answer: **NO** - Simple sorting, no pattern discovery

**(d) A system that calculates total sales from records**
- Answer: **NO** - Simple aggregation (sum)

**(e) A system that extracts frequencies from sound waves**
- Answer: **NO** - Signal processing, not data mining

**(f) A system that predicts future stock prices from historical data**
- Answer: **YES** - Prediction (regression) using learned patterns

**(g) A system that monitors heart rate for abnormalities**
- Answer: **YES** - Anomaly detection

**(h) A system that predicts seismic waves for earthquakes**
- Answer: **YES** - Prediction using patterns

**(i) A system that identifies fraud transactions different from others**
- Answer: **YES** - Anomaly detection

### KEY DISTINCTIONS TO REMEMBER:

**NOT Data Mining:**
- Simple queries (SELECT, WHERE)
- Sorting
- Aggregation (SUM, COUNT, AVG)
- Grouping by explicit attributes
- Signal processing

**IS Data Mining:**
- Prediction (classification, regression)
- Clustering (finding unknown groups)
- Association rules (finding relationships)
- Anomaly detection (finding unusual patterns)
- Recommendation systems

---

## QUICK REVIEW CHECKLIST

- [ ] Can define data mining and KDD
- [ ] Know the difference between Predictive and Descriptive tasks
- [ ] Can identify all 5 main data mining tasks and their types
- [ ] Understand Classification with examples
- [ ] Understand Clustering with examples
- [ ] Understand Association Rules with examples
- [ ] Know difference between Classification and Regression
- [ ] Can identify what IS and IS NOT data mining
- [ ] Know the KDD process steps
- [ ] Understand challenges of data mining
- [ ] Know why data mining is important

---

*END OF TOPIC 1*
