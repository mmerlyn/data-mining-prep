# ASSOCIATION ANALYSIS PART B (ADVANCED TOPICS)
## CS653: DATA MINING
### Detailed Study Notes - Lecture 06-B

---

## TABLE OF CONTENTS
1. Recap from Part A
2. Association Rules for Categorical Attributes
3. Association Rules for Continuous Attributes
   - 3.1 Discretization-based Methods
   - 3.2 Statistics-based Methods
   - 3.3 Non-discretization Based: Min-Apriori
4. Multi-level Association Rules
5. Sequential Pattern Mining
6. Frequent Subgraph Mining (Optional)
7. Examples and Problems
8. Summary

---

## 1. RECAP FROM PART A

### KEY CONCEPTS FROM PART A
- Association Rule: {Milk, Diaper} -> {Beer}
- Two-step Association Rule Mining:
  1. Frequent itemset generation
  2. Rule generation
- Rule evaluation using support and confidence

### SAMPLE TRANSACTION DATA

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

### IMPORTANT ASSUMPTION IN PART A
We assumed every attribute has one unique value - but this is NOT always true!
- Real-world data has CATEGORICAL attributes (multiple possible values)
- Real-world data has CONTINUOUS attributes (infinite possible values)


---

## 2. ASSOCIATION RULES FOR CATEGORICAL ATTRIBUTES

### THE PROBLEM
How do we handle categorical attributes in association rule mining?

**Example Dataset:**

| Session Id | Country | Session Length(sec) | Number of Web Pages | Gender | Browser Type | Buy |
|------------|---------|---------------------|---------------------|--------|--------------|-----|
| 1 | USA | 982 | 8 | Male | IE | No |
| 2 | China | 811 | 10 | Female | Netscape | No |
| 3 | USA | 2125 | 45 | Female | Mozilla | Yes |
| 4 | Germany | 596 | 4 | Male | IE | Yes |
| 5 | Australia | 123 | 9 | Male | Mozilla | No |
| ... | ... | ... | ... | ... | ... | ... |

**Example Association Rule:**
{Number of Pages in [5,10) AND Browser=Mozilla} -> {Buy = No}

### SOLUTION: TRANSFORM TO ASYMMETRIC BINARY VARIABLES

**Method:** Introduce a NEW "item" for each distinct attribute-value pair

Example: Replace "Browser Type" attribute with:
- Browser Type = Internet Explorer
- Browser Type = Mozilla
- Browser Type = Netscape

Each becomes a separate binary item (0 or 1)

### POTENTIAL ISSUES AND SOLUTIONS

**ISSUE 1: Attribute has many possible values**
- Example: "Country" attribute has more than 200 possible values
- Many attribute values may have VERY LOW SUPPORT
- **Solution:** AGGREGATE the low-support attribute values
  - Group countries by region (Asia, Europe, Americas, etc.)

**ISSUE 2: Distribution of attribute values is highly skewed**
- Example: 95% of visitors have Buy = No
- Problem: Most items will be associated with (Buy=No) item
- **Solution:** DROP the highly frequent items
  - Or use interest measures to filter uninteresting rules


---

## 3. ASSOCIATION RULES FOR CONTINUOUS ATTRIBUTES

### DIFFERENT KINDS OF RULES FOR CONTINUOUS DATA

**Type 1: Discretized intervals in rules**
- Age in [21,35) AND Salary in [70k,120k) -> Buy

**Type 2: Statistics-based rules**
- Salary in [70k,120k) AND Buy -> Age: mean=28, std=4

### THREE APPROACHES FOR HANDLING CONTINUOUS ATTRIBUTES
- A. Discretization-based
- B. Statistics-based
- C. Non-discretization based: minApriori

---

### 3.1 DISCRETIZATION-BASED METHODS

### DISCRETIZATION METHODS

**UNSUPERVISED METHODS:**
- Equal-width binning: Divide range into k bins of equal width
- Equal-depth binning: Divide into k bins with equal number of records
- Clustering: Use clustering to determine bins

**SUPERVISED METHODS:**
- Use class labels to guide discretization
- Example: Separate bins for Anomalous vs Normal classes

**Example Table:**

| Class | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 |
|-------|----|----|----|----|----|----|----|----|-----|
| Anomalous | 0 | 0 | 20 | 10 | 20 | 0 | 0 | 0 | 0 |
| Normal | 150 | 100 | 0 | 0 | 0 | 100 | 100 | 150 | 100 |

Bins:     |--bin1--|--bin2--|-------bin3-------|

### THE INTERVAL SIZE PROBLEM

Size of discretized intervals affects both SUPPORT and CONFIDENCE:

**If intervals TOO SMALL:**
- May not have ENOUGH SUPPORT
- Example: {Refund = No, Income = $51,250} -> {Cheat = No}
  - Very specific, few transactions match

**If intervals TOO LARGE:**
- May not have ENOUGH CONFIDENCE
- Example: {Refund = No, 0K <= Income <= 1B} -> {Cheat = No}
  - Too general, loses predictive power

**POTENTIAL SOLUTION: Use ALL POSSIBLE INTERVALS**
- {Refund = No, Income = $51,250} -> {Cheat = No}
- {Refund = No, 60K <= Income <= 80K} -> {Cheat = No}
- {Refund = No, 0K <= Income <= 1B} -> {Cheat = No}

### PROBLEMS WITH ALL POSSIBLE INTERVALS

**1. EXECUTION TIME**
   - If intervals contain n values, there are O(n^2) possible ranges
   - Computationally expensive

**2. TOO MANY RULES**
   - Many redundant rules generated
   - Example: Multiple rules for same pattern with different granularity

### APPROACH BY SRIKANT & AGRAWAL

**Step 1: PREPROCESS THE DATA**
- Discretize attribute using equi-depth partitioning
- Use "partial completeness measure" to determine number of partitions
- Merge adjacent intervals as long as support is less than max-support

**Step 2: APPLY EXISTING ASSOCIATION RULE MINING**
- Run Apriori or other algorithm on preprocessed data

**Step 3: DETERMINE INTERESTING RULES**
- Filter output rules based on interest measures

---

### 3.2 STATISTICS-BASED METHODS

### KEY IDEA
Rule consequent consists of a CONTINUOUS VARIABLE, characterized by their
STATISTICS (mean, median, standard deviation, etc.)

**Example Rule:**
Browser=Mozilla AND Buy=Yes -> Age: mean=23

This tells us: Among Mozilla users who buy, the average age is 23

### APPROACH

**Step 1:** Withhold the TARGET VARIABLE from the rest of the data
- Separate the continuous variable you want to predict

**Step 2:** Apply EXISTING FREQUENT ITEMSET GENERATION on rest of data
- Find frequent itemsets without the target variable

**Step 3:** For each frequent itemset, COMPUTE DESCRIPTIVE STATISTICS
- Calculate mean, median, std for target variable
- The frequent itemset becomes a rule by introducing target as consequent

**Step 4:** Apply STATISTICAL TEST to determine interestingness
- Use hypothesis testing to verify significance
- Example: T-test to compare mean with population mean

### EXAMPLE
If {Browser=Mozilla, Buy=Yes} is a frequent itemset with support 100:
- Look at the 100 transactions matching this pattern
- Calculate: mean(Age) = 23, std(Age) = 5
- Rule: {Browser=Mozilla, Buy=Yes} -> Age: mean=23, std=5

This approach preserves the continuous nature of the target variable!

---

### 3.3 NON-DISCRETIZATION BASED: MIN-APRIORI

### THE PROBLEM
Data contains only continuous attributes of the SAME TYPE.
Example: Document-term matrix (frequency of words in documents)

**Document-Term Matrix:**

| TID | W1 | W2 | W3 | W4 | W5 |
|-----|----|----|----|----|-----|
| D1 | 2 | 2 | 0 | 0 | 1 |
| D2 | 0 | 0 | 1 | 2 | 2 |
| D3 | 2 | 3 | 0 | 0 | 0 |
| D4 | 0 | 0 | 1 | 0 | 1 |
| D5 | 1 | 1 | 1 | 0 | 2 |

Goal: Find associations among words (e.g., W1 and W2 tend to appear together)

### WHY CAN'T WE USE STANDARD APPROACHES?

**Option 1: Convert to 0/1 matrix**
- Problem: LOSES word frequency information
- W1 appearing 10 times vs 1 time are treated the same

**Option 2: Discretization**
- Problem: Users want association among WORDS, not ranges of words
- Doesn't make sense to have "W1 in [1,3]" as an item

### SOLUTION: MIN-APRIORI

**KEY INSIGHT:** How to determine the support of a word?

Problem: If we simply SUM UP frequency, support count will be GREATER than
         total number of documents!

**Solution:** NORMALIZE the word vectors using L1 norm

### L1 NORMALIZATION
For each document (row), divide each value by the row sum.

**Original Matrix → Normalized Matrix:**

| TID | W1 | W2 | W3 | W4 | W5 | | TID | W1 | W2 | W3 | W4 | W5 |
|-----|----|----|----|----|----| |-----|------|------|------|------|------|
| D1 | 2 | 2 | 0 | 0 | 1 | → | D1 | 0.40 | 0.33 | 0.00 | 0.00 | 0.17 |
| D2 | 0 | 0 | 1 | 2 | 2 | → | D2 | 0.00 | 0.00 | 0.33 | 1.00 | 0.33 |
| D3 | 2 | 3 | 0 | 0 | 0 | → | D3 | 0.40 | 0.50 | 0.00 | 0.00 | 0.00 |
| D4 | 0 | 0 | 1 | 0 | 1 | → | D4 | 0.00 | 0.00 | 0.33 | 0.00 | 0.17 |
| D5 | 1 | 1 | 1 | 0 | 2 | → | D5 | 0.20 | 0.17 | 0.33 | 0.00 | 0.33 |

After normalization: Each word has a support that equals to 1.0
(Column sums to 1.0)

### NEW DEFINITION OF SUPPORT

For an itemset C (collection of words):
```
sup(C) = SUM over all documents i [ MIN over all words j in C { D(i,j) } ]
```

Where D(i,j) is the normalized value for document i and word j.

### EXAMPLE CALCULATION

Using normalized matrix above:

```
Sup(W1,W2,W3) = min(0.40,0.33,0.00) + min(0.00,0.00,0.33) +
                min(0.40,0.50,0.00) + min(0.00,0.00,0.33) +
                min(0.20,0.17,0.33)
             = 0 + 0 + 0 + 0 + 0.17
             = 0.17
```

### ANTI-MONOTONE PROPERTY VERIFICATION

Example calculations:
```
Sup(W1) = 0.4 + 0 + 0.4 + 0 + 0.2 = 1.0
Sup(W1, W2) = 0.33 + 0 + 0.4 + 0 + 0.17 = 0.9
Sup(W1, W2, W3) = 0 + 0 + 0 + 0 + 0.17 = 0.17
```

**PROPERTY HOLDS:** Sup(W1) >= Sup(W1,W2) >= Sup(W1,W2,W3)

The anti-monotone property is preserved because MIN operation can only
decrease or stay the same when adding more items!


---

## 4. MULTI-LEVEL ASSOCIATION RULES

### CONCEPT HIERARCHY
Items can be organized in a hierarchy (taxonomy):

```
                    Food                          Electronics
                   /    \                         /         \
               Bread    Milk                 Computers     Home
              /    \    /   \                /    |    \   /    \
          Wheat  White Skim  2%         Desktop Laptop Accessory TV DVD
                           /   \                         /    \
                      Foremost Kemps              Printer  Scanner
```

### WHY INCORPORATE CONCEPT HIERARCHY?

1. We need levels of patterns/knowledge for effective decision making
   - "People who buy food also buy electronics" (too general)
   - "People who buy milk also buy bread" (useful)
   - "People who buy Foremost skim milk also buy white bread" (specific)

2. Rules at lower levels may not have enough support
   - Very specific items may be infrequent individually

3. Rules at lower levels of hierarchy are overly specific
   - skim milk -> white bread
   - 2% milk -> wheat bread
   - skim milk -> wheat bread
   These are all indicative of association between milk and bread!

### SUPPORT AND CONFIDENCE IN CONCEPT HIERARCHY

**KEY PROPERTIES:**

**Property 1:** Parent support >= sum of children supports
- If X is parent of X1 and X2: s(X) <= s(X1) + s(X2)
- The inequality because a transaction containing both X1 and X2
  is only counted once for X

**Property 2:** Support propagates upward
- If s(X1 U Y1) >= minsup, and X is parent of X1, Y is parent of Y1
- Then: s(X U Y1) >= minsup
        s(X1 U Y) >= minsup
        s(X U Y) >= minsup

**Property 3:** Confidence propagates to parent consequent
- If conf(X1 => Y1) >= minconf
- Then conf(X1 => Y) >= minconf
  (moving consequent to higher level increases confidence)

### APPROACH 1: TRANSACTION AUGMENTATION

**Method:** Extend current association rule formulation by AUGMENTING each
transaction with higher level items.

Example:
- Original Transaction: {skim milk, wheat bread}
- Augmented Transaction: {skim milk, wheat bread, milk, bread, food}

**Issues with this approach:**
1. Items at higher levels have MUCH HIGHER support counts
   - If support threshold is low, too many frequent patterns from high levels
2. INCREASED DIMENSIONALITY of the data
   - More items to consider

### APPROACH 2: LEVEL-WISE MINING

**Method:**
1. Generate frequent patterns at HIGHEST level first
2. Then generate frequent patterns at NEXT highest level, and so on
3. Use discovered patterns to guide search at lower levels

**Issues with this approach:**
1. I/O requirements INCREASE dramatically
   - Need to perform more passes over the data
2. May MISS some potentially interesting CROSS-LEVEL association patterns
   - Patterns involving items from different levels


---

## 5. SEQUENTIAL PATTERN MINING

### SEQUENCE DATA DEFINITION

A sequence is an ORDERED LIST of ELEMENTS (transactions):
```
s = < e1 e2 e3 ... >
```

Each element contains a collection of EVENTS (items):
```
ei = {i1, i2, ..., ik}
```

Each element is attributed to a specific TIME or LOCATION.

### EXAMPLE SEQUENCE DATABASE

| Object | Timestamp | Events |
|--------|-----------|--------|
| A | 10 | 2, 3, 5 |
| A | 20 | 6, 1 |
| A | 23 | 1 |
| B | 11 | 4, 5, 6 |
| B | 17 | 2 |
| B | 21 | 7, 8, 1, 2 |
| B | 28 | 1, 6 |
| C | 14 | 1, 8, 7 |

Timeline visualization:
```
Object A: |---{2,3,5}------|----{6,1}--|--{1}--|
Object B: |--{4,5,6}--|--{2}--|--{7,8,1,2}--|--{1,6}--|
Object C: |------{1,8,7}------|
```

### EXAMPLES OF SEQUENCE DATA

| Sequence Database | Sequence | Element | Event |
|-------------------|----------|---------|-------|
| Customer | Purchase history | Items bought at t | Books, CDs |
| Web Data | Browsing activity | Files viewed | Home page |
| Event data | Sensor event history | Events at time t | Alarm types |
| Genome sequences | DNA sequence | DNA element | Bases A,T,G,C |

### KEY TERMINOLOGY

**LENGTH of a sequence |s|:** Number of ELEMENTS in the sequence

**k-SEQUENCE:** A sequence that contains k EVENTS (items total, not elements)

Example:
```
s = <{a b} {c d e} {f} {g h i}>
```
- Has 4 elements
- Has 9 items (k=9 for this 9-sequence)

### EXAMPLES OF SEQUENCES

**Web sequence:**
```
< {Homepage} {Electronics} {Digital Cameras} {Canon Digital Camera}
  {Shopping Cart} {Order Confirmation} {Return to Shopping} >
```

**Library book sequence:**
```
< {Fellowship of the Ring} {The Two Towers} {Return of the King} >
```

### SUBSEQUENCE DEFINITION

A sequence <a1 a2 ... an> is CONTAINED IN another sequence <b1 b2 ... bm>
(where m >= n) if there exist integers i1 < i2 < ... < in such that:
- a1 is subset of b_i1
- a2 is subset of b_i2
- ...
- an is subset of b_in

### CONTAINMENT EXAMPLES

| Data sequence | Subsequence | Contain? |
|---------------|-------------|----------|
| <{2,4} {3,5,6} {8}> | <{2} {3,5}> | YES |
| <{1,2} {3,4}> | <{1} {2}> | NO (1 and 2 must be in same element or 1 before 2) |
| <{2,4} {2,4} {2,5}> | <{2} {4}> | YES |

### SUPPORT AND FREQUENT SUBSEQUENCE

**SUPPORT** of a subsequence w: Fraction of data sequences that CONTAIN w

**SEQUENTIAL PATTERN:** A frequent subsequence
                   (a subsequence whose support >= minsup)

### SEQUENTIAL PATTERN MINING TASK

**Given:**
- A database of sequences
- A user-specified minimum support threshold (minsup)

**Task:**
- Find ALL subsequences with support >= minsup

### THE COMPUTATIONAL CHALLENGE

Given a sequence: <{a b} {c d e} {f} {g h i}>

Examples of subsequences:
- <{a} {c d} {f} {g}>
- <{c d e}>
- <{b} {g}>

How many k-subsequences can be extracted from a given n-sequence?

For n=9, k=4:
```
<{a b} {c d e} {f} {g h i}>
  |  |   | | |  |   | | |
  v  v   v v v  v   v v v
  Y  _   _ Y Y  _   _ _ Y  -> <{a} {d e} {i}>
```

Answer: C(n,k) = C(9,4) = 126 possible 4-subsequences!

### SEQUENTIAL PATTERN MINING EXAMPLE

**Sequence Database:**

| Object | Timestamp | Events |
|--------|-----------|--------|
| A | 1 | 1,2,4 |
| A | 2 | 2,3 |
| A | 3 | 5 |
| B | 1 | 1,2 |
| B | 2 | 2,3,4 |
| C | 1 | 1, 2 |
| C | 2 | 2,3,4 |
| C | 3 | 2,4,5 |
| D | 1 | 2 |
| D | 2 | 3, 4 |
| D | 3 | 4, 5 |
| E | 1 | 1, 3 |
| E | 2 | 2, 4, 5 |

**With Minsup = 50%**

Examples of Frequent Subsequences:
```
< {1,2} >       s=60%
< {2,3} >       s=60%
< {2,4} >       s=80%
< {3} {5} >     s=80%
< {1} {2} >     s=80%
< {2} {2} >     s=60%
< {1} {2,3} >   s=60%
< {2} {2,3} >   s=60%
< {1,2} {2,3} > s=60%
```

### GENERALIZED SEQUENTIAL PATTERN (GSP) ALGORITHM

**Step 1:** First pass
- Scan sequence database D to find all 1-element frequent sequences

**Step 2:** Repeat until no new frequent sequences found:

   a. **CANDIDATE GENERATION:**
      - Merge pairs of frequent subsequences from (k-1)th pass
      - Generate candidate sequences containing k items

   b. **CANDIDATE PRUNING:**
      - Prune candidate k-sequences containing infrequent (k-1)-subsequences

   c. **SUPPORT COUNTING:**
      - New pass over database D to find support for candidates

   d. **CANDIDATE ELIMINATION:**
      - Eliminate k-sequences whose actual support < minsup

### CANDIDATE GENERATION IN GSP

**BASE CASE (k=2):**
Merging two frequent 1-sequences <{i1}> and <{i2}> produces:
- <{i1} {i2}> (i1 occurs before i2)
- <{i1 i2}>   (i1 and i2 occur together)

**GENERAL CASE (k>2):**
A frequent (k-1)-sequence w1 is merged with w2 to produce a candidate
k-sequence IF:
- Subsequence obtained by removing FIRST event in w1 EQUALS
- Subsequence obtained by removing LAST event in w2

Example: W1 = (A,B,C,D), W2 = (B,C,D,E) -> Candidate = (A,B,C,D,E)

**RULES FOR EXTENDING w1 WITH LAST EVENT OF w2:**
- If last two events in w2 belong to SAME element:
  -> Last event in w2 becomes part of last element in w1
- Otherwise:
  -> Last event in w2 becomes a SEPARATE element appended to w1

### CANDIDATE GENERATION EXAMPLES

**Example 1:**
```
w1 = <{1} {2 3} {4}>  and  w2 = <{2 3} {4 5}>
- Remove first from w1: <{2 3} {4}>
- Remove last from w2: <{2 3} {4}>
- They match!
- Last two events in w2 (4 and 5) are in SAME element
- Result: <{1} {2 3} {4 5}>
```

**Example 2:**
```
w1 = <{1} {2 3} {4}>  and  w2 = <{2 3} {4} {5}>
- Remove first from w1: <{2 3} {4}>
- Remove last from w2: <{2 3} {4}>
- They match!
- Last two events in w2 (4 and 5) in DIFFERENT elements
- Result: <{1} {2 3} {4} {5}>
```

**Example 3 (NO MERGE):**
```
w1 = <{1} {2 6} {4}>  and  w2 = <{1} {2} {4 5}>
- Remove first from w1: <{2 6} {4}>
- Remove last from w2: <{1} {2} {4}>
- They DON'T match (different subsequences)
- Don't merge these
```

### GSP EXAMPLE WALKTHROUGH

**Frequent 3-sequences:**
- <{1} {2} {3}>
- <{1} {2 5}>
- <{1} {5} {3}>
- <{2} {3} {4}>
- <{2 5} {3}>
- <{3} {4} {5}>
- <{5} {3 4}>

**After Candidate Generation:**
- <{1} {2} {3} {4}>
- <{1} {2 5} {3}>
- <{1} {5} {3 4}>
- <{2} {3} {4} {5}>
- <{2 5} {3 4}>

**After Candidate Pruning (check all 3-subsequences):**
- <{1} {2 5} {3}> survives (all 3-subsequences are frequent)
- Others may be pruned if any 3-subsequence is infrequent

### TIMING CONSTRAINTS IN GSP

**Additional parameters:**
- **xg:** MAX-GAP (maximum time between consecutive elements)
- **ng:** MIN-GAP (minimum time between consecutive elements)
- **ms:** MAXIMUM SPAN (maximum time from first to last element)
- **ws:** WINDOW SIZE (elements within window can be merged)

```
Pattern: {A B} {C} {D E}
         |<--xg-->|<-ng->|
         |<------ms------>|
```

### CONTAINMENT WITH TIMING CONSTRAINTS

xg = 2, ng = 0, ms = 4

| Data sequence | Subsequence | Contain? |
|---------------|-------------|----------|
| <{2,4} {3,5,6} {4,7} {4,5} {8}> | <{6} {5}> | YES |
| <{1} {2} {3} {4} {5}> | <{1} {4}> | NO (gap > 2) |
| <{1} {2,3} {3,4} {4,5}> | <{2} {3} {5}> | YES |
| <{1,2} {3} {2,3} {3,4} {2,4} {4,5}> | <{1,2} {5}> | NO (span > 4) |

### APRIORI PRINCIPLE WITH TIMING CONSTRAINTS

**PROBLEM:** Anti-monotone property may NOT hold with max-gap constraint!

Example with xg=1 (max-gap), ng=0 (min-gap), ms=5 (max span), minsup=60%:

<{2} {5}> has support = 40%
BUT
<{2} {3} {5}> has support = 60%!

Why? The intermediate element {3} helps satisfy the max-gap constraint!

### SOLUTION: MODIFIED CANDIDATE PRUNING

**Without maxgap constraint:**
- Prune if ANY (k-1)-subsequence is infrequent

**With maxgap constraint:**
- Prune only if a CONTIGUOUS (k-1)-subsequence is infrequent

### CONTIGUOUS SUBSEQUENCE DEFINITION

s is a contiguous subsequence of w = <e1><e2>...<ek> if any of these hold:

1. s is obtained from w by deleting an item from either e1 or ek
2. s is obtained from w by deleting an item from any element ei
   that contains more than 2 items
3. s is a contiguous subsequence of s' and s' is a contiguous
   subsequence of w (recursive)

Examples for s = <{1} {2}>:
- IS contiguous subsequence of: <{1} {2 3}>, <{1 2} {2} {3}>, <{3 4} {1 2} {2 3} {4}>
- IS NOT contiguous subsequence of: <{1} {3} {2}>, <{2} {1} {3} {2}>

### WINDOW SIZE CONSTRAINT

xg = 2, ng = 0, ws = 1, ms = 5

Window size allows events from adjacent timestamps to be grouped!

| Data sequence | Subsequence | Contain? |
|---------------|-------------|----------|
| <{2,4} {3,5,6} {4,7} {4,6} {8}> | <{3} {5}> | NO |
| <{1} {2} {3} {4} {5}> | <{1,2} {3}> | YES (ws allows grouping) |
| <{1,2} {2,3} {3,4} {4,5}> | <{1,2} {3,4}> | YES |

### MODIFIED SUPPORT COUNTING

Given candidate pattern: <{a, c}>

Data sequences that contribute to support:
- <... {a c} ...>              (a and c in same element)
- <... {a} ... {c} ...>        where time({c}) - time({a}) <= ws
- <... {c} ... {a} ...>        where time({a}) - time({c}) <= ws

Window size allows temporal flexibility in matching!

### FREQUENT EPISODE MINING

Alternative formulation when there's only ONE very long time series:

**Applications:**
- Monitoring network traffic events for attacks
- Monitoring telecommunication alarm signals

**Goal:** Find frequent sequences of events in the time series

Timeline:
```
E1 E3    E1      E1 E2 E4    E1 E2      E1
E2 E4    E2 E3 E4  E2 E3 E5    E2 E3 E5    E2 E3 E1
|--------|--------|----------|----------|----------|
```

Pattern: <E1> <E3> - appears multiple times in the timeline


---

## 6. FREQUENT SUBGRAPH MINING (OPTIONAL)

### MOTIVATION
Extend association rule mining to finding frequent SUBGRAPHS.

**Applications:**
- Web Mining (link structure analysis)
- Computational chemistry (molecular structures)
- Bioinformatics (protein structures)
- Spatial data sets (geographic patterns)

### GRAPH DEFINITIONS

**LABELED GRAPH:** Graph with labels on vertices and/or edges

**SUBGRAPH:** A graph G' is a subgraph of G if:
- Vertices of G' are subset of vertices of G
- Edges of G' are subset of edges of G

**INDUCED SUBGRAPH:** A subgraph where ALL edges between the selected
vertices in original graph are included

### REPRESENTING TRANSACTIONS AS GRAPHS

Each transaction is a CLIQUE of items (all items connected to each other)

**Transaction:**

| TID | Items |
|-----|-------|
| 1 | {A,B,C,D} |
| 2 | {A,B,E} |
| 3 | {B,C} |
| 4 | {A,B,D,E} |
| 5 | {B,C,D} |

TID=1 as graph:
```
    A
   /|\
  B-+-C
   \|/
    D
```

### REPRESENTING GRAPHS AS TRANSACTIONS

Each edge becomes an "item" in the form (vertex1, vertex2, edge_label)

Example: 4 graphs G1, G2, G3, G4 can be converted to:

| | (a,b,p) | (a,b,q) | (a,b,r) | (b,c,p) | ... | (d,e,r) |
|---|---------|---------|---------|---------|-----|---------|
| G1 | 1 | 0 | 0 | 0 | ... | 0 |
| G2 | 1 | 0 | 0 | 0 | ... | 0 |
| G3 | 0 | 0 | 1 | 1 | ... | 0 |
| G4 | 0 | 0 | 0 | 0 | ... | 0 |

### CHALLENGES IN FREQUENT SUBGRAPH MINING

1. Nodes may contain DUPLICATE LABELS
   - Same label on multiple nodes causes complexity

2. SUPPORT and CONFIDENCE definition
   - Support: Number of graphs containing a particular subgraph

3. Additional constraints imposed by pattern structure
   - Assumption: Frequent subgraphs must be CONNECTED

4. Apriori-like approach considerations:
   - What is k?
   - VERTEX GROWING: k = number of vertices
   - EDGE GROWING: k = number of edges

### APRIORI PRINCIPLE FOR GRAPHS

Support is ANTI-MONOTONE for subgraphs:
- If a subgraph is infrequent, all its supergraphs are infrequent

**Level-wise approach:**
1. Find frequent 1-subgraphs
2. Generate candidate (k+1)-subgraphs from frequent k-subgraphs
3. Prune candidates containing infrequent k-subgraphs
4. Count support, eliminate infrequent candidates
5. Repeat

### VERTEX GROWING

Join two frequent (k-1)-subgraphs that share (k-2) vertices.

Example:
G1 (4 vertices): a-b-a (triangle with e connected by q)
                  \e/
                   p

G2 (4 vertices): a-a-d (path with triangular structure)
                  \/
                   a

Result G3 (5 vertices): Join where 3 vertices overlap

Adjacency matrix grows by one row and column.

### EDGE GROWING

Join two frequent (k-1)-subgraphs that share (k-2) edges.

Example:
G1 (4 edges): Triangle plus one edge
G2 (4 edges): Different configuration with 3 shared edges
Result G3 (5 edges): Combined structure

### MULTIPLICITY OF CANDIDATES

**PROBLEM:** Merging two k-subgraphs may produce MORE THAN ONE candidate!

**Case 1: IDENTICAL VERTEX LABELS**
- Same label on different positions creates multiple valid joins

**Case 2: CORE CONTAINS IDENTICAL LABELS**
- The (k-1) subgraph common between joined graphs has repeated labels
- Multiple ways to align the structures

**Case 3: CORE MULTIPLICITY**
- Multiple ways to identify the common (k-1) subgraph

### GRAPH ISOMORPHISM PROBLEM

The SAME graph can be represented by MANY different adjacency matrices!

Example: An 8-vertex graph can have vertices numbered 1-8 in any order,
producing different but equivalent matrices.

### WHEN IS ISOMORPHISM TEST NEEDED?

1. During **CANDIDATE GENERATION:**
   - Check if a candidate has already been generated

2. During **CANDIDATE PRUNING:**
   - Check if (k-1)-subgraphs are frequent

3. During **CANDIDATE COUNTING:**
   - Check if candidate is contained within another graph

### CANONICAL LABELING SOLUTION

Map each graph to an ORDERED STRING representation (code) such that
two ISOMORPHIC graphs map to the SAME canonical encoding.

Example: Lexicographically largest adjacency matrix

```
Matrix:         Canonical Matrix:
[0 0 1 0]       [0 1 1 1]
[0 0 1 1]  ->   [1 0 1 0]
[1 1 0 1]       [1 1 0 0]
[0 1 1 0]       [1 0 0 0]
```

String: 0010001111010110  ->  Canonical: 0111101011001000

Two isomorphic graphs will produce the SAME canonical string!


---

## 7. EXAMPLES AND PROBLEMS

### PROBLEM 1: Min-Apriori Support Calculation

**Given normalized document-term matrix:**

| TID | W1 | W2 | W3 | W4 | W5 |
|-----|------|------|------|------|------|
| D1 | 0.40 | 0.33 | 0.00 | 0.00 | 0.17 |
| D2 | 0.00 | 0.00 | 0.33 | 1.00 | 0.33 |
| D3 | 0.40 | 0.50 | 0.00 | 0.00 | 0.00 |
| D4 | 0.00 | 0.00 | 0.33 | 0.00 | 0.17 |
| D5 | 0.20 | 0.17 | 0.33 | 0.00 | 0.33 |

Calculate: Sup(W1), Sup(W1,W2), Sup(W1,W2,W3)

**Solution:**
```
Sup(W1) = 0.40 + 0.00 + 0.40 + 0.00 + 0.20 = 1.0
Sup(W1,W2) = min(0.40,0.33) + min(0.00,0.00) + min(0.40,0.50) +
             min(0.00,0.00) + min(0.20,0.17)
           = 0.33 + 0 + 0.40 + 0 + 0.17 = 0.90
Sup(W1,W2,W3) = min(0.40,0.33,0.00) + min(0.00,0.00,0.33) +
                min(0.40,0.50,0.00) + min(0.00,0.00,0.33) +
                min(0.20,0.17,0.33)
              = 0 + 0 + 0 + 0 + 0.17 = 0.17
```

Verify anti-monotone: 1.0 >= 0.90 >= 0.17 (YES!)

---

### PROBLEM 2: Sequential Pattern Containment

Determine if subsequence is contained in data sequence:

**a) Data: <{2,4} {3,5,6} {8}>  Subsequence: <{2} {3,5}>**
   - {2} subset of {2,4}? YES
   - {3,5} subset of {3,5,6}? YES
   - Order preserved? YES
   **Answer: CONTAINED**

**b) Data: <{1,2} {3,4}>  Subsequence: <{1} {2}>**
   - Need {1} in some element and {2} in LATER element
   - {1} and {2} are in SAME element {1,2}
   **Answer: NOT CONTAINED**

**c) Data: <{2,4} {2,4} {2,5}>  Subsequence: <{2} {4}>**
   - {2} subset of {2,4}? YES (first element)
   - {4} subset of {2,4}? YES (second element, comes after)
   **Answer: CONTAINED**

---

### PROBLEM 3: GSP Candidate Generation

**Given frequent 3-sequences:**
<{1} {2} {3}>, <{1} {2 5}>, <{1} {5} {3}>, <{2} {3} {4}>,
<{2 5} {3}>, <{3} {4} {5}>, <{5} {3 4}>

Which pairs can be merged? Show resulting candidates.

**Solution:**

Merge <{1} {2} {3}> and <{2} {3} {4}>:
- Remove first from first: <{2} {3}>
- Remove last from second: <{2} {3}>
- Match! Result: <{1} {2} {3} {4}>

Merge <{1} {2 5}> and <{2 5} {3}>:
- Remove first from first: <{2 5}>
- Remove last from second: <{2 5}>
- Match! Result: <{1} {2 5} {3}>

Merge <{1} {5} {3}> and <{5} {3 4}>:
- Remove first from first: <{5} {3}>
- Remove last from second: <{5} {3}>
- Match! Last two events in second (3,4) in same element
- Result: <{1} {5} {3 4}>

---

### PROBLEM 4: Timing Constraints

Given xg = 2, ng = 0, ms = 4

**Does <{1} {2} {3} {4} {5}> contain <{1} {4}>?**

**Solution:**
- Match {1} at position 1 (time 1)
- Match {4} at position 4 (time 4)
- Gap between consecutive elements: positions 1 to 4 = gap of 3
- Max-gap constraint: xg = 2
- Gap 3 > 2
**Answer: NO, violates max-gap constraint**

**Does <{1} {2,3} {3,4} {4,5}> contain <{2} {3} {5}>?**

**Solution:**
- Match {2} in {2,3} at position 2
- Match {3} in {3,4} at position 3
- Match {5} in {4,5} at position 4
- Gaps: 2->3 = 1, 3->4 = 1 (both <= xg=2)
- Span: 4-2 = 2 (<= ms=4)
**Answer: YES**

---

### PROBLEM 5: Multi-level Association Rules

**Given hierarchy:**
```
         Dairy
        /     \
     Milk    Cheese
    /    \
  Skim   2%
```

**Transactions:**
1: {Skim milk, Bread}
2: {2% milk, Eggs}
3: {Cheese, Bread}
4: {Skim milk, Cheese}
5: {2% milk, Bread}

With minsup = 40%

**a) What is support of {Milk}?**
   Milk appears when Skim or 2% appears
   TID 1 (Skim), TID 2 (2%), TID 4 (Skim), TID 5 (2%)
   **Support = 4/5 = 80%**

**b) What is support of {Dairy}?**
   Dairy appears when Milk or Cheese appears
   TID 1 (Skim->Milk->Dairy), TID 2 (2%->Milk->Dairy),
   TID 3 (Cheese->Dairy), TID 4 (both), TID 5 (2%->Milk->Dairy)
   **Support = 5/5 = 100%**

**c) What is support of {Skim milk}?**
   TID 1, TID 4
   **Support = 2/5 = 40%**

**d) Why might {Skim milk, Bread} not be frequent but {Milk, Bread} is frequent?**
   {Skim milk, Bread}: TID 1 only = 20% (not frequent)
   {Milk, Bread}: TID 1, TID 5 = 40% (frequent!)

   Lower level rules may lack support that higher level rules have!


---

## 8. SUMMARY

### KEY CONCEPTS

**1. CATEGORICAL ATTRIBUTES:**
   - Transform to asymmetric binary variables
   - Create new item for each attribute-value pair
   - Watch for: many values (aggregate), skewed distribution (drop frequent)

**2. CONTINUOUS ATTRIBUTES - THREE APPROACHES:**

   **A. Discretization-based:**
      - Equal-width, equal-depth, or supervised binning
      - Trade-off between support (small intervals) and confidence (large)
      - Srikant & Agrawal: equi-depth + merge + mine + filter

   **B. Statistics-based:**
      - Consequent is continuous, characterized by statistics
      - Rule: antecedent -> target: mean=X, std=Y
      - Preserves continuous nature of target

   **C. Min-Apriori:**
      - For same-type continuous attributes (e.g., word frequencies)
      - Normalize using L1 norm
      - Support = SUM of MIN values
      - Anti-monotone property preserved

**3. MULTI-LEVEL ASSOCIATION RULES:**
   - Concept hierarchy organizes items
   - Higher levels have higher support
   - Approach 1: Augment transactions with ancestor items
   - Approach 2: Level-wise mining (top-down)
   - Cross-level patterns may be missed

**4. SEQUENTIAL PATTERN MINING:**
   - Sequence = ordered list of elements (sets of events)
   - Subsequence containment considers order
   - GSP Algorithm: level-wise, similar to Apriori
   - Timing constraints: max-gap, min-gap, max-span, window-size
   - Contiguous subsequences for pruning with max-gap

**5. FREQUENT SUBGRAPH MINING:**
   - Extend itemset mining to graph structures
   - Vertex growing vs edge growing
   - Graph isomorphism is key challenge
   - Canonical labeling for isomorphism detection
   - Support still anti-monotone

### FORMULAS TO REMEMBER

**Min-Apriori Support:**
```
sup(C) = SUM_i [ MIN_j in C { D(i,j) } ]
```

**Sequence Containment:**
<a1...an> contained in <b1...bm> if exists i1<i2<...<in where ak subset of b_ik

**Number of k-subsequences from n-sequence:** C(n,k)

### ALGORITHM COMPARISON

| Problem Type | Algorithm/Approach |
|--------------|-------------------|
| Categorical attr | Binary transformation |
| Continuous attr | Discretization / Statistics / Min-Apriori |
| Hierarchical items | Multi-level Apriori |
| Temporal sequences | GSP with timing constraints |
| Graph patterns | Vertex/Edge growing with isomorphism check |

### COMMON PITFALLS

1. Not handling skewed distributions in categorical attributes
2. Choosing wrong interval size for discretization
3. Forgetting that max-gap breaks anti-monotone property
4. Confusing subsequence containment (order matters!)
5. Not checking graph isomorphism (duplicate candidates)

---

## QUICK REVIEW CHECKLIST

### Categorical Attributes
- [ ] Do I know how to transform categorical attributes to binary items?
- [ ] Can I handle many-valued attributes (aggregate low-support values)?
- [ ] Can I handle skewed distributions (drop highly frequent items)?

### Continuous Attributes
- [ ] Can I explain discretization-based approach (equal-width, equal-depth)?
- [ ] Do I understand the trade-off between support and confidence in interval size?
- [ ] Can I explain statistics-based approach (mean, std in consequent)?
- [ ] Can I calculate Min-Apriori support: sup(C) = Σ min values?
- [ ] Do I know to normalize using L1 norm for Min-Apriori?

### Multi-level Association Rules
- [ ] Do I understand concept hierarchy (Skim milk → Milk → Dairy)?
- [ ] Can I explain why higher levels have higher support?
- [ ] Do I know the two approaches (augment transactions vs level-wise)?

### Sequential Pattern Mining
- [ ] Can I define a sequence: ordered list of elements (itemsets)?
- [ ] Can I determine if subsequence is contained in a sequence?
- [ ] Do I understand GSP algorithm structure (like Apriori)?
- [ ] Can I perform GSP candidate generation (merge pairs)?
- [ ] Do I know when to merge: remove first from w1 = remove last from w2?

### Timing Constraints
- [ ] Do I understand max-gap (xg), min-gap (ng), max-span (ms), window-size (ws)?
- [ ] Can I check containment with timing constraints?
- [ ] Do I know max-gap BREAKS anti-monotone property?
- [ ] Do I understand contiguous subsequence for pruning with max-gap?

### Frequent Subgraph Mining (if covered)
- [ ] Do I know vertex growing vs edge growing approaches?
- [ ] Do I understand graph isomorphism problem?
- [ ] Can I explain canonical labeling solution?

### Problem Solving
- [ ] Can I calculate Min-Apriori support from a normalized table?
- [ ] Can I determine subsequence containment?
- [ ] Can I perform GSP candidate generation and pruning?
- [ ] Can I check timing constraints (max-gap, max-span)?

---

*END OF NOTES*
