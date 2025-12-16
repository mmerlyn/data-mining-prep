# TOPIC 5A: ALTERNATIVE CLASSIFICATION TECHNIQUES
## (Rule-Based Methods & Nearest Neighbor)
### CS653 - Data Mining (SDSU)
### Instructor: Dr. Xiaobai Liu

---

## TABLE OF CONTENTS
1. Overview of Alternative Classification Techniques
2. Rule-Based Classifier - Introduction
3. Rule Coverage and Accuracy (WITH EXAMPLES)
4. How Rule-Based Classifiers Work
5. Characteristics of Rule-Based Classifiers
6. Converting Decision Trees to Rules
7. Ordered Rule Sets (Decision Lists)
8. Rule Ordering Schemes
9. Building Classification Rules - Direct Method
10. Building Classification Rules - Indirect Method
11. C4.5rules Method
12. Advantages of Rule-Based Classifiers
13. Instance-Based Classifiers
14. K-Nearest Neighbor (k-NN) Classifier
15. Distance Metrics
16. Choosing the Value of k
17. Scaling Issues in k-NN
18. PEBLS Algorithm (WITH WORKED EXAMPLES)
19. Quiz Questions & Worked Examples


---

## 1. OVERVIEW OF ALTERNATIVE CLASSIFICATION TECHNIQUES

### CLASSIFICATION TECHNIQUES COVERED IN THIS LECTURE:
1. Rule-based method          <-- Covered in Part A
2. Nearest Neighbor (k-NN)    <-- Covered in Part A
3. Support Vector Machine     <-- Covered in Part B
4. Neural Network             <-- Covered in Part B
5. Naive Bayes Classifier     <-- Covered in Part B
6. Boosting                   <-- Covered in Part B


---

## 2. RULE-BASED CLASSIFIER - INTRODUCTION

### DEFINITION:
Classify records by using a collection of "if...then..." rules

### RULE FORMAT:
```
    (Condition) --> y
```

Where:
- Condition (LEFT-HAND SIDE / ANTECEDENT): Conjunction of attributes
- y (RIGHT-HAND SIDE / CONSEQUENT): Class label


### EXAMPLES OF CLASSIFICATION RULES:
Rule 1: (Blood Type=Warm) AND (Lay Eggs=Yes) --> Birds
Rule 2: (Taxable Income < 50K) AND (Refund=Yes) --> Evade=No


---

### COMPLETE EXAMPLE: Animal Classification

**RULE SET:**

| Rule | Condition | Class |
|------|-----------|-------|
| R1 | (Give Birth = no) AND (Can Fly = yes) | Birds |
| R2 | (Give Birth = no) AND (Live in Water = yes) | Fishes |
| R3 | (Give Birth = yes) AND (Blood Type = warm) | Mammals |
| R4 | (Give Birth = no) AND (Can Fly = no) | Reptiles |
| R5 | (Live in Water = sometimes) | Amphibians |

**TRAINING DATA:**

| Name | Blood Type | Give Birth | Can Fly | Live in Water | Class |
|------|------------|------------|---------|---------------|-------|
| human | warm | yes | no | no | mammals |
| python | cold | no | no | no | reptiles |
| salmon | cold | no | no | yes | fishes |
| whale | warm | yes | no | yes | mammals |
| frog | cold | no | no | sometimes | amphibians |
| komodo | cold | no | no | no | reptiles |
| bat | warm | yes | yes | no | mammals |
| pigeon | warm | no | yes | no | birds |
| cat | warm | yes | no | no | mammals |
| leopard shark | cold | yes | no | yes | fishes |
| turtle | cold | no | no | sometimes | reptiles |
| penguin | warm | no | no | sometimes | birds |
| porcupine | warm | yes | no | no | mammals |
| eel | cold | no | no | yes | fishes |
| salamander | cold | no | no | sometimes | amphibians |
| gila monster | cold | no | no | no | reptiles |
| platypus | warm | no | no | no | mammals |
| owl | warm | no | yes | no | birds |
| dolphin | warm | yes | no | yes | mammals |
| eagle | warm | no | yes | no | birds |


---

## 3. RULE COVERAGE AND ACCURACY (CRITICAL FOR EXAM!)

### RULE COVERAGE:

**DEFINITION:** A rule r COVERS an instance x if the attributes of the instance
            SATISFY THE CONDITION (antecedent) of the rule

**FORMULA:**
```
                    |records satisfying antecedent|
    Coverage(r) = ------------------------------------
                         |total records|
```


### RULE ACCURACY:

**DEFINITION:** Fraction of records that satisfy BOTH the antecedent AND consequent

**FORMULA:**
```
                    |records satisfying antecedent AND consequent|
    Accuracy(r) = -------------------------------------------------
                         |records satisfying antecedent|
```


---

### WORKED EXAMPLE: Coverage and Accuracy Calculation

**DATASET:**

| Tid | Refund | Marital Status | Income | Class |
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

**RULE:** (Status=Single) --> No

**STEP 1:** Find records satisfying antecedent (Status=Single)
- Tid 1: Single --> MATCHES
- Tid 3: Single --> MATCHES
- Tid 8: Single --> MATCHES
- Tid 10: Single --> MATCHES
- Total matching antecedent: 4 records

**STEP 2:** Calculate COVERAGE
Coverage = 4/10 = 40%

**STEP 3:** Find records satisfying antecedent AND consequent (Single AND Class=No)
- Tid 1: Single, Class=No --> MATCHES BOTH
- Tid 3: Single, Class=No --> MATCHES BOTH
- Tid 8: Single, Class=Yes --> Only matches antecedent
- Tid 10: Single, Class=Yes --> Only matches antecedent
- Total matching both: 2 records

**STEP 4:** Calculate ACCURACY
Accuracy = 2/4 = 50%

**ANSWER:** Coverage = 40%, Accuracy = 50%


---

### APPLYING RULES TO CLASSIFY NEW INSTANCES

**NEW TEST DATA:**

| Name | Blood Type | Give Birth | Can Fly | Live in Water | Class |
|------|------------|------------|---------|---------------|-------|
| hawk | warm | no | yes | no | ? |
| grizzly bear | warm | yes | no | no | ? |

**RULES:**
R1: (Give Birth = no) AND (Can Fly = yes) --> Birds
R2: (Give Birth = no) AND (Live in Water = yes) --> Fishes
R3: (Give Birth = yes) AND (Blood Type = warm) --> Mammals
R4: (Give Birth = no) AND (Can Fly = no) --> Reptiles
R5: (Live in Water = sometimes) --> Amphibians

**CLASSIFICATION:**
- Hawk: Give Birth=no, Can Fly=yes --> Triggers R1 --> CLASS: Birds
- Grizzly Bear: Give Birth=yes, Blood Type=warm --> Triggers R3 --> CLASS: Mammals


---

## 4. HOW RULE-BASED CLASSIFIERS WORK

### PROBLEM CASES:

**TEST DATA:**

| Name | Blood Type | Give Birth | Can Fly | Live in Water | Class |
|------|------------|------------|---------|---------------|-------|
| lemur | warm | yes | no | no | ? |
| turtle | cold | no | no | sometimes | ? |
| dogfish shark | cold | yes | no | yes | ? |

**ANALYSIS:**

**1. LEMUR:**
   - Give Birth=yes, Blood Type=warm
   - Triggers R3: (Give Birth=yes) AND (Blood Type=warm) --> Mammals
   - CLASSIFICATION: Mammal (ONE RULE TRIGGERED)

**2. TURTLE:**
   - Give Birth=no, Can Fly=no, Live in Water=sometimes
   - Triggers R4: (Give Birth=no) AND (Can Fly=no) --> Reptiles
   - Triggers R5: (Live in Water=sometimes) --> Amphibians
   - PROBLEM: MULTIPLE RULES TRIGGERED! (CONFLICT)

**3. DOGFISH SHARK:**
   - Give Birth=yes, Blood Type=cold, Live in Water=yes
   - Check R1: Give Birth=no? NO (Give Birth=yes) --> Does NOT trigger
   - Check R2: Give Birth=no? NO --> Does NOT trigger
   - Check R3: Give Birth=yes AND Blood Type=warm? NO (Blood Type=cold) --> Does NOT trigger
   - Check R4: Give Birth=no? NO --> Does NOT trigger
   - Check R5: Live in Water=sometimes? NO (Live in Water=yes) --> Does NOT trigger
   - PROBLEM: NO RULES TRIGGERED! (UNCOVERED INSTANCE)


---

## 5. CHARACTERISTICS OF RULE-BASED CLASSIFIERS

---

### 5.1 MUTUALLY EXCLUSIVE RULES

**DEFINITION:**
- Rules are INDEPENDENT of each other
- Every record is covered by AT MOST ONE rule
- NO CONFLICTS possible

**BENEFIT:** Easy classification - each record matches exactly zero or one rule


---

### 5.2 EXHAUSTIVE RULES

**DEFINITION:**
- Classifier accounts for EVERY POSSIBLE combination of attribute values
- Each record is covered by AT LEAST ONE rule
- NO UNCOVERED INSTANCES

**BENEFIT:** Every record can be classified


### IDEAL RULE SET:
Both MUTUALLY EXCLUSIVE and EXHAUSTIVE:
- Each record is covered by EXACTLY ONE rule
- Rules from decision trees have this property!


---

## 6. CONVERTING DECISION TREES TO RULES

Every path from root to leaf becomes a rule.

### EXAMPLE DECISION TREE:
```
                    [Refund]
                   /        \
                Yes          No
                 |            |
               [NO]       [MarSt]
                         /       \
              {Single,       {Married}
              Divorced}          |
                    |          [NO]
                [TaxInc]
                /      \
            <80K      >80K
              |          |
            [NO]       [YES]
```


### EQUIVALENT CLASSIFICATION RULES:

Rule 1: (Refund=Yes) ==> No

Rule 2: (Refund=No, Marital Status={Single,Divorced},
         Taxable Income<80K) ==> No

Rule 3: (Refund=No, Marital Status={Single,Divorced},
         Taxable Income>80K) ==> Yes

Rule 4: (Refund=No, Marital Status={Married}) ==> No


### KEY PROPERTY:
Rules derived from decision trees are MUTUALLY EXCLUSIVE and EXHAUSTIVE!

- Mutually Exclusive: Each path in tree is unique
- Exhaustive: All leaf nodes cover all possibilities

The rule set contains AS MUCH INFORMATION as the tree.


---

## 7. ORDERED RULE SETS (DECISION LISTS)

### DEFINITION:
Rules are RANK ORDERED according to their PRIORITY

Also known as: DECISION LIST


### CLASSIFICATION PROCESS:
When a test record is presented:
1. Check rules in order (highest priority first)
2. Assign class label of the HIGHEST RANKED rule that triggers
3. If NO RULES fire, assign to DEFAULT CLASS


### EXAMPLE:

Rules (in priority order):
R1: (Give Birth = no) AND (Can Fly = yes) --> Birds
R2: (Give Birth = no) AND (Live in Water = yes) --> Fishes
R3: (Give Birth = yes) AND (Blood Type = warm) --> Mammals
R4: (Give Birth = no) AND (Can Fly = no) --> Reptiles
R5: (Live in Water = sometimes) --> Amphibians

**TEST RECORD - Turtle:**

| Name | Blood Type | Give Birth | Can Fly | Live in Water | Class |
|------|------------|------------|---------|---------------|-------|
| turtle | cold | no | no | sometimes | ? |

**CLASSIFICATION PROCESS:**
- Check R1: Give Birth=no AND Can Fly=yes? NO (Can Fly=no)
- Check R2: Give Birth=no AND Live in Water=yes? NO (Live in Water=sometimes)
- Check R3: Give Birth=yes? NO
- Check R4: Give Birth=no AND Can Fly=no? YES --> TRIGGERS!

**RESULT:** Turtle classified as REPTILE (R4 has higher priority than R5)


---

## 8. RULE ORDERING SCHEMES

### TWO APPROACHES:

---

### 8.1 RULE-BASED ORDERING
- Individual rules ranked based on their QUALITY
- Quality measures: accuracy, coverage, or combination
- Best quality rules evaluated first

**EXAMPLE:**
(Refund=Yes) ==> No
(Refund=No, Marital Status={Single,Divorced}, Taxable Income<80K) ==> No
(Refund=No, Marital Status={Single,Divorced}, Taxable Income>80K) ==> Yes
(Refund=No, Marital Status={Married}) ==> No


---

### 8.2 CLASS-BASED ORDERING
- Rules belonging to the SAME CLASS appear TOGETHER
- All rules for Class A, then all rules for Class B, etc.
- Within each class, rules may still be ordered by quality

**EXAMPLE:**
(Refund=Yes) ==> No
(Refund=No, Marital Status={Single,Divorced}, Taxable Income<80K) ==> No
(Refund=No, Marital Status={Married}) ==> No
(Refund=No, Marital Status={Single,Divorced}, Taxable Income>80K) ==> Yes

Note: All "No" rules grouped together, then "Yes" rules


---

## 9. BUILDING CLASSIFICATION RULES - DIRECT METHOD

### DIRECT METHOD:
Extract rules DIRECTLY from data without building intermediate model

**EXAMPLES:** RIPPER, CN2, Holte's 1R


---

### SEQUENTIAL COVERING ALGORITHM

**ALGORITHM:**
1. Start from an EMPTY rule
2. GROW a rule using the Learn-One-Rule function
3. REMOVE training records covered by the rule
4. REPEAT Steps 2-3 until stopping criterion is met


### VISUAL EXAMPLE:

**Step 1: Original Data**
```
+-------------------+
|  -  +  -  -       |
|   + +    -    -   |
|  + +     -        |
|  -        -    -  |
|       -  -        |
| -        -  + -   |
|  - + -      + +   |
|  + + + -   + +    |
+-------------------+
(+ = Class 1, - = Class 2)
```

**Step 2: Find rule R1 covering positive instances**
```
+-------------------+
|  ........  -       |
|  :+ + : -    -   |
|  :+ +:    -        |
|  :+ +:-            |
|  :..:     -    -  |
|       -  -        |
| -        -  + -   |
|  - + -      + +   |
|  + + + -   + +    |
+-------------------+
R1 covers region with positive instances
```

**Step 3:** Remove covered instances, find next rule
Remaining data after removing instances covered by R1...

**Step 4:** Find rule R2 covering remaining positive instances
Continue until all positive instances covered or stopping criterion met


### ASPECTS OF SEQUENTIAL COVERING:
1. Rule Growing - How to add conditions to a rule
2. Instance Elimination - Remove covered instances
3. Rule Evaluation - Measure rule quality
4. Stopping Criterion - When to stop adding rules
5. Rule Pruning - Remove unnecessary conditions


### SUMMARY OF DIRECT METHOD:
1. Grow a single rule
2. Remove instances covered by rule
3. Prune the rule (if necessary)
4. Add rule to current rule set
5. Repeat


---

## 10. BUILDING CLASSIFICATION RULES - INDIRECT METHOD

### INDIRECT METHOD:
Extract rules from OTHER CLASSIFICATION MODELS (decision trees, neural networks)

**EXAMPLE:** C4.5rules


### EXAMPLE - Converting Tree to Rules:

**DECISION TREE:**
```
              P
            /   \
          No     Yes
          |       |
          Q       R
         / \     / \
        No Yes  No  Yes
        |   |   |    |
        -   +   +    Q
                    / \
                   No  Yes
                   |    |
                   -    +
```


**EQUIVALENT RULE SET:**
r1: (P=No, Q=No) ==> -
r2: (P=No, Q=Yes) ==> +
r3: (P=Yes, R=No) ==> +
r4: (P=Yes, R=Yes, Q=No) ==> -
r5: (P=Yes, R=Yes, Q=Yes) ==> +


---

## 11. C4.5RULES METHOD (INDIRECT)

### ALGORITHM:

1. Extract rules from an UNPRUNED decision tree

2. For each rule r: A --> y
   - Consider alternative rule r': A' --> y
   - Where A' is obtained by REMOVING one conjunct from A

3. Compare PESSIMISTIC ERROR RATE for r against all r's

4. PRUNE if one of the r's has LOWER pessimistic error rate

5. REPEAT until no improvement in generalization error


### EXAMPLE:

Original Rule: (A=1) AND (B=2) AND (C=3) --> Class1

Alternative Rules (removing one condition each):
- r1': (B=2) AND (C=3) --> Class1
- r2': (A=1) AND (C=3) --> Class1
- r3': (A=1) AND (B=2) --> Class1

If r2' has lower pessimistic error rate than original rule,
then PRUNE to: (A=1) AND (C=3) --> Class1


---

## 12. ADVANTAGES OF RULE-BASED CLASSIFIERS

1. As HIGHLY EXPRESSIVE as decision trees
   - Can represent same concepts as trees

2. EASY TO INTERPRET
   - Human-readable if-then rules
   - Good for explaining decisions

3. EASY TO GENERATE
   - Can be extracted from trees
   - Or learned directly from data

4. Can classify new instances RAPIDLY
   - Just check rule conditions

5. PERFORMANCE COMPARABLE to decision trees
   - Similar accuracy in many cases


---

## 13. INSTANCE-BASED CLASSIFIERS

### DEFINITION:
- STORE the training records
- Use training records to PREDICT class label of unseen cases
- NO explicit model is built!


### CONCEPTUAL DIAGRAM:
```
Set of Stored Cases              Unseen Case
+------+-----+-------+           +------+-----+
| Atr1 | ... | Class |           | Atr1 | ... |
+------+-----+-------+           +------+-----+
|      |     |   A   |<----------|      |     |
|      |     |   B   |<----------|      |     |
|      |     |   B   |           +------+-----+
|      |     |   C   |<----------Find similar
|      |     |   A   |           training cases
|      |     |   C   |
|      |     |   B   |
+------+-----+-------+
```


### TYPES OF INSTANCE-BASED CLASSIFIERS:

**1. ROTE-LEARNER:**
   - Memorizes ENTIRE training data
   - Classifies only if attributes EXACTLY MATCH a training example
   - Very restrictive!

**2. NEAREST NEIGHBOR:**
   - Uses k "CLOSEST" points (nearest neighbors)
   - More flexible and practical


---

## 14. K-NEAREST NEIGHBOR (k-NN) CLASSIFIER

### BASIC IDEA:
"If it walks like a duck, quacks like a duck, then it's probably a duck"

Find similar training examples and use their class labels.


### THREE REQUIREMENTS:
1. The set of STORED RECORDS (training data)
2. DISTANCE METRIC to compute distance between records
3. The value of K (number of nearest neighbors to retrieve)


### CLASSIFICATION PROCESS:
To classify an UNKNOWN record:
1. Compute DISTANCE to all training records
2. Identify K nearest neighbors (k closest records)
3. Use class labels of nearest neighbors to determine class
   - Typically by MAJORITY VOTE


### VISUAL EXAMPLE:
```
                   Unknown record
                        |
+----------------------v-----------------------+
|                      *                       |
|  -      +  +       ....                      |
|      + +         .    .                      |
|    + +         .  -    .     -               |
|      +       .  +       .                    |
|  -           .    -      .                   |
|                 -    -                   -   |
|  -       -                  -                |
|                  -   + -                     |
|     -  +                        -     +      |
| + + +  -          -       +  +               |
+-----------------------+----------------------+
```

If k=5: Find 5 nearest neighbors
Count: 3 positive (+), 2 negative (-)
PREDICTION: Positive (majority vote)


### DEFINITION OF K-NEAREST NEIGHBORS:
K-nearest neighbors of record x are data points that have the
K SMALLEST DISTANCES to x

**EXAMPLES:**
- 1-nearest neighbor: Only the SINGLE closest point
- 2-nearest neighbor: Two closest points
- 3-nearest neighbor: Three closest points

The region of influence creates a VORONOI DIAGRAM where each cell
contains points closest to a particular training example.


---

## 15. DISTANCE METRICS

---

### 15.1 EUCLIDEAN DISTANCE (Most Common)

**FORMULA:**
```
                    _____________________
    d(p, q) = sqrt(SUM (p_i - q_i)^2)
                     i
```

Where p_i and q_i are the i-th attributes of records p and q.


### EXAMPLE:
Record p = (1, 2, 3)
Record q = (4, 0, 3)

d(p, q) = sqrt((1-4)^2 + (2-0)^2 + (3-3)^2)
        = sqrt(9 + 4 + 0)
        = sqrt(13)
        = 3.61


---

### 15.2 DETERMINING CLASS FROM NEIGHBORS

**METHOD 1: MAJORITY VOTE**
- Count class labels among k nearest neighbors
- Assign most common class

**METHOD 2: WEIGHTED VOTE**
- Weight factor: w = 1/d^2
- Closer neighbors have MORE influence
- Further neighbors have LESS influence

### EXAMPLE:
k=3, neighbors at distances 1, 2, 3 with classes A, B, A

Majority vote: A wins (2 vs 1)

Weighted vote:
- Class A: 1/1^2 + 1/3^2 = 1 + 0.111 = 1.111
- Class B: 1/2^2 = 0.25
- Class A wins with higher weighted vote


---

## 16. CHOOSING THE VALUE OF K

### THE CRITICAL TRADE-OFF:

| K Value | Issue |
|---------|-------|
| Too SMALL | SENSITIVE to noise points. Single noisy neighbor can cause misclassification |
| Too LARGE | Neighborhood may include points from OTHER CLASSES. Loses local structure |


### VISUAL EXAMPLE (k too large):
```
                    -       +     -
                +   -   -    +    -

              +   -  + + +   -
                     + + x

                +  -          -
                       -           +
                  -   -   -   -
```

If x is positive class, but k is very large,
neighborhood includes many negative examples,
and x gets misclassified as negative.


### BEST PRACTICE:
- Use CROSS-VALIDATION to find optimal k
- Typically use ODD values of k for binary classification
  (to avoid ties in majority vote)
- Common starting point: k = sqrt(n) where n is training size


---

## 17. SCALING ISSUES IN K-NN

### PROBLEM:
Attributes with DIFFERENT SCALES can DOMINATE distance calculations

### EXAMPLE:
- Height: ranges from 1.5m to 1.8m (range = 0.3)
- Weight: ranges from 90lb to 300lb (range = 210)
- Income: ranges from $10K to $1M (range = $990K)

Without scaling, INCOME will completely dominate the distance!
A $1000 difference in income would swamp any difference in height.


### SOLUTION: NORMALIZATION
Scale all attributes to similar ranges before computing distances

Common methods:
1. Min-Max Normalization: Scale to [0, 1]
2. Z-score Standardization: Mean=0, StdDev=1


---

## 18. K-NN CHARACTERISTICS

### K-NN IS A "LAZY LEARNER":
- Does NOT build models explicitly
- Simply stores training data
- All work done at CLASSIFICATION TIME

### COMPARISON:

| Lazy Learners | Eager Learners |
|---------------|----------------|
| k-NN | Decision Trees |
| Store training data, no model built | Rule-based systems |
| | Build model during training |


### IMPLICATIONS:
- Training: FAST (just store data)
- Classification: EXPENSIVE (compute all distances)
- Memory: HIGH (must store all training data)


---

## 19. PEBLS ALGORITHM (CRITICAL EXAMPLE!)

PEBLS: Parallel Examplar-Based Learning System (Cost & Salzberg)

### CHARACTERISTICS:
1. Works with BOTH continuous AND nominal features
2. Uses Modified Value Difference Metric (MVDM) for nominal features
3. Each record assigned a WEIGHT FACTOR
4. Number of nearest neighbors: k = 1


---

### 19.1 MODIFIED VALUE DIFFERENCE METRIC (MVDM)

For NOMINAL attributes, cannot use Euclidean distance!

**FORMULA:**
```
                    |n_1i     n_2i|
    d(V1, V2) = SUM |----  -  ----|
                 i  |n_1      n_2 |
```

Where:
- n_1i = count of value V1 with class i
- n_1 = total count of value V1
- n_2i = count of value V2 with class i
- n_2 = total count of value V2


---

### WORKED EXAMPLE: MVDM Distance Calculation

**DATASET:**

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


**STEP 1: Build Count Matrix for Marital Status**

| Class | Single | Married | Divorced |
|-------|--------|---------|----------|
| Yes | 2 | 0 | 1 |
| No | 2 | 4 | 1 |
| Total | 4 | 4 | 2 |


**STEP 2: Calculate d(Single, Married)**
```
d(Single, Married) = |2/4 - 0/4| + |2/4 - 4/4|
                   = |0.5 - 0| + |0.5 - 1|
                   = 0.5 + 0.5
                   = 1
```


**STEP 3: Calculate d(Single, Divorced)**
```
d(Single, Divorced) = |2/4 - 1/2| + |2/4 - 1/2|
                    = |0.5 - 0.5| + |0.5 - 0.5|
                    = 0 + 0
                    = 0
```

**INTERPRETATION:** Single and Divorced have SIMILAR class distributions!


**STEP 4: Calculate d(Married, Divorced)**
```
d(Married, Divorced) = |0/4 - 1/2| + |4/4 - 1/2|
                     = |0 - 0.5| + |1 - 0.5|
                     = 0.5 + 0.5
                     = 1
```


**STEP 5: Build Count Matrix for Refund**

| Class | Yes | No |
|-------|-----|-----|
| Yes | 0 | 3 |
| No | 3 | 4 |
| Total | 3 | 7 |


**STEP 6: Calculate d(Refund=Yes, Refund=No)**
```
d(Yes, No) = |0/3 - 3/7| + |3/3 - 4/7|
           = |0 - 0.429| + |1 - 0.571|
           = 0.429 + 0.429
           = 6/7 (approximately 0.857)
```


---

### 19.2 PEBLS DISTANCE BETWEEN RECORDS

**FORMULA:**
```
                         d
    Delta(X, Y) = wX * wY * SUM d(Xi, Yi)^2
                        i=1
```

Where:
- wX = Weight factor for record X
- d = Number of attributes


**WEIGHT FACTOR:**
```
                    Number of times X is used for prediction
    wX = ------------------------------------------------
              Number of times X predicts correctly
```


**INTERPRETATION:**
- wX approximately 1: X makes accurate predictions most of the time
- wX > 1: X is NOT reliable for making predictions


### EXAMPLE:

| Tid | Refund | Marital Status | Income | Cheat |
|-----|--------|----------------|--------|-------|
| X | Yes | Single | 125K | No |
| Y | No | Married | 100K | No |

If wX = wY = 1 (both reliable predictors):

```
Delta(X, Y) = 1 * 1 * [d(Yes,No)^2 + d(Single,Married)^2 + d(125K,100K)^2]
            = 1 * [(6/7)^2 + 1^2 + (normalized income diff)^2]
```


---

## 20. QUIZ QUESTIONS & WORKED EXAMPLES

---

### QUESTION 1: Rule Coverage and Accuracy

Given the dataset:

| Tid | Status | Class |
|-----|--------|-------|
| 1 | A | Yes |
| 2 | A | Yes |
| 3 | A | No |
| 4 | B | No |
| 5 | B | No |
| 6 | C | Yes |

For rule: (Status=A) --> Yes

Calculate Coverage and Accuracy.

**SOLUTION:**
Records with Status=A: Tid 1, 2, 3 (3 records)
Total records: 6

Coverage = 3/6 = 50%

Records with Status=A AND Class=Yes: Tid 1, 2 (2 records)
Records with Status=A: 3

Accuracy = 2/3 = 66.7%


---

### QUESTION 2: Decision Tree to Rules

Convert this decision tree to rules:

```
           [A]
          /   \
        Yes    No
         |      |
       [B]     [+]
       / \
      1   2
      |   |
     [-] [+]
```

**SOLUTION:**
Rule 1: (A=Yes) AND (B=1) --> -
Rule 2: (A=Yes) AND (B=2) --> +
Rule 3: (A=No) --> +


---

### QUESTION 3: k-NN Classification

Given training data:
Point A: (1, 1), Class = +
Point B: (2, 1), Class = +
Point C: (4, 3), Class = -
Point D: (5, 4), Class = -
Point E: (3, 2), Class = +

Classify point X: (3, 3) using k=3

**SOLUTION:**
Step 1: Calculate distances from X=(3,3) to each point

d(X, A) = sqrt((3-1)^2 + (3-1)^2) = sqrt(4+4) = sqrt(8) = 2.83
d(X, B) = sqrt((3-2)^2 + (3-1)^2) = sqrt(1+4) = sqrt(5) = 2.24
d(X, C) = sqrt((3-4)^2 + (3-3)^2) = sqrt(1+0) = sqrt(1) = 1.00
d(X, D) = sqrt((3-5)^2 + (3-4)^2) = sqrt(4+1) = sqrt(5) = 2.24
d(X, E) = sqrt((3-3)^2 + (3-2)^2) = sqrt(0+1) = sqrt(1) = 1.00

Step 2: Sort by distance
C: 1.00 (Class -)
E: 1.00 (Class +)
B: 2.24 (Class +)
D: 2.24 (Class -)
A: 2.83 (Class +)

Step 3: Take k=3 nearest neighbors
C (Class -), E (Class +), B (Class +)  [or D instead of B - tie]

Step 4: Majority vote
If neighbors are C(-), E(+), B(+): 2 positive, 1 negative
PREDICTION: + (Positive class)


---

### QUESTION 4: MVDM Calculation

Given:

| Class | Value1 | Value2 |
|-------|--------|--------|
| A | 3 | 1 |
| B | 2 | 4 |
| Total | 5 | 5 |

Calculate d(Value1, Value2)

**SOLUTION:**
```
d(Value1, Value2) = |3/5 - 1/5| + |2/5 - 4/5|
                  = |0.6 - 0.2| + |0.4 - 0.8|
                  = 0.4 + 0.4
                  = 0.8
```


---

### QUESTION 5: Why are rules from decision trees mutually exclusive?

**ANSWER:**
Rules from decision trees are mutually exclusive because:
1. Each rule corresponds to a UNIQUE PATH from root to leaf
2. No two paths share the same combination of conditions
3. At each internal node, branches are mutually exclusive
   (e.g., "Refund=Yes" vs "Refund=No" cannot both be true)
4. Therefore, a record can only satisfy conditions of ONE path


---

### QUESTION 6: k-NN Trade-offs

Explain what happens when k is:
a) Too small (k=1)
b) Too large (k=N where N is total training size)

**ANSWER:**
a) k=1 (Too small):
   - Very sensitive to NOISE
   - Single outlier can cause misclassification
   - Decision boundary is very jagged
   - HIGH VARIANCE, LOW BIAS

b) k=N (Too large):
   - Always predicts MAJORITY CLASS of entire dataset
   - Ignores local structure completely
   - Underfits the data
   - LOW VARIANCE, HIGH BIAS


---

### QUESTION 7: Ordered Rule Set

Given rules in order:
R1: (A=1) --> Class X
R2: (B=2) --> Class Y
R3: (A=1) AND (B=2) --> Class Z

Classify record with A=1, B=2

**SOLUTION:**
Check R1: A=1? YES --> TRIGGERS!
STOP checking, assign Class X

Note: Even though R3 is more specific, R1 fires first
because rules are checked IN ORDER.


---

## COMPARISON TABLE: RULE-BASED VS K-NN

| Aspect | Rule-Based | k-NN |
|--------|------------|------|
| Model Type | Eager (builds model) | Lazy (no model) |
| Training Time | Slow (learn rules) | Fast (just store data) |
| Classification Time | Fast (check conditions) | Slow (compute distances) |
| Interpretability | High (readable rules) | Low (no explicit rules) |
| Memory Usage | Low (store rules only) | High (store all data) |
| Handles Nominal Attributes | Naturally | Needs MVDM or encoding |
| Sensitive to Irrelevant Features | Less sensitive | Very sensitive |


---

## KEY FORMULAS SUMMARY

**1. RULE COVERAGE:**
   Coverage = |records satisfying antecedent| / |total records|

**2. RULE ACCURACY:**
   Accuracy = |records satisfying antecedent AND consequent| /
              |records satisfying antecedent|

**3. EUCLIDEAN DISTANCE:**
   d(p, q) = sqrt(SUM_i (p_i - q_i)^2)

**4. WEIGHTED VOTE:**
   w = 1/d^2

**5. MVDM (Modified Value Difference Metric):**
   d(V1, V2) = SUM_i |n_1i/n_1 - n_2i/n_2|

**6. PEBLS DISTANCE:**
   Delta(X, Y) = wX * wY * SUM_i d(Xi, Yi)^2

**7. PEBLS WEIGHT:**
   wX = (# times X used for prediction) / (# times X predicts correctly)


---

## QUICK REVIEW CHECKLIST

### RULE-BASED CLASSIFIERS:
- [ ] Can define a classification rule (condition --> class)
- [ ] Can calculate rule COVERAGE
- [ ] Can calculate rule ACCURACY
- [ ] Understand MUTUALLY EXCLUSIVE rules
- [ ] Understand EXHAUSTIVE rules
- [ ] Can convert decision tree to rules
- [ ] Understand ordered rule sets (decision lists)
- [ ] Know difference between rule-based and class-based ordering
- [ ] Know Sequential Covering algorithm (direct method)
- [ ] Know C4.5rules algorithm (indirect method)
- [ ] Can list advantages of rule-based classifiers

### K-NEAREST NEIGHBOR:
- [ ] Understand instance-based (lazy) learning concept
- [ ] Know the three requirements for k-NN
- [ ] Can calculate Euclidean distance
- [ ] Understand majority vote classification
- [ ] Understand weighted voting (w = 1/d^2)
- [ ] Know trade-offs in choosing k value
- [ ] Understand scaling/normalization issues
- [ ] Know PEBLS algorithm concepts
- [ ] Can calculate MVDM distance for nominal attributes
- [ ] Understand PEBLS weight factor

### COMPARISON:
- [ ] Can compare lazy vs eager learners
- [ ] Can compare rule-based vs k-NN classifiers


---

*END OF TOPIC 5A*
