# ASSOCIATION ANALYSIS (ASSOCIATION RULE METHOD)
## CS653: DATA MINING
### Detailed Study Notes - Lecture 06-A

---

## TABLE OF CONTENTS
1. Introduction to Association Rules
2. Key Definitions
3. Association Rule Mining Task
4. Frequent Itemset Generation
5. Apriori Principle and Algorithm
6. Rule Generation
7. Rule Pruning and Evaluation
8. Interestingness Measures
9. Examples and Problems
10. Summary

---

## 1. INTRODUCTION TO ASSOCIATION RULES

### WHAT IS AN ASSOCIATION RULE?
- An implication expression of the form: X → Y
- Where X and Y are itemsets (collections of items)
- X is called the ANTECEDENT (left-hand side)
- Y is called the CONSEQUENT (right-hand side)

**IMPORTANT:** Implication means CO-OCCURRENCE, NOT CAUSALITY!
- The rule {Diaper} → {Beer} means diapers and beer are frequently
  bought together, NOT that buying diapers causes someone to buy beer

### CLASSIC EXAMPLE: Market-Basket Transactions

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

Example Association Rules from this data:
- {Diaper} → {Beer}
- {Milk, Bread} → {Eggs, Coke}
- {Beer, Bread} → {Milk}


---

## 2. KEY DEFINITIONS

### ITEMSET
- A collection of one or more items
- Example: {Milk, Bread, Diaper}

### k-ITEMSET
- An itemset that contains exactly k items
- {Milk} is a 1-itemset
- {Milk, Bread} is a 2-itemset
- {Milk, Bread, Diaper} is a 3-itemset

### SUPPORT COUNT (σ)
- The FREQUENCY of occurrence of an itemset
- The number of transactions that contain the itemset
- Example: σ({Milk, Bread, Diaper}) = 2 (appears in TID 4 and 5)

### SUPPORT (s)
- FRACTION of transactions that contain an itemset
- Formula: s(X) = σ(X) / |T|
  where |T| is the total number of transactions

- Example: s({Milk, Bread, Diaper}) = 2/5 = 0.4 = 40%

### FREQUENT ITEMSET
- An itemset whose support is GREATER THAN OR EQUAL TO a minimum
  support threshold (minsup)
- If minsup = 40%, then {Milk, Bread, Diaper} is frequent (support = 40%)

### CONFIDENCE (c)
- Measures how often items in Y appear in transactions that contain X
- Formula: c(X → Y) = σ(X ∪ Y) / σ(X)
- Or equivalently: c(X → Y) = s(X ∪ Y) / s(X)

### EXAMPLE: Computing Support and Confidence

For rule: {Milk, Diaper} → {Beer}

Using the market-basket data above:
- σ({Milk, Diaper, Beer}) = 2 (TID 3 and TID 4)
- σ({Milk, Diaper}) = 3 (TID 3, 4, and 5)
- |T| = 5

Support:
s = σ(Milk, Diaper, Beer) / |T| = 2/5 = 0.4 = 40%

Confidence:
c = σ(Milk, Diaper, Beer) / σ(Milk, Diaper) = 2/3 = 0.67 = 67%


---

## 3. ASSOCIATION RULE MINING TASK

### GOAL
Given a set of transactions T, find ALL rules having:
- support ≥ minsup threshold
- confidence ≥ minconf threshold

### BRUTE-FORCE APPROACH (Naive Method)
1. List all possible association rules
2. Compute support and confidence for each rule
3. Prune rules that fail the minsup and minconf thresholds

Problem: COMPUTATIONALLY PROHIBITIVE!

### WHY IS BRUTE-FORCE EXPENSIVE?

Given d unique items:
- Total number of itemsets = 2^d
- Total number of possible association rules:
```
  R = Σ(k=1 to d-1) [ C(d,k) × Σ(j=1 to d-k) C(d-k,j) ]
    = 3^d - 2^(d+1) + 1
```

Example: If d = 6, R = 602 rules!

The lattice of all possible itemsets for 5 items {A,B,C,D,E}:
- Level 0: null (empty set)
- Level 1: A, B, C, D, E (5 itemsets)
- Level 2: AB, AC, AD, AE, BC, BD, BE, CD, CE, DE (10 itemsets)
- Level 3: ABC, ABD, ABE, ACD, ACE, ADE, BCD, BCE, BDE, CDE (10 itemsets)
- Level 4: ABCD, ABCE, ABDE, ACDE, BCDE (5 itemsets)
- Level 5: ABCDE (1 itemset)
Total = 2^5 = 32 itemsets (including empty set)

Complexity: O(NMw)
- N = number of transactions
- M = number of candidate itemsets = 2^d
- w = transaction width (average items per transaction)


---

## 4. SMARTER APPROACH: TWO-STEP METHOD

### KEY OBSERVATION
All rules derived from the same itemset have IDENTICAL SUPPORT but can
have DIFFERENT CONFIDENCE.

Example: Rules from itemset {Milk, Diaper, Beer}
- {Milk, Diaper} → {Beer}  (s=0.4, c=0.67)
- {Milk, Beer} → {Diaper}  (s=0.4, c=1.0)
- {Diaper, Beer} → {Milk}  (s=0.4, c=0.67)
- {Beer} → {Milk, Diaper}  (s=0.4, c=0.67)
- {Diaper} → {Milk, Beer}  (s=0.4, c=0.5)
- {Milk} → {Diaper, Beer}  (s=0.4, c=0.5)

All have support = 0.4, but confidence varies!

### TWO-STEP APPROACH

**Step 1: FREQUENT ITEMSET GENERATION**
- Generate all itemsets whose support ≥ minsup

**Step 2: RULE GENERATION**
- Generate high confidence rules from each frequent itemset
- Each rule is a binary partitioning of a frequent itemset

This decouples support and confidence requirements!


---

## 5. FREQUENT ITEMSET GENERATION

### THE COMPUTATIONAL CHALLENGE
Even frequent itemset generation alone is computationally expensive because:
- Given d items, there are 2^d possible candidate itemsets
- Must count support by scanning database for each candidate
- Complexity ~ O(NMw) where M = 2^d

### STRATEGIES TO REDUCE COMPUTATION

**a. REDUCE NUMBER OF CANDIDATES (M)**
   - Complete search: M = 2^d
   - Use pruning techniques to reduce M
   - Key technique: APRIORI PRINCIPLE

**b. REDUCE NUMBER OF COMPARISONS (NM)**
   - Use efficient data structures (hash trees, etc.)
   - No need to match every candidate against every transaction

**c. REDUCE NUMBER OF TRANSACTIONS (N)**
   - Reduce size of N as itemset size increases
   - Transactions shorter than k items cannot contain any k-itemsets

---

### APRIORI PRINCIPLE

**DEFINITION:**
If an itemset is FREQUENT, then ALL of its SUBSETS must also be FREQUENT.

Equivalently (contrapositive):
If an itemset is INFREQUENT, then ALL of its SUPERSETS must also be INFREQUENT.

**WHY DOES THIS WORK?**
Due to the ANTI-MONOTONE property of support:
- Support of an itemset NEVER EXCEEDS the support of its subsets

Mathematical formulation:
For all X, Y: (X ⊆ Y) ⟹ s(X) ≥ s(Y)

**INTUITION:**
If {A,B} appears in only 2 transactions, then {A,B,C} can appear in
AT MOST 2 transactions (because every transaction containing {A,B,C}
must also contain {A,B}).

### ILLUSTRATION OF APRIORI PRUNING
If {AB} is found to be INFREQUENT, we can PRUNE all supersets:
- {ABC}, {ABD}, {ABE}
- {ABCD}, {ABCE}, {ABDE}
- {ABCDE}

These don't need to be checked!

### EXAMPLE: Apriori Pruning with Minimum Support = 3

**Step 1: Count 1-itemsets**

| Item | Count | Frequent? |
|------|-------|-----------|
| Bread | 4 | Yes (≥3) |
| Coke | 2 | NO (<3) |
| Milk | 4 | Yes |
| Beer | 3 | Yes |
| Diaper | 4 | Yes |
| Eggs | 1 | NO (<3) |

Prune: Coke and Eggs (don't generate candidates involving these)

**Step 2: Count 2-itemsets (only from frequent 1-itemsets)**

| Itemset | Count | Frequent? |
|---------|-------|-----------|
| {Bread, Milk} | 3 | Yes |
| {Bread, Beer} | 2 | NO |
| {Bread, Diaper} | 3 | Yes |
| {Milk, Beer} | 2 | NO |
| {Milk, Diaper} | 3 | Yes |
| {Beer, Diaper} | 3 | Yes |

**Step 3: Count 3-itemsets (only from frequent 2-itemsets)**
- Can only generate {Bread, Milk, Diaper} (all its 2-subsets are frequent)
- Cannot generate {Bread, Milk, Beer} because {Bread, Beer} is infrequent
- Cannot generate {Milk, Beer, Diaper} because {Milk, Beer} is infrequent

| Itemset | Count | Frequent? |
|---------|-------|-----------|
| {Bread, Milk, Diaper} | 3 | Yes |

**SAVINGS:**
- Without pruning: 6C1 + 6C2 + 6C3 = 6 + 15 + 20 = 41 itemsets
- With pruning: 6 + 6 + 1 = 13 itemsets checked

---

### APRIORI ALGORITHM

**PSEUDOCODE:**
1. Let k = 1
2. Generate frequent itemsets of length 1 (scan DB, count items)
3. REPEAT until no new frequent itemsets are identified:
   a. Generate length (k+1) candidate itemsets from length k frequent itemsets
   b. Prune candidates containing subsets of length k that are infrequent
   c. Count support of each candidate by scanning the database
   d. Eliminate candidates that are infrequent
   e. k = k + 1

**CANDIDATE GENERATION (Fk-1 × Fk-1 Method):**
- Join two frequent (k-1)-itemsets that share (k-2) items
- Example: {A,B,C} and {A,B,D} can be joined to form {A,B,C,D}

**CANDIDATE PRUNING:**
- After generating candidate, check all (k-1)-subsets
- If any subset is not in Fk-1, prune the candidate


---

## 6. RULE GENERATION

### GOAL
Given a frequent itemset L, find all non-empty subsets f ⊂ L such that
f → (L - f) satisfies the minimum confidence requirement.

### EXAMPLE: Rules from {A,B,C,D}

If {A,B,C,D} is frequent, candidate rules are:

**1-item consequent:**
- ABC → D,  ABD → C,  ACD → B,  BCD → A

**2-item consequent:**
- AB → CD,  AC → BD,  AD → BC,  BC → AD,  BD → AC,  CD → AB

**3-item consequent:**
- A → BCD,  B → ACD,  C → ABD,  D → ABC

Total: 2^k - 2 candidate rules (excluding L → ∅ and ∅ → L)
For k=4: 2^4 - 2 = 14 candidate rules

### CONFIDENCE ANTI-MONOTONE PROPERTY

**IMPORTANT:** In general, confidence does NOT have anti-monotone property.
c(ABC → D) can be larger or smaller than c(AB → D)

**HOWEVER:** For rules generated from the SAME itemset, confidence IS
anti-monotone with respect to the CONSEQUENT size.

For itemset L = {A,B,C,D}:
c(ABC → D) ≥ c(AB → CD) ≥ c(A → BCD)

**WHY?**
```
c(ABC → D) = σ(ABCD) / σ(ABC)
c(AB → CD) = σ(ABCD) / σ(AB)
```

Since ABC ⊂ AB implies σ(ABC) ≤ σ(AB), we get:
σ(ABCD) / σ(ABC) ≥ σ(ABCD) / σ(AB)

### RULE GENERATION USING LATTICE

For itemset {A,B,C,D}, the lattice of rule consequents:

```
Level 0:        ABCD → {} (not valid)
                    |
Level 1:  BCD→A  ACD→B  ABD→C  ABC→D
                    \    |    /
Level 2:  CD→AB  BD→AC  BC→AD  AD→BC  AC→BD  AB→CD
                    \    |    /
Level 3:    D→ABC   C→ABD   B→ACD   A→BCD
```

If a rule at Level 1 (e.g., BCD → A) has LOW confidence,
then all rules below it in the lattice that extend the
consequent can be PRUNED:
- CD → AB, BD → AC, BC → AD (Level 2)
- D → ABC, C → ABD, B → ACD (Level 3)

### RULE GENERATION ALGORITHM
1. Start with rules having 1-item consequent
2. For each frequent itemset, compute confidence for rules with 1-item consequent
3. Keep rules satisfying minconf
4. Use frequent consequents to generate candidates with 2-item consequent
5. Repeat until no more rules can be generated

**Candidate Generation:**
- Merge rules sharing same prefix in consequent
- join(CD → AB, BD → AC) produces candidate D → ABC
- Prune D → ABC if subset rule AD → BC does not have high confidence


---

## 7. RULE PRUNING AND EVALUATION

### EFFECT OF SUPPORT DISTRIBUTION
Many real datasets have SKEWED support distribution:
- Few items have very high support (popular items)
- Many items have low support (rare items)

**Problem with single minsup threshold:**
- If minsup is TOO HIGH: miss itemsets with interesting rare items
  (e.g., expensive products like diamonds)
- If minsup is TOO LOW: computationally expensive, too many itemsets

### MULTIPLE MINIMUM SUPPORT

**Solution:** Assign different minimum support to different items.

MS(i) = minimum support for item i

Examples:
- MS(Milk) = 5%
- MS(Coke) = 3%
- MS(Broccoli) = 0.1%
- MS(Salmon) = 0.5%

For an itemset, use the MINIMUM of individual item supports:
MS({Milk, Broccoli}) = min(MS(Milk), MS(Broccoli)) = min(5%, 0.1%) = 0.1%

### CHALLENGE: Support is No Longer Anti-Monotone!

Example:
- Support(Milk, Coke) = 1.5%
- Support(Milk, Coke, Broccoli) = 0.5%
- MS(Milk, Coke) = min(5%, 3%) = 3%  → {Milk, Coke} is INFREQUENT
- MS(Milk, Coke, Broccoli) = min(5%, 3%, 0.1%) = 0.1% → FREQUENT!

A superset can be frequent even if a subset is infrequent!

### SOLUTION: Modified Apriori

1. Order items by their minimum support (ascending order)
   Example: Broccoli, Salmon, Coke, Milk

2. Modify pruning step:
   - In traditional Apriori: prune if ANY k-subset is infrequent
   - In modified Apriori: prune only if subset CONTAINING THE FIRST ITEM
     is infrequent

Example:
- Candidate = {Broccoli, Coke, Milk}
- {Broccoli, Coke} is frequent
- {Broccoli, Milk} is frequent
- {Coke, Milk} is INFREQUENT

Traditional Apriori: Would prune {Broccoli, Coke, Milk}
Modified Apriori: Do NOT prune because {Coke, Milk} doesn't contain
                  first item (Broccoli)


---

## 8. INTERESTINGNESS MEASURES

### WHY DO WE NEED MORE MEASURES?
- Association rule algorithms produce TOO MANY rules
- Many rules are uninteresting or redundant
- Redundant: {A,B,C} → {D} and {A,B} → {D} with same support & confidence

Interestingness measures help prune/rank derived patterns.

### CONTINGENCY TABLE

For rule X → Y:

```
           |    Y    |   ~Y    | Total
    -------|---------|---------|-------
      X    |   f11   |   f10   |  f1+
     ~X    |   f01   |   f00   |  f0+
    -------|---------|---------|-------
    Total  |   f+1   |   f+0   |  |T|
```

Where:
- f11 = support count of X AND Y
- f10 = support count of X AND NOT Y
- f01 = support count of NOT X AND Y
- f00 = support count of NOT X AND NOT Y

### LIMITATION OF CONFIDENCE

**EXAMPLE: Tea → Coffee**

```
           | Coffee | ~Coffee | Total
    -------|--------|---------|-------
      Tea  |   15   |    5    |   20
     ~Tea  |   75   |    5    |   80
    -------|--------|---------|-------
    Total  |   90   |   10    |  100
```

Confidence(Tea → Coffee) = P(Coffee|Tea) = 15/20 = 0.75 = 75%

BUT: P(Coffee) = 90/100 = 0.9 = 90%

The confidence is HIGH (75%), but the rule is MISLEADING!
- People who drink tea are LESS likely to drink coffee (75%)
  than the general population (90%)
- Actually: P(Coffee|~Tea) = 75/80 = 93.75% > P(Coffee|Tea)

### STATISTICAL INDEPENDENCE

Two events X and Y are statistically independent if:
P(X ∧ Y) = P(X) × P(Y)

Example: Population of 1000 students
- 600 know how to swim (S)
- 700 know how to bike (B)
- 420 know both (S ∧ B)

P(S ∧ B) = 420/1000 = 0.42
P(S) × P(B) = 0.6 × 0.7 = 0.42

Since P(S ∧ B) = P(S) × P(B), swimming and biking are INDEPENDENT.

**Correlation:**
- P(X ∧ Y) = P(X) × P(Y)  → INDEPENDENT
- P(X ∧ Y) > P(X) × P(Y)  → POSITIVELY CORRELATED
- P(X ∧ Y) < P(X) × P(Y)  → NEGATIVELY CORRELATED

---

### STATISTICAL-BASED MEASURES

### LIFT (Interest Factor)
```
           P(Y|X)       P(X,Y)
Lift = ----------- = -----------
           P(Y)       P(X) × P(Y)
```

**Interpretation:**
- Lift = 1: X and Y are independent
- Lift > 1: X and Y are positively correlated
- Lift < 1: X and Y are negatively correlated

**EXAMPLE: Tea → Coffee (from contingency table above)**

```
Lift = P(Coffee|Tea) / P(Coffee)
     = 0.75 / 0.9
     = 0.833
```

Since Lift < 1, Tea and Coffee are NEGATIVELY ASSOCIATED!

**QUIZ:** Is ~Tea → Coffee positively or negatively associated?

```
P(Coffee|~Tea) = 75/80 = 0.9375
P(Coffee) = 0.9
Lift = 0.9375 / 0.9 = 1.04 > 1
```

Therefore, ~Tea → Coffee is POSITIVELY ASSOCIATED.

### INTEREST (Same as Lift)
```
              P(X,Y)
Interest = -----------
           P(X) × P(Y)
```

### PS (Piatetsky-Shapiro)
```
PS = P(X,Y) - P(X) × P(Y)
```

- PS = 0: Independent
- PS > 0: Positively correlated
- PS < 0: Negatively correlated

### PHI-COEFFICIENT (φ)
```
                     P(X,Y) - P(X) × P(Y)
φ-coefficient = --------------------------------
                √[P(X)(1-P(X))P(Y)(1-P(Y))]
```

Similar to correlation coefficient, ranges from -1 to +1.


### OBJECTIVE VS. SUBJECTIVE MEASURES

**OBJECTIVE MEASURES:**
- Rank patterns based on statistics computed from data
- Examples: support, confidence, lift, Gini, mutual information,
  Jaccard, Laplace, etc.

**SUBJECTIVE MEASURES:**
- Rank patterns according to user's interpretation
- A pattern is subjectively interesting if:
  - It CONTRADICTS the expectation of a user
  - It is ACTIONABLE (can lead to profitable decisions)


---

## 9. EXAMPLES AND PROBLEMS

### PROBLEM 1: Computing Support and Confidence

Given transactions:

| TID | Items |
|-----|-------|
| 1 | A, B, C |
| 2 | A, B, D |
| 3 | A, B, C, D |
| 4 | B, C, D |
| 5 | A, C, D |

Compute support and confidence for: {A,B} → {C}

**Solution:**
- σ({A,B,C}) = 2 (TID 1 and TID 3)
- σ({A,B}) = 3 (TID 1, 2, and 3)
- |T| = 5

Support = σ({A,B,C})/|T| = 2/5 = 0.4 = 40%
Confidence = σ({A,B,C})/σ({A,B}) = 2/3 = 0.67 = 67%

---

### PROBLEM 2: Apriori Algorithm

Given items: {A, B, C, D, E} with minsup = 2

Transactions:

| TID | Items |
|-----|-------|
| 1 | A, B, D |
| 2 | B, C, E |
| 3 | A, B, C, E |
| 4 | B, E |
| 5 | A, B, C, E |

Find all frequent itemsets.

**Solution:**

**Step 1: Count 1-itemsets**

| Item | Count | Frequent? |
|------|-------|-----------|
| A | 3 | Yes |
| B | 5 | Yes |
| C | 3 | Yes |
| D | 1 | No (prune) |
| E | 4 | Yes |

L1 = {A, B, C, E}

**Step 2: Generate and count 2-itemsets**

| Itemset | Count | Frequent? |
|---------|-------|-----------|
| {A,B} | 3 | Yes |
| {A,C} | 2 | Yes |
| {A,E} | 2 | Yes |
| {B,C} | 3 | Yes |
| {B,E} | 4 | Yes |
| {C,E} | 3 | Yes |

L2 = {{A,B}, {A,C}, {A,E}, {B,C}, {B,E}, {C,E}}

**Step 3: Generate and count 3-itemsets**
Candidates: {A,B,C}, {A,B,E}, {A,C,E}, {B,C,E}

| Itemset | Count | Frequent? |
|---------|-------|-----------|
| {A,B,C} | 2 | Yes |
| {A,B,E} | 2 | Yes |
| {A,C,E} | 2 | Yes |
| {B,C,E} | 3 | Yes |

L3 = {{A,B,C}, {A,B,E}, {A,C,E}, {B,C,E}}

**Step 4: Generate and count 4-itemsets**
Candidate: {A,B,C,E}

| Itemset | Count | Frequent? |
|---------|-------|-----------|
| {A,B,C,E} | 2 | Yes |

L4 = {{A,B,C,E}}

**Step 5:** Generate 5-itemsets - Cannot (only 4 frequent items)

**Final frequent itemsets:** L1 ∪ L2 ∪ L3 ∪ L4

---

### PROBLEM 3: Rule Generation from Frequent Itemset

Given frequent itemset: {A, B, C, D}
- σ({A,B,C,D}) = 100
- σ({A,B,C}) = 150
- σ({A,B,D}) = 120
- σ({A,C,D}) = 200
- σ({B,C,D}) = 125

Generate rules with 1-item consequent and calculate confidence.

**Solution:**

Rule: ABC → D
c = σ({A,B,C,D})/σ({A,B,C}) = 100/150 = 0.67

Rule: ABD → C
c = σ({A,B,C,D})/σ({A,B,D}) = 100/120 = 0.83

Rule: ACD → B
c = σ({A,B,C,D})/σ({A,C,D}) = 100/200 = 0.50

Rule: BCD → A
c = σ({A,B,C,D})/σ({B,C,D}) = 100/125 = 0.80

If minconf = 0.6, accepted rules: ABC→D (0.67), ABD→C (0.83), BCD→A (0.80)

---

### PROBLEM 4: Lift Calculation

Given contingency table:

```
           | Buy Y | ~Buy Y | Total
    -------|-------|--------|-------
    Buy X  |  400  |   100  |  500
   ~Buy X  |  200  |   300  |  500
    -------|-------|--------|-------
    Total  |  600  |   400  | 1000
```

Calculate Lift for X → Y.

**Solution:**
P(Y|X) = 400/500 = 0.8
P(Y) = 600/1000 = 0.6

Lift = P(Y|X)/P(Y) = 0.8/0.6 = 1.33

Since Lift > 1, X and Y are POSITIVELY CORRELATED.
Buying X increases the likelihood of buying Y.

---

### PROBLEM 5: Anti-Monotone Property Quiz

Given itemset L = {A,B,C,D}, does the following hold?
c(ABC → D) ≥ c(AB → CD) ≥ c(A → BCD)

**Answer:** YES, this holds!

**Proof:**
```
c(ABC → D) = σ(ABCD)/σ(ABC)
c(AB → CD) = σ(ABCD)/σ(AB)
c(A → BCD) = σ(ABCD)/σ(A)
```

Since A ⊆ AB ⊆ ABC, we have σ(A) ≥ σ(AB) ≥ σ(ABC)
(anti-monotone property of support)

Therefore: σ(ABCD)/σ(ABC) ≥ σ(ABCD)/σ(AB) ≥ σ(ABCD)/σ(A)

The confidence is anti-monotone with respect to consequent size
for rules generated from the SAME frequent itemset.


---

## 10. SUMMARY

### KEY CONCEPTS

1. **ASSOCIATION RULE:** X → Y (co-occurrence, not causality)

2. **METRICS:**
   - Support: s(X→Y) = σ(X∪Y)/|T| (fraction of transactions with X and Y)
   - Confidence: c(X→Y) = σ(X∪Y)/σ(X) (conditional probability)

3. **TWO-STEP APPROACH:**
   - Step 1: Frequent Itemset Generation
   - Step 2: Rule Generation

4. **APRIORI PRINCIPLE:**
   - If itemset is frequent, all subsets are frequent
   - If itemset is infrequent, all supersets are infrequent
   - Based on ANTI-MONOTONE property of support

5. **APRIORI ALGORITHM:**
   - Level-wise search (k-itemsets before (k+1)-itemsets)
   - Generate candidates from frequent itemsets
   - Prune candidates with infrequent subsets
   - Count support by database scan

6. **RULE GENERATION:**
   - Generate rules from frequent itemsets
   - Confidence is anti-monotone w.r.t. consequent size
     (for rules from same itemset)

7. **RULE PRUNING:**
   - Confidence/Support thresholds
   - Multiple minimum support for skewed data
   - Statistical measures (Lift, Interest, PS, φ)

8. **INTERESTINGNESS MEASURES:**
   - Lift = P(Y|X)/P(Y)
     - Lift = 1: Independent
     - Lift > 1: Positively correlated
     - Lift < 1: Negatively correlated

### FORMULAS TO REMEMBER
```
Support:     s(X) = σ(X)/|T|
Confidence:  c(X→Y) = σ(X∪Y)/σ(X) = s(X∪Y)/s(X)
Lift:        Lift(X→Y) = c(X→Y)/s(Y) = s(X∪Y)/(s(X)×s(Y))
Interest:    Interest(X,Y) = P(X,Y)/(P(X)×P(Y))
PS:          PS = P(X,Y) - P(X)×P(Y)

Number of itemsets: 2^d
Number of rules: 3^d - 2^(d+1) + 1
```

### COMMON PITFALLS
1. Confusing implication with causality
2. Using only confidence without considering base rate (need Lift)
3. Setting minsup too high (miss rare but important patterns)
4. Setting minsup too low (too many patterns, computationally expensive)
5. Ignoring statistical significance of rules

---

## QUICK REVIEW CHECKLIST

### Basic Definitions
- [ ] Can I define support: s(X) = σ(X)/|T|?
- [ ] Can I define confidence: c(X→Y) = σ(X∪Y)/σ(X)?
- [ ] Do I understand that implication means CO-OCCURRENCE, not CAUSALITY?
- [ ] Can I identify frequent itemsets given a minsup threshold?

### Apriori Principle & Algorithm
- [ ] Can I state the Apriori principle (if itemset infrequent, all supersets infrequent)?
- [ ] Do I understand the anti-monotone property of support?
- [ ] Can I perform candidate generation (Fk-1 × Fk-1 method)?
- [ ] Can I apply candidate pruning (check all k-1 subsets)?
- [ ] Can I trace through Apriori algorithm step by step?

### Support Counting
- [ ] Do I understand brute force complexity: O(NMw)?
- [ ] Do I know how hash tree improves support counting?
- [ ] Can I calculate theoretical number of itemsets: 2^d?
- [ ] Can I calculate number of association rules: 3^d - 2^(d+1) + 1?

### Rule Generation
- [ ] Do I understand rule generation from frequent itemsets?
- [ ] Do I know confidence is anti-monotone w.r.t. consequent size (same itemset)?
- [ ] Can I generate rules that meet minconf threshold?

### Interestingness Measures
- [ ] Can I calculate Lift = P(Y|X)/P(Y)?
- [ ] Do I know Lift = 1 means independent?
- [ ] Do I know Lift > 1 means positively correlated?
- [ ] Do I know Lift < 1 means negatively correlated?
- [ ] Can I interpret a contingency table for rule evaluation?

### Problem Solving
- [ ] Can I compute support and confidence from transaction data?
- [ ] Can I identify which rules satisfy minsup and minconf?
- [ ] Can I calculate Lift and interpret the result?

---

*END OF NOTES*
