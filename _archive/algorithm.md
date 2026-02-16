# Genetic Algorithm Description

This document describes the genetic algorithm implementation used in the
classifier system simulations.

## General Overview

The genetic algorithm takes as input a classifier system and evolves it
over time. Some classifiers, those with high fitness or large frequency,
are chosen for mating and reproduction. Others, those with low strength
or little use, are chosen for elimination. Thus, stronger classifiers
which yield a high return are the survivors.

Reproduction of classifiers involves selecting pairs of "parents" via a
roulette wheel. Stronger or more fit classifiers occupy a good percentage
of the wheel and thus have a greater chance of being selected. A pair
produces two offspring that have a portion of each parent's genes. Also,
there may be some mutation of genes.

Keeping a constant number of classifiers in a system entails replacing
classifiers with new offspring. Only a subset of the classifier systems
are considered for elimination. Then, a certain proportion (possibly all)
are analyzed to find the "worst" (i.e., in terms of strength) among them.
A pair considered to be the worst after some sampling is then replaced by
a pair of new offspring.

## Algorithm Parameters

The inputs to the genetic algorithm are:

- **CS** = the classifier system
  - `CS_c` is the consume classifier system
  - `CS_e` is the exchange/trade classifier
  - In the code: `ctype=1` implies trade, `ctype=2` implies consume

- **n_select** = the number of pairs of classifiers chosen for reproduction

- **p_cross** = the probability that crossover occurs

- **p_mutation** = the probability that mutation occurs

- **crowdingfactor** = the number of times that a search for the worst
  (i.e., has lowest strength) classifier is conducted

- **crowdingsubpop** = the proportion of the classifier system used in
  a search for the worst classifier

- **n_class** = the number of classifiers in CS

- **l_cond** = condition length (6 for trade, 3 for consume classifiers)

- **last** = a matrix giving indices for the consume classifier systems
  for rewarding bids in period t-1 (when the current period is t)

- **iteration** = the current period

- **S_min** (uratio[0]) = the first criterion for choosing the subset
  of candidates for killing

- **f_u** (uratio[1]) = the second criterion for choosing candidates
  for killing based on usage frequency

## Kill Eligibility

Only a subset of the classifier system is allowed to be "eliminated."
Classifiers are eligible for elimination if:

1. Strength less than S_min, OR
2. Used less than f_u × max times (where max is the maximum usage count)

If the number of classifiers to be replaced (2 × n_select) exceeds the
number satisfying these constraints, n_select is adjusted downward.

## Main Loop

Given that n_select is positive, the following steps are performed
n_select times:

### 1. Select Candidate Parents

Spin a roulette wheel `p_used × n_class` times without replacement,
choosing candidate classifiers for mating according to the number of
times they have been used. This selects the most-used classifiers as
parent candidates.

### 2. Select Parent Pair

Spin a roulette wheel 2 times with replacement to get a pair of parents
from the candidates.

### 3. Crossover Decision

Generate a random number from a uniform distribution. If the random
number is less than p_cross, then crossover in reproduction will occur.

### 4. Crossover Point

If there is crossover, choose a number between 1 and l_cond (6 for trade
classifiers, 3 for consume classifiers). Call this j_cross.

### 5. Reproduction

Create two children from the parent pair:

- **Child k** has:
  - Bits 1 through j_cross from parent k
  - Bits (j_cross+1) through l_cond from parent ℓ (where k≠ℓ)
  - Action bit from parent k

**Example**: If the parents are:

```
Parent 1: # 0 1 # 1 0 | 1
Parent 2: 1 # 0 0 1 # | 0
```

And the crossover point is 2, then the children are:

```
Child 1:  # 0 | 0 0 1 # | 1
Child 2:  1 # | 1 # 1 0 | 0
```

### 6. Mutation

Generate random numbers for each gene position. If a random number is
less than p_mutation, mutation occurs:

- For condition genes: randomly change:
  - `#` (wildcard) to 0 or 1
  - `0` to `#` or 1
  - `1` to `#` or 0

All alternatives occur with equal probability.

### 7. Crowding Replacement

Choose a candidate from those eligible to be replaced:

1. Pick `crowdingsubpop` random indices from the eligible classifiers
2. Find the one with the worst (lowest) strength
3. Count similar condition bits and dissimilar action bits between this
   classifier and the replacing child
4. Store the index and match count
5. Repeat `crowdingfactor` times
6. Replace the classifier with the overall worst strength

## Post-Processing

After replacing classifiers with new offspring:

1. Report statistics on crossover and mutation counts
2. Rescale classifier strengths if any are negative

## GA Variants

### GA3 (ga_v3)

- Single-point crossover
- Kill eligibility based on strength and usage
- Standard roulette wheel selection

### GA4 (ga_v4)

- Two-point crossover
- Same kill eligibility as GA3
- Creates more genetic diversity

## Classifier Encoding

### Trade Classifiers (l_cond = 6)

```
[c1, c2, c3 | c4, c5, c6 | action | strength | trades | used | iteration]
 ^^^^^^^^^   ^^^^^^^^^     ^^^^
 own good    other's good  0=no trade, 1=trade
```

### Consume Classifiers (l_cond = 3)

```
[c1, c2, c3 | action | strength | used | iteration]
 ^^^^^^^^^   ^^^^
 own good    0=no consume, 1=consume
```

### Condition Encoding

- `-1` = wildcard (#) - matches any value
- `0` = match only 0
- `1` = match only 1
