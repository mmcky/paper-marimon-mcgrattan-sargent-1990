---
title: Money as a Medium of Exchange in an Economy with Artificially Intelligent Agents
authors:
  - name: Ramon Marimon
    affiliation: University of Minnesota, Minneapolis, MN 55455, USA
  - name: Ellen McGrattan
    affiliation: Duke University, Durham, NC 27706, USA
  - name: Thomas J. Sargent
    affiliation: Hoover Institution, Stanford, CA 94305, USA
date: 1990-01-01
venue: Journal of Economic Dynamics and Control 14 (1990) 329-373
bibliography: references.bib
---

# Abstract

We study the exchange economies of {cite}`kiyotaki1989` in which agents must use a commodity or fiat money as a medium of exchange if trade is to occur. Our agents are artificially intelligent and are modeled as using classifier systems to make decisions. In the assignment of credit within the classifier systems, we introduce some innovations designed to study sequential decision problems in a multi-agent environment. For most economies that we have simulated, trading and consumption patterns converge to a stationary Nash equilibrium even if agents start with random rules. In economies with multiple equilibria, the only equilibrium that emerges in our simulations is the one in which goods with low storage costs play the role of medium of exchange (i.e., the 'fundamental equilibrium' of Kiyotaki and Wright).

## 1. Introduction

{cite}`kiyotaki1989` studied how three classes of rational agents would interact within a Nash-Markov equilibrium in a world of repeated 'Wicksell triangles'. This paper studies how a collection of artificially intelligent agents would learn to coordinate their activities if they were thrust into the economic environment described by Kiyotaki and Wright. Our agents' behavior is determined by {cite}`holland1975` classifier systems.

Thus, while Kiyotaki and Wright studied stationary equilibria in which beliefs about 'media of exchange' are consistent with trading patterns, we study economies in which particular commodities emerge as media of exchange. We also study an economy (Economy C) in which a good from which no agent derives utility emerges as fiat money.

We want to learn whether our artificially intelligent agents can learn to play a Markovian Nash equilibrium of the Kiyotaki-Wright model. When there are multiple Nash equilibria (e.g., the 'fundamental' and 'speculative' equilibria of Kiyotaki and Wright), we want to know whether the system might converge to some but not others of these equilibria. In addition, if classifier systems do converge to Nash-Markov equilibria, they might be used to compute equilibria in other economies for which it is difficult to obtain analytic solutions. To this end, we also study enlarged versions of the Kiyotaki-Wright model.

A *classifier system* is a collection of potential decision rules, together with an accounting system for selecting which rules to use. The accounting system credits rules generating good outcomes and debits rules generating bad outcomes. The system is designed to select a 'co-adaptive' set of rules that work well over the range of situations that an agent typically encounters. In a multiple-agent environment like Kiyotaki and Wright's, the range of situations encountered by one agent depends on the actions taken by other agents. This means that the collections of rules used by the collection of agents must co-adapt jointly.

We study two sorts of classifier systems, with the distinction between them being induced by the fact that for many problems enumerating all possible rules would require a very long list:

1. **Complete enumeration classifier system**: A classifier system in which a complete enumeration of all possible rules is carried along. For many problems, it is not feasible to use a complete enumeration classifier system because the state and action spaces are so large that the set of all possible rules is much too big. For us, a complete enumeration is tractable because the Kiyotaki-Wright model has low-dimensional state and action spaces.

2. **Genetic algorithm classifier system**: Designed for situations in which it is not efficient to carry along a complete enumeration of decision rules. A modified version of the genetic algorithm of Holland is used as a device for periodically eliminating some rules and injecting new rules into the population of rules to be operated on by the accounting system. The genetic-algorithm version of the classifier models learning in the face of unforeseen contingencies. When an unprecedented state arises which is not covered by the existing set of rules, the system contains a procedure for manufacturing and experimenting with new rules that apply in the new situation.

In this paper, we set up classifier systems for the Kiyotaki-Wright environment and simulate them on a computer. We also formulate definitions of equilibrium and stability for classifier systems, and pose some convergence questions. Although we suggest possible convergence theorems, we prove no theorems in this paper.

**Paper Organization:**
- Section 2 describes the Kiyotaki-Wright environment
- Section 3 describes the classifier systems without genetics
- Section 4 describes behavior in the stationary equilibria of Kiyotaki and Wright and also the types of classifier systems that could support that behavior
- Section 5 defines stability criteria for sets of classifier systems, and discusses how these definitions can be used to formulate questions about whether systems of classifier systems converge to a stationary equilibrium
- Section 6 describes classifier systems operating with genetics
- Section 7 describes our simulations of systems of classifier systems (including economies with five types of agents and five goods)
- Section 8 concludes the paper

## 2. The Kiyotaki-Wright Environment

There are three types of agents, with types being indexed by $i = 1, 2, 3$. Type $i$ agents get utility only from consuming type $i$ good. Type $i$ agent has access to a technology for producing type $i^*$ good, where $i^* \neq i$. We initially specify $(i, i^*)$ according to Kiyotaki-Wright's 'model A', namely, as follows:

| $i$ | $i^*$ |
|-----|-------|
| 1   | 2     |
| 2   | 3     |
| 3   | 1     |

This specification assumes no 'double coincidence of wants' and seems to call for a multilateral trading arrangement.

All goods are indivisible. Each agent can store one and only one unit of only one good from one period to the next. When an agent of type $i$ consumes good $i$ at time $t$, he immediately produces good $i^*$, which he carries over to the next period. The net utility to an agent of type $i$ of consuming good $y$ and producing good $i^*$ is given by $u_i(y)$. We assume that an agent of type $i$ does not know his utility function, but does recognize utility when he experiences it.

The goods are costly to store. Storing good $k$ ($k = 1, 2, 3$) from $t$ to $t+1$ imposes costs at $t$ of $s_k$. Following Kiyotaki and Wright, we assume that $s_3 > s_2 > s_1 > 0$. We summarize this cost function by saying that $s(y)$ is the one-period cost of storing one unit of good $y$. We assume that individuals do not know this cost function, but that they do recognize costs when they bear them.

The economy and each agent within it live forever. There are large and equal numbers of agents of each type. We modify Kiyotaki and Wright's model by assuming that each agent cares about his long-run average level of utility. (Kiyotaki and Wright assumed that agents have preferences ordered by expected discounted future utilities.)

Each period, there is a random matching process that assigns each and every agent in the economy to a pair with one and only one other agent in the economy. The random matching technology matches agents without regard to type. Only pairs of agents matched together can trade at a point in time.

The economy begins at $t = 0$ with agents being endowed with an arbitrary and randomly generated initial distribution of holdings of goods. At each date $t \geq 0$, each agent $i$ has to make two decisions sequentially:

1. First, given the good he is holding and given the good held by the partner with whom he is matched at $t$, he must decide whether to propose to trade. Trade occurs only if both parties propose to do so.

2. Second, the agent must decide whether or not to consume the good with which he exits the trading process. If he doesn't consume, he simply carries the good into period $t + 1$, experiencing cost $s_k$. If an agent of type $i$ does consume good $y$ at $t$, he experiences net utility $u_i(y_t)$, produces good $i^*$, and experiences carrying costs $s_{i^*}$.

We assume that $u_i(k) = 0$ if $k \neq i$ and $u_i(i) = u_i > 0$. We follow Kiyotaki and Wright both in adopting their specification of the physical environment and in considering only Markov strategies, but we drop the assumption of rational agents. Instead, our agents are assumed to be 'artificially intelligent', in the sense that they use versions of the *classifier system* introduced by {cite}`holland1986`.

### 2.1 Notation

There is a collection of agents $\mathscr{A} = \{1, 2, \ldots, A\}$. A typical element of $\mathscr{A}$ is denoted $a$. Kiyotaki and Wright assumed that there was a continuum of each type of agent, while we assume that there is a finite number of each type. The first $A_1$ agents are of type I, the next $A_2$ are of type II, and the following $A_3$ are of type III. Here $A = 3A_i$.

At the beginning of time $t$, agent $a$ is carrying good $x_{at}$. The variable $x_{at}$ characterizes the pre-match state of agent $a$ at time $t$. There is a random matching process which each period matches each agent $a \in \mathscr{A}$ with a distinct agent $\rho_t(a) \in \mathscr{A}$. For each agent in each period, the matching process induces a function $\rho_t(a): \mathscr{A} \rightarrow \mathscr{A}$. After the matching process, the pre-trade state of agent $a$ is $(x_{at}, x_{\rho_t(a)t})$.

The pair $(x_{at}, x_{\rho_t(a)t}) = z_{at}$ records what agent $a$ is carrying and what the agent $\rho_t(a)$ with whom $a$ is matched at $t$ is carrying.

### 2.2 Trading and Consumption Decisions

At $t$, after being matched with agent $\rho_t(a)$, agent $a$ decides whether or not to propose to trade. We let $\lambda_{at}$ denote the trading decision of agent $a$ at time $t$, where

$$
\lambda_{at} = \begin{cases}
1 & \text{if } a \text{ proposes to trade } x_{at} \text{ for } x_{\rho_t(a)t} \\
0 & \text{if } a \text{ refuses to trade}
\end{cases}
$$ (eq-trade-decision)

Similarly, $\lambda_{\rho_t(a)t}$ summarizes the trading decision of the agent $\rho_t(a)$ with whom $a$ is paired at $t$. Trade takes place if and only if $\lambda_{at} \cdot \lambda_{\rho_t(a)t} = 1$.

Let $x_{at}^*$ denote the post-trade (but pre-consuming decision) holdings of agent $a$ at $t$. We then have that

$$
x_{at}^* = (1 - \lambda_{at}\lambda_{\rho_t(a)t})x_{at} + \lambda_{at}\lambda_{\rho_t(a)t}x_{\rho_t(a)t}
$$ (eq-post-trade)

After leaving the trading process with holdings $x_{at}^*$, agent $a$ must decide whether to consume $x_{at}^*$ or to carry it into the next period. We let $y_{at}$ denote the consumption decision of agent $a$ at $t$ where

$$
y_{at} = \begin{cases}
1 & \text{if } a \text{ decides to consume } x_{at}^* \\
0 & \text{if } a \text{ decides not to consume}
\end{cases}
$$ (eq-consume-decision)

If agent $a$ decides to consume, he automatically produces good $f(a)$, which he carries into $(t + 1)$. From the specification of $i^*$ as a function of $i$ described above for Kiyotaki and Wright's model A, we have that $f(a)$ is a good of type 2 if $a$ is a type I agent, $f(a)$ is a good of type 3 if $a$ is a type II agent, and $f(a)$ is a good of type 1 if $a$ is a type III agent. It follows that beginning-of-period holdings of $a$ at time $t + 1$ are described by

$$
x_{a,t+1} = y_{at} f(a) + (1 - y_{at})((1 - \lambda_{at}\lambda_{\rho_t(a)t})x_{at} + \lambda_{at}\lambda_{\rho_t(a)t}x_{\rho_t(a)t})
$$ (eq-holdings-evolution)

## 3. Classifier Systems for the Kiyotaki-Wright Environment

We now describe the *classifier systems* that agents use to make their trading and consumption decisions. Agents sequentially use two interconnected classifier systems each period:

1. A first **trading classifier system** receives input in the form of the pre-trade state $z_{at} = (x_{at}, x_{\rho_t(a)t})$. This classifier system determines the trading decision $\lambda_{at}$, which interacts with the trading decision $\lambda_{\rho_t(a)t}$ of the agent $\rho_t(a)$ with whom $a$ is paired at $t$ to determine $x_{at}^*$ [see eq. {eq}`eq-post-trade`].

2. A second **consumption classifier system** takes $x_{at}^*$ as input and determines the consumption decision $y_{at}$.

The two classifier systems are used sequentially in this way each period, and their accounting systems are interconnected in ways to be described below.

:::{figure} figures/concepts/classifier_flow.png
:name: fig-classifier-flow
:align: center

Example of flow of payments in classifier systems for a type I agent. The message containing the agent's holdings and partner's holdings is matched against the classifier list, and the highest-strength matching classifier determines the action.
:::

### 3.1 Components of a Classifier System

A classifier system consists of the following objects:

1. A collection of trinary strings, called 'classifiers'.

2. A system for interpreting or decoding the strings or classifiers as instructions mapping states into decisions. The first part of a string encodes a particular state or condition, while the second part encodes a particular action. Thus, an individual classifier or string is just an encoding of a single (state, action) pair. For a given state, there can be many classifiers present in a classifier system.

3. A list of 'strengths' attached to each classifier at each point in time $t = 0, 1, \ldots$

4. A system for reading in external 'messages' and determining which classifiers are 'matched' with the current state.

5. An 'auction' system for determining which classifier gets to make the decision given the state.

6. An accounting system for updating strengths of classifiers.

7. (Optional) A 'genetic algorithm' which periodically eliminates some strings and creates new strings.

### 3.2 Encoding

We let integers stand for goods: 1 stands for 'good 1', 2 for 'good 2', and 3 for 'good 3'. We consider trinary codes for describing which good an agent is holding, shown in Table 1.

:::{table} Encoding of goods in classifiers
:label: tab-encoding

| Code | Meaning |
|------|---------|
| 1 0 0 | Good 1 |
| 0 1 0 | Good 2 |
| 0 0 1 | Good 3 |
| # 0 0 | Not good 1 |
| 0 # 0 | Not good 2 |
| 0 0 # | Not good 3 |
:::

The coding is written in the trinary alphabet $(1, 0, \#)$, where $\#$ means 'don't care'. Because the '$\#$' symbol can match either 0 or 1, the code '$\#\ 0\ 0$' matches both '$1\ 0\ 0$' and '$0\ 0\ 0$'. The code '$0\ 0\ 0$' does not match any good (only one good can be held), so '$\#\ 0\ 0$' in effect matches all goods except good 1.

### 3.3 Exchange Classifier System

To see how these codes are applied to encode particular trading classifiers, we consider the following two classifiers:

| Own storage | Partner storage | Trading decision |
|-------------|-----------------|------------------|
| 1 0 0       | 0 0 1           | 1                |
| 1 0 0       | # # 0           | 0                |

The first classifier instructs an agent who is carrying good 1 and who is matched with someone carrying good 3 to offer to trade. The second classifier instructs an agent who is carrying good 1 not to trade if he is matched with someone who is not carrying good 3.

The exchange classifier system of agent $a$ consists of a list of such classifiers. Let $e = 1, 2, \ldots, E_e$ index this collection of classifiers. A given classifier system has a fixed number of classifiers.

For the exchange classifier system, how many distinct classifiers are possible? Evidently, $3^3 \times 3^3 \times 2 = 1458$. However, most of these classifiers are redundant in the sense that only a subset of them is needed to represent all possible trading decision rules defined on the state space $z_{at} = (x_{at}, x_{\rho_t(a)t})$ when there are three goods. All rules can be written in terms of pairs of conditions drawn from Table 1. Thus, only $6 \times 6 \times 2 = 72$ strings are required to represent the complete set of possible rules for this system.

Assigned to each exchange classifier $e \in \{1, 2, \ldots, E_e\}$ is a strength, denoted $S_e^a(t)$. The strength $S_e^a(t)$ evolves over time in a way determined by the accounting system.

For a given state or 'condition' $z_{at} = (x_{at}, x_{\rho_t(a)t})$, there is a collection of classifiers within the classifier system whose conditions are satisfied. We denote the set of such classifiers by $M_e(z_{at})$, defined as

$$
M_e(z_{at}) = \{e : z_{at} \text{ matches the condition part of classifier } e\}
$$ (eq-matched-classifiers)

The members of $M_e(z_{at})$ form a class of potential 'bidders' in an 'auction' whose purpose is to determine which classifier makes the decision of agent $a$ at time $t$. When state $z_{at}$ is encountered at time $t$ by agent $a$, the classifier belonging to $M_e(z_{at})$ that has the highest strength makes the decision. Let $e_t(z_{at})$ denote the index of the classifier to be used in deciding whether to trade at $t$. Then,

$$
e_t(z_{at}) = \arg\max\{S_e^a(t) : e \in M_e(z_{at})\}
$$ (eq-auction-winner)

We denote the action (trade or no trade) taken by classifier $e_t(z_{at})$ as $\lambda_{at}$. Equations {eq}`eq-matched-classifiers` and {eq}`eq-auction-winner` describe the 'auction system' by which the highest strength rule that applies in a given state is given the right to decide for agent $a$ at $t$.

### 3.4 Consumption Classifier System

The consumption classifier system is a collection of trinary strings of length 4. The first three positions encode $x_{at}^*$ using the same code that was described in Table 1. The condition part of the strings is written in terms of the trinary alphabet $(0, 1, \#)$. The fourth position of a consumption string is the 'action' part, taking values of 1 (meaning consume) and 0 (meaning don't consume).

We let consumption classifier strings be indexed by $c \in \{1, 2, \ldots, C_c\}$. The strength assigned to classifier $c$ of agent $a$ is $S_c^a(t)$. The set of 'matched' classifiers is denoted by $M_c(z_{at})$, where

$$
M_c(z_{at}) = \{c : x_{at}^* \text{ matches the condition part of classifier } c\}
$$ (eq-consume-matched)

Let $c_t(z_{at})$ denote the classifier that makes the consumption decision at $t$. The highest-strength classifier gets to make the decision:

$$
c_t(z_{at}) = \arg\max\{S_c^a(t) : c \in M_c(z_{at})\}
$$ (eq-consume-winner)

### 3.5 Accounting System

We attach to each exchange classifier $e$ a 'counter' $\tau_e(t)$ which records the cumulative number of times that classifier $e$ has won the auction as of date $t$. We shall change the strength of classifier $e$ only when it actually wins the auction and thereby gets to make the exchange decision. The counter $\tau_e(t)$ for classifier $e$ is defined recursively in terms of the indicator $I_e^a(t)$, which records whether classifier $e$ wins the auction:

$$
I_e^a(t) = \begin{cases}
1 & \text{if } e \text{ wins the auction (unless classifier } e \\
  & \text{sets } \lambda_{at} = 1 \text{ while } \lambda_{\rho_t(a)t} = 0, \\
  & \text{so that the offer to trade is not reciprocated)} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\tau_e^a(t) = \sum_{s=0}^{t} I_e^a(s) + 1
$$ (eq-counter)

Notice that we initialize the counter of each classifier at unity. The strength of classifier $e$ at $t$ will be represented as $\bar{S}_{\tau_e(t)}^{a,e}$.

Similarly, we attach to each consumption classifier $c$ a 'counter' $\tau_c(t)$ which records the cumulative number of times that classifier $c$ has won the auction as of date $t$.

#### Bid Functions

At date $t$, if classifier $e$'s condition is matched [i.e., if $e \in M_e(z_{at})$], then classifier $e$ makes a bid of $b_1(e)S_e^a(t)$, where $b_1(e)$ is a positive fraction that can depend on $e$. If classifier $e$ wins the auction, its bid will be deducted from its strength. The winning bid will be allocated to augment the strength of other classifiers whose actions drove the system to the state that satisfied $e$'s condition.

We choose the particular bid function

$$
b_1(e) = b_{10} + b_{11}\alpha_e
$$ (eq-bid1)

where $b_{10}$ and $b_{11}$ are positive constants adding up to less than one, and $\alpha_e$ is a fraction which is proportional to the specificity of a particular classifier. In particular, we choose

$$
\alpha_e = \frac{1}{1 + \text{number of } \#\text{'s in the string}}
$$

Similarly, we define a function $b_2(c)$ as

$$
b_2(c) = b_{20} + b_{21}\alpha_c
$$ (eq-bid2)

where $\alpha_c = 1/(1 + \text{number of } \#\text{'s in the string})$. By the above choices of $b_1(e)$ and $b_2(c)$, we favor specific rules over more general rules that can be activated by a particular state.

#### Strength Update Equations

The bid of the winning exchange classifier at $t$ is paid to the winning consumption classifier at $t - 1$, which is the classifier that is to be credited with setting the time $t$ state to $z_{at}$. The bid of the winning consumption classifier at $t$ is paid to the winning exchange classifier at time $t$, which is to be credited with setting the post-exchange state at $t$ at $x_{at}^*$, thereby giving the winning consumption classifier a chance to bid.

We represent these payments in terms of the following difference equations:

$$
\bar{S}_{\tau_c}^{a,c} = \bar{S}_{\tau_c-1}^{a,c} + \frac{1}{\tau_c}\left[U_a(y_{at}^c) + b_1(e_t)S_{\tau_e(t)}^{a,e_t} - b_2(c)S_{\tau_c-1}^{a,c} - \bar{S}_{\tau_c-1}^{a,c}\right]
$$ (eq-strength-consume)

$$
\bar{S}_{\tau_e}^{a,e} = \bar{S}_{\tau_e-1}^{a,e} + \frac{1}{\tau_e}\left[b_2(c_{t-1})S_{\tau_c(t-1)}^{a,c_{t-1}} - b_1(e)S_{\tau_e-1}^{a,e} - \bar{S}_{\tau_e-1}^{a,e}\right]
$$ (eq-strength-exchange)

In {eq}`eq-strength-consume`, $U_a(y_{at}^c)$ is the external payoff when the winning consumption classifier $c$ makes final consumption decision $y_{at}^c$. If the post-exchange state at $t$ is $x_{at}^*$, then we have

$$
U_a(y_{at}^c) = y_{at}^c[u_i(x_{at}^*) - s(f(a))] + (1 - y_{at}^c)s(x_{at}^*)
$$ (eq-external-payoff)

There are several features of {eq}`eq-strength-consume` and {eq}`eq-strength-exchange` that bear emphasizing:

1. Equations {eq}`eq-strength-consume` and {eq}`eq-strength-exchange` are recursive formulas that make $\bar{S}_{\tau_c}^{a,c}$ and $\bar{S}_{\tau_e}^{a,e}$ averages of past payoffs (external rewards plus bids received from other classifiers) minus payments (bids made to other classifiers). Use of cumulative average net payoffs in this way (as opposed to total payoffs) is a departure from the existing literature on classifiers.

2. Notice how the term $I_c^a(t)b_1(e_t)S_{\tau_e(t)}^{a,e_t}$ expresses the condition that only the winning exchange classifier at $t$ pays the winning consumption classifier at $t - 1$.

3. Notice how the use of the counters $\tau_c^a(t)$ and $\tau_e^a(t)$ causes changes to be made only to the strengths attached to the winning classifiers.

## 4. Classifier Systems for Supporting Kiyotaki and Wright's Stationary Equilibria

Before describing the simulation results, we discuss behavior in the stationary equilibria of Kiyotaki and Wright and also the types of classifier systems that could support that behavior.

### 4.1 Equilibrium Probabilities

We can characterize equilibria in terms of the following probabilities:

$$
\pi_t^i(k) = \text{probability that a type } i \text{ agent is holding good } k \text{ at } t
$$

$$
\pi_t^h(k) = \text{probability that a randomly selected agent is holding good } k
$$

$$
\pi_t^i(jk) = \text{probability that type } i \text{ agent holds } j, \text{ meets } k, \text{ and trades}
$$

In a stationary equilibrium, these probabilities are constant over time.

### 4.2 The Fundamental Equilibrium

For their model A, Kiyotaki and Wright define as the *fundamental equilibrium* the trading pattern in which good 1, which has the lowest storage cost, serves as the general medium of exchange.

**Equilibrium trading patterns in the fundamental equilibrium:**
- Type I agents: only store good 2, exchange only for good 1
- Type III agents: only store good 1, exchange only for good 3
- Type II agents: half of the time store good 1 (exchange for good 2), half of the time store good 3 (exchange for good 1)

:::{figure} figures/a/exchange_pattern_type1.png
:name: fig-exchange-fundamental
:align: center

Exchange pattern diagram showing the trading pattern in the fundamental equilibrium. Good 1 (with lowest storage cost) serves as the medium of exchange.
:::

### 4.3 The Speculative Equilibrium

In speculative equilibria, goods with higher storage costs can serve as media of exchange under certain parameter configurations. This occurs when the probability of meeting someone with whom trade is beneficial is sufficiently high to overcome the higher storage costs.

## 5. Concepts of Convergence for Classifier Systems Playing Games

We formalize concepts of convergence and stability for multi-agent classifier systems.

### 5.1 Nash Equilibrium Definition

A collection of classifier systems $\{CS_a\}_{a \in \mathscr{A}}$ is in *Nash equilibrium* if no individual agent can improve their expected payoff by unilaterally changing their decision rules, given the decision rules of all other agents.

### 5.2 Stochastic Approximation

The strength update equations {eq}`eq-strength-consume` and {eq}`eq-strength-exchange` can be analyzed using stochastic approximation methods. The recursive averaging formulas make $\bar{S}_{\tau}$ converge to the expected value of net payoffs under appropriate conditions.

## 6. Incomplete Enumeration Classifiers and the Genetic Algorithm

When the state and action spaces are large, it is not feasible to enumerate all possible rules. We use a genetic algorithm to:

1. Periodically eliminate weak rules
2. Inject new rules through crossover and mutation
3. Handle unforeseen contingencies by manufacturing new rules

### 6.1 Genetic Operations

**Selection**: Roulette wheel selection weighted by classifier strength. Each classifier's probability of being selected is proportional to its strength.

**Crossover**: Two-point crossover with probability $p_{\text{cross}}$. Two parent classifiers exchange genetic material within a randomly selected interval.

:::{figure} figures/concepts/mating_process.png
:name: fig-mating-process
:align: center

The mating process for exchange classifiers. Two parent classifiers (binary strings of length 7) undergo two-point crossover, exchanging genetic material within the randomly drawn interval to produce two offspring classifiers.
:::

**Mutation**: Bit-flip mutation with probability $p_{\text{mutation}}$ per bit. Each bit in a classifier string has a small probability of being flipped.

**Crowding**: New classifiers replace similar weak classifiers to maintain diversity in the population. The crowding factor determines how many classifiers are compared when finding a replacement target.

### 6.2 Creating New Classifiers

When an unprecedented state arises not covered by existing rules, the system manufactures new rules:

1. Create a classifier with condition matching the new state
2. Initialize with random action and default strength
3. Apply genetic operations to evolve better rules

## 7. Simulation Results

The sets of parameters defining the economies under study are summarized in Table 4. We report whether the Nash-Markov stationary equilibrium of Kiyotaki and Wright is speculative or fundamental for our parameter settings.

Our economies differ from one another in parameter settings, and whether initial classifier systems contain a complete enumeration of states and actions or whether they are randomly generated, in which case the genetic algorithm described in Section 6 is active. For all the simulations, we keep the number of exchange classifiers fixed at 72 for each type of agent, and the number of consumption classifiers fixed at 12 for each type of agent.

:::{table} Parameter specifications for simulated economies
:label: tab-economies

| Economy | $s_1$ | $s_2$ | $s_3$ | Agents | Equilibrium Type |
|---------|-------|-------|-------|--------|------------------|
| A1      | 0.1   | 1     | 20    | 150    | Fundamental      |
| A2      | 0.1   | 1     | 4     | 150    | Fund. or Spec.   |
| B       | 0.05  | 0.5   | 2     | 150    | Fundamental      |
| C       | *     | *     | *     | 150    | Fiat Money       |
| D       | varies| varies| varies| 150    | 5-good economy   |
:::

### 7.1 Economy A1

:::{table} Parameter values for Economy A1.1
:label: tab-economy-a1-params

| Parameter | Value |
|-----------|-------|
| No. of agents | $A_i = 50$ |
| No. of classifiers | $E_e = 72$, $C_c = 12$ |
| Storage costs | $s_1 = 0.1$, $s_2 = 1$, $s_3 = 20$ |
| Utility | $u_i = 100$ |
| Initial strengths | $\bar{S}_e^a(0) = 0$, $\bar{S}_c^a(0) = 0$ |
| Bids | $b_{10} = 0.025$, $b_{11} = 0.025$, $b_{20} = 0.25$, $b_{21} = 0.25$ |
:::

Table 6 and {numref}`fig-holdings-distribution` report the results of a long simulation of Economy A1.1 (Economy A1 with complete enumeration of classifiers). The frequencies reported are ten-period moving averages ending at the indicated time periods.

:::{figure} figures/a/holdings_distribution.png
:name: fig-holdings-distribution
:align: center

Distribution of holdings for Economy A1.1. Each panel shows the percentage of time each agent type holds each good over the simulation. The distribution converges to the fundamental equilibrium where good 1 (with lowest storage cost) serves as the medium of exchange.
:::

The empirical frequencies converge closely to the theoretical equilibrium values shown in Table 5. In particular:
- Type I agents hold good 2 with probability approaching 1
- Type II agents hold goods 1 and 3 with approximately equal probability (0.5 each)
- Type III agents hold good 1 with probability approaching 1

:::{figure} figures/a/trade_prob_type1.png
:name: fig-trade-prob
:align: center

Trading probability matrix for Type 1 agents in Economy A1.1. The heatmap shows P(trade | own holding, partner holding). The pattern confirms convergence to the fundamental equilibrium trading strategy.
:::

### 7.2 Economy A2

Economy A2 uses parameter values for which both the fundamental and speculative equilibria exist. The storage costs are $s_1 = 0.1$, $s_2 = 1$, $s_3 = 4$ (compared to $s_3 = 20$ in Economy A1).

In all our simulations of Economy A2, the fundamental equilibrium emerges. This occurs both with complete enumeration classifiers (Economy A2.1) and with incomplete enumeration plus genetic algorithm (Economy A2.2).

### 7.3 Economy B

Economy B uses different storage cost parameters to test robustness. The fundamental equilibrium again emerges in all simulations.

### 7.4 Economy C: Fiat Money

In Economy C, a fourth good (fiat money) is introduced. This good:
- Provides no utility to any agent: $u_i(\text{fiat}) = 0$ for all $i$
- Has zero storage cost: $s_{\text{fiat}} = 0$

Under these conditions, fiat money emerges as the medium of exchange in our simulations. Agents learn to accept fiat money not for its intrinsic value (which is zero) but because they expect others to accept it in future trades.

### 7.5 Economy D: Five Agents, Five Goods

We extend the model to 5 types of agents and 5 goods to test scalability. The production and preference structure maintains the 'Wicksell triangle' pattern: type $i$ produces good $i+1$ (mod 5) and desires to consume good $i$.

With 5 goods, the complete enumeration approach requires $6^{10} \times 2 \approx 121$ million classifiers, which is not feasible. We therefore use only the genetic algorithm classifier system for this economy.

Despite the larger state space, classifier systems still converge to Nash equilibria, though convergence is slower. The good with lowest storage cost again emerges as the medium of exchange.

## 8. Conclusions

Our simulations demonstrate that:

1. **Convergence to Nash equilibrium**: Multi-agent systems of classifiers can learn to play Nash-Markov equilibria in the Kiyotaki-Wright environment.

2. **Selection among equilibria**: In economies with multiple equilibria, the fundamental equilibrium (in which goods with low storage costs serve as media of exchange) is the only one that emerges in our simulations.

3. **Fiat money emergence**: Fiat money can emerge as a medium of exchange even though it provides no direct utility, when it has lower storage costs than other goods.

4. **Scalability**: The approach scales to larger economies (5 goods, 5 types), though the genetic algorithm is necessary for computational tractability.

5. **Robustness**: Results are robust across different parameter configurations and whether complete enumeration or genetic algorithm classifiers are used.

Future work will focus on:
- Developing analytical convergence results for classifier systems
- Applying classifier systems to more complex economic environments
- Improving genetic algorithms for faster convergence in larger state spaces

## Acknowledgments

This research began with visits by Marimon and Sargent to the Santa Fe Institute. We thank Brian Arthur and John Holland for several helpful discussions about genetic algorithms at the Santa Fe Institute. We also thank Randall Wright and Nancy Stokey for helpful comments on an earlier draft. Sargent's research was supported by a grant from the National Science Foundation to the National Bureau of Economic Research. Marimon's research was supported by a grant from the National Science Foundation and by the National Fellows Program at the Hoover Institution.

## References

```{bibliography}
```
