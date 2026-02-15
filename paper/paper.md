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
acknowledgments: This research began with visits by Marimon and Sargent to the Santa Fe Institute. We thank Brian Arthur and John Holland for several helpful discussions about genetic algorithms at the Santa Fe Institute. We also thank Randall Wright and Nancy Stokey for helpful comments on an earlier draft. Sargent's research was supported by a grant from the National Science Foundation to the National Bureau of Economic Research. Marimon's research was supported by a grant from the National Science Foundation and by the National Fellows Program at the Hoover Institution. This paper is an abbreviated version of a Hoover Institution working paper with the same title, which is available from the authors upon request.
---

+++ {"part": "abstract"}
We study the exchange economies of {cite}`kiyotaki1989` in which agents must use a commodity or fiat money as a medium of exchange if trade is to occur. Our agents are artificially intelligent and are modeled as using classifier systems to make decisions. In the assignment of credit within the classifier systems, we introduce some innovations designed to study sequential decision problems in a multi-agent environment. For most economies that we have simulated, trading and consumption patterns converge to a stationary Nash equilibrium even if agents start with random rules. In economies with multiple equilibria, the only equilibrium that emerges in our simulations is the one in which goods with low storage costs play the role of medium of exchange (i.e., the 'fundamental equilibrium' of Kiyotaki and Wright).
+++

(sec-introduction)=
## 1. Introduction

{cite}`kiyotaki1989` studied how three classes of rational agents would interact within a Nash-Markov equilibrium in a world of repeated 'Wicksell triangles'. This paper studies how a collection of artificially intelligent agents would learn to coordinate their activities if they were thrust into the economic environment described by Kiyotaki and Wright. Our agents' behavior is determined by {cite}`holland1975` classifier systems.[^fn1]

[^fn1]: The way in which we model agents as searching for rewarding decision rules attributes much less knowledge and rationality to them than is typical in the literature on least-squares learning in the context of linear rational-expectations models. See {cite}`bray1982`, {cite}`bray1986`, {cite}`fourgeaud1986`, and {cite}`marcet1989a`. In the least-squares learning literature, agents are posited to be almost fully rational and knowledgeable about the system they operate within, lacking knowledge only about particular parameters in laws of motion of a set of variables exogenous to their own decisions, which they learn about through the recursive application of least squares. But given their estimates, agents are supposed to compute optimal decision rules by using dynamic programming or the calculus of variations.

Thus, while Kiyotaki and Wright studied stationary equilibria in which beliefs about 'media of exchange' are *consistent* with trading patterns, we study economies in which particular commodities *emerge* as media of exchange. We also study an economy (Economy C) in which a good from which no agent derives utility emerges as fiat money. We want to learn whether our artificially intelligent agents can learn to play a Markovian Nash equilibrium of the Kiyotaki-Wright model. When there are multiple Nash equilibria (e.g., the 'fundamental' and 'speculative' equilibria of Kiyotaki and Wright), we want to know whether the system might converge to some but not others of these equilibria. In addition, if classifier systems do converge to Nash-Markov equilibria, they might be used to compute equilibria in other economies for which it is difficult to obtain analytic solutions. To this end, we also study enlarged versions of the Kiyotaki-Wright model.

A *classifier system* is a collection of potential decision rules, together with an accounting system for selecting which rules to use. The accounting system credits rules generating good outcomes and debits rules generating bad outcomes. The system is designed to select a 'co-adaptive' set of rules that work well over the range of situations that an agent typically encounters. In a multiple-agent environment like Kiyotaki and Wright's, the range of situations encountered by one agent depends on the actions taken by other agents. This means that the collections of rules used by the collection of agents must co-adapt jointly.

We study two sorts of classifier systems, with the distinction between them being induced by the fact that for many problems enumerating all possible rules would require a very long list. The first kind of classifier system is one in which a complete enumeration of all possible rules is carried along. For many problems, it is not feasible to use a complete enumeration classifier system because the state and action spaces are so large that the set of all possible rules is much too big. For us, a complete enumeration is tractable because the Kiyotaki-Wright model has low-dimensional state and action spaces.

The second kind of classifier system is designed for situations in which it is not efficient to carry along a complete enumeration of decision rules. In this case, a modified version of the *genetic algorithm* of Holland is used as a device for periodically eliminating some rules and injecting new rules into the population of rules to be operated on by the accounting system. The genetic-algorithm version of the classifier models learning in the face of unforeseen contingencies. When an unprecedented state arises which is not covered by the existing set of rules, the system contains a procedure for manufacturing and experimenting with new rules that apply in the new situation. Even though it is feasible for us to work with a complete enumeration of rules, we also study the genetic-algorithm version of the classifier.[^fn2]

[^fn2]: When the state and action spaces are small, as in the Kiyotaki-Wright model, the list of all possible rules mapping states into actions is not too long. In such a situation, one can follow {cite}`axelrod1987` and model an agent's strategy as a single binary bit string. A pure genetic algorithm can then be applied, as in Axelrod (1987). This approach has been applied to the Kiyotaki-Wright model by Marimon and Miller (1989) and by Knez and Litterman (1989). While much easier to implement than classifier systems, this pure genetic approach also has some drawbacks. First, it is difficult to extend to situations involving 'unforeseen contingencies' and in which it is not practical to enumerate all of the histories that might be encountered in the course of play. Second, relative to the classifier, the pure genetic algorithm encodes information wastefully in the sense that an entire strategy is encoded, while the experience of play will typically throw up observations only on a small subset of the possible states. While the classifier system concentrates 'effort' on learning rules to use in frequently encountered states, the pure genetic algorithm purports to be learning about entire strategies. Third, classifiers can learn general rules that apply to several states and that offer potential informational economies.

In this paper, we set up classifier systems for the Kiyotaki-Wright environment and simulate them on a computer. We also formulate definitions of equilibrium and stability for classifier systems, and pose some convergence questions. Although we suggest possible convergence theorems, we prove no theorems in this paper. {ref}`sec-kw-environment` describes the Kiyotaki-Wright environment. {ref}`sec-classifier-systems` describes the classifier systems without genetics. {ref}`sec-stationary-equilibria` describes behavior in the stationary equilibria of Kiyotaki and Wright and also the types of classifier systems that could support that behavior. {ref}`sec-convergence` defines stability criteria for sets of classifier systems, and discusses how these definitions can be used to formulate questions about whether systems of classifier systems converge to a stationary equilibrium. {ref}`sec-genetic-algorithm` describes classifier systems operating with genetics. {ref}`sec-simulations` describes our simulations of systems of classifier systems. In addition to the economies studied by Kiyotaki and Wright (including one with fiat money), we study an economy with five types of agents and five goods. {ref}`sec-conclusions` concludes the paper.

(sec-kw-environment)=
## 2. The Kiyotaki-Wright Environment

There are three types of agents, with types being indexed by $i = 1, 2, 3$. Type $i$ agents get utility only from consuming type $i$ good. Type $i$ agent has access to a technology for producing type $i^*$ good, where $i^* \neq i$. We initially specify $(i, i^*)$ according to Kiyotaki-Wright's 'model A', namely, as follows:

| $i$ | $i^*$ |
|-----|-------|
| 1   | 2     |
| 2   | 3     |
| 3   | 1     |

This specification assumes no 'double coincidence of wants' and seems to call for a multilateral trading arrangement. All goods are indivisible. Each agent can store one and only one unit of only one good from one period to the next. When an agent of type $i$ consumes good $i$ at time $t$, he immediately produces good $i^*$, which he carries over to the next period. The net utility to an agent of type $i$ of consuming good $y$ and producing good $i^*$ is given by $u_i(y)$. We assume that an agent of type $i$ does not know his utility function, but does recognize utility when he experiences it. The goods are costly to store. Storing good $k$ ($k = 1, 2, 3$) from $t$ to $t+1$ imposes costs at $t$ of $s_k$. Following Kiyotaki and Wright, we assume that $s_3 > s_2 > s_1 > 0$. We summarize this cost function by saying that $s(y)$ is the one-period cost of storing one unit of good $y$. We assume that individuals do not know this cost function, but that they do recognize costs when they bear them.

The economy and each agent within it live forever. There are large and equal numbers of agents of each type. We modify Kiyotaki and Wright's model by assuming that each agent cares about his long-run average level of utility. (Kiyotaki and Wright assumed that agents have preferences ordered by expected discounted future utilities.) Each period, there is a random matching process that assigns each and every agent in the economy to a pair with one and only one other agent in the economy. The random matching technology matches agents without regard to type. Only pairs of agents matched together can trade at a point in time.

The economy begins at $t = 0$ with agents being endowed with an arbitrary and randomly generated initial distribution of holdings of goods. At each date $t \geq 0$, each agent $i$ has to make two decisions sequentially. First, given the good he is holding and given the good held by the partner with whom he is matched at $t$, he must decide whether to propose to trade. Trade occurs only if both parties propose to do so. Second, the agent must decide whether or not to consume the good with which he exits the trading process. If he doesn't consume, he simply carries the good into period $t + 1$, experiencing cost $s_k$. If an agent of type $i$ does consume good $y$ at $t$, he experiences net utility $u_i(y_t)$, produces good $i^*$, and experiences carrying costs $s_{i^*}$. We assume that $u_i(k) = 0$ if $k \neq i$ and $u_i(i) = u_i > 0$. We follow Kiyotaki and Wright both in adopting their specification of the physical environment and in considering only Markov strategies, but we drop the assumption of rational agents. Instead, our agents are assumed to be 'artificially intelligent', in the sense that they use versions of the *classifier system* introduced by {cite}`holland1986`.

Before describing the classifier systems used by our agents, we introduce some notation for naming agents and for describing the state of each individual agent in the economy. There is a collection of agents $\mathscr{A} = \{1, 2, \ldots, A\}$. A typical element of $\mathscr{A}$ is denoted $a$. Kiyotaki and Wright assumed that there was a continuum of each type of agent, while we assume that there is a finite number of each type. The first $A_1$ agents are of type I, the next $A_2$ are of type II, and the following $A_3$ are of type III. Here $A = 3A_i$.

At the beginning of time $t$, agent $a$ is carrying good $x_{at}$. The variable $x_{at}$ characterizes the pre-match state of agent $a$ at time $t$. There is a random matching process which each period matches each agent $a \in \mathscr{A}$ with a distinct agent $\rho_t(a) \in \mathscr{A}$. For each agent in each period, the matching process induces a function $\rho_t(a): \mathscr{A} \rightarrow \mathscr{A}$. After the matching process, the pre-trade state of agent $a$ is $(x_{at}, x_{\rho_t(a)t})$. The pair $(x_{at}, x_{\rho_t(a)t}) = z_{at}$ records what agent $a$ is carrying and what the agent $\rho_t(a)$ with whom $a$ is matched at $t$ is carrying.

At $t$, after being matched with agent $\rho_t(a)$, agent $a$ decides whether or not to propose to trade. We let $\lambda_{at}$ denote the trading decision of agent $a$ at time $t$, where

$$
\lambda_{at} = \begin{cases}
1 & \text{if } a \text{ proposes to trade } x_{at} \text{ for } x_{\rho_t(a)t} \\
0 & \text{if } a \text{ refuses to trade}
\end{cases}
$$ (eq-trade-decision)

Similarly, $\lambda_{\rho_t(a)t}$ summarizes the trading decision of the agent $\rho_t(a)$ with whom $a$ is paired at $t$. Trade takes place if and only if $\lambda_{at} \cdot \lambda_{\rho_t(a)t} = 1$.

Let $x_{at}^+$ denote the post-trade (but pre-consuming decision) holdings of agent $a$ at $t$. We then have that

$$
x_{at}^+ = (1 - \lambda_{at} \cdot \lambda_{\rho_t(a)t}) x_{at} + \lambda_{at} \lambda_{\rho_t(a)t} x_{\rho_t(a)t}
$$ (eq-post-trade)

After leaving the trading process with holdings $x_{at}^+$, agent $a$ must decide whether to consume $x_{at}^+$ or to carry it into the next period. We let $\gamma_{at}$ denote the consumption decision of agent $a$ at $t$ where

$$
\gamma_{at} = \begin{cases}
1 & \text{if } a \text{ decides to consume } x_{at}^+ \\
0 & \text{if } a \text{ decides not to consume}
\end{cases}
$$ (eq-consumption-decision)

If agent $a$ decides to consume, he automatically produces good $f(a)$, which he carries into $(t + 1)$. From the specification of $i^*$ as a function of $i$ described above for Kiyotaki and Wright's model A, we have that $f(a)$ is a good of type 2 if $a$ is a type I agent, $f(a)$ is a good of type 3 if $a$ is a type II agent, and $f(a)$ is a good of type 1 if $a$ is a type III agent. It follows that beginning-of-period holdings of $a$ at time $t + 1$ are described by[^fn3]

$$
x_{a,t+1} = \gamma_{at} f(a) + (1 - \gamma_{at}) \left( (1 - \lambda_{at} \cdot \lambda_{\rho_t(a)t}) x_{at} + \lambda_{at} \cdot \lambda_{\rho_t(a)t} x_{\rho_t(a)t} \right)
$$ (eq-holdings-evolution)

[^fn3]: A pure genetic algorithm could be applied to the trading decision as follows. For each agent, there are $3 \times 3 = 9$ possible values that his current state prior to trade can attain. There are two possible actions: trade or don't trade. Letting 1 mean trade and 0 mean don't trade, a binary string of length 9 can be used to encode a Markovian strategy. This is done by numbering the set of possible states $j = 1, \ldots, 9$ and then letting the $j$th entry in the bit string denote the decision to be made when the $j$th state is encountered. The genetic algorithm is then applied to select the best binary string (i.e., the best strategy). This approach, which has been applied to the Kiyotaki-Wright model [see Marimon and Miller (1989) and Knez and Litterman (1989)], has shortcomings as a starting point for studying more complex systems. Because the optimization is done over entire strategies, which include descriptions of decisions to be made in *all* possible states, even rarely visited ones, the encoding of information is wasteful and does not facilitate more intensive learning about frequently visited states.

(sec-classifier-systems)=
## 3. Classifier Systems for the Kiyotaki-Wright Environment

We now describe the *classifier systems* that agents use to make their trading and consumption decisions.[^fn4] Agents sequentially use two interconnected classifier systems each period. A first trading classifier system receives input in the form of the pre-trade state $z_{at} = (x_{at}, x_{\rho_t(a)t})$. This classifier system determines the trading decision $\lambda_{at}$, which interacts with the trading decision $\lambda_{\rho_t(a)t}$ of the agent $\rho_t(a)$ with whom $a$ is paired at $t$ to determine $x_{at}^+$ [see eq. {eq}`eq-post-trade`]. A second consumption classifier system takes $x_{at}^+$ as input and determines the consumption decision $\gamma_{at}$. The two classifier systems are used sequentially in this way each period, and their accounting systems are interconnected in ways to be described below.

[^fn4]: See {cite}`goldberg1989` for a description of classifier systems and a survey of some of their applications.

A *classifier system* consists of the following objects:

1. A collection of trinary strings, called 'classifiers'.
2. A system for interpreting or decoding the strings or classifiers as instructions mapping states into decisions. The first part of a string encodes a particular state or *condition*, while the second part encodes a particular action. Thus, an individual classifier or string is just an encoding of a single (state, action) pair. For a given state, there can be many classifiers present in a classifier system.
3. A list of 'strengths' attached to each classifier at each point in time $t = 0, 1, \ldots$
4. A system for reading in external messages or 'states' and for determining the set of classifiers that are pertinent, i.e., the set of classifiers whose 'conditions' are satisfied or 'matched' at that state. For a given state, the system can sometimes contain several classifiers with distinct actions.
5. An 'auction system' for determining which of the pertinent classifiers is allowed to make a decision at $t$. Among all classifiers whose condition part matches the actual state at $t$, the 'highest bidder' in the auction actually 'makes the decision' in real time.[^fn5]
6. An accounting system for updating the strengths of the collection of classifiers in response to the net rewards that flow into the system as a result of the decisions that are made.

And possibly:

7. A genetic algorithm for introducing new strings and extinguishing old strings. This algorithm will be called whenever states are encountered for which no existing classifiers have their conditions met. The algorithm may also be applied from time to time as a device to promote experimentation.

[^fn5]: This rule could easily be modified to permit a 'stochastic auction' in which the probability of winning the auction varies directly with strength.

We now describe each of these elements as applied to the Kiyotaki-Wright environment. We first consider an exchange classifier system, which maps the pre-trade state $z_{at}$ into a trading decision. This classifier system consists of a list of trinary strings of length 7. The first three digits represent own storage, the next three represent trading partner's storage, and the seventh represents the trading decision.

```{list-table} Encoding of goods in classifiers
:header-rows: 1
:name: tbl-encoding

* - Code
  - Meaning
* - 1 0
  - Good 1
* - 0 1
  - Good 2
* - 0 0
  - Good 3
* - 0 #
  - Not good 1
* - # 0
  - Not good 2
* - # #
  - Not good 3
```

The coding is written in the trinary alphabet $(1, 0, \#)$, where $\#$ means 'don't care'. For the trading decision, the code is in the binary alphabet $(1, 0)$, where 1 means trade, while 0 means don't trade.

To illustrate how the codes in {numref}`tbl-encoding` are applied to encode particular trading classifiers, we consider the following two classifiers:

| Own storage | Partner storage | Trading decision |
|-------------|-----------------|------------------|
| 1 0 0       | 0 0 1           | 1                |
| 1 0 0       | # # 0           | 0                |

The first classifier instructs an agent who is carrying good 1 and who is matched with someone carrying good 3 to offer to trade. The second classifier instructs an agent who is carrying good 1 not to trade if he is matched with someone who is not carrying good 3.

The exchange classifier system of agent $a$ consists of a list of such classifiers. Let $e = 1, 2, \ldots, E_a$ index this collection of classifiers. A given classifier system has a fixed number of classifiers. For the exchange classifier system, how many distinct classifiers are possible? Evidently, $3^3 \times 3^3 \times 2 = 1458$. However, most of these classifiers are redundant in the sense that only a subset of them is needed to represent all possible trading decision rules defined on the state space $z_{at} = (x_{at}, x_{\rho_t(a)t})$ when there are three goods. All rules can be written in terms of pairs of conditions drawn from {numref}`tbl-encoding`. Thus, only $6 \times 6 \times 2 = 72$ strings are required to represent the complete set of possible rules for this system.

Assigned to each exchange classifier $e \in \{1, 2, \ldots, E_a\}$ is a *strength*, denoted $S_e^a(t)$. The strength $S_e^a(t)$ evolves over time in a way determined by the accounting system. The strengths attached to classifiers are used to determine the decisions made by the classifier system at $t$. For a given state or 'condition' $z_{at} = (x_{at}, x_{\rho_t(a)t})$, there is a collection of classifiers within the classifier system whose conditions are satisfied. We denote the set of such classifiers by $M_e(z_{at})$, defined as

$$
M_e(z_{at}) = \{ e : z_{at} \text{ matches the condition part of classifier } e \}
$$ (eq-matched-classifiers)

The members of $M_e(z_{at})$ form a class of potential 'bidders' in an 'auction' whose purpose is to determine which classifier makes the decision of agent $a$ at time $t$. When state $z_{at}$ is encountered at time $t$ by agent $a$, the classifier belonging to $M_e(z_{at})$ that has the highest strength makes the decision. Let $e_t(z_{at})$ denote the index of the classifier to be used in deciding whether to trade at $t$. Then,

$$
e_t(z_{at}) = \arg\max\{S_e^a(t): e \in M_e(z_{at})\}
$$ (eq-auction-winner)

We denote the action (trade or no trade) taken by classifier $e_t(z_{at})$ as $\lambda_{at}$. Equations {eq}`eq-matched-classifiers` and {eq}`eq-auction-winner` describe the 'auction system' by which the highest strength rule that applies in a given state is given the right to decide for agent $a$ at $t$.

Because the trade and consumption classifier systems will be making payments to one another, we have to describe the consumption classifier system before describing the accounting system that updates strengths of the trade classifier system. The consumption classifier system is a collection of trinary strings of length 4. The first three positions encode $x_{at}^+$ using the same code that was described in {numref}`tbl-encoding`. The condition part of the strings is written in terms of the trinary alphabet $(0, 1, \#)$. The fourth position of a consumption string is the 'action' part, taking values of 1 (meaning consume) and 0 (meaning don't consume).

We let consumption classifier strings be indexed by $c \in \{1, 2, \ldots, C_a\}$. The strength assigned to classifier $c$ of agent $a$ is $S_c^a(t)$. By virtue of eq. {eq}`eq-post-trade`, the state $x_{at}^+$ can be expressed in terms of $z_{at} = (x_{at}, x_{\rho_t(a)t})$. Therefore it suffices to denote the set of 'matched' classifiers by $M_c(z_{at})$, where

$$
M_c(z_{at}) = \{c: x_{at}^+ \text{ matches the condition part of classifier } c\}
$$ (eq-matched-consumption)

Let $c_t(z_{at})$ denote the classifier that makes the consumption decision at $t$. The highest-strength classifier gets to make the decision:

$$
c_t(z_{at}) = \arg\max\{S_c^a(t): c \in M_c(z_{at})\}
$$ (eq-consumption-auction)

Given the decisions determined by {eq}`eq-auction-winner` and {eq}`eq-consumption-auction`, $x_{at}^+$ evolves according to the law of motion {eq}`eq-holdings-evolution`.

### 3.1 Counters and Strength Evolution

We attach to each exchange classifier $e$ a 'counter' $\tau_e^a(t)$ which records the cumulative number of times that classifier $e$ has won the auction as of date $t$. We shall change the strength of classifier $e$ only when it actually wins the auction and thereby gets to make the exchange decision, which is why we need the counter $\tau_e^a(t)$. The counter $\tau_e^a(t)$ for classifier $e$ is defined recursively in terms of the indicator $I_e^a(t)$, which records whether classifier $e$ wins the auction:

$$
I_e^a(t) = \begin{cases}
1 & \text{if } e \text{ wins the auction (unless classifier } e \text{ sets } \lambda_{at} = 1 \\
  & \text{while } \lambda_{\rho_t(a)t} = 0, \text{ so that the offer to trade is not reciprocated)} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\tau_e^a(t) = \sum_{s=0}^{t} I_e^a(s) + 1
$$ (eq-counter-exchange)

Notice that we initialize the counter of each classifier at unity. The strength of classifier $e$ at $t$ will be represented as $S_e^a(t) = S_{e\tau_e^a(t)}^a$.

Similarly, we attach to each consumption classifier $c$ a 'counter' $\tau_c^a(t)$ which records the cumulative number of times that classifier $c$ has won the auction as of date $t$. The counter $\tau_c^a(t)$ is defined by

$$
I_c^a(t) = \begin{cases}
1 & \text{if } c \text{ wins the auction} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\tau_c^a(t) = \sum_{s=0}^t I_c^a(s) + 1
$$ (eq-counter-consumption)

The strength of classifier $c$ at $t$ will be represented as $S_c^a(t) = S_{c\tau_c^a(t)}^a$.

### 3.2 Bid Functions and Strength Updates

The counters $\tau_e^a(t)$ and $\tau_c^a(t)$ induce a transformation of time in terms of which the strengths of classifiers $c$ and $e$ are updated. At date $t$, a strength $S_{c\tau_c^a(t)}^a$ is attached to classifier $c$, while a strength $S_{e\tau_e^a(t)}^a$ is attached to classifier $e$ of agent $a$. At date $t$, if classifier $e$'s condition is matched [i.e., if $e \in M_e(z_{at})$], then classifier $e$ makes a *bid* of $b_1(e)S_e^a(t)$, where $b_1(e)$ is a positive fraction that can depend on $e$. If classifier $e$ wins the auction, its bid will be deducted from its strength. The winning bid will be allocated to augment the strength of other classifiers whose actions drove the system to the state that satisfied $e$'s condition. We choose the particular bid function

$$
\begin{aligned}
b_1(e) &= b_{11} + b_{12}\sigma_e \\
b_2(c) &= b_{21} + b_{22}\sigma_c
\end{aligned}
$$ (eq-bid-functions)

where $b_{11}$ and $b_{12}$ are positive constants adding up to less than one, and $\sigma_e$ is a fraction which is proportional to the specificity of a particular classifier. In particular, we choose $\sigma_e = 1/(1 + \text{number of } \#\text{'s in the string})$. Similarly, we define a function $b_2(c)$ with $\sigma_c = 1/(1 + \text{number of } \#\text{'s in the string})$. By the above choices of $b_1(e)$ and $b_2(c)$, we favor specific rules over more general rules that can be activated by a particular state. When $c \in M_c(z_{at})$, classifier $c$ makes a bid of $b_2(c)S_c^a(t)$.

Only winning classifiers pay their bids by having them deducted from their strengths. The bid of the winning exchange classifier at $t$ is paid to the winning consumption classifier at $t - 1$, which is the classifier that is to be credited with setting the time $t$ state to $z_{at}$. The bid of the winning consumption classifier at $t$ is paid to the winning exchange classifier at time $t$, which is to be credited with setting the post-exchange state at $t$ at $x_{at}^+$, thereby giving the winning consumption classifier a chance to bid.

We represent these payments in terms of the following difference equations:

$$
S_{c\tau_c^a(t)}^a = S_{c\tau_c^a(t)-1}^a - \frac{1}{(\tau_c^a(t)-1)} \left[ (1+b_2(c)) S_{c\tau_c^a(t)-1}^a - \sum_e I_e^a(t) b_1(e) S_{e\tau_e^a(t)}^a - U_a(\gamma_{ct}^a) \right]
$$ (eq-strength-consumption)

$$
S_{e\tau_e^a(t)+1}^a = S_{e\tau_e^a(t)}^a - \frac{1}{\tau_e^a(t)} \left[ (1+b_1(e)) S_{e\tau_e^a(t)}^a - \sum_c I_c^a(t) b_2(c) S_{c\tau_c^a(t)}^a \right]
$$ (eq-strength-exchange)

In {eq}`eq-strength-consumption`, $U_a(\gamma_{ct}^a)$ is the external payoff when the winning consumption classifier $c$ makes final consumption decision $\gamma_{ct}^a$. If the post-exchange state at $t$ is $x_{at}^+$, then we have

$$
U_a(\gamma_{ct}^a) = \gamma_{ct}^a \left[ u_i(x_{at}^+) - s(f(a)) \right] + (1 - \gamma_{ct}^a) s(x_{at}^+)
$$ (eq-external-payoff)

There are several features of {eq}`eq-strength-consumption` and {eq}`eq-strength-exchange` that bear emphasizing. First, these are recursive formulas that make $S_{c\tau_c^a(t)}^a$ and $S_{e\tau_e^a(t)}^a$ averages of past payoffs (external rewards plus bids received from other classifiers) minus payments (bids made to other classifiers). Use of cumulative average net payoffs in this way (as opposed to total payoffs) is a departure from the existing literature on classifiers. Second, notice how the term $\sum_e I_e^a(t) b_1(e) S_{e\tau_e^a(t)}^a$ expresses the condition that only the winning exchange classifier at $t$ pays the winning consumption classifier at $t - 1$. Third, notice how the use of the counters $\tau_e^a(t)$ and $\tau_c^a(t)$ causes changes to be made only to the strengths attached to the winning classifiers.

```{figure} figures/fig1_classifier_flow.png
:name: fig-classifier-flow
:width: 80%

Example of flow of payments in classifier systems for type I agent: transfer payments denoted in bold lines, decision flows denoted in thin lines.
```

{numref}`fig-classifier-flow` illustrates how the exchange and consumption classifier systems interact for an individual of type I. Suppose that the state at time $t$ is encoded 010 100. This means that the type I individual is storing good 2 and has been matched with someone who is storing good 1. The individual has two classifiers whose conditions are matched, one with strength 50, the other with strength 10. The strength 50 classifier makes the decision, which is '1', namely, offer to trade. If the individual's trading partner also offers to trade, the individual exits the trading process with state 100, which is the input into type I's consumption classifier. With this input, there are two consumption classifiers whose conditions are matched, one with strength 80 that says 'consume', the other with strength -1 that says 'don't consume'. The individual consumes, initiating an external payoff. The bold lines indicate the flows of payoffs. In addition to the external payoff to the winning consumption classifier, there is a payment from the winning consumption classifier at $t$ to the winning exchange classifier at $t$, and a payment from the winning exchange classifier at $t$ to the winning consumption classifier at $t - 1$.

With {eq}`eq-strength-consumption`, {eq}`eq-strength-exchange`, and {eq}`eq-external-payoff`, we have completed our description of the classifier system for agent $a$ when that classifier system contains a complete enumeration of possible rules. Our version of the Kiyotaki-Wright model is formed by assuming that each agent $a$ of type $i$ uses the same classifier system to make decisions.[^fn6] Thus, we can replace $a$ superscripts and subscripts in the descriptions of strengths by $i$'s. We start with a set of initial strengths $S^i(0)$ for $i = 1, 2, 3$. We temporarily introduce the index $T$ to index calendar time, while $t$ now temporarily indexes the cumulative number of matches which have occurred rather than calendar time. Our artificially intelligent agents then interact as follows for each $T = 1, 2, \ldots, L$, where $L$ is the length of our simulation. All $A$ agents are randomly matched into $A/2$ pairs. For each pair $(a, a')$, the exchange and classifier systems for agents whose types are $a$ and $a'$, respectively, are used mutually to determine the trading and consumption decisions of agents $a$ and $a'$. After the consumption decisions have been made for the pair, the counters and the strengths of the classifier systems for both types of agents are updated according to {eq}`eq-counter-exchange`, {eq}`eq-counter-consumption`, {eq}`eq-strength-consumption`, and {eq}`eq-strength-exchange`. Also, after each pair has played, the synthetic time counter $t$ is augmented by one. After all $A/2$ pairs have played, the counter $T$ is augmented by one, and the entire process is repeated. The counter $T$ records the passage of physical time, while the counter $t$ records the cumulative number of matches that have occurred. The outcomes of all matches $t \in [(T-1)(A/2) + 1, T(A/2)]$ are interpreted as having occurred during period $T$.

In {ref}`sec-simulations`, we shall be interested in recording various frequency distributions across agents as a function of calendar time. That is, there will be distinct frequency distributions of, e.g., holdings of goods across agents for each date in calendar time. At the risk of confusion, we choose to index calendar time by $t$ (rather than $T$) in subsequent sections.

[^fn6]: We make this specification in order to economize on computer time and space. It is possible that this specification speeds convergence relative to a setting in which each agent $a \in \mathscr{A}$ uses his own classifier system. However, relative to a setup in which each agent uses his own classifier system and so can experiment individually, the common classifier setup causes all type $i$ agents to experiment simultaneously. This feature might delay convergence.

(sec-stationary-equilibria)=
## 4. Classifier Systems for Supporting Kiyotaki and Wright's Stationary Equilibria

### 4.1 The Fundamental Equilibrium

For their model A, Kiyotaki and Wright define as the *fundamental equilibrium* the trading pattern defined by the triangle depicted in {numref}`fig-fundamental`. In this trading pattern, good 1, which has the lowest storage cost, serves as the general medium of exchange. In terms of storage, agents of type I only store good 2, which they exchange only for good 1; agents of type III only store good 1, which they exchange only for good 3; and agents of type II half of the time store good 1, which they exchange for good 2, and half of the time good 3, which they exchange for good 1.

```{figure} figures/fig2_fundamental_equilibrium.png
:name: fig-fundamental
:width: 60%

Exchange patterns in fundamental equilibrium of model A.
```

We can characterize equilibria in terms of a set of probabilities that describe holding and exchanging patterns. We define

$$
\begin{aligned}
\pi_{it}^h(k) &= \text{probability that a type } i \text{ agent is holding } k \text{ at } t \\
\pi_t^h(k) &= \text{probability that a randomly selected agent is holding } k \text{ at } t \\
\pi_{it}^e(kj) &= \text{probability that a type } i \text{ agent exchanges } k \text{ for } j \text{ at } t \\
\pi_{it}^e(kj|k) &= \text{probability that a type } i \text{ agent exchanges } k \text{ for } j \text{ given that } i \text{ is holding } k \\
\pi_{it}^c(k) &= \text{probability that a type } i \text{ agent consumes } k \text{ at } t \\
\pi_{it}^c(k|k) &= \text{probability that a type } i \text{ agent consumes } k \text{ given that he holds } k \text{ after trading} \\
\pi_t(i|k) &= \text{probability that an agent is of type } i \text{ given that he holds } k \text{ at } t
\end{aligned}
$$ (eq-probabilities)

We denote by $\Pi_t$ the entire set of probabilities defined in {eq}`eq-probabilities` for all $i, k, j$. Associated with Kiyotaki and Wright's stationary equilibrium is a time-invariant $\Pi_t = \Pi$. In {numref}`tbl-economy-a1-equilibrium` we display the $\Pi$'s for the fundamental equilibrium of Kiyotaki and Wright's model A.

```{list-table} Equilibrium probabilities for Economy A1
:header-rows: 1
:name: tbl-economy-a1-equilibrium

* - (a) Probability that $i$ holds $j$, $\pi_i^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 0
  - 1
  - 0
* - $i=2$
  - 0.5
  - 0
  - 0.5
* - $i=3$
  - 1
  - 0
  - 0
```

```{list-table} Joint exchange probabilities $\pi_i^e(jk)$ in the stationary equilibrium of Economy A1
:header-rows: 1
:name: tbl-economy-a1-joint-exchange

* - $\pi_i^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0, 0, 0)$
  - $(0.167, [0,0.333], 0)$
  - $(0, 0, 0)$
* - $i=2$
  - $([0,0.25], 0.167, 0)$
  - $(0, 0, 0)$
  - $(0.167, 0, [0,0.083])$
* - $i=3$
  - $([0,0.5], 0, 0.167)$
  - $(0, 0, 0)$
  - $(0, 0, 0)$
```

The $(i,j)$ entry in {numref}`tbl-economy-a1-joint-exchange` is the triple $(\pi_i^e(j1), \pi_i^e(j2), \pi_i^e(j3))$, representing the probability that a type $i$ agent holds $j$, meets $k$, and trades.

```{list-table} Equilibrium exchange strategies for Economy A1
:header-rows: 1
:name: tbl-economy-a1-exchange

* - $\tilde{\pi}_i^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $([0,1],0,0)?$
  - $(1,[0,1],0)$
  - $(1,1,[0,1])?$
* - $i=2$
  - $([0,1],1,0)$
  - $(0,[0,1],0)?$
  - $(1,1,[0,1])$
* - $i=3$
  - $([0,1],0,1)$
  - $(1,[0,1],1)?$
  - $(0,0,[0,1])?$
```

Note: A question mark denotes sequential equilibrium strategies for events of zero probability. The $(i,j)$ entry in the table is the triple $(\tilde{\pi}_i^e(j1|j), \tilde{\pi}_i^e(j2|j), \tilde{\pi}_i^e(j3|j))$, representing the probability that a type $i$ agent *is willing* to exchange good $j$ for goods 1, 2, or 3 respectively, given that $i$ is holding $j$.

```{list-table} Consumption probabilities for Economy A1
:header-rows: 1
:name: tbl-economy-a1-consumption

* - $\pi_i^c(j|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 1
  - 0
  - 0
* - $i=2$
  - 0
  - 1
  - 0
* - $i=3$
  - 0
  - 0
  - 1
```

{numref}`tbl-economy-a1-consumption` shows that in any stationary equilibrium, each type of agent consumes only its own desired good.

We now describe how the strategies that support Kiyotaki and Wright's 'fundamental equilibrium' can be represented as a classifier system. For the classifiers we introduce the following additional notation. Let $e^i_{k,j,d}$ denote the exchange classifier for a type $i$ agent that reads 'If I am storing good $k$ and I meet an agent with good $j$, then my exchange decision is $d$'; $d \in \{0, 1\}$. Similarly, specific consumption classifiers for type $i$ agents are of the form $c^i_{k,d}$ for 'If, at the end of the exchange subperiod, I am storing good $k$, then my consumption action is $d$'. For more general rules, we use the subindex $-j$ to denote 'not good $j$' and $\#$ to denote 'any good'. That is, $e^1_{2,-1,0}$ is the classifier for type I agents that reads 'If I am storing good 2 and I meet an agent not carrying good 1, then do not trade', and $c^1_{\#,0}$ is a classifier for type I that reads 'No matter what I am storing at the end of the exchange subperiod, do not consume'.

We can simplify the study of classifier systems by considering only a small complete set of rules. For type I agents these rules are: $e^1_{2,1,1}$; $e^1_{2,-1,0}$; $e^1_{3,1,1}$; $e^1_{3,-1,0}$; $c^1_{1,1}$; $c^1_{-1,0}$. If $\{e^1_{2,1,1}; e^1_{2,-1,0}; c^1_{1,1}; c^1_{-1,0}\}$ are the only classifiers being used by agents of type I, we can say that type I agents support the fundamental equilibrium. We can also describe classifiers for type II and type III agents that support the fundamental equilibrium. These are, for type II: $e^2_{3,1,1}$; $e^2_{3,2,1}$; $e^2_{3,-1,0}$; $e^2_{1,-2,0}$; $c^2_{2,1}$; $c^2_{-2,0}$; and for type III: $e^3_{1,3,1}$; $e^3_{1,-3,0}$; $c^3_{3,1}$; $c^3_{-3,0}$.

It is useful to define formally the sense in which these fixed lists of classifiers support Kiyotaki and Wright's fundamental equilibrium. Let $D_a$ be a fixed set of classifiers (exchange and consumption classifiers) for agent $a$. Let the set $D_a$ have no redundancies in the sense that for every state $z_{at}$, $D_a$ uniquely determines actions. Then the fixed set of classifiers $D_a$ represents a strategy for agent $a$ defined on $z_{at}$. We use the following definition of optimality for a fixed set of classifiers:

**Definition.** *Given the probability distributions $\Pi$ and fixed sets of classifiers of the other agents $D_{a'}$, for all $a' \neq a$, a fixed set of classifiers $D_a$ is said to be optimal for agent $a$ if there exists no other set $\tilde{D}_a$ which yields higher long-run average utility for agent $a$.*

A stationary Nash equilibrium can then be defined as follows:

**Definition.** *A stationary Nash equilibrium is a set of probability distributions $\Pi$ and fixed sets of classifiers $D_a$, $a = 1, \ldots, A$, such that:*

1. *Given $\Pi$ and $D_{a'}$, for $a' \neq a$, $D_a$ is optimal for agent $a$.*
2. *$\{D_a, a = 1, \ldots, A\}$ and the random matching technology imply that $\Pi_t = \Pi$ for all $t$.*

*This definition is just an alternative representation of Kiyotaki and Wright's definition in terms of fixed lists of classifiers.*

{numref}`tbl-type-i-fundamental` illustrates the sequence of events under the circumstance that the only active (i.e., auction-winning) classifiers are those that support the fundamental equilibrium for agents of type I. A natural question is whether the classifier systems of the three types of agents will actually converge to a situation in which the fundamental equilibrium of Kiyotaki and Wright is supported. Before introducing some concepts that will help us think about this question, we shall briefly describe another kind of equilibrium of Kiyotaki and Wright's model and the kind of classifier system that could support it.

```{list-table} Behavior of type I agent in a fundamental equilibrium
:header-rows: 1
:name: tbl-type-i-fundamental

* - State
  - Action
  - Next State
* - Holding good 2
  - Meet agent with good 1 → Trade
  - Holding good 1
* - Holding good 1
  - Consume → Produce good 2
  - Holding good 2
```

### 4.2 Kiyotaki and Wright's 'Speculative Equilibrium'

Kiyotaki and Wright show that for their model A the trading pattern associated with a fundamental equilibrium is not the only possible one in equilibrium. They show that there can occur a *speculative equilibrium* with a trading pattern depicted in {numref}`fig-speculative`.

```{figure} figures/fig4_speculative_equilibrium.png
:name: fig-speculative
:width: 60%

Exchange patterns in speculative equilibrium of Economy A.
```

Kiyotaki and Wright establish a condition under which the unique stationary rational-expectations equilibrium is a fundamental equilibrium. The condition, for the limiting case as the discount rate converges to zero, is that the following inequality is satisfied:[^fn7]

$$
s_3 - s_2 > \left(\pi_1^h(3) - \pi_1^h(2)\right) \frac{1}{3} u_1
$$ (eq-fundamental-condition)

[^fn7]: Notice that this condition involves both parameter values ($s_3$, $s_2$, $u_1$) and the quantities ($\pi_1^h(3)$, $\pi_1^h(2)$) which are endogenous variables that are determined in equilibrium. Kiyotaki and Wright show that when the above inequality is reversed, then the unique stationary rational-expectations equilibrium is a speculative equilibrium. Vis-à-vis the fundamental equilibrium, the only change in trading patterns is that agents of type I 'speculate' by holding good 3, which is not too costly because in equilibrium inequality {eq}`eq-fundamental-condition` is reversed.

In the speculative equilibrium, {numref}`tbl-type-i-speculative` depicts the flow of events in the stationary rational-expectations equilibrium for a classifier system of a type I agent. Notice that the flows depicted reduce to a set of fundamental classifiers if the dotted link is removed, i.e., if type I agents decide not to trade good 2 for good 3.

```{list-table} Behavior of type I agent in a speculative equilibrium
:header-rows: 1
:name: tbl-type-i-speculative

* - State
  - Action
  - Next State
* - Holding good 2
  - Meet agent with good 1 → Trade
  - Holding good 1
* - Holding good 2
  - Meet agent with good 3 → Trade*
  - Holding good 3
* - Holding good 1
  - Consume → Produce good 2
  - Holding good 2
* - Holding good 3
  - Meet agent with good 1 → Trade
  - Holding good 1
```

Note: The asterisk (*) denotes the speculative move that distinguishes this from the fundamental equilibrium.

In {ref}`sec-simulations` we shall provide an example of a simulated classifier economy where {eq}`eq-fundamental-condition` is not satisfied ($u_1$ is too high), but where nevertheless when agents start with a complete set of classifiers (with homogeneous strengths), the classifier systems converge to the fundamental equilibrium and not to the speculative equilibrium. Before turning to these simulations, we define some notions of stability that are designed to distinguish alternative senses in which a set of classifier systems may be said to converge.

(sec-convergence)=
## 5. Concepts of Convergence for Classifier Systems Playing Games

Given the decision rules being employed by other agents, {eq}`eq-strength-consumption`-{eq}`eq-strength-exchange` for agent $a$ forms a system of stochastic difference equations in the strengths $S^a(t)$. Since the classifier systems of the three types of agents $i = 1, 2, 3$ are operating simultaneously, the behavior of the entire system is determined by the system of difference equations {eq}`eq-strength-consumption`-{eq}`eq-strength-exchange` for agent types $i = 1, 2, 3$. Let $S^i(t)$ denote the strengths for an agent of type $i$. It is natural to inquire whether the system formed by {eq}`eq-strength-consumption`-{eq}`eq-strength-exchange` for $a = i = 1, 2, 3$ converges to a stationary point in the space of strengths, and if it does so, whether that stationary point supports the fundamental equilibrium of Kiyotaki and Wright.

Although we have not yet carried out an analysis of the convergence of this system, it seems useful at this point to report our ideas about how such an analysis could be structured. In this section we indicate how ideas from the stochastic-approximation literature might be used to study convergence in the space of strengths.[^fn8]

[^fn8]: It was considerations from the stochastic-approximation literature that led us to alter Holland's specification of the laws of motion of strengths so that they measure cumulative *average* past net rewards, not total rewards. The use of average rewards makes it possible for strengths to converge. {cite}`arthur1989` uses average rewards for strengths for the same reason.

From the stochastic-approximation literature,[^fn9] we know that any limit points $(S_c^*, S_e^*)$ must satisfy

[^fn9]: {cite}`ljung1983` is a useful reference on stochastic approximation and on the 'ordinary differential-equations approach' to proving almost sure convergence. See {cite}`marcet1989a` and {cite}`marcet1989b` for some applications in economics.

$$
E\left[ (1 + b_2(c)) S_c^a - \sum_e I_e^a(t) b_1(e) S_e^a - U_a(\gamma_c^a) \right] = 0
$$

$$
E\left[ (1 + b_1(e)) S_e^a - \sum_c I_c^a(t) b_2(c) S_c^a \right] = 0
$$ (eq-limit-points)

for all $c$ and $e$. For a given $a$ we define the solutions $(S_c^*, S_e^*)$ of {eq}`eq-limit-points` as a *stationary* set of strengths for agent $a$. Given a stationary set of strengths for agent $a$, one can determine the set of classifiers $\{e, c\} = W_a(z_{at})$ that would win the auction for each $z_{at}$. If $a$ has the fixed set of classifiers $W_a(z_{at}) = D_a$, where $D_a$ is the fixed set of classifiers that supports $a$'s behavior in a Nash equilibrium, then the stationary set of strengths associated with $W_a(z_{at})$ can be said to support the stationary Nash equilibrium behavior of agent $a$.

These definitions inspire several questions. First, given stationary sets of strengths for agents of types 1, 2, 3, is it true that $W_a(z_{at}) = D_a$ for a type $i$ agent, for each $i = 1, 2, 3$? That is, does a stationary set of strengths for the classifier systems for agents of type $i = 1, 2, 3$ support a stationary Nash equilibrium?

Second, given fixed sets of classifiers $D_a$ for $a$ of type $i = 1, 2, 3$ that support a stationary Nash equilibrium, do these sets of classifiers imply the existence of a set of strengths $S_a$ for $a$ of type $i = 1, 2, 3$ that solve {eq}`eq-limit-points`? That is, is every Nash equilibrium supported as a stationary point in strengths?

Third, if we let the classifier systems of agents of type $i = 1, 2, 3$ run in real time, does the joint system formed by {eq}`eq-strength-consumption`-{eq}`eq-strength-exchange` for agents of each type converge almost surely, and if so, to which solution of {eq}`eq-limit-points` does it converge? We know from the theory of stochastic approximation that *if* it converges, it must converge to solutions of {eq}`eq-limit-points`.

Associated with the third of these questions is the notion of global asymptotic stability. In future work, we plan to apply results from the stochastic approximation literature to study the global asymptotic stability of versions of our interacting classifier systems.

We also intend to study a form of stability that is distinct from and weaker than global asymptotic stability. This form of stability is summarized in the following:

*Nash-like stability test.* Take a stationary equilibrium fixed by $\Pi$. Place a single agent of a given type $i$ in the environment formed by the fixed probabilities $\Pi$. Let a classifier system for that single agent operate until it converges. Check to see whether it converges to imply the collection of classifiers $D_a$ for an agent $a$ of type $i$ which are required to support the equilibrium probabilities. In this experiment, in running the classifier system for the single agent, the decision rules of all other agents of his type are being held fixed at their equilibrium values. This experiment is to be repeated for a single agent of each type.

In the remainder of this paper, we shall not pursue the formal analysis of the stability and convergence questions formulated above.[^fn10] Instead, we shall report computer simulations. In these computer simulations, convergence can be based either on the limiting behavior of the strengths, as indicated above, or on the convergence of key aspects of the empirical frequency distribution of holdings $\Pi_t$, which we defined in {ref}`sec-stationary-equilibria`.

[^fn10]: Brian Arthur and Carl Simon have used stochastic approximation arguments to establish convergence of strengths for a fixed system of two classifiers playing a two-armed bandit. Their arguments do not apply to our system because their system is one in which there are competing classifiers with distinct actions, whose condition parts are identical and are always met.

(sec-genetic-algorithm)=
## 6. Incomplete Enumeration Classifiers and the 'Genetic Algorithm'

We now describe how the classifier system is modified when we deal with systems in which there is not a complete enumeration of rules. In order to complete the system, two aspects must be added to the above description of the classifier system under complete enumeration. The first is a way of generating an initial population of rules. The second is a way of deleting old rules from the system and adding new rules to the system. To generate the initial population of rules, we simply use a random number generator to generate bit strings by choosing successive bits independently.

The process of adding new rules and deleting old ones is accomplished by adding four major operations to the previous description of the workings of the classifier system. We call these steps 'creation', 'diversification', 'specialization', and 'generalization', respectively.

**Creation**: The *creation* operation is activated when there is no classifier matching the current state $z_{at}$, i.e., $M_e(z_{at})$ is empty. In this case, a new classifier is created with its condition part defined by the current state and its action part randomly selected. A 'weak' classifier from the set of all classifiers is deleted to keep the number of classifiers constant.

**Diversification**: The *diversification* operation is designed to inject sufficient diversity into the range of actions called for by different classifiers in a given situation. For the exchange classifier, this step occurs at the stage at which $z_{at}$ has been observed and the set $M_e(z_{at})$ has been constructed. Recall that $M_e(z_{at})$ is the set of exchange classifiers whose condition parts match $z_{at}$. If all $e \in M_e(z_{at})$ have the same action (0 or 1), the diversification operation creates a classifier encoding the specific state $z_{at}$ and an action opposite to that taken by the other classifiers in $M_e(z_{at})$. The strength of the winning classifier is assigned to this new classifier. This new classifier is added to the system, while simultaneously a 'weak' classifier from the set $M_e(z_{at})$ is deleted from the system. The weakness of a classifier is measured in terms of a combination of strength and cumulated number of times the classifier has won the bidding in the past. The diversification operation is used each time the classifier system is called upon. Notice that if this step were to be added to a complete enumeration system, it would have no effect on the population of rules because sufficient diversity is present from the beginning.

**Specialization**: The *specialization* operation is called randomly with a probability that we choose to be diminishing over time. The random variable governing specialization calls this step into action just after the winning bid has been determined. The winning classifier is checked to determine whether there are any $\#$'s in its condition part. A new classifier is synthesized by exposing each $\#$ in the condition part to a probability of being switched to a 0 or a 1, whichever one specifically encodes the particular $z_{at}$ that was just observed or matched. This new and (probably) more specific rule is then used to replace a weak rule from the set $e \in M_e(z_{at})$ [or $c \in M_c(z_{at})$ in the case of the consumption classifier], where weakness is measured by a combination of strengths and cumulative number of times the auction has been won.

**Generalization**: The fourth step, which we call *generalization*, is a version of a 'genetic algorithm'. This step is called randomly, according to a probability that we specify to be declining through time. At a point when the step is called, the process is initiated by generating two distinct populations of classifiers: 'potential parents' and 'potential exterminants'. The potential parents are chosen to be a population of classifiers of specified size. They are chosen according to a 'fitness criterion' that weights strength and cumulative number of times that the classifier won the auction. The potential exterminants are chosen to be a population of specified size, whose members are of low fitness as measured by a fitness criterion weighting strengths and cumulative victories in past bidding.

From the population of potential parents, a specified number of pairs of parents are randomly drawn to engage in 'mating' for the purpose of creating children. How a pair of parents mates can be illustrated for two exchange classifiers, which are bit strings of length 7, as depicted in {numref}`fig-mating`.

The pair mates in two steps. First the pair draws two random integers uniformly distributed on $[1, 7]$. The two integers signify the positions between bits depicted in {numref}`fig-mating`. Next, a Bernoulli random variable with probability 0.5 is drawn, assuming values of 'in' or 'out'. If 'in' is drawn, we focus on the bits *between* the two lines, i.e., the bits inside the randomly drawn interval. If 'out' is drawn, we focus on the bits *outside* the randomly drawn interval. Within the interval of focus, we complete the mating process via the following version of genetic crossover. For each of the parent strings in the area of focus, we change to a $\#$ any bits for which the parent strings disagree. (Here, $\#$ agrees with either 0 or 1, while 0 disagrees with 1). This process produces two children. The children are each assigned strengths that are the average of their parents' strengths. The children are introduced into the population of classifiers. For each child added to the population, a randomly selected individual from the population of 'potential exterminants' is deleted from the population.

This completes our description of the algorithm to be used for updating the classifier system when we deal with an incomplete enumeration classifier system. In the simulations to be reported in {ref}`sec-simulations`, we shall employ both complete and incomplete enumeration classifier systems.

```{figure} figures/fig5_mating_process.png
:name: fig-mating
:width: 70%

The mating process for exchange classifiers who have drawn '3,6' and 'in'.
```

(sec-simulations)=
## 7. Simulation Results

The sets of parameters defining the economies under study are summarized in {numref}`tbl-economies`.

```{list-table} Description of the economies
:header-rows: 1
:name: tbl-economies

* - Economy
  - Production
  - Storage costs
  - Utility
  - Initial CS
  - Equil. type
* - A1.1
  - 2
  - $s_1=0.1, s_2=1, s_3=20$
  - $u_i=100$
  - F
  - F
* - A1.2
  - 2
  - $s_1=0.1, s_2=1, s_3=20$
  - $u_i=100$
  - R
  - F
* - A2.1
  - 2
  - $s_1=0.1, s_2=1, s_3=20$
  - $u_i=500$
  - F
  - S
* - A2.2
  - 2
  - $s_1=0.1, s_2=1, s_3=20$
  - $u_i=500$
  - R
  - S
* - B.1
  - 3
  - $s_1=1, s_2=4, s_3=9$
  - $u_i=100$
  - F
  - F/S
* - B.2
  - 3
  - $s_1=1, s_2=4, s_3=9$
  - $u_i=100$
  - R
  - F/S
* - C
  - 2
  - $s_1=9, s_2=14, s_3=29, s_0=0$
  - $u_i=100$
  - R
  - F
* - D
  - 3
  - $s_1=1, s_2=4, s_3=9, s_4=16, s_5=30$
  - $u_i=200$
  - R
  - —
```

Notes: Utility levels $u_i$ are set equal for $i = 1, 2, 3$. CS denotes 'classifier system'. 'F' implies fixed enumeration and 'R' implies randomly generated rules.

### 7.1 Economy A1

```{list-table} Parameter values used for Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 72, C_a = 12$
* - Storage costs
  - $s_1 = 0.1, s_2 = 1, s_3 = 20$
* - Utility
  - $u_i = 100, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.25, b_{22} = 0.25$
```

Economy A1.1 (Economy A1 with complete enumeration of classifiers) shows that the distribution of holdings rapidly converges to the stationary equilibrium distributions. Similarly, the exchange and consumption strategies implemented by the system of winning classifiers virtually coincide with the equilibrium strategies. In {numref}`tbl-economy-a11-holdings`, and in all subsequent tables of empirical frequencies, the frequencies reported are ten-period moving averages ending at the indicated time periods.[^fn11]

[^fn11]: When '—' appears in the table, it means that the event in question occurred so rarely in our simulation that no reliable frequency can be computed.

```{list-table} Frequency with which $i$ holds $j$ at $t=500$ and $t=1000$ for Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-holdings

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - 0
  - 1
  - 0
* - $i=2$ ($t=500$)
  - 0.502
  - 0
  - 0.498
* - $i=3$ ($t=500$)
  - 1
  - 0
  - 0
* - $i=1$ ($t=1000$)
  - 0
  - 1
  - 0
* - $i=2$ ($t=1000$)
  - 0.506
  - 0
  - 0.494
* - $i=3$ ($t=1000$)
  - 1
  - 0
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=500$ and $t=1000$ for Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-exchange-freq

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - $(0,0,0)$
  - $(0.16,0,0)$
  - $(0,0,0)$
* - $i=2$ ($t=500$)
  - $(0.26,0.16,0)$
  - $(0,0,0)$
  - $(0.17,0,0)$
* - $i=3$ ($t=500$)
  - $(0.52,0,0.17)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=1$ ($t=1000$)
  - $(0,0,0)$
  - $(0.17,0,0)$
  - $(0,0,0)$
* - $i=2$ ($t=1000$)
  - $(0.27,0.17,0)$
  - $(0,0,0)$
  - $(0.19,0,0)$
* - $i=3$ ($t=1000$)
  - $(0.51,0,0.19)$
  - $(0,0,0)$
  - $(0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=1000$ for Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-winning

* - $\tilde{\pi}_{it}^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - —
  - $(1,0,0)$
  - —
* - $i=2$
  - $(1,1,0)$
  - —
  - $(1,1,0)$
* - $i=3$
  - $(1,0,1)$
  - —
  - —
```

```{list-table} Highest-strength consumption classifiers for type I at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type1-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 1
  - 45.8
* -
  - \# \# 0 0
  - −0.47
* -
  - 0 \# \# 1
  - −9.16
```

```{list-table} Highest-strength exchange classifiers for type I at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type1-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 0 \# \# 1 0 0 1
  - 11.22
* -
  - 0 \# \# 0 1 0 0
  - −0.08
* -
  - \# \# 0 0 0 1 0
  - −0.08
```

```{list-table} Highest-strength consumption classifiers for type II at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type2-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 0
  - 0.0089
* -
  - 0 1 0 1
  - 49.04
* -
  - \# 0 \# 0
  - −10.90
```

```{list-table} Highest-strength exchange classifiers for type II at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type2-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - \# \# 0 \# \# 0 1
  - 4.86
* -
  - \# \# 0 \# 0 \# 0
  - 0.002
* -
  - 0 0 1 1 0 0 1
  - 0.002
* -
  - 0 \# \# 0 1 0 1
  - 12.72
* -
  - 0 \# \# 0 \# \# 0
  - −1.79
```

```{list-table} Highest-strength consumption classifiers for type III at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type3-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - \# 0 \# 0
  - −0.024
* -
  - \# \# 0 0
  - −0.412
* -
  - 0 0 1 1
  - 31.62
```

```{list-table} Highest-strength exchange classifiers for type III at $t=1000$, Economy A1.1
:header-rows: 1
:name: tbl-economy-a11-cs-type3-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - \# 0 \# \# 0 \# 1
  - −0.01
* -
  - \# 0 \# 0 1 0 0
  - −0.01
* -
  - \# 0 \# 0 0 1 1
  - 7.78
```

```{figure} figures/fig6_economy_a11.png
:name: fig-economy-a11
:width: 80%

Distribution of holdings for Economy A1.1.
```

```{list-table} Parameter values used for Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 72, C_a = 12$
* - Storage costs
  - $s_1 = 0.1, s_2 = 1, s_3 = 20$
* - Utility
  - $u_i = 100, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.25, b_{22} = 0.25$
* - Specialization
  - $f_s(t) = 1/2\sqrt{t}, p_s = 0.01$
* - Generalization
  - $f_g(t) = 1/2\sqrt{t}, p_1 = 0.2, p_2 = 0.7, p_3 = 0.2, p_4 = 0.5, S = 0, N_e = 8, N_c = 4$
```

Economy A1.2 is identical to Economy A1.1, except that we use a randomly generated list of rules initially, and rely on a genetic algorithm to inject new rules into the classifier's system.[^fn12]

[^fn12]: More standard Holland algorithms were first tried without much success, prompting us to produce the modified algorithm described in {ref}`sec-genetic-algorithm`.

```{list-table} Frequency with which $i$ holds $j$ at $t=1000$ and $t=2000$ for Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-holdings

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=1000$)
  - 0
  - 0.992
  - 0.008
* - $i=2$ ($t=1000$)
  - 0.226
  - 0
  - 0.774
* - $i=3$ ($t=1000$)
  - 1
  - 0
  - 0
* - $i=1$ ($t=2000$)
  - 0
  - 0.98
  - 0.02
* - $i=2$ ($t=2000$)
  - 0.318
  - 0
  - 0.682
* - $i=3$ ($t=2000$)
  - 1
  - 0
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=1000$ and $t=2000$ for Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-exchange-freq

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=1000$)
  - $(0,0,0)$
  - $(0.08,0.14,0.01)$
  - $(0,0,0)$
* - $i=2$ ($t=1000$)
  - $(0,0.08,0)$
  - $(0,0,0)$
  - $(0.28,0.01,0.09)$
* - $i=3$ ($t=1000$)
  - $(0,0,0.28)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=1$ ($t=2000$)
  - $(0,0,0)$
  - $(0.09,0.19,0.03)$
  - $(0,0,0)$
* - $i=2$ ($t=2000$)
  - $(0,0.08,0)$
  - $(0,0,0)$
  - $(0.2,0.03,0.06)$
* - $i=3$ ($t=2000$)
  - $(0,0.01,0.2)$
  - $(0,0,0)$
  - $(0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=1000$ and $t=2000$ for Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-winning

* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=1000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,\text{—},\text{—})$
  - $(1,1,0)$
  - $(1,1,0)$
* - $i=2$
  - $(0,1,0)$
  - $(\text{—},\text{—},\text{—})$
  - $(1,0,1)$
* - $i=3$
  - $(0,0,1)$
  - $(\text{—},\text{—},1)$
  - $(\text{—},\text{—},1)$
* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=2000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(\text{—},\text{—},\text{—})$
  - $(1,0,1)$
  - $(1,1,1)$
* - $i=2$
  - $(0,1,1)$
  - $(\text{—},\text{—},\text{—})$
  - $(1,1,1)$
* - $i=3$
  - $(0,1,1)$
  - $(\text{—},\text{—},1)$
  - —
```

```{list-table} Consumption frequency $\pi_{it}^c(j|j)$ at $t=1000$ and $t=2000$ for Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-consumption-freq

* - $\pi_{it}^c(j|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=1000$)
  - 1
  - 0.011
  - 0.049
* - $i=2$ ($t=1000$)
  - 0.45
  - 1
  - 0.366
* - $i=3$ ($t=1000$)
  - 0.033
  - —
  - 1
* - $i=1$ ($t=2000$)
  - 1
  - 0
  - 0.69
* - $i=2$ ($t=2000$)
  - 0.23
  - 1
  - 0.641
* - $i=3$ ($t=2000$)
  - 0.84
  - 1
  - 1
```

```{list-table} Highest-strength consumption classifiers for type I, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type1-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 1
  - 38.11
* -
  - 0 1 \# 0
  - −0.44
* -
  - 0 0 1 1
  - −4.26
* - 2000
  - 1 0 0 1
  - 41.65
* -
  - 0 1 0 0
  - −0.43
* -
  - 0 0 1 0
  - −8.86
```

```{list-table} Highest-strength exchange classifiers for type I, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type1-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 0 1 0 1 0 0 1
  - 8.27
* -
  - 0 1 0 0 1 0 1
  - −0.09
* -
  - 0 1 0 0 0 1 0
  - −0.10
* - 2000
  - 0 1 0 1 0 0 1
  - 9.05
* -
  - 0 1 0 0 1 0 0
  - −0.10
* -
  - 0 1 0 0 0 1 1
  - −0.10
```

```{list-table} Highest-strength consumption classifiers for type II, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type2-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - \# \# 0 0
  - 1.88
* -
  - 0 \# 0 1
  - 11.48
* -
  - 0 0 1 1
  - −9.29
* - 2000
  - \# \# 0 0
  - 8.05
* -
  - 0 \# 0 1
  - 38.40
* -
  - 0 0 1 1
  - −9.01
```

```{list-table} Highest-strength exchange classifiers for type II, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type2-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 1 0 0 0
  - 0.29
* -
  - 1 0 0 0 1 0 1
  - 1.67
* -
  - 1 0 0 0 0 1 0
  - 0.28
* -
  - 0 0 1 \# \# 0 1
  - 2.23
* -
  - 0 \# 1 0 1 0 0
  - 4.08
* -
  - 0 0 1 0 0 1 1
  - −2.08
* - 2000
  - 1 0 0 1 0 0 0
  - 1.32
* -
  - 1 0 0 0 1 0 1
  - 6.15
* -
  - 1 0 0 0 0 1 0
  - 1.25
* -
  - 0 0 1 1 0 0 1
  - 1.34
* -
  - 0 \# 1 0 1 0 1
  - 2.98
* -
  - 0 0 1 0 0 1 1
  - −1.99
```

```{list-table} Highest-strength consumption classifiers for type III, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type3-cons

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 0
  - 0.00
* -
  - 0 0 1 1
  - 30.94
* - 2000
  - 1 0 0 1
  - 8.97
* -
  - 0 0 1 1
  - 30.55
```

```{list-table} Highest-strength exchange classifiers for type III, Economy A1.2
:header-rows: 1
:name: tbl-economy-a12-cs-type3-exch

* - Iteration
  - Classifier
  - Strength
* - 1000
  - 1 0 0 \# \# \# 0
  - 0.28
* -
  - 1 0 0 0 \# \# 0
  - 1.77
* -
  - 1 \# \# 0 \# 1 1
  - 7.39
* - 2000
  - 1 0 0 1 \# 0 0
  - 0.14
* -
  - 1 0 0 0 1 0 1
  - 0.42
* -
  - 1 \# \# 0 \# 1 1
  - 7.25
```

As we can see from {numref}`fig-economy-a12` and the above tables, the distribution of holdings for agents of type I and III rapidly converge to their equilibrium levels. Type II agents only hold goods 1 and 3 as predicted but it takes longer to converge to the equilibrium values. The exchange decisions are the correct ones (even in period 1000), except for a couple of events of very small probability (type I agents trade good 2 for 3 and type III agents trade good 1 for 2).

Consumption decisions are basically correct, except that II consumes 3 and III consumes 1. Overeating has been a minor problem in several of our economies. In most such cases that we have encountered, such wrong behavior has been associated with low-frequency states for which it is difficult for the classifier system to acquire the experience required to reward better classifiers and 'right' actions.

In summary, both with complete enumeration and without it, our simulations of Economy A1 seem to be converging to the stationary equilibrium, although this convergence is slower when initial classifiers are randomly generated and new classifiers are synthesized.

```{figure} figures/fig7_economy_a12.png
:name: fig-economy-a12
:width: 80%

Distribution of holdings for Economy A1.2.
```

### 7.2 Economy A2

```{list-table} Parameter values used for Economy A2.1
:header-rows: 1
:name: tbl-economy-a21-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 72, C_a = 12$
* - Storage costs
  - $s_1 = 0.1, s_2 = 1, s_3 = 20$
* - Utility
  - $u_i = 500, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.25, b_{22} = 0.25$
```

Economy A2 differs from Economy A1 only in that agents receive 500 utils from consuming their desired good instead of 100. With this change of parameters, for high enough discount factors the unique stationary equilibrium of Kiyotaki and Wright's economy is the so-called speculative equilibrium. The probabilities that characterize the speculative equilibrium are summarized in {numref}`tbl-economy-a2-spec-holdings`.

```{list-table} Speculative equilibrium: probability that $i$ holds $j$, $\pi_i^h(j)$, for Economy A2
:header-rows: 1
:name: tbl-economy-a2-spec-holdings

* - $\pi_i^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 0
  - 0.707
  - 0.293
* - $i=2$
  - 0.586
  - 0
  - 0.414
* - $i=3$
  - 1
  - 0
  - 0
```

```{list-table} Speculative equilibrium: joint exchange probability $\pi_i^e(jk)$ for Economy A2
:header-rows: 1
:name: tbl-economy-a2-spec-exchange

* - $\pi_i^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,0,0)$
  - $(0.138,[0,0.167],0.097)$
  - $(0.097,0,[0,0.029])$
* - $i=2$
  - $([0,0.269],0.138,0.057)$
  - $(0,0,0)$
  - $(0.138,0.098,[0,0.098])$
* - $i=3$
  - $([0,0.528],0,0.235)$
  - $(0,0,0)$
  - $(0,0,0)$
```

```{list-table} Speculative equilibrium exchange strategies $\tilde{\pi}_i^e(jk|j)$ for Economy A2
:header-rows: 1
:name: tbl-economy-a2-spec-strategies

* - $\tilde{\pi}_i^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $([0,1],0,0)?$
  - $(1,[0,1],1)$
  - $(1,0,[0,1])$
* - $i=2$
  - $([0,1],1,0)$
  - $(0,[0,1],0)?$
  - $(1,1,[0,1])$
* - $i=3$
  - $([0,1],0,1)$
  - $(1,[0,1],1)?$
  - $(0,0,[0,1])?$
```

```{list-table} Frequency with which $i$ holds $j$ at $t=500$ and $t=1000$ for Economy A2.1
:header-rows: 1
:name: tbl-economy-a21-holdings

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - 0
  - 1
  - 0
* - $i=2$ ($t=500$)
  - 0.504
  - 0
  - 0.496
* - $i=3$ ($t=500$)
  - 1
  - 0
  - 0
* - $i=1$ ($t=1000$)
  - 0
  - 1
  - 0
* - $i=2$ ($t=1000$)
  - 0.466
  - 0
  - 0.534
* - $i=3$ ($t=1000$)
  - 1
  - 0
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=500$ and $t=1000$ for Economy A2.1
:header-rows: 1
:name: tbl-economy-a21-exchange-freq

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - $(0,0,0)$
  - $(0.16,0.17,0.15)$
  - $(0,0,0)$
* - $i=2$ ($t=500$)
  - $(0,0.16,0)$
  - $(0,0,0)$
  - $(0.17,0.15,0.04)$
* - $i=3$ ($t=500$)
  - $(0.34,0,0.17)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=1$ ($t=1000$)
  - $(0,0,0)$
  - $(0.18,0.13,0.19)$
  - $(0,0,0)$
* - $i=2$ ($t=1000$)
  - $(0,0.18,0)$
  - $(0,0,0)$
  - $(0.18,0.19,0.04)$
* - $i=3$ ($t=1000$)
  - $(0.34,0,0.18)$
  - $(0,0,0)$
  - $(0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=500$ and $t=1000$ for Economy A2.1
:header-rows: 1
:name: tbl-economy-a21-winning

* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=500$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,0,0)$
  - $(1,0,1)$
  - $(0,0,0)$
* - $i=2$
  - $(0,1,0)$
  - $(0,1,0)$
  - $(1,1,0)$
* - $i=3$
  - $(1,0,1)$
  - $(1,1,1)$
  - $(1,0,1)$
* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=1000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,0,0)$
  - $(1,0,1)$
  - $(0,0,0)$
* - $i=2$
  - $(0,1,0)$
  - $(0,1,0)$
  - $(1,1,0)$
* - $i=3$
  - $(1,0,1)$
  - $(1,1,1)$
  - $(1,0,1)$
```

{numref}`tbl-economy-a21-holdings` depicts a pattern of holdings characteristic of a fundamental equilibrium, not a speculative one. Compare with the theoretical predictions of {numref}`tbl-economy-a2-spec-holdings`–{numref}`tbl-economy-a2-spec-exchange`. Recall that a crucial difference between a speculative and a fundamental equilibrium is that in the former, type I agents are willing to exchange good 2 for good 3, while in the latter they are not. {numref}`tbl-economy-a21-winning` shows that type I agents *are* willing to exchange good 2 for 3, i.e., $\tilde{\pi}_{it}^e(jk|2) = (1,0,1)$ for $i=\text{I}$. Then how is it that the distribution of holding patterns fails to support a speculative equilibrium? The answer is to be found in the consumption classifiers of type I agents, which are too general (have too many $\#$ signs) to distinguish among goods stored. It happens that the winning classifiers embody levels of generality that result in overconsumption of good 3 and a failure to transmit the information from the consumption classifier to the exchange classifier that would be required to support a speculative equilibrium.[^fn13]

[^fn13]: To conserve space, these classifiers are not reported here. They are contained in a longer version of this paper which is available from the authors as a working paper.

We shall only briefly summarize the results for Economy A2.2 with random generation of initial classifiers, and refer the reader to the working paper for tables giving the numerical results. After 1000 iterations the economy had not converged to a steady state. Trading patterns were closer to the fundamental equilibrium than to the speculative equilibrium in period 1000. (The reverse was true in period 500.) However, the classifier system of type I agents did distinguish between storage of different goods for consumption, and the classifier $c^1_{\#,0}$ was slowly gaining strength over time. Nevertheless, our simulations provided no support for a hope that the economy would converge to the speculative equilibrium if it were to last longer.

While the results for Economy A2 are fairly inconclusive, they illustrate two interesting points. First, they expose some deficiencies of the genetic algorithm which we have used (we return to this in {ref}`sec-conclusions`). Second, they raise the issue of whether our artificially intelligent agents are too impatient.

*Patience requires experience.* The transfer system inside the classifier system is designed to converge to a set of long-run average strengths. In the limit, the artificially intelligent agents should behave as long-run average payoff maximizers, since the steady-state strengths weight payoffs by their relative frequencies. It takes time, however, for optimal rules to achieve the desired strengths. The behavior of our artificially intelligent agents can be very myopic at the beginning. In economies, such as Economy A2, in which the nature of the equilibrium changes with the discount rate, this early myopia might have a perverse effect in diverting the economy towards a low discount-factor stationary equilibrium, such as the fundamental equilibrium. The underlying algorithm has to provide enough experimentation to avoid an early 'lock in'.[^fn14][^fn15] The present algorithm seems defective in that it has too little experimentation to support the speculative equilibrium even in the long simulations we have run.

[^fn14]: A fundamental equilibrium exists for Economy A2 only if the discount factor is sufficiently low.

[^fn15]: Marimon and Miller (1989) report that in their experiments with the genetic algorithm 25 out of 30 runs of Economy A2 converged to the speculative equilibrium.

### 7.3 Economy B

```{list-table} Parameter values used for Economy B.1
:header-rows: 1
:name: tbl-economy-b1-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 72, C_a = 12$
* - Storage costs
  - $s_1 = 1, s_2 = 4, s_3 = 9$
* - Utility
  - $u_i = 100, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.25, b_{12} = 0.25, b_{21} = 0.25, b_{22} = 0.25$
```

Economy B has a different production pattern than Economy A (I produces 3, II produces 1, and III produces 2). For the specified parameters two stationary equilibria are possible: fundamental and speculative. Stability tests based on classifier systems are potentially useful as selection criteria in economies with multiple equilibria. {numref}`tbl-economy-b-fundamental-holdings` and {numref}`tbl-economy-b-speculative-holdings` summarize the trading patterns for the fundamental and the speculative equilibrium, respectively.

```{figure} figures/fig8_economy_b.png
:name: fig-economy-b
:width: 80%

Exchange patterns in Economy B: Fundamental Equilibrium (left) and Speculative Equilibrium (right).
```

```{list-table} Fundamental equilibrium: probability that $i$ holds $j$, $\pi_i^h(j)$, for Economy B
:header-rows: 1
:name: tbl-economy-b-fundamental-holdings

* - $\pi_i^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 0
  - 0.293
  - 0.707
* - $i=2$
  - 1
  - 0
  - 0
* - $i=3$
  - 0.586
  - 0.414
  - 0
```

```{list-table} Fundamental equilibrium: joint exchange probability $\pi_i^e(jk)$ for Economy B
:header-rows: 1
:name: tbl-economy-b-fundamental-exchange

* - $\pi_i^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,0,0)$
  - $(0.098,[0,0.069],0)$
  - $(0.138,0.098,[0,0.167])$
* - $i=2$
  - $([0,0.529],0.236,0)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=3$
  - $([0,0.310],0,0.138)$
  - $(0.138,[0,0.098],0.098)$
  - $(0,0,0)$
```

```{list-table} Fundamental equilibrium exchange strategies $\tilde{\pi}_i^e(jk|j)$ for Economy B
:header-rows: 1
:name: tbl-economy-b-fundamental-strategies

* - $\tilde{\pi}_i^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $([0,1],0,0)?$
  - $(1,[0,1],0)$
  - $(1,1,[0,1])$
* - $i=2$
  - $([0,1],1,0)$
  - $(0,[0,1],0)?$
  - $(1,1,[0,1])?$
* - $i=3$
  - $([0,1],0,1)$
  - $(1,[0,1],1)$
  - $(0,0,[0,1])?$
```

```{list-table} Speculative equilibrium: probability that $i$ holds $j$, $\pi_i^h(j)$, for Economy B
:header-rows: 1
:name: tbl-economy-b-speculative-holdings

* - $\pi_i^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 0
  - 0.586
  - 0.414
* - $i=2$
  - 0.707
  - 0
  - 0.293
* - $i=3$
  - 0
  - 1
  - 0
```

```{list-table} Speculative equilibrium: joint exchange probability $\pi_i^e(jk)$ for Economy B
:header-rows: 1
:name: tbl-economy-b-speculative-exchange

* - $\pi_i^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(0,0,0)$
  - $(0.138,[0,0.310],0)$
  - $(0.098,0.138,[0,0.098])$
* - $i=2$
  - $([0,0.167],0.138,0.098)$
  - $(0,0,0)$
  - $(0,0.098,[0,0.069])$
* - $i=3$
  - $(0,0,0)$
  - $(0,[0,0.529],0.138)$
  - $(0,0,0)$
```

```{list-table} Speculative equilibrium exchange strategies $\tilde{\pi}_i^e(jk|j)$ for Economy B
:header-rows: 1
:name: tbl-economy-b-speculative-strategies

* - $\tilde{\pi}_i^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $([0,1],0,0)?$
  - $(1,[0,1],0)$
  - $(1,1,[0,1])$
* - $i=2$
  - $([0,1],1,1)$
  - $(0,[0,1],0)?$
  - $(0,1,[0,1])$
* - $i=3$
  - $([0,1],0,1)?$
  - $(0,[0,1],1)$
  - $(0,0,[0,1])?$
```

Economy B.1 displays an interesting pattern of evolution. At iteration 500 the distribution of holdings and, especially, the trading patterns correspond to the speculative equilibrium. However, the economy moves away from this state and by iteration 1000 has practically converged to the fundamental equilibrium.

```{list-table} Frequency with which $i$ holds $j$ at $t=500$ and $t=1000$ for Economy B.1
:header-rows: 1
:name: tbl-economy-b1-holdings

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - 0
  - 0.464
  - 0.536
* - $i=2$ ($t=500$)
  - 0.994
  - 0
  - 0.006
* - $i=3$ ($t=500$)
  - 0
  - 1
  - 0
* - $i=1$ ($t=1000$)
  - 0
  - 0.28
  - 0.72
* - $i=2$ ($t=1000$)
  - 0.994
  - 0
  - 0.006
* - $i=3$ ($t=1000$)
  - 0.526
  - 0.474
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=500$ and $t=1000$ for Economy B.1
:header-rows: 1
:name: tbl-economy-b1-exchange-freq

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=500$)
  - $(0,0,0)$
  - $(0.16,0.05,0.01)$
  - $(0.01,0.19,0.04)$
* - $i=2$ ($t=500$)
  - $(0,0.16,0.01)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=3$ ($t=500$)
  - $(0,0,0)$
  - $(0,0,0.18)$
  - $(0,0,0)$
* - $i=1$ ($t=1000$)
  - $(0,0,0)$
  - $(0.08,0.02,0.01)$
  - $(0.15,0.11,0.07)$
* - $i=2$ ($t=1000$)
  - $(0.41,0.25,0.01)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=3$ ($t=1000$)
  - $(0.11,0.01,0.15)$
  - $(0.18,0.03,0.10)$
  - $(0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=500$ and $t=1000$ for Economy B.1
:header-rows: 1
:name: tbl-economy-b1-winning

* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=500$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(1,0,0)$
  - $(1,1,1)$
  - $(1,1,1)$
* - $i=2$
  - $(0,1,0)$
  - $(1,1,0)$
  - $(1,1,1)$
* - $i=3$
  - $(0,0,1)$
  - $(0,0,1)$
  - $(0,0,0)$
* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=1000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(1,0,0)$
  - $(1,0,0)$
  - $(1,1,1)$
* - $i=2$
  - $(1,1,1)$
  - $(1,1,0)$
  - $(1,1,0)$
* - $i=3$
  - $(1,0,1)$
  - $(1,1,1)$
  - $(0,0,0)$
```

```{list-table} Consumption frequency $\pi_{it}^c(j|j)$ at $t=500$ for Economy B.1
:header-rows: 1
:name: tbl-economy-b1-consumption-freq

* - $\pi_{it}^c(j|j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - 1
  - 0.07
  - 0.508
* - $i=2$
  - 0.009
  - 1
  - 0.998
* - $i=3$
  - 0.050
  - 0.632
  - 1
```

Economy B.2 with random initial classifiers had not converged after 2000 periods. Nevertheless, the economy seems to be moving towards the fundamental equilibrium.

```{list-table} Parameter values used for Economy B.2
:header-rows: 1
:name: tbl-economy-b2-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 72, C_a = 12$
* - Storage costs
  - $s_1 = 1, s_2 = 4, s_3 = 9$
* - Utility
  - $u_i = 100, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.25, b_{22} = 0.25$
* - Specialization
  - $f_s(t) = 1/2\sqrt{t}, p_s = 0.01$
* - Generalization
  - $f_g(t) = 1/2\sqrt{t}, p_1 = 0.2, p_2 = 0.7, p_3 = 0.2, p_4 = 0.5, S = 0, N_e = 8, N_c = 4$
```

```{list-table} Frequency with which $i$ holds $j$ at $t=1000$ and $t=2000$ for Economy B.2
:header-rows: 1
:name: tbl-economy-b2-holdings

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=1000$)
  - 0
  - 0.434
  - 0.566
* - $i=2$ ($t=1000$)
  - 0.89
  - 0
  - 0.11
* - $i=3$ ($t=1000$)
  - 0.088
  - 0.912
  - 0
* - $i=1$ ($t=2000$)
  - 0
  - 0.354
  - 0.646
* - $i=2$ ($t=2000$)
  - 0.996
  - 0
  - 0.004
* - $i=3$ ($t=2000$)
  - 0.268
  - 0.732
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=1000$ and $t=2000$ for Economy B.2
:header-rows: 1
:name: tbl-economy-b2-exchange-freq

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$ ($t=1000$)
  - $(0,0,0)$
  - $(0.14,0.08,0.02)$
  - $(0.18,0.18,0.05)$
* - $i=2$ ($t=1000$)
  - $(0.14,0.41,0.19)$
  - $(0,0,0)$
  - $(0.03,0.05,0.1)$
* - $i=3$ ($t=1000$)
  - $(0.01,0,0.02)$
  - $(0.26,0.16,0.20)$
  - $(0,0,0)$
* - $i=1$ ($t=2000$)
  - $(0,0,0)$
  - $(0.14,0.04,0.01)$
  - $(0.06,0.16,0.07)$
* - $i=2$ ($t=2000$)
  - $(0.13,0.36,0)$
  - $(0,0,0)$
  - $(0,0,0)$
* - $i=3$ ($t=2000$)
  - $(0.04,0,0.06)$
  - $(0.22,0.12,0.16)$
  - $(0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=1000$ and $t=2000$ for Economy B.2
:header-rows: 1
:name: tbl-economy-b2-winning

* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=1000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(\text{—},0,\text{—})?$
  - $(1,1,1)$
  - $(1,1,0)$
* - $i=2$
  - $(1,1,1)$
  - $(\text{—},\text{—},\text{—})$
  - $(1,1,1)$
* - $i=3$
  - $(0,1,1)$
  - $(\text{—},\text{—},1)$
  - —
* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=2000$)
  - $j=1$
  - $j=2$
  - $j=3$
* - $i=1$
  - $(\text{—},1,\text{—})?$
  - $(1,1,0)$
  - $(1,1,0)$
* - $i=2$
  - $(1,0,1)$
  - $(\text{—},\text{—},\text{—})$
  - $(1,1,1)$
* - $i=3$
  - $(1,0,1)$
  - $(1,\text{—},1)$
  - —
```

In particular, with the exception of a zero-frequency event (II is willing to trade 1 for 3) the trading patterns in Economy B.2 correspond to the fundamental equilibrium.

These two simulations for Economy B provide examples in which the classifier systems seem to select the fundamental equilibrium over the speculative equilibrium.[^fn16] Furthermore, the results for Economy B.1 indicate that this is not the result of myopic behavior.

[^fn16]: In Marimon and Miller (1989) 30 out of 30 experiments resulted in convergence to the fundamental equilibrium.

### 7.4 Economy C (Fiat Money)

```{list-table} Parameter values used for Economy C
:header-rows: 1
:name: tbl-economy-c-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 150, C_a = 20$
* - Storage costs
  - $s_1 = 9, s_2 = 14, s_3 = 29, s_0 = 0$
* - Utility
  - $u_i = 100, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.025, b_{22} = 0.25$
* - Specialization
  - $f_s(t) = 1/2\sqrt{t}, p_s = 0.01$
* - Generalization
  - $f_g(t) = 1/2\sqrt{t}, p_1 = 0.2, p_2 = 0.7, p_3 = 0.2, p_4 = 0.5, S = 0, N_e = 8, N_c = 4$
```

In Economy C a new good, good 0, with the characteristics of 'fiat money' is introduced. No agent derives utility from consuming good 0. There are no storage costs associated with good 0.

Fiat money is introduced into the system by forcing some agents to store good 0 in period 0. In particular, 48 units of good 0 are randomly allocated to 48 agents. To avoid fluctuations in the quantity of money, agents are not allowed to consume good 0. A modified setup, not pursued here, would give agents the opportunity to learn not to eat money.

The production technologies are as in Economy A (see {numref}`tbl-economies`). {numref}`tbl-economy-c-eq-holdings` provides the probabilities that characterize the unique (fundamental) equilibrium of Economy C. The equilibrium trading patterns are depicted in {numref}`fig-economy-c`.

```{list-table} Fundamental equilibrium: probability that $i$ holds $j$, $\pi_i^h(j)$, for Economy C
:header-rows: 1
:name: tbl-economy-c-eq-holdings

* - $\pi_i^h(j)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - 0
  - 0.26
  - 0.62
* - $j=2$
  - 0.74
  - 0
  - 0
* - $j=3$
  - 0
  - 0.42
  - 0
* - $j=0$
  - 0.26
  - 0.32
  - 0.38
```

```{list-table} Fundamental equilibrium: joint exchange probability $\pi_i^e(jk)$ for Economy C
:header-rows: 1
:name: tbl-economy-c-eq-exchange

* - $\pi_i^e(jk)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(0,0,0,0)$
  - $([0,0.08],0.064,0,0.023)$
  - $([0,0.18],0,0.087,0.034)$
* - $j=2$
  - $(0.064,[0,0.08],0,0.079)$
  - $(0,0,0,0)$
  - $(0,0,0,0)$
* - $j=3$
  - $(0,0,0,0)$
  - $(0.087,0,[0,0.06],0.053)$
  - $(0,0,0,0)$
* - $j=0$
  - $(0.076,0,0,[0,0.08])$
  - $(0,0.079,0,[0,0.1])$
  - $(0,0,0.053,[0,0.12])$
```

```{list-table} Fundamental equilibrium exchange strategies $\tilde{\pi}_i^e(jk|j)$ for Economy C
:header-rows: 1
:name: tbl-economy-c-eq-strategies

* - $\tilde{\pi}_i^e(jk|j)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(\text{—},0,0,0)?$
  - $(\text{—},1,0,1)$
  - $(\text{—},0,1,1)$
* - $j=2$
  - $(1,\text{—},0,1)$
  - $(0,\text{—},0,0)?$
  - $(1,\text{—},1,1)?$
* - $j=3$
  - $(1,1,\text{—},1)?$
  - $(1,1,\text{—},1)$
  - $(0,0,\text{—},0)?$
* - $j=0$
  - $(1,0,0,\text{—})$
  - $(0,1,0,\text{—})$
  - $(0,0,1,\text{—})$
```

```{figure} figures/fig9_economy_c.png
:name: fig-economy-c
:width: 60%

Exchange pattern in Economy C.
```

```{list-table} Frequency with which $i$ holds $j$ at $t=750$ and $t=1250$ for Economy C
:header-rows: 1
:name: tbl-economy-c-holdings

* - $\pi_{it}^h(j)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$ ($t=750$)
  - 0
  - 0.18
  - 0.58
* - $j=2$ ($t=750$)
  - 0.68
  - 0
  - 0.01
* - $j=3$ ($t=750$)
  - 0
  - 0.58
  - 0
* - $j=0$ ($t=750$)
  - 0.32
  - 0.23
  - 0.41
* - $j=1$ ($t=1250$)
  - 0
  - 0.18
  - 0.77
* - $j=2$ ($t=1250$)
  - 0.54
  - 0
  - 0.01
* - $j=3$ ($t=1250$)
  - 0
  - 0.53
  - 0
* - $j=0$ ($t=1250$)
  - 0.46
  - 0.28
  - 0.21
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=750$ for Economy C
:header-rows: 1
:name: tbl-economy-c-exchange-freq-750

* - $\pi_{it}^e(jk)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(0,0,0,0)$
  - $(0,0.052,0.004,0.008)$
  - $(0.008,0.008,0.112,0.06)$
* - $j=2$
  - $(0.06,0,0,0.062)$
  - $(0,0,0,0)$
  - $(0,0,0.002,0.004)$
* - $j=3$
  - $(0,0,0,0)$
  - $(0.116,0.002,0.052,0.048)$
  - $(0,0,0,0)$
* - $j=0$
  - $(0.064,0,0,0)$
  - $(0,0.058,0,0)$
  - $(0.004,0.008,0.048,0.044)$
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=1250$ for Economy C
:header-rows: 1
:name: tbl-economy-c-exchange-freq-1250

* - $\pi_{it}^e(jk)$
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(0,0,0,0)$
  - $(0.016,0.028,0.002,0.028)$
  - $(0,0.016,0.028,0.002)$
* - $j=2$
  - $(0.044,0,0,0.066)$
  - $(0,0,0,0)$
  - $(0,0,0,0)$
* - $j=3$
  - $(0,0,0,0)$
  - $(0.014,0,0.044,0.044)$
  - $(0,0,0,0)$
* - $j=0$
  - $(0.06,0,0,0)$
  - $(0,0.066,0,0)$
  - $(0,0,0.044,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=750$ and $t=1250$ for Economy C
:header-rows: 1
:name: tbl-economy-c-winning

* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=750$)
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(\text{—},\text{—},\text{—},\text{—})$
  - $(1,1,0,1)$
  - $(0,0,1,0)$
* - $j=2$
  - $(1,0,0,1)$
  - $(\text{—},\text{—},\text{—},\text{—})$
  - $(0,0,1,1)$
* - $j=3$
  - $(0,0,0,0)$
  - $(1,1,0,1)$
  - $(0,0,0,1)$
* - $j=0$
  - $(1,0,0,0)$
  - $(0,1,0,0)$
  - $(0,0,1,1)$
* - $\tilde{\pi}_{it}^e(jk|j)$ ($t=1250$)
  - $i=1$
  - $i=2$
  - $i=3$
* - $j=1$
  - $(\text{—},\text{—},\text{—},\text{—})$
  - $(1,1,1,1)$
  - $(0,0,1,0)$
* - $j=2$
  - $(1,0,0,1)$
  - $(\text{—},\text{—},\text{—},\text{—})$
  - $(0,0,0,0)$
* - $j=3$
  - $(0,1,0,0)$
  - $(1,1,0,1)$
  - $(\text{—},\text{—},\text{—},\text{—})$
* - $j=0$
  - $(1,0,0,0)$
  - $(0,1,0,0)$
  - $(0,0,1,0)$
```

The economy converges remarkably fast to the fundamental equilibrium. Given the complexity of the exchange patterns with fiat money, the results for Economy C indicate the ability of our artificially intelligent agents to set up complex social arrangements like fiat money.[^fn17]

[^fn17]: A version of Economy C with the storage costs of Economy A did not converge, probably because the low cost of storing good 1 (0.1) made this good too good a substitute for fiat money. These results for Economy C have been successfully replicated in Marimon and Miller (1989).

### 7.5 Economy D (Five Goods, Five Types)

```{list-table} Parameter values used for Economy D
:header-rows: 1
:name: tbl-economy-d-params

* - Parameter
  - Value
* - No. of agents
  - $A_i = 50$
* - No. of classifiers
  - $E_a = 180, C_a = 20$
* - Storage costs
  - $s_1 = 1, s_2 = 4, s_3 = 9, s_4 = 16, s_5 = 30$
* - Utility
  - $u_i = 200, \forall i$
* - Initial strengths
  - $S_{e\tau_e^a(0)}^a = 0, S_{c\tau_c^a(0)}^a = 0$
* - Bids
  - $b_{11} = 0.025, b_{12} = 0.025, b_{21} = 0.25, b_{22} = 0.25$
* - Specialization
  - $f_s(t) = 1/2\sqrt{t}, p_s = 0.01$
* - Generalization
  - $f_g(t) = 1/2\sqrt{t}, p_1 = 0.2, p_2 = 0.7, p_3 = 0.2, p_4 = 0.5, S = 0, N_e = 8, N_c = 4$
```

In Economy D we enhance complexity by considering five goods and five types. The production technologies are described by {numref}`fig-economy-d-production`. That is, type I produces good 3, type II produces good 4, type III produces good 5, type IV produces good 1, and type V produces good 2. As before, each type of agent only derives utility from consuming the good of the same number. Storage costs are ranked in increasing order. For this economy we do not start with any characterization of a stationary equilibrium. With enough work, one could obtain an analytic solution for the fundamental equilibrium for such an economy, but here our purpose is to let our artificially intelligent agents suggest to us what an equilibrium might be. Given this suggestion, we could then presumably verify whether it is an equilibrium.

```{figure} figures/fig10_economy_d_production.png
:name: fig-economy-d-production
:width: 60%

Production patterns in Economy D; an arrow shows that the agent at the origin of the arrow produces the good at the point of the arrow.
```

```{list-table} Frequency with which $i$ holds $j$ at $t=500$ for Economy D
:header-rows: 1
:name: tbl-economy-d-holdings-500

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - 0
  - 0.336
  - 0.626
  - 0.032
  - 0.006
* - $i=2$
  - 0.322
  - 0
  - 0.038
  - 0.614
  - 0.026
* - $i=3$
  - 0.118
  - 0.180
  - 0
  - 0.054
  - 0.648
* - $i=4$
  - 0.922
  - 0.044
  - 0.002
  - 0
  - 0.032
* - $i=5$
  - 0.210
  - 0.720
  - 0.038
  - 0.032
  - 0
```

```{list-table} Frequency with which $i$ holds $j$ at $t=1750$ for Economy D
:header-rows: 1
:name: tbl-economy-d-holdings-1750

* - $\pi_{it}^h(j)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - 0
  - 0.832
  - 0.158
  - 0.004
  - 0.006
* - $i=2$
  - 0.096
  - 0
  - 0.004
  - 0.870
  - 0.030
* - $i=3$
  - 0.048
  - 0.418
  - 0
  - 0.052
  - 0.482
* - $i=4$
  - 0.988
  - 0.008
  - 0.002
  - 0
  - 0.002
* - $i=5$
  - 0
  - 0.974
  - 0.024
  - 0.002
  - 0
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=500$ for Economy D
:header-rows: 1
:name: tbl-economy-d-exchange-freq-500

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - $(0,0,0,0,0)$
  - $(0.03,0.02,0.01,0.01,0)$
  - $(0.06,0.05,0.03,0,0)$
  - $(0.01,0,0,0,0)$
  - $(0,0,0,0,0)$
* - $i=2$
  - $(0.08,0.11,0.04,0.01,0.02)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0.17,0.04,0.01,0.03,0.03)$
  - $(0,0,0,0,0)$
* - $i=3$
  - $(0.04,0,0.02,0.02,0)$
  - $(0.03,0,0.01,0.02,0.01)$
  - $(0,0,0,0,0)$
  - $(0.01,0,0,0.01,0)$
  - $(0.09,0.09,0.01,0.03,0.03)$
* - $i=4$
  - $(0.24,0.04,0,0.14,0.03)$
  - $(0,0,0,0,0.01)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
* - $i=5$
  - $(0.03,0,0,0.03,0.03)$
  - $(0.09,0.02,0.03,0,0.08)$
  - $(0,0,0,0.01,0.01)$
  - $(0.01,0,0,0,0)$
  - $(0,0,0,0,0)$
```

```{list-table} Exchange frequency $\pi_{it}^e(jk)$ at $t=1750$ for Economy D
:header-rows: 1
:name: tbl-economy-d-exchange-freq-1750

* - $\pi_{it}^e(jk)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - $(0,0,0,0,0)$
  - $(0.03,0.17,0.01,0.01,0)$
  - $(0,0.05,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
* - $i=2$
  - $(0.01,0.04,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0.18,0.04,0,0.10,0.03)$
  - $(0,0,0,0,0)$
* - $i=3$
  - $(0.01,0.01,0,0.01,0)$
  - $(0.01,0.05,0.02,0.04,0.02)$
  - $(0,0,0,0,0)$
  - $(0.01,0,0,0,0)$
  - $(0,0.1,0,0.03,0.03)$
* - $i=4$
  - $(0.09,0.03,0,0.19,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
* - $i=5$
  - $(0,0,0,0,0)$
  - $(0.03,0.26,0.04,0,0.07)$
  - $(0.01,0,0,0,0)$
  - $(0,0,0,0,0)$
  - $(0,0,0,0,0)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=500$ for Economy D
:header-rows: 1
:name: tbl-economy-d-winning-500

* - $\tilde{\pi}_{it}^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - $(1,\text{—},\text{—},\text{—},0)$
  - $(1,1,1,0,0)$
  - $(1,0,1,0,0)$
  - $(1,1,1,1,\text{—})$
  - $(0,\text{—},\text{—},0,1)$
* - $i=2$
  - $(1,1,1,1,0)$
  - $(1,1,1,\text{—},\text{—})$
  - $(1,1,1,0,0)$
  - $(1,1,1,0,0)$
  - $(1,1,1,1,\text{—})$
* - $i=3$
  - $(1,0,1,1,0)$
  - $(1,0,1,1,1)$
  - $(1,\text{—},1,1,\text{—})$
  - $(1,\text{—},1,1,1)$
  - $(1,1,1,1,0)$
* - $i=4$
  - $(1,0,0,1,1)$
  - $(1,\text{—},0,1,1)$
  - $(1,\text{—},1,1,\text{—})$
  - $(1,\text{—},1,1,\text{—})$
  - $(0,1,1,1,\text{—})$
* - $i=5$
  - $(1,0,0,1,1)$
  - $(1,0,0,0,1)$
  - $(1,0,0,1,1)$
  - $(1,1,1,1,1)$
  - $(\text{—},\text{—},\text{—},\text{—},1)$
```

```{list-table} Winning classifier actions $\tilde{\pi}_{it}^e(jk|j)$ at $t=1750$ for Economy D
:header-rows: 1
:name: tbl-economy-d-winning-1750

* - $\tilde{\pi}_{it}^e(jk|j)$
  - $j=1$
  - $j=2$
  - $j=3$
  - $j=4$
  - $j=5$
* - $i=1$
  - $(1,\text{—},\text{—},\text{—},0)$
  - $(1,1,0,1,0)$
  - $(1,1,0,0,0)$
  - $(1,1,1,1,1)$
  - $(0,1,1,1,0)$
* - $i=2$
  - $(1,1,1,0,0)$
  - $(\text{—},\text{—},\text{—},\text{—},\text{—})$
  - $(1,1,0,\text{—},1)$
  - $(1,1,1,0,0)$
  - $(1,1,1,1,1)$
* - $i=3$
  - $(1,0,1,1,\text{—})$
  - $(0,1,1,0,0)$
  - $(\text{—},\text{—},\text{—},\text{—},\text{—})$
  - $(0,0,1,0,\text{—})$
  - $(1,1,1,1,0)$
* - $i=4$
  - $(0,0,0,1,0)$
  - $(1,0,1,1,0)$
  - $(0,0,0,1,0)$
  - $(\text{—},\text{—},\text{—},\text{—},\text{—})$
  - $(1,1,1,1,1)$
* - $i=5$
  - $(1,0,\text{—},0,1)$
  - $(1,1,1,0,1)$
  - $(0,1,0,1,1)$
  - $(1,0,\text{—},1,1)$
  - $(\text{—},\text{—},\text{—},\text{—},\text{—})$
```

From the simulation results, we can see that the trading patterns nearly seem to describe a fundamental equilibrium in which agents are only willing to trade for commodities of lower cost than the one currently in storage, except that they always accept the commodity of their type. These fundamental trading patterns are shown in {numref}`fig-economy-d-exchange`. Some speculative moves can be detected. An example of this is that agents of type II accept (from I) good 3 for good 1 in order to trade 3 for 2 with type III. We do not pursue here a full analysis of partially speculative equilibria for this economy.

```{figure} figures/fig11_economy_d_exchange.png
:name: fig-economy-d-exchange
:width: 80%

Exchange patterns in Economy D exhibited by classifier systems.
```

(sec-conclusions)=
## 8. Conclusions

The work described above is presented as the first steps of our project to use the classifier systems of Holland to model learning in multi-agent environments. Classifier systems have previously been applied to solve some complex optimization problems [see {cite}`machinelearning1988`]. Our application has involved some extensions to handle the fact that ours is a multi-agent environment in which agents are solving Markovian dynamic decision problems.

The work described in this paper has accomplished two objectives that are parts of our broader project. First, our simulations have demonstrated by example that multi-agent systems of classifiers can exhibit interesting behavior, and can eventually learn to play Nash-Markov equilibria. Second, the process of making classifier systems cope with the Kiyotaki-Wright environment has prompted us to formalize a variety of concepts and definitions that our subsequent work shall build on. These definitions will facilitate formal analyses of systems of multi-agent classifier systems, which we intend to pursue.

Among the unfinished aspects which we leave for future papers, two interrelated ones are quite important. First, our experiments with the genetic-type algorithms in our incomplete-enumeration classifier systems have convinced us that existing algorithms can be improved. These improvements, which will increase the amount of experimentation that is done at the 'right times', will have the effect of improving the capacity of the classifier systems to settle upon optimal outcomes more rapidly, and to avoid becoming locked in to suboptimal patterns of interaction. The improved algorithms will also help with a second major piece of unfinished business, which is to develop analytical results on the convergence of classifier systems. The new algorithms will be set up in such a way that we can apply the stochastic approximation approach to convergence analysis which we alluded to in {ref}`sec-convergence`.

## References

```{bibliography}
:filter: docname in docnames
```
