[//]: # (References)

<!-- IMAGES -->
[img_results_experiment_1_all_runs]: https://wpumacay.github.io/research_blog/imgs/img_results_experiment_1_all_runs.png
[img_results_experiment_1_all_runs_std]: https://wpumacay.github.io/research_blog/imgs/img_results_experiment_1_all_runs_std.png
[img_results_experiment_2_all_runs_std]: https://wpumacay.github.io/research_blog/imgs/img_results_experiment_2_all_runs_std.png
[img_results_experiment_3_all_runs_std]: https://wpumacay.github.io/research_blog/imgs/img_results_experiment_3_all_runs_std.png
[img_results_submission_single_pytorch]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_single_pytorch.png
[img_results_submission_single_tensorflow]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_single_tensorflow.png
[img_results_submission_all_runs]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs.png
[img_reulsts_submission_all_runs_pytorch_std]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs_pytorch_std.png
[img_results_submission_all_runs_tensorflow_std]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs_tensorflow_std.png

[img_rl_pg_objective]: imgs/img_rl_pg_objective.png
[img_rl_pg_gradient_ascent]: imgs/img_rl_pg_gradient_ascent.png
[img_rl_pg_stochastic_policy_gradient_theorem]: imgs/img_rl_pg_stochastic_policy_gradient_theorem.png
[img_rl_pg_deterministic_policy_gradient_theorem]: imgs/img_rl_pg_deterministic_policy_gradient_theorem.png
[img_rl_pg_log_likelihood_policy_gradient]: imgs/img_rl_pg_log_likelihood_policy_gradient.png
[img_rl_pg_baselines]: imgs/img_rl_pg_baselines.png
[img_rl_ddpg_learning_critic]: imgs/img_rl_ddpg_learning_critic.png
[img_rl_ddpg_learning_actor]: imgs/img_rl_ddpg_learning_actor.png
[img_rl_ddpg_algorithm]: imgs/img_rl_ddpg_algorithm.png
[img_rl_ddpg_polyak_averaging]: imgs/img_rl_ddpg_polyak_averaging.png
[img_rl_q_learning_ccontrol]: imgs/img_rl_q_learning_ccontrol.png
[img_rl_ddpg_learning_maximizer]: imgs/img_rl_ddpg_learning_maximizer.png
[img_ddpg_network_architecture_actor]: imgs/img_ddpg_network_architecture_actor.png
[img_ddpg_network_architecture_critic]: imgs/img_ddpg_network_architecture_critic.png

<!-- URLS -->
[url_readme]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/README.md
[url_torchsummary]: https://github.com/sksq96/pytorch-summary

# Using DDPG to solve the Reacher environment from ML-Agents

This is an accompanying report for the project-2 submission. Below we list some key details we
will cover in our submission:

* [Agent description](#1-agent-description)
    * [DDPG algorithm overview](#11-ddpg-algorithm-overview)
    * [Models architecture](#12-models-architecture)
    
* [Implementation](#2-implementation)
    * [Agent implementation](#21-agent)
    * [Model implementation](#22-model)
    * [Utilities implementation](#23-utilities)
    * [Trainer implementation](#24-trainer)
    * [Choice of hyperparameters](#25-hyperparameters)

* [Results](#3-results)
    * [Running a pretrained agent](#30-running-a-pretrained-agent)
    * [Submission results](#31-submission-results)

* [Future work](#4-future-work)

## 1. Agent description

In this section we give a brief description of the DDPG algorithm from [2]. There 
are various details to take into account in order to understand the algorithm properly, 
so below I try to give a small overview that might be helpful to understand a bit 
better the algorithm and contrast it with other policy based algorithms in the literature.

### 1.1 Deep Deterministic Policy Gradients overview

**TL;DR**
> DDPG [2] is an actor-critic algorithm that can be thought as an **approximate version of DQN** 
> for continuous action spaces, in which we learn both the action-value function
> **Q(s,a;&phi;)** and a deterministic policy **&mu;<sub>&theta;</sub>(s)** by using
> the **Deterministic Policy Gradients** theorem from [3]. It builds on top of the DPG
> algorithm by using the features introduced in [4] to stabilize Q-learning when using
> DNN as function approximators, namely *Experience Replay* and *Target Networks*.

The *Deep Deterministic Policy Gradients* algorithm is an off-policy model-free policy-based 
actor-critic RL method. Let's try to explain each part of this statement in more detail:

* **Policy-based**: Unlike its value-based counterparts (like DQN), this method tries to
  learn the policy that the agent should use to maximize its objective directly. Recall
  that value-based methods (like Q-learning) try to learn an action-value function 
  to then recover the implict policy (greedy policy).

* **Actor-critic**: Although this algorithm tries to learn the policy, it also learns
  an action-value function to help learning. In various actor-critic methods this is used
  to reduce variance by not using MC estimates of the return to compute advantange estimates
  used for the gradient estimation of the objective. However, as we will see later, this is not
  that similar to the stochastic policy case, and that's why you could actually call this an approximate
  version of DQN instead of an actual actor-critic algorithm.

* **Model-free**: We do not need access to the dynamics of the environment. This algorithm
  learns the policy using samples taken from the environment. We learn the action-value function
  (critic) by using *Q-learning* over samples taken from the world, and the policy by
  using the *Deterministic Policy Gradients* theorem over those samples.

* **Off-policy**: The sample experiences that we use for learning do not necessarily come
  from the actual policy we are learning, but instead come from a different policy (exploratory
  policy). As in DQN, we store these experiences in a replay buffer and learn from
  samples of this buffer, which might come from different timesteps (and potentially from
  different versions of the exploratory policy).

To get a better understanding of this algorithm, and contrast it with other different
policy gradient based algorithms found in the literature, let's do a quick recap of
the results from the Stochastic Policy Gradients theorem and the Vanilla Policy Gradients
algorithm.

### **Policy Gradients**

In this setup we usually deal with stochastic policies **&pi;<sub>&theta;</sub>(a|s)**, and
we try to get the parameters **&theta;** that maximize an objective function **J(&theta;)**, 
as shown below:

![pg-objective][img_rl_pg_objective]

We achieve this by doing gradient ascent on the objective, i.e. computing an estimate of
the gradient of our objective and updating our parameters with it:

![pg-gradient-ascent][img_rl_pg_gradient_ascent]

Unfortunately, in order to compute this gradient directly we would need to have 
knowledge of the distribution **p(&tau;,&theta;)** of trajectories, i.e. compute
this gradient: &nabla;<sub>&theta;</sub> &#8721;<sub>&tau;</sub> p(&tau;,&theta;) R(&tau;). It would
be nice to be able to compute an estimate of this gradient by just using some sample
trajectories &tau; obtained from interactions with the environment. Luckily, we can
compute an unbiased estimate of this gradient by using the **Policy Gradient Theorem**.

![pg-stochastic-policy-gradient-theorem][img_rl_pg_stochastic_policy_gradient_theorem]

The states **s** are still sampled from a stationary distribution d<sup>&pi;</sup>, which
we could get at estimate of by samples, but we would still need to compute the quantity inside
the expectation using a sum over all actions (or an integral on the continuous case), which
is information that we don't actually have as we don't take all actions during a
trajectory. To fix this we can make use of the likelihood-ratio trick, in which we replace
the gradient &nabla;<sub>&theta;</sub>&pi;<sub>&theta;</sub>(a|s) by its equivalent 
&pi;<sub>&theta;</sub>(a|s)&nabla;<sub>&theta;</sub>log &pi;<sub>&theta;</sub>(a|s)
and then get rid of the sum over actions (or integral) by replacing it with an expectation,
as shown below:

![pg-log-likelihood-policy-gradient][img_rl_pg_log_likelihood_policy_gradient]

We can now compute this estimate of the gradient by using samples from our trajectories,
namely state action pairs **(s,a)** where states **s** in the trajectory are samples 
coming from d<sup>&pi;</sup>, and actions **a** are actions taken at that state sampled
from the distribution given by the policy. 

We have many options to compute Q<sup>&pi;(s,a)</sup>, and one such approach would be
to compute an unbiased estimate of this value using the return G<sub>t</sub> (a Monte-Carlo
estimate). This yields the REINFORCE algorithm, shown below:

> **Algorithm: REINFORCE (Monte-Carlo policy gradients)**
>  * Initialize the policy parameters &theta;
>  * For *episode* = 0,1,2, ...
>    * Generate an episode &tau; = (s<sub>0</sub>,a<sub>0</sub>,r<sub>1</sub>,...,s<sub>T-1</sub>,a<sub>T-1</sub>,r<sub>T</sub>) using policy &pi;<sub>&theta;</sub>
>    * For t = 0,...,T-1
>       * Compute estimate of Q<sup>&pi;</sup>(s<sub>t</sub>,a<sub>t</sub>) using the return G<sub>t</sub> (MC estimate)
>       * Update paremeters &theta; using gradient ascent: &theta;:= &theta; + &alpha; G<sub>t</sub> &nabla;<sub>&theta;</sub>log &pi;(a|s)

Another such variant comes from the fact that we can substract a term that depends
only on the states **s** from the estimate of Q<sup>&pi;</sup> and we would still
get an unbiased estimate. Such terms are called **baselines**, and the resulting terms
after substracting the baselines are called **advantages**. The usage of baselines is
that a good baseline can help reduce variance. Once such baseline could be the average 
return, which is usually a good idea to use. The idea of a baseline is shown in the 
equation below:

![pg-baselines][img_rl_pg_baselines]

Another option for the baseline could be a value-function **V<sub>&phi;</sub>(s)** 
whose parameters we can learn by fitting to targets (either MC estimates or TD estimates).
We could then go a bit further and use this value function to bootstrap the value of Q<sup>&pi;</sup>
using a TD-estimate like in TD(0). In this case our value function becomes a **critic**
and we end up with an actor critic algorithm. For example, one such variant (but 
perhaps not good one due to convergence issues like in fitted Q-learning) is shown
in the algorithm below:

> **Algorithm: TD(0) Actor-Critic**
>  * Initialize **actor** &pi;<sub>&theta;</sub>(a|s) and **critic** V<sub>&phi;</sub>(s)
>  * For *episode* = 0,1,2, ...
>    * Generate an episode &tau; = (s<sub>0</sub>,a<sub>0</sub>,r<sub>1</sub>,...,s<sub>T-1</sub>,a<sub>T-1</sub>,r<sub>T</sub>) using policy &pi;<sub>&theta;</sub>
>    * For t = 0,...,T-1
>       * Compute *Advantage Estimate* using TD(0) : A<sub>t</sub> = &delta;<sub>t</sub> = r<sub>t+1</sub> + &gamma;V<sub>&phi;</sub>(s<sub>t+1</sub>,a<sub>t+1</sub>) - V<sub>&phi;</sub>(s<sub>t</sub>)
>       * Update paremeters &theta; using gradient ascent: &theta; := &theta; + &alpha; A<sub>t</sub> &nabla;<sub>&theta;</sub>log &pi;(a<sub>t</sub>|s<sub>t</sub>)
>       * Update parameters &phi; using gradient descent on MSE-loss: &phi; := &phi; + &beta; &delta;<sub>t</sub> &nabla;<sub>&phi;</sub>V<sub>&phi;</sub>(s<sub>t</sub>)
> 

### **Deterministic Policy Gradients**

Unlike the stochastic policies used earlier, what if we wanted to use a deterministic
policy?. Let's say we have a deterministic policy &mu;<sub>&theta;</sub>(s) that we
want to learn. Luckily, we can use the **Deterministic Policy Gradient Theorem**
from [3], which give us a way of computing the gradient of our objective (still we 
want to maximize expected return) w.r.t. the policy parameters &theta;, as follows:

![pg-deterministic-policy-gradient-theorem][img_rl_pg_deterministic_policy_gradient_theorem]

Notice that this gradient is the expectation of the gradient of the Q-function w.r.t.
the action taken at state **s** (chosen by the policy &mu;<sub>&theta;</sub>) multiplied
by the gradient of the policy, which resembles a kind of chain rule when using compositions
like f(g(x)) (a fact that we will later use when we implement it using autodiff).

The authors in [2] built on top of this result and the improvements presented by
DQN in [4] to devise the **Deep Deterministic Policy Gradient** algorithm, in which
we learn both an actor &mu;<sub>&theta;</sub>(s) and a critic Q<sub>&phi;</sub>(s,a)
in a stable way using both **Experience Replay** and **Target Networks** from [4],
and **Soft-Updates** (via Polyak averaging). The key features of this algorithm are
the following:

* The critic learns in a way similar to DQN, but instead of maximizing over actions 
  to take the best Q-value, it uses actions given by the actor instead, avoiding 
  having to do expensive optimization steps to compute *max<sub>a</sub> Q(s,a)*.

![ddpg-learning-critic][img_rl_ddpg_learning_critic]

* The actor learns by using the *deterministic policy gradient theorem*, using the
  critic to compute the required gradients. Notice that we compound the actor inside
  the critic in the theorem expression to make it easier for Deep Learning packages
  to compute the gradient using autodiff.

![ddpg-learning-actor][img_rl_ddpg_learning_actor]

* Instead of updating the target networks at a certain frequency (hard-updates),
  the authors proposed to use **Polyak averaging** to softly update the target
  networks (for both actor and critic).

![ddpg-polyak-averaging][img_rl_ddpg_polyak_averaging]

By combining all these features, and the use of batch normalization for the networks
to improve learning, we get the DDPG algorithm from [2]:

![ddpg-algorithm][img_rl_ddpg_algorithm]

### **An approximate version of DQN**

It's useful to think of the DDPG algorithm as an approximate version of the DQN
algorithm for continuous actions spaces. Recall the problem we had when trying to 
use Value-based methods for continuous action spaces: it requires to solve an optimization
problem to get an action from the q-function.

![ddpg-q-learning-issue][img_rl_q_learning_ccontrol]

One approach to avoid this maximization step (which would be costly, and would be
required every learning step) is to **learn a maximizer** (the actor):

![ddpg-learning-maximizer][img_rl_ddpg_learning_maximizer]

So, it's like we were actually still using DQN, but "adapted" for continuous action
spaces by learning a maximizer to evaluate the actions for the Critic. As we will
see in the implementation section, the code for our DDPG agent is very similar to
the DQN code from the previous project, with the difference that we now have an actor
&mu;<sub>&theta;</sub>(s), which brings two more networks (including its corresponding
target network) to our implementation.

### 1.2 Models architecture

As starting we decided to use similar architectures to the ones presented in the
example code provided by udacity, with the small difference in the amount and size
of the layers used, and the usage of **Batch Normalization** as in the original DDPG
paper, which was also suggested in various forums.

### **Actor-network architecture**

Our actor consist of a deterministic policy, so instead of producing as outputs
the parameters of a distribution of a stochastic policy (like the mean and stddev
of a gaussian) or similar, it outputs an action from the action space as a one-to-one
mapping. 

As we saw in the [README.md][url_readme], our environment consists of an observation
space of vector observations of 33 dimensions (rank-1 tensor of size 33), and an action
space of vector actions of 4 dimensions (rank-1 tensor of size 4). Hence, our network
has as inputs vectors of size 33, and outputs of size 4. In between there are fully
connected layers with ReLU activations and Batch Normalization layers just before
the activations (except at the inputs, where we use batchnorm right away on the inputs).
Below we show an image of the architecture, and also a small summary of the network
printed using [torchsummary][url_torchsummary].

![ddpg-actor-model-architecture][img_ddpg_network_architecture_actor]

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
       BatchNorm1d-1                   [-1, 33]              66
            Linear-2                  [-1, 256]           8,704
       BatchNorm1d-3                  [-1, 256]             512
            Linear-4                  [-1, 128]          32,896
       BatchNorm1d-5                  [-1, 128]             256
            Linear-6                    [-1, 4]             516
================================================================
Total params: 42,950
Trainable params: 42,950
Non-trainable params: 0
----------------------------------------------------------------

```

### **Critic-network architecture**

Our critic consists of a Q-network Q<sub>&phi;</sub>(s,a) that must combine both 
states and actions pairs (or batches) and produce a single scalar (or batch of 
scalars) as outputs representing the q-values for those pairs, unlike the DQN from
the previous project (output all q-values for all actions)

Our network receives as inputs the observations of the agent, it processes them
using some layers (just one fully connected in this case) and then concatenates
the hidden states in that layer and concatenates them with the actions taken by
the agent into a single vector, and then continues processing it using two fully
connected layers. Batchnorm is only used for the inputs in this case, as it worked
well in our first experiments. At first we used batchnorm in all layers, as in the
actor network, but after some tuning and changes to the architecture, we ended up
having a working architecture for the task that had fewer layers and with less tricks
as possible (the one we show in the image below).

![ddpg-critic-model-architecture][img_ddpg_network_architecture_critic]

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
       BatchNorm1d-1                   [-1, 33]              66
            Linear-2                  [-1, 128]           4,352
            Linear-3                  [-1, 128]          17,024
            Linear-4                    [-1, 1]             129
================================================================
Total params: 21,571
Trainable params: 21,571
Non-trainable params: 0
----------------------------------------------------------------
```

## 2. Implementation

In this section we give a brief description of our implementation of DDPG, which 
is based in the algorithm presented in [2], and in various DDPG implementations online,
like [4] and [5].

### 2.1 Agent


### 2.2 Models


### 2.3 Trainer


### 2.4 Hyperparameters


## 3. Results



### 3.0 Running a pretrained agent

We provide a trained agent (trained with the config_submission.json configuration).
To use it, just run the following in your terminal:

```bash
python trainer.py test --sessionId=banana_submission
```

The weights of the trained agent are provided in the **results/banana_submission** folder,
and are stored in the **banana_submission_model_pytorch.pth** file.

Also, you can check [this](https://youtu.be/ng7e61LNNLs) video of a the pretrained 
agent collecting bananas.

### 3.1 Submission results


### 3.2 Experiments


## 4. Future Work

Finally, below we mention some of the improvements we consider making in following
updates to this post:

* TODO

## References

* [1] Sutton, Richard & Barto, Andrew. [*Reinforcement Learning: An introduction.*](http://incompleteideas.net/book/RLbook2018.pdf)
* [2][*Continuous control through deep reinforcement learning* paper by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf)
* [3][*Deterministic Policy Gradients Algorithms* paper by Silver et. al.](http://proceedings.mlr.press/v32/silver14.pdf)
* [4][DDPG implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
* [5][Post on *Deep Deterministic Policy Gradients* from OpenAI's **Spinning Up in RL**](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [6][Post on *Policy Gradient Algorithms* by **Lilian Weng**](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
* [7][*Lecture 8: Advanced Q-Learning Algorithms* from Berkeley's cs294 DeepRL course](https://youtu.be/hP1UHU_1xEQ?t=4365)
* [8] [Udacity DeepRL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)