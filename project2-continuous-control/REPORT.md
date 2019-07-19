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

<!-- URLS -->


# Using DDPG to solve the Reacher environment from ML-Agents

This is an accompaying report for the project-2 submission. Below we list some key details we
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

In this section we give a brief description of the DDPG algorithm from [2], as our banana collector 
agent is based on the DQN agent from that paper. This description is a brief adaptation from the overview we gave
in [this][url_project_1_post_part_1] post (part 1, section 3). We gave an intro and various details in that post so 
please, for further info about the algorithm refer to that post for a more detailed explanation.

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
  that similar to the stochastic policy case, and that's you could actually call this an approximate
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

> **Algorithm: REINFORCE (Monte-Carlo policy gradients)**
>  * Initialize the policy parameters &theta;
>  * For *episode* = 0,1,2, ...
>    * Generate an episode &tau; = (s<sub>0</sub>,a<sub>0</sub>,r<sub>1</sub>,...,s<sub>T-1</sub>,a<sub>T-1</sub>,r<sub>T</sub>) using policy &pi;<sub>&theta;</sub>
>    * For i = 1,...,T

### 1.2 Q-network architecture


## 2. Implementation


### 2.1 Agent


### 2.2 Model


### 2.3 Replay buffer


### 2.4 Trainer


### 2.5 Hyperparameters


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
* [2] 