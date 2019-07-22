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
[url_stable_baselines]: https://github.com/hill-a/stable-baselines
[url_dopamine]: https://github.com/google/dopamine

[url_impl_agent]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/core/agent.py
[url_impl_models_interface]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/core/model.py
[url_impl_models_pytorch]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/models/pytorch.py
[url_impl_config]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/config.py
[url_impl_util_replay_buffer]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/replaybuffer.py
[url_impl_util_noise]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/noise.py
[url_impl_util_env_wrapper]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/envs/mlagents.py
[url_impl_trainer]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/trainer.py

[url_agent_method_constructor]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/agent.py#L28
[url_agent_method_act]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/agent.py#L87
[url_agent_method_update]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/agent.py#L111
[url_agent_method_learn]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/agent.py#L142

[url_models_core_backbone]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/model.py#L10
[url_models_core_actor_head]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/model.py#L65
[url_models_core_critic_head]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/core/model.py#L189

[url_pytorch_base_backbone]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L29
[url_pytorch_actor_backbone]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L72
[url_pytorch_critic_backbone]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L116
[url_pytorch_actor_head]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L170
[url_pytorch_critic_head]: https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L246

[url_config_default_gin_file]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/configs/ddpg_reacher_multi_default.gin
[url_config_mechanism]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/config.py

[url_utils_replay_buffer]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/replaybuffer.py
[url_utils_noise]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/ddpg/utils/noise.py
[url_utils_env_wrapper]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/ccontrol/envs/mlagents.py

[url_trainer_method_main]: https://github.com/wpumacay/DeeprlND-projects/blob/cbbab21d795b8c0b1d38cacc5f07662efae6352b/project2-continuous-control/trainer.py#L151
[url_trainer_method_train]: https://github.com/wpumacay/DeeprlND-projects/blob/cbbab21d795b8c0b1d38cacc5f07662efae6352b/project2-continuous-control/trainer.py#L38

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
like [4] and [5]. We followed a similar approach to the previous projects and decided 
to abstract away some of the functionality of the agent and models (based also in some
rl-frameworks online, like [this][url_stable_baselines] one). The implementation consists 
of the following components:

* **Agent** : Base DDPG agent class, which instantiates all components and implements DDPG.
* **Model** : Abstract classes (interfaces) from which backend-specific models will inherit.
* **Config** : Configuration structures linked with the [gin-config][url_gin_config] package 
               to easily use, tweak and share hyperparameters.
* **Utils**: Some helper functions and classes used throughout the implementation, like replay
             buffers, and env. wrappers.
* **Trainer** : Script in charge of loading configuration structures, instantiating both
                environment and agent, implementing the training loop, and using the required
                logging functionality to save training results.

Below we give a more detailed explanation of these components:

### 2.1 **Agent**

**Implementation: [agent.py][url_impl_agent]**

The agent implementation consists of a base class container that composes all functionality
required for our DDPG agent (networks, noise process, replay buffer, etc.) and implements
the DDPG algorithm at a high-level, delegating various backend-specific operations to the
appropriate implementations, like pytorch implementations of the models. Let's see each part
of the implementation in more detail:

* First, the constructor of the agent is in charge of creating and composing all required 
  components for our DDPG implementation. It receives a configuration structure with some
  hyperparameters, and also receives the created actor and critic models which will be used
  for the DDPG algorithm. Notice we create the corresponding target networks for both actor
  and critic. We then create the requested noise process, either by using a Ornstein-Uhlenbeck 
  process, or just a simple normal distribution noise sampler. Also, we create the replay buffer,
  which in this case is a simple Circular Replay Buffer (no priority for now).

```python
class DDPGAgent( object ) : 
    r"""DDPG core agent class

    Implements a DDPG agent as in the paper 'Continuous control with deep reinforcement learning'
    by Lillicrap et. al., which implements an approximate DQN for continuous action
    spaces by making an 'actor' that approximates the maximizer used for a 'critic'. Both
    are trained using off-policy data from an exploratory policy based on the actor.

    Args:
        config (config.DDPGAgentConfig) : configuration object for this agent
        actorModel (model.IDDPGActor)   : model used for the actor of the ddpg-agent
        criticModel (model.IDDPGCritic) : model used for the critic of the ddpg agent

    """
    def __init__( self, agentConfig, actorModel, criticModel ) :
        super( DDPGAgent, self ).__init__()

        # keep the references to both actor and critic
        self._actor = actorModel
        self._critic = criticModel
        # and create some copies for the target networks
        self._actorTarget = self._actor.clone()
        self._criticTarget = self._critic.clone()
        # hint these target networks that they are actual target networks,
        # which is kind of a HACK to ensure batchnorm is called during eval
        # when using these networks to compute some required tensors
        self._actorTarget.setAsTargetNetwork( True )
        self._criticTarget.setAsTargetNetwork( True )

        # keep the reference to the configuration object
        self._config = agentConfig

        # directory where to save both actor and critic models
        self._savedir = './results/session_default'

        # step counter
        self._istep = 0

        # replay buffer to be used
        self._rbuffer = replaybuffer.DDPGReplayBuffer( self._config.replayBufferSize )

        # noise generator to be used
        if self._config.noiseType == 'ounoise' :
            self._noiseProcess = noise.OUNoise( self._config.actionsShape,
                                                self._config.noiseOUMu,
                                                self._config.noiseOUTheta,
                                                self._config.noiseOUSigma,
                                                self._config.seed )
        else :
            self._noiseProcess = noise.Normal( self._config.actionsShape,
                                               self._config.noiseNormalStddev,
                                               self._config.seed )

        # epsilon factor used to adjust exploration noise
        self._epsilon = 1.0

        # mode of the agent, either train or test
        self._mode = 'train'
```

* The next important method is the [**act**][url_agent_method_act], which is in
  charge of picking an action by using the actor and the state information given
  as inputs to the method. During testing we make sure to add some noise to help 
  with exploration. The noise is then annealed to a small value over the training
  period, in a similar way to the e-greedy schedule we had for DQN in the previous
  project.

```python
    def act( self, state ) :
        r"""Returns an action to take in state(s) 'state'

        Args:
            state (np.ndarray) : state (batch of states) to be evaluated by the actor

        Returns:
            (np.ndarray) : action (batch of actions) to be taken at that situation

        """
        assert state.ndim > 1, 'ERROR> state should have a batch dimension (even if it is a single state)'

        _action = self._actor.eval( state )
        # during training add some noise (per action in the batch, to incentivize more exploration)
        if self._mode == 'train' :
            _noise = np.array( [ self._epsilon * self._noiseProcess.sample() \
                                    for _ in range( len( state ) ) ] ).reshape( _action.shape )
            _action += _noise
            _action = np.clip( _action, -1., 1. )

        return _action
```

* The [**update**][url_agent_method_update] is in charge of handling the logic of the
  DDPG algorithm, like storing transitions in the replay buffer, checking when to take
  a learning step and updating the noise scaler using the appropriate schedule. We also
  save the models for both actor and critic just in case.


```python
    def update( self, transitions ) :
        r"""Updates the internals of the agent given some new batch of transitions

        Args:
            transitions (list) : a batch of transitions of the form (s,a,r,s',done)

        """
        for transition in transitions :
            self._rbuffer.store( transition )

        if self._istep >= self._config.trainingStartingStep and \
           self._istep % self._config.trainFrequencySteps == 0 and \
           len( self._rbuffer ) > self._config.batchSize :
            # do the required number of learning steps
            for _ in range( self._config.trainNumLearningSteps ) :
                self._learn()

            # save the current model
            self._actor.save()
            self._critic.save()

        self._istep += 1

        # update epsilon using the required schedule
        if self._config.epsilonSchedule == 'linear' :
            self._epsilon = max( 0.025, self._epsilon - self._config.epsilonFactorLinear )
            ## self._actionScaler = min( 1.0, self._actionScaler + self._config.epsilonFactorLinear )
        else :
            self._epsilon = max( 0.025, self._epsilon * self._config.epsilonFactorGeom )
            ## self._actionScaler = min( 1.0, self._actionScaler * self._config.epsilonFactorGeom )
```

* Finally, the [**learn**][url_agent_method_learn] is in charge of implementing the following 
  high-level parts of the learning step of the DDPG algorithm:

    * Grabbing a minibatch of experience from the replay buffer.
    * Delegating learning updates of the critic network to its appropriate backend implementation.
    * Delegating learning updates of the actor network to its appropriate backend implementation.
    * Delegating target-network updates of both actor and critic to their appropriate backend implementations.

```python
    def _learn( self ) :
        r"""Takes a learning step on a batch from the replay buffer

        """
        # 0) grab some experience tuples from the replay buffer
        _states, _actions, _rewards, _statesNext, _dones = self._rbuffer.sample( self._config.batchSize )

        # 1) train the critic (fit q-values to q-targets)
        #
        #   minimize mse-loss of current q-value estimations and the ...
        #   corresponding TD(0)-estimates used as "true" q-values
        #
        #   * pi  -> actor parametrized by weights "theta"
        #       theta
        #
        #   * pi  -> actor target parametrized by weights "theta-t"
        #       theta-t
        #
        #   * Q   -> critic parametrized by weights "phi"
        #      phi
        #
        #   * Q   -> critic-target parametrized by weights "phi-t"
        #      phi-t
        #                           __                 ___                          2
        #   phi := phi + lrCritic * \/    ( 1 / |B| )  \    || Qhat(s,a) - Q(s,a) ||
        #                             phi              /__
        #                                         (s,a,r,s',d) in B
        #
        #   where:
        #      * Q(s,a) = Q (s,a) -> q-values from the critic
        #                phi
        #
        #      * a' = pi(s') -> max. actions from the target actor
        #               theta-t
        #
        #      * Qhat(s,a) = r + (1 - d) * gamma * Q (s',a') -> q-targets from the target critic
        #                                           phi-t
        #
        # so: compute q-target, and used them as true labels in a supervised-ish learning process
        #
        _actionsNext = self._actorTarget.eval( _statesNext )
        _qtargets = _rewards + ( 1. - _dones ) * self._config.gamma * self._criticTarget.eval( _statesNext, _actionsNext )
        self._critic.train( _states, _actions, _qtargets )

        # 2) train the actor (its gradient comes from the critic in a pathwise way)
        #
        #   compute gradients for the actor from gradients of the critic ...
        #   based on the deterministic policy gradients theorem:
        #
        #   dJ / d = E [ dQ / du * du / dtheta ]
        #
        #   __            __  
        #   \/  J   = E [ \/     Q( s, a ) |  ]
        #     theta        theta  phi      |s=st, a=pi(st)
        #                                             theta
        #
        #   which can be further reduced to :
        #
        #   __            __                            __
        #   \/  J   = E [ \/  Q( s, a ) |               \/  pi(s) |  ]
        #     theta        a   phi      |s=st, a=pi(st)   theta   |s=st
        #                                         theta
        #
        #   so: compute gradients of the actor from one of the expression above:
        #
        #    * for pytorch: just do composition Q(s,pi(s)), like f(g(x)), ...
        #                   and let pytorch's autograd do the job of ...
        #                   computing df/dg * dg/dx
        #
        #    * for tensorflow: compute gradients from both and combine them ...
        #                      using tf ops and tf.gradients
        #
        self._actor.train( _states, self._critic )
        
        # 3) apply soft-updates using polyak averaging
        self._actorTarget.copy( self._actor, self._config.tau )
        self._criticTarget.copy( self._critic, self._config.tau )
```

### 2.2 **Models**

**Implementation: 
    * Interface: [model.py][url_impl_models_interface] | 
    * Pytorch implementation: [pytorch.py][url_impl_models_pytorch]**

We decided to separate the models in a similar way to the previous projects,
by implementing both abstract and concrete model classes. We went a bit further
and tried to follow the implementation from various online rl-frameworks that
separate the models in a more modular way to allow reuse. Following that approach
we separate the model into two components:

* **Backbone**: a component that implements the bulk of the network, like a multi-layer
                perceptron, or a convolutional network like Resnet34.

* **Head**: a container that wraps the backbone, and adds operations specific to
            the algorithm to be implemented, e.g. implementing a DPG theorem-like
            update for the actor, or implementing an update of the critic network using
            fitter Q-learning.

The interface for this models can be found in the [model.py][url_impl_models_interface] 
file, and it defines abstract classes with the required methods to be implemented by the 
backend-specific implementations. There are three abstract classes declared in this
file, and these are:

* [**DDPGModelBackbone**][url_models_core_backbone]: interface for all backbone models to be used, which must define
                                                     the bulk of the network to be used for the models. Below there's
                                                     a snippet of the implementation.

```python
class DDPGModelBackbone( abc.ABC ) :
    r"""DDPG backbone architecture for all models (either actor|critic)

    Args:
        backboneConfig (config.DDPGModelBackboneConfig) : configuration of the backbone to be used

    """
    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGModelBackbone, self ).__init__()

        self._config = backboneConfig


    @abc.abstractmethod
    def forward( self, inputs ) :
        r"""Executes a forward pass over the backbone at a given state (or state-batch)

        Args:
            inputs (list): a list of np.ndarrays, which represents the inputs to the model

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Copies softly (with polyak averaging) model weights from another model

        Args:
            other (DDPGModelBackbone)   : model from whom to copy the weights
            tau (float)                 : averaging factory (soft-update with polyak averaging)

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates an exact deep-replica of this model

        Returns:
            (DDPGModelBackbone) : replica of this model

        """
        pass


    @property
    def config( self ) :
        r"""Returns the configuration properties of the backbone

        """
        return self._config
```

* [**IDDPGActor**][url_models_core_actor_head]: interface for the *actor-head*, which must define additional operations
                                                needed to train the actor-network. Below there's a snippet of this interface.
                                                Notice that this component is wrapping a given backbone for an actor-network.

```python
class IDDPGActor( abc.ABC ) :
    r"""DDPG core actor-model class

    Abstract class that represents an actor for a DDPG-based agent, 
    composed of operations on top of a backbone architecture defined
    by the user as he requests (either defining a custom model or
    through layer definitions)

    Args:
        backbone (DDPGModelBackbone)    : model backbone architecture
        learningRate (float)            : learning rate used for the optimizer

    """
    def __init__( self, backbone, learningRate, **kwargs ) :
        super( IDDPGActor, self ).__init__()

        self._backbone = backbone
        self._learningRate = learningRate
        self._isTargetNetwork = False
        self._savedir = './results/session_default'


    def setAsTargetNetwork( self, isTarget ) :
        r"""Sets the target-mode of the network

        It hints the network to whether or not the network is a target network,
        which changes the behaviour a bit of this network in evaluation mode,
        namely removing the constraint of deactivating the batch-norm when 
        calling eval on a batch of states

        Args:
            isTarget (boolean) : target-mode to be used by this network 

        """
        self._isTargetNetwork = isTarget


    @abc.abstractmethod
    def eval( self, state ) :
        r"""Returns the action to take at a given state (batch of states)

        Args:
            state (np.ndarray) : state (batch of states) at which we want to act

        Returns:
            (np.ndarray) : action(s) to take at the given state(s)

        """
        pass

    @abc.abstractmethod
    def train( self, states, critic ) :
        r"""Takes a learning step to update the parameters of the actor

        This method must implement the update of the parameters of the actor
        by using the deterministic policy gradients theorem, which computes
        the gradients through the critic

        Args:
            states (np.ndarray)     : batch of states sampled from the replay buffer
            critic (IDDPGCritic)    : appropriate critic to be used to compute gradients from

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Updates the parameters of the actor from another one using polyak averaging

        Args:
            other (IDDPGActor)  : actor from whom we want to copy the parameters
            tau (float)         : polyak averaging factor for soft-updates

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates a replica of this actor (usually for a target actor)

        Returns:
            (IDDPGActor) : a replica of this actor

        """
        pass


    def setSaveDir( self, savedir ) :
        r"""Sets the directory where to save actor model

        Args:
            savedir (string) : folder where to save the actor model

        """
        self._savedir = savedir


    @abc.abstractmethod
    def save( self ) :
        r"""Saves the actor model into disk

        """
        pass


    @abc.abstractmethod
    def load( self ) :
        r"""Loads the actor model from disk

        """
        pass


    @property
    def backbone( self ) :
        r"""Returns a reference to the backbone model

        """
        return self._backbone
```

* [**IDDPGCritic**][url_models_core_critic_head]: interface for the *critic-head*, which must define additional operations
                                                  needed to train the critic-network. Below there's a snippet of this interface.
                                                  Notice that this component is wrapping a given backbone for an critic-network.

```python
class IDDPGCritic( abc.ABC ) :
    r"""DDPG core critic-model class

    Abstract class that represents a critic for a DDPG-based agent, 
    composed of operations on top of a backbone architecture defined
    by the user as he requests (either defining a custom model or
    through layer definitions)

    Args:
        backbone (DDPGModelBackbone)    : model backbone architecture
        learningRate (float)            : learning rate used for the optimizer

    Args:

    """
    def __init__( self, backbone, learningRate, **kwargs ) :
        super( IDDPGCritic, self ).__init__()

        self._backbone = backbone
        self._learningRate = learningRate
        self._isTargetNetwork = False
        self._savedir = './results/session_default'


    def setAsTargetNetwork( self, isTarget ) :
        r"""Sets the target-mode of the network

        It hints the network to whether or not the network is a target network,
        which changes the behaviour a bit of this network in evaluation mode,
        namely removing the constraint of deactivating the batch-norm when 
        calling eval on a batch of states

        Args:
            isTarget (boolean) : target-mode to be used by this network 

        """
        self._isTargetNetwork = isTarget


    @abc.abstractmethod
    def eval( self, state, action ) :
        r"""Returns the q-values Q(s,a) at the given state(s) and action(s)

        Args:
            state (np.ndarray)  : state (batch of states) at which we want to evaluate Q(s,a)
            action (np.ndarray) : action (batch of actions) at which we want to evaluate Q(s,a)

        Returns:
            (np.ndarray) : q-value (batch of Q(s,a))

        """
        pass


    @abc.abstractmethod
    def train( self, states, actions, qtargets ) :
        r"""Takes a learning step to update the parameters of the critic

        This method must implement the update of the parameters of the critic
        by using fitted Q-learning, using the given Q-targets as the true values
        of Q(s,a) at those states and actions given

        Args:
            states (np.ndarray)     : batch of states to compute Q(s,a)
            actions (np.ndarray)    : batch of actions to compute Q(s,a)
            qtargets (np.ndarray)   : batch of qtargets Qhat(s,a) = r + gamma * Q(s,u(s))

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Updates the parameters of the critic from another one using polyak averaging

        Args:
            other (IDDPGCritic) : critic from whom we want to copy the parameters
            tau (float)         : polyak averaging factor for soft-updates

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates a replica of this actor (usually for a target actor)

        Returns:
            (IDDPGActor) : a replica of this actor

        """
        pass


    def setSaveDir( self, savedir ) :
        r"""Sets the directory where to save critic model

        Args:
            savedir (string) : folder where to save the critic model

        """
        self._savedir = savedir


    @abc.abstractmethod
    def save( self ) :
        r"""Saves the critic model into disk

        """
        pass


    @abc.abstractmethod
    def load( self ) :
        r"""Loads the critic model from disk

        """
        pass


    @property
    def backbone( self ) :
        return self._backbone
```

Regarding the backend-specific implementations, we currently have support for **pytorch**
backbones and heads, and we'll add support for **tensorflow** in later updates. This
implementation is located in the [pytorch.py][url_impl_models_pytorch] file, which currently
implements **custom/specific MLP backbones** for our actor and critic (same architecture as
explained in the previous section on architectures used), and **generic heads** for the
actor and critic. The idea was to allow the user to pass a definition of the architecture
as a dictionary of layers, and then instantiate these using a generic backbone, kind of like
keras (this feature will be implemented in later updates as well). The features implemented 
for this project submission are explained below:

* [**Base Pytorch backbones**][url_pytorch_base_backbone]: We first implement a base backbone for the pytorch backend, that contains
                                                           some functionality that will be required by all backbones to be implemented,
                                                           like copies using soft-updates, etc.. Notice that we also define a custom 
                                                           weight initializer, which is similar to the *Lecun Uniform initializer*. This
                                                           comes from our first code reference from the Udacity Pendulum DDPG implementation,
                                                           which also worked fine for our experiments.

```python
def lecunishUniformInitializer( layer ) :
    _fanIn = layer.weight.data.size()[0]
    _limit = np.sqrt( 2. / _fanIn )
    
    return ( -_limit, _limit )


class DDPGModelBackbonePytorch( model.DDPGModelBackbone, nn.Module ) :


    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGModelBackbonePytorch, self ).__init__( backboneConfig, **kwargs )

        self._seed = torch.manual_seed( self._config.seed )
        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE

    @abc.abstractmethod
    def _resetParameters( self ) :
        r"""Resets the parameters of the network by using the appropriate initializers

        """
        pass


    def copy( self, other, tau = 1.0 ) :
        r"""Copies softly (with polyak averaging) model weights from another model

        Args:
            other (DDPGModelBackbone)   : model from whom to copy the weights
            tau (float)                 : averaging factory (soft-update with polyak averaging)

        """
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


    def clone( self ) :
        r"""Creates an exact deep-replica of this model

        Returns:
            (DDPGModelBackbone) : replica of this model

        """
        _replica = self.__class__( self._config, device = self._device )
        _replica.to( self._device )
        _replica.copy( self )

        return _replica
```

* [**MLP Actor-Backbone**][url_pytorch_actor_backbone]: We implemented a custom backbone for the actor-network,
                                                        using the architecture explained earlier. It consists of
                                                        a MLP with batchnorm (if requested) and ReLU activations.



```python
class DDPGMlpModelBackboneActor( DDPGModelBackbonePytorch ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGMlpModelBackboneActor, self ).__init__( backboneConfig, **kwargs )

        self._fc1 = nn.Linear( self._config.inputShape[0], 256 )
        self._fc2 = nn.Linear( 256, 128 )
        self._fc3 = nn.Linear( 128, self._config.outputShape[0] )

        if self._config.useBatchnorm :
            self._bn0 = nn.BatchNorm1d( self._config.inputShape[0] )
            self._bn1 = nn.BatchNorm1d( 256 )
            self._bn2 = nn.BatchNorm1d( 128 )

        # initialize the parameters of the backbone
        self._resetParameters()


    def _resetParameters( self ) :
        self._fc1.weight.data.uniform_( *lecunishUniformInitializer( self._fc1 ) )
        self._fc2.weight.data.uniform_( *lecunishUniformInitializer( self._fc2 ) )
        self._fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, inputs ) :
        # sanity check: actor network is deterministic, and receives only states as input
        assert len( inputs ) == 1, 'ERROR> this network expects only one input (states)'

        # grab the actual input to the model (states)
        _states = inputs[0]

        if self._config.useBatchnorm :
            x = self._bn0( _states )
            x = F.relu( self._bn1( self._fc1( x ) ) )
            x = F.relu( self._bn2( self._fc2( x ) ) )
            x = F.tanh( self._fc3( x ) )
        else :
            x = F.relu( self._fc1( _states ) )
            x = F.relu( self._fc2( x ) )
            x = F.tanh( self._fc3( x ) )

        return x
```

* [**MLP Critic-Backbone**][url_pytorch_critic_backbone]: We also implemented a custom backbone for the critic-network,
                                                          using the architecture explained earlier. It consists of
                                                          a MLP with batchnorm (if requested) and ReLU activations. Notice
                                                          it concatenates the actions taken by the agent, to the hidden 
                                                          states after the first Fully Connected layer. This effectively
                                                          allows to compute Q<sub>&phi;</sub>(s,a).

```python
class DDPGMlpModelBackboneCritic( DDPGModelBackbonePytorch ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGMlpModelBackboneCritic, self ).__init__( backboneConfig, **kwargs )

        # sanity check: ensure sizes are exactly what we expect for this case
        assert self._config.outputShape[0] == 1, \
            'ERROR> Critic model should only output 1 value for Q(s,a)'

        self._fc1 = nn.Linear( self._config.inputShape[0], 128 )
        self._fc2 = nn.Linear( 128 + self._config.actionsShape[0], 128 )
        self._fc3 = nn.Linear( 128, self._config.outputShape[0] )

        if self._config.useBatchnorm :
            self._bn0 = nn.BatchNorm1d( self._config.inputShape[0] )

        # initialize the parameters of the backbone
        self._resetParameters()


    def _resetParameters( self ) :
        self._fc1.weight.data.uniform_( *lecunishUniformInitializer( self._fc1 ) )
        self._fc2.weight.data.uniform_( *lecunishUniformInitializer( self._fc2 ) )
        self._fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, inputs ) :
        # sanity check: critic network expects both state and action batches as inputs
        assert len( inputs ) == 2, 'ERROR> this network expects two inputs (states,actions)'

        # grab the actual inputs to the model (states and actions)
        _states = inputs[0]
        _actions = inputs[1]

        if self._config.useBatchnorm :
            x = self._bn0( _states )
            x = F.relu( self._fc1( x ) )
            x = torch.cat( [x, _actions], dim = 1 )
            x = F.relu( self._fc2( x ) )
            x = self._fc3( x )
        else :
            x = F.relu( self._fc1( _states ) )
            x = torch.cat( [x, _actions], dim = 1 )
            x = F.relu( self._fc2( x ) )
            x = self._fc3( x )

        return x
```

* [**Actor-head**][url_pytorch_actor_head]: The actor-head is implemented in the **DDPGActor** class, which
                                            contains the required backend-specific operations to train the
                                            wrapped backbone for the actor-network using the Deterministic
                                            Policy Gradient update (see the [train](https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L202) 
                                            method).

```python
class DDPGActor( model.IDDPGActor ) :

    def __init__( self, backbone, learningRate, **kwargs ) :
        super( DDPGActor, self ).__init__( backbone, learningRate, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE
        # send the backbone model to the appropriate device
        self._backbone.to( self._device )

        self._optimizer = opt.Adam( self._backbone.parameters(), self._learningRate )


    def eval( self, state ) :
        with autograd.detect_anomaly() :
            # transform to torch tensors
            state = torch.from_numpy( state ).float().to( self._device )
    
            if not self._isTargetNetwork :
                # set in evaluation mode, as we might be using batch-norm
                self._backbone.eval()
    
            # do not compute gradients for the actor just yet
            with torch.no_grad() :
                _action = self._backbone( [state] )
    
            if not self._isTargetNetwork :
                # get back to training mode, as we might be using batch-norm
                self._backbone.train()
    
            return _action.cpu().data.numpy()


    def train( self, states, critic ) :
        with autograd.detect_anomaly() :
            # transform to torch tensors
            states = torch.from_numpy( states ).float().to( self._device )
    
            self._optimizer.zero_grad()
            # compute actions taken in these states by the actor
            _actionsPred = self._backbone( [states] )
            # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
            _lossActor = -critic( states, _actionsPred ).mean()
            _lossActor.backward()
            # take a step with the optimizer
            self._optimizer.step()


    def copy( self, other, tau = 1.0 ) :
        self._backbone.copy( other.backbone, tau )


    def clone( self ) :
        _replica = self.__class__( self._backbone.clone(),
                                   self._learningRate,
                                   device = self._device )

        return _replica


    def save( self ) :
        torch.save( self._backbone.state_dict(), os.path.join( self._savedir, 'checkpoint_actor.pth' ) )


    def load( self ) :
        self._backbone.load_state_dict( torch.load( os.path.join( self._savedir, 'checkpoint_actor.pth' ) ) )


    def __call__( self, states ) :
        return self._backbone( [states] )
```

* [**Critic-head**][url_pytorch_critic_head]: The critic-head is implemented in the **DDPGCritic** class, which
                                              contains the required backend-specific operations to train the
                                              wrapped backbone for the critic-network using fitted Q-learning as
                                              in DQN (see the [train](https://github.com/wpumacay/DeeprlND-projects/blob/fda2c59f348a0712efe9dc8234830f879a2ef6d8/project2-continuous-control/ccontrol/ddpg/models/pytorch.py#L279) 
                                              method).

```python
class DDPGCritic( model.IDDPGCritic ) :

    def __init__( self, backbone, learningRate, **kwargs ) :
        super( DDPGCritic, self ).__init__( backbone, learningRate, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE
        # send the backbone model to the appropriate device
        self._backbone.to( self._device )

        self._optimizer = opt.Adam( self._backbone.parameters(), self._learningRate )


    def eval( self, state, action ) :
        with autograd.detect_anomaly() :
            # transform to torch tensors
            state = torch.from_numpy( state ).float().to( self._device )
            action = torch.from_numpy( action ).float().to( self._device )
    
            if not self._isTargetNetwork :
                # set in evaluation mode, as we might be using batch-norm
                self._backbone.eval()
    
            # do not compute gradients for the critic in this stage
            with torch.no_grad() :
                _qvalue = self._backbone( [state, action] )
    
            if not self._isTargetNetwork :
                # get back to training mode, as we might be using batch-norm
                self._backbone.train()
    
            return _qvalue.cpu().data.numpy()


    def train( self, states, actions, qtargets ) :
        with autograd.detect_anomaly() :
            # transform to torch tensors
            states = torch.from_numpy( states ).float().to( self._device )
            actions = torch.from_numpy( actions ).float().to( self._device )
            qtargets = torch.from_numpy( qtargets ).float().to( self._device )
    
            # compute q-values for Q(s,a), where s,a come from the given ...
            # states and actions batches passed along the q-targets
            _qvalues = self._backbone( [states, actions] )
    
            # compute loss for the critic
            self._optimizer.zero_grad()
            _lossCritic = F.mse_loss( _qvalues, qtargets )
            _lossCritic.backward()
            if self._backbone.config.clipGradients :
                nn.utils.clip_grad_norm( self._backbone.parameters(), 
                                         self._backbone.config.gradientsClipNorm )
            # take a step with the optimizer
            self._optimizer.step()


    def copy( self, other, tau = 1.0 ) :
        self._backbone.copy( other.backbone, tau )


    def clone( self ) :
        _replica = self.__class__( self._backbone.clone(),
                                   self._learningRate,
                                   device = self._device )

        return _replica


    def save( self ) :
        torch.save( self._backbone.state_dict(), os.path.join( self._savedir, 'checkpoint_critic.pth' ) )


    def load( self ) :
        self._backbone.load_state_dict( torch.load( os.path.join( self._savedir, 'checkpoint_critic.pth' ) ) )


    def __call__( self, states, actions ) :
        return self._backbone( [states, actions] )
```

Finally, we have added also a small example of how to instantiate these backbones and heads. Recall
that we mentioned we were using the **gin-config** framework to easily use configuration structures
and pass them during construction of our required objects. Below we show a snippet of how to instantiate
such models, which is located in the same [pytorch.py][url_impl_models_pytorch] file as an example:

```python
if __name__ == '__main__' :
    import gin
    gin.parse_config_file( '../../../configs/ddpg_reacher_multi_default.gin' )

    from ccontrol.ddpg.utils.config import DDPGModelBackboneConfig
    with gin.config_scope( 'actor' ) :
        _actorNetBackboneConfig = DDPGModelBackboneConfig()
    with gin.config_scope( 'critic' ) :
        _criticNetBackboneConfig = DDPGModelBackboneConfig()

    print( '-------------------------- ACTOR --------------------------------' )
    _actorNetBackbone = DDPGMlpModelBackboneActor( _actorNetBackboneConfig )
    print( _actorNetBackbone )

    print( '-------------------------- CRITIC -------------------------------' )
    _criticNetBackbone = DDPGMlpModelBackboneCritic( _criticNetBackboneConfig )
    print( _criticNetBackbone )
```

### 2.3 **Config**

**Implementation: [config.py][url_config_mechanism]**

A major change from the previous project was to use a different configuration mechanism. 
We decided to follow the same approach used in the [dopamine][url_dopamine] an configure 
our models and agents using the [gin-config][url_gin_config] framework. In this way we can 
save the hyperparameters for our agents in .gin files and used them as we need them in our 
trainer by parsing the appropriate configuration file.

To make use of gin, we defined some configuration structures that will hold the configuration
parameters and hyperparameters of our models and agents. These structures can be found in the
[config.py][url_config_mechanism] file, and consist of the following:

* **Trainer-config**: the structure holding parameters to configure the trainer, like how many episodes
                      to use, max. length of an episode, id of the training run (to create appropriate
                      folders for the results), etc.. Below there is a snippet containing this structure.
                      Notice how we just had to label the structure as gin-configurable to be able to
                      pass the configuration parameters from gin-config files.

```python
@gin.configurable
class DDPGTrainerConfig( object ) :
    r"""Configuration options for the DDPG trainer

    Args:
        numTrainingEpisodes (int)   : number of episodes used for training
        maxStepsInEpisode (int)     : maximum number of steps per episode
        logWindowSize (int)         : size of the logging averaging window (in episodes)
        seed (int)                  : seed for the random-number generators
        sessionID (str)             : name of the session used for training, used as 
                                      savedir or loaddir during training or testing respectively

    """
    def __init__( self,
                  numTrainingEpisodes = 2000,
                  maxStepsInEpisode = 3000,
                  logWindowSize = 100,
                  seed = 0,
                  sessionID = 'session_default' ) :
        super( DDPGTrainerConfig, self ).__init__()

        self.numTrainingEpisodes    = numTrainingEpisodes
        self.maxStepsInEpisode      = maxStepsInEpisode
        self.logWindowSize          = logWindowSize
        self.seed                   = seed
        self.sessionID              = sessionID
```

* **Agent-config**: the structure holding configuration parameters for the agent, like the size
                    of the replay buffer, type of noise to use, size of the minibatch used for
                    learning, etc.. Below there's a snippet showing all possible configuration
                    parameters.

```python
@gin.configurable
class DDPGAgentConfig( object ) :
    r"""Configuration options for DDPG based agents

    Args:
        observationsShape (tuple)   : shape of the observations provided to the agent
        actionsShape (tuple)        : shape of the actions that the agent can take
        seed (int)                  : random seed use to initialize the random number generators
        gamma (float)               : discount factor
        tau (float)                 : polyak averaging factor used for soft-updates
        replayBufferSize (int)      : size of the replay buffer
        lrActor (float)             : learning rate to be used for the actor
        lrCritic (float)            : learning rate to be used for the critic
        batchSize (int)             : size of the batch taken from the replay buffer at each learning step
        trainFrequencySteps (int)   : frequency (in steps) at which to take learning steps
        trainNumLearningSteps (int) : number of learning steps to take when learning is required
        noiseType (str)             : type of noise to be used, either (ounoise|normal)
        noiseOUMu (float)           : mu factor for the Ornstein-Uhlenbeck noise process
        noiseOUTheta (float)        : theta factor for the Ornstein-Uhlenbeck noise process
        noiseOUSigma (float)        : sigma factor for the Ornstein-Uhlenbeck noise process
        noiseNormalStddev (float)   : standard deviation of the zero-mean gaussian noise
        epsilonSchedule (str)       : type of schedule to be used to decay epsilon (noise), either 'linear' or 'geometric'
        epsilonFactorGeom (float)   : decay factor (multiplicative) used for the geometric schedule
        epsilonFactorLinear (float) : decay factor (decrement) used for the linear schedule
        trainingStartingStep (int)  : step number at which training actually starts

    """
    def __init__( self,
                  observationsShape = (2,),
                  actionsShape = (2,),
                  seed = 0,
                  gamma = 0.99,
                  tau = 0.001,
                  replayBufferSize = 1000000,
                  lrActor = 0.001,
                  lrCritic = 0.001,
                  batchSize = 256,
                  trainFrequencySteps = 20,
                  trainNumLearningSteps = 10,
                  noiseType = 'ounoise',
                  noiseOUMu = 0.0,
                  noiseOUTheta = 0.15,
                  noiseOUSigma = 0.2,
                  noiseNormalStddev = 0.25,
                  epsilonSchedule = 'linear',
                  epsilonFactorGeom = 0.999,
                  epsilonFactorLinear = 1e-5,
                  trainingStartingStep = 0 ) :
        super( DDPGAgentConfig, self ).__init__()

        self.observationsShape = observationsShape
        self.actionsShape = actionsShape
        self.seed = 0
        self.gamma = gamma
        self.tau = tau
        self.replayBufferSize = replayBufferSize
        self.lrActor = lrActor
        self.lrCritic = lrCritic
        self.batchSize = batchSize
        self.trainFrequencySteps = trainFrequencySteps
        self.trainNumLearningSteps = trainNumLearningSteps
        self.noiseType = noiseType
        self.noiseOUMu = noiseOUMu
        self.noiseOUTheta = noiseOUTheta
        self.noiseOUSigma = noiseOUSigma
        self.noiseNormalStddev = noiseNormalStddev
        self.epsilonSchedule = epsilonSchedule
        self.epsilonFactorGeom = epsilonFactorGeom
        self.epsilonFactorLinear = epsilonFactorLinear
        self.trainingStartingStep = trainingStartingStep
```

* **Backbone-config**: the structure holding configuration parameters for the backbones
                       of the models used by the actors and critics. It primarily contains
                       some information from the architecture of the model, and in future
                       updates will contain the full description of the architecture through
                       the *layersDefs* attribute. As this same class (not same object) is 
                       used for both actor and critics, we have to take this into account in
                       the gin-config file, as by default it only allows to configure a single
                       class, unlike our case that we need two flavors of the same class (one
                       for the actor and one for the critic). As we will see later, this can
                       be easily fixed using a feature of gin-config called *scopes*. Below
                       we show a snippet showing all possible configuration parameters for
                       this structure.

```python
@gin.configurable
class DDPGModelBackboneConfig( object ) :
    r"""Configuration options of the backbone of models used with DDPG based agents

    Args:
        observationsShape (tuple)   : shape of the observation space
        actionsShape (tuple)        : shape of the action space
        inputShape (tuple)          : shape of the input to the model
        outputShape (tuple)         : shape of the output of the model
        layersDefs (list)           : a list of dictionaries each describing a layer of the model
        useBatchnorm (boolean)      : whether or not to use batchnorm in the backbone of the model
        clipGradients (boolean)     : whether or not to clip the norm of the gradients in the layers
        gradientsClipNorm (float)   : norm to which to clip the gradients (if applicable)
        seed (int)                  : seed for random number generators to use

    """
    def __init__( self,
                  observationsShape = (2,),
                  actionsShape = (2,),
                  inputShape = (2,),
                  outputShape = (2,),
                  layersDefs = [],
                  useBatchnorm = True,
                  clipGradients = False,
                  gradientsClipNorm = 1.,
                  seed = 0 ) :
        super( DDPGModelBackboneConfig, self ).__init__()

        self.observationsShape = copy.copy( observationsShape )
        self.actionsShape = copy.copy( actionsShape )
        self.inputShape = copy.copy( inputShape )
        self.outputShape = copy.copy( outputShape )
        self.layersDefs = copy.copy( layersDefs )
        self.useBatchnorm = useBatchnorm
        self.clipGradients = clipGradients
        self.gradientsClipNorm = gradientsClipNorm
        self.seed = seed
```

An example of the usage of the configuration mechanism was shown in the previous section, 
when we described how to instantiate a model using our implementation. There we used as 
configuration file a default config file called [ddpg_reacher_multi_default.gin][url_config_default_gin_file], 
whose contents are shown below. Notice how we make use of scopes for the actor and critic 
by using the **actor/** and **critic/** scopes to configure the appropriate structures.

```python

import ccontrol.ddpg.utils.config

SEED = 0

ccontrol.ddpg.utils.config.DDPGTrainerConfig.numTrainingEpisodes    = 2000
ccontrol.ddpg.utils.config.DDPGTrainerConfig.maxStepsInEpisode      = 3000
ccontrol.ddpg.utils.config.DDPGTrainerConfig.logWindowSize          = 100
ccontrol.ddpg.utils.config.DDPGTrainerConfig.seed                   = %SEED
ccontrol.ddpg.utils.config.DDPGTrainerConfig.sessionID              = 'session_default'

ccontrol.ddpg.utils.config.DDPGAgentConfig.observationsShape        = (33,)
ccontrol.ddpg.utils.config.DDPGAgentConfig.actionsShape             = (4,)
ccontrol.ddpg.utils.config.DDPGAgentConfig.seed                     = %SEED
ccontrol.ddpg.utils.config.DDPGAgentConfig.gamma                    = 0.99
ccontrol.ddpg.utils.config.DDPGAgentConfig.tau                      = 0.001
ccontrol.ddpg.utils.config.DDPGAgentConfig.replayBufferSize         = 1000000
ccontrol.ddpg.utils.config.DDPGAgentConfig.lrActor                  = 0.001
ccontrol.ddpg.utils.config.DDPGAgentConfig.lrCritic                 = 0.001
ccontrol.ddpg.utils.config.DDPGAgentConfig.batchSize                = 256
ccontrol.ddpg.utils.config.DDPGAgentConfig.trainFrequencySteps      = 20
ccontrol.ddpg.utils.config.DDPGAgentConfig.trainNumLearningSteps    = 10
ccontrol.ddpg.utils.config.DDPGAgentConfig.noiseType                = 'ounoise'
ccontrol.ddpg.utils.config.DDPGAgentConfig.noiseOUMu                = 0.0
ccontrol.ddpg.utils.config.DDPGAgentConfig.noiseOUTheta             = 0.15
ccontrol.ddpg.utils.config.DDPGAgentConfig.noiseOUSigma             = 0.2
ccontrol.ddpg.utils.config.DDPGAgentConfig.noiseNormalStddev        = 0.25
ccontrol.ddpg.utils.config.DDPGAgentConfig.epsilonSchedule          = 'linear'
ccontrol.ddpg.utils.config.DDPGAgentConfig.epsilonFactorGeom        = 0.999
ccontrol.ddpg.utils.config.DDPGAgentConfig.epsilonFactorLinear      = 1e-5
ccontrol.ddpg.utils.config.DDPGAgentConfig.trainingStartingStep     = 0

actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.observationsShape = (33,)
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.actionsShape = (4,)
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.inputShape = (33,)
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.outputShape = (4,)
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.layersDefs = []
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.useBatchnorm = True
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.clipGradients = False
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.gradientsClipNorm = 1.
actor/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.seed = %SEED

critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.observationsShape = (33,)
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.actionsShape = (4,)
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.inputShape = (33,)
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.outputShape = (1,)
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.layersDefs = []
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.useBatchnorm = True
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.clipGradients = True
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.gradientsClipNorm = 1.
critic/ccontrol.ddpg.utils.config.DDPGModelBackboneConfig.seed = %SEED
```

### 2.4 **Utils**

**Implementation:
    * Replay Buffer: [replaybuffer.py][url_utils_replay_buffer]
    * Noise helpers: [noise.py][url_utils_noise]
    * Env-wrapper: [mlagents.py][url_utils_env_wrapper]**

As explained earlier we used various utilities throughout our implementation. The
main utilities we used are listed below:

* **Replay Buffer**: Implementation of a circular replay buffer without
                     prioritization, similar to the one used for our 
                     experiments from our previous DQN project. 

* **Noise Process**: Implementation of a noise helpers. It contains the implementation
                     of the *Ornstein-Uhlenbeck* noise process, and also a simple gaussian
                     random noise generator. We used both during our experiments, and
                     with a good set of hyperparameters, both gave us similar training results.

* **Gym-like wrapper**: Implementation of a simple Environment Wrapper for the unity
                        mlagents environment. It helped us to standarize the code with
                        a **gym-like interface**, which allowed us to check that the implementation
                        worked in both simple gym environments and in our unity environment.

### 2.4 **Trainer**

**Implementation: [trainer.py][url_impl_trainer]**

Finally, we implemented a simple trainer, similar to the one from the previous project. This
trainer is in charge of instantiating and configuring the environment, models, agents, managing
the training and testing loop, and calling the appropriate logging functionality to check training
progress. There are two main sections of the trainer that we will discuss in more detail below:

* [**Entry point**][url_trainer_method_main]: This is the *main* method of our trainer, in which we instantiate
                                              all required objects, like the configuration files using gin-config, 
                                              instantiate the environment and set its seed, create the models' 
                                              backbones, and compose the backbones into our DDPG agent. This method
                                              then delegates according to the training mode to either train or test
                                              using the already configured environment and agent. Below there is a
                                              snippet of this functionality:

```python
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument( 'mode', help='mode to run the script (train|test)', type=str, choices=['train','test'], default='train' )
    parser.add_argument( '--config', help='gin-config file with the trainer configuration', type=str, default='./configs/ddpg_reacher_multi_default.gin' )
    parser.add_argument( '--seed', help='random seed to be used all over the place', type=int, default=None )

    args = parser.parse_args()

    gin.parse_config_file( args.config )

    # grab training mode
    TRAIN = ( args.mode == 'train' )

    # grab configuration from gin
    trainerConfig = DDPGTrainerConfig()
    agentConfig = DDPGAgentConfig()
    with gin.config_scope( 'actor' ) :
        actorBackboneConfig = DDPGModelBackboneConfig()
    with gin.config_scope( 'critic' ) :
        criticBackboneConfig = DDPGModelBackboneConfig()

    # check for command line seed. if so, override all seeds in gin files
    if args.seed is not None :
        trainerConfig.seed          = args.seed
        agentConfig.seed            = args.seed
        actorBackboneConfig.seed    = args.seed
        criticBackboneConfig.seed   = args.seed
        # slightly modify the session id
        trainerConfig.sessionID += ( '_seed_' + str( args.seed ) )

    env = UnityEnvWrapper( EXECUTABLE_FULL_PATH,
                           numAgents = NUMBER_OF_AGENTS, 
                           mode = 'training' if TRAIN else 'testing', 
                           workerID = 0, 
                           seed = trainerConfig.seed )

    # in case the results directory for this session does not exist, create a new one
    SESSION_FOLDER = os.path.join( './results', trainerConfig.sessionID )
    if not os.path.exists( SESSION_FOLDER ) :
        os.makedirs( SESSION_FOLDER )

    TRAINING_EPISODES = trainerConfig.numTrainingEpisodes
    MAX_STEPS_IN_EPISODE = trainerConfig.maxStepsInEpisode
    TRAINING_SESSION_ID = trainerConfig.sessionID

    # create the backbones for both actor and critic
    actorBackbone = DDPGMlpModelBackboneActor( actorBackboneConfig )
    criticBackbone = DDPGMlpModelBackboneCritic( criticBackboneConfig )

    # create both actor and critics
    actor = DDPGActor( actorBackbone, agentConfig.lrActor )
    critic = DDPGCritic( criticBackbone, agentConfig.lrCritic )

    # create the agent
    agent = DDPGAgent( agentConfig, actor, critic )
    agent.setSaveDir( SESSION_FOLDER )

    env.seed( trainerConfig.seed )
    random.seed( trainerConfig.seed )
    np.random.seed( trainerConfig.seed )

    if TRAIN :
        train( env, agent, TRAINING_EPISODES )
    else :
        test( env, agent )

```

* [**Train loop**][url_trainer_method_train]: This is the method in charge of grabbing information from the environment
                                              by taking steps using the current agent, updating the agent with the recent
                                              transitions, and logging all required information for later usage. Below we
                                              show a snippet of this functionality:

```python
def train( env, agent, numEpisodes ) :
    r"""Training loop for our given agent in a given environment

    Args:
        env (gym.Env)               : Gym-like environment (or wrapper) used to train the agent
        agent (ddpg.core.DDPGAgent) : ddpg-based agent to be trained
        numEpisodes (int)           : number of episodes to train

    """
    progressbar = tqdm( range( 1, numEpisodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestMeanScore = -np.inf
    bestSingleScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( os.path.join( SESSION_FOLDER, 'tensorboard_summary' ) )

    for iepisode in progressbar :

        _scoresPerAgent = np.zeros( NUMBER_OF_AGENTS )
        _ss = env.reset()
        agent.noiseProcess.reset()

        for i in range( MAX_STEPS_IN_EPISODE ) :
            # get the action(s) to take
            _aa = agent.act( _ss )

            # take action in the environment and grab bounty
            _ssnext, _rr, _dd, _ = env.step( _aa )

            # pack the transitions for the agent
            _transitions = []
            for _s, _a, _r, _snext, _done in zip( _ss, _aa, _rr, _ssnext, _dd ) :
                if i == MAX_STEPS_IN_EPISODE - 1 :
                    _transitions.append( ( _s, _a, _r, _snext, True ) )
                else :
                    if _dd.any() :
                        # this is used for environments in which different ...
                        #'done' signals are received for different agents, ...
                        # like the crawler case (some might end, but some not)
                        _transitions.append( ( _s, _a, _r, _snext, True ) )
                    else :
                        _transitions.append( ( _s, _a, _r, _snext, _done ) )

            # update the agent
            agent.update( _transitions )

            # book keeping for next iteration
            _ss = _ssnext
            _scoresPerAgent += _rr

            if _dd.any() :
                break

        # update some info for logging
        _meanScore = np.mean( _scoresPerAgent )
        bestMeanScore = max( bestMeanScore, _meanScore )
        bestSingleScore = max( bestSingleScore, np.max( _scoresPerAgent ) )
        scoresWindow.append( _meanScore )

        if iepisode >= LOG_WINDOW :
            avgScore = np.mean( scoresWindow )
            scoresAvgs.append( avgScore )
            message = 'Training> best-mean: %.2f - best-single: %.2f - current-mean-window: %.2f - current-mean: %.2f'
            progressbar.set_description( message % ( bestMeanScore, bestSingleScore, avgScore, _meanScore ) )
            progressbar.refresh()
        else :
            message = 'Training> best-mean: %.2f - best-single: %.2f - current-mean : %.2f'
            progressbar.set_description( message % ( bestMeanScore, bestSingleScore, _meanScore ) )
            progressbar.refresh()

        writer.add_scalar( 'log_1_mean_score', _meanScore, iepisode )
        writer.add_scalar( 'log_2_mean_score_window', np.mean( scoresWindow ), iepisode )
        writer.add_scalar( 'log_3_buffer_size', len( agent.replayBuffer ), iepisode )
        writer.add_scalar( 'log_4_epsilon', agent.epsilon, iepisode )
```

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