[//]: # (References)

<!-- IMAGES -->
[gif_project_3_tennis_agent]: imgs/gif_project_3_tennis_agent.gif

[img_rl_actor_representation]: imgs/img_rl_actor_representation.png
[img_rl_critic_representation]: imgs/img_rl_critic_representation.png
[img_rl_ddpg_actor_representation]: imgs/img_rl_ddpg_actor_network.png
[img_rl_ddpg_critic_representation]: imgs/img_rl_ddpg_critic_network.png
[img_rl_ddpg_critic_training]: imgs/img_rl_ddpg_critic_training.png
[img_rl_ddpg_actor_training]: imgs/img_rl_ddpg_actor_training.png
[img_rl_maddpg_actor_representation]: imgs/img_rl_maddpg_actor_representation.png
[img_rl_maddpg_critic_representation]: imgs/img_rl_maddpg_critic_representation.png
[img_rl_maddpg_actor_training]: imgs/img_rl_maddpg_actor_training.png
[img_rl_maddpg_critic_training]: imgs/img_rl_maddpg_critic_training.png
[img_rl_maddpg_algorithm]: imgs/img_rl_maddpg_algorithm.png
[img_rl_maddpg_network_architecture_actor]: imgs/img_rl_maddpg_network_architecture_actor.png
[img_rl_maddpg_network_architecture_critic]: imgs/img_rl_maddpg_network_architecture_critic.png
[img_results_submission_1]: imgs/img_results_passing_submission_1.png
[img_results_submission_2]: imgs/img_results_passing_submission_2.png
[img_results_initial_implementation]: imgs/img_results_initial_implementation.png
[img_results_original_implementation]: imgs/img_results_original_implementation.png

<!-- URLS -->
[url_readme]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project3-collaboration/README.md
[url_report_proj2]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/REPORT.md
[url_impl_model_actor]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L63
[url_impl_model_critic]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L110
[url_impl_replay_buffer]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L158
[url_impl_replay_noisegen_ouproc]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L252
[url_impl_replay_noisegen_normal]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L281
[url_impl_trainer]: https://github.com/wpumacay/DeeprlND-projects/blob/f73d74997ef9ab5c44e2d717a74d4e96874ebbd4/project3-collaboration/maddpg_tennis.py#L304

# Using MADDPG to solve the Tennis environment from ML-Agents

This is an accompanying report for the project-3 submission. Below we list some key details we
will cover in our submission:

* [Algorithm description](#1-algorithm-description)
    * [MADDPG algorithm overview](#11-maddpg-algorithm-overview)
    * [Models architecture](#12-models-architecture)
    
* [Implementation](#2-implementation)
    * [Models implementation](#21-models)
    * [Utilities implementation](#22-utils)
    * [Trainer implementation](#23-trainer)

* [Results](#3-results)
    * [Running a pretrained agent](#30-running-a-pretrained-agent)
    * [Hyperparameters](#31-hyperparameters)
    * [Submission results](#32-submission-results)
    * [Experiments results](#33-experiments-results)

* [Future work](#4-future-work)

## 1. Algorithm description

In this section we give a brief description of the MADDPG algorithm from [2], which 
is an extension of the DDPG algorithm [3] for multi-agent rl setups. We recommend to
take a look at section 1 from the report of the previous project (you can find
it [here][url_report_proj2]), as it provides a description of the DDPG algorithm that
might be useful in case you need a refresher.


### 1.1 MADDPG algorithm overview

**TL;DR**
> MADDPG [2] is an multi-agent actor-critic algorithm based on DDPG [3] that makes
> use of the centralized-trainining with descentralized-execution framework to train
> multiple agents to solve a multi-agent environment. Centralized training allows
> agents to train their critics using joint-information from all agents, and
> descentralized-execution makes agents act using their actors and only the local
> observations given to each agent.

Before we explain some of the details of the MADDPG algorithm, let's start by
analyzing the multi-agent rl-problem a bit more and give a brief explanation
of why this problem is potentially harder than the single-agent case.

* In the multi-agent setup we have various agents interacting with an environment,
  each receiving **its own observations** of the environment (local observations)
  and each acting according to **its own policy**.

* Because for a specific agent all other agents are part of the environment, then
  the environment for that agent looks non-stationary. For example, if one agent
  comes up with some policy that kind of works, then if all of a sudden some of
  the other agent starts acting differently (because it learns something new,
  or if swaps its policy, etc.) then, as the observations that the former agent 
  receives are still the same, this agent would take the same actions, but the 
  outcomes might be very different.

* These issues might have the effect for Q-learning agents that make use of experience
  replay-based (like DQN) that experiences from early learning steps might not have the
  same meaning or be as representative in later learning steps.

* For policy-gradient based agents there are also issues, with the intuition being that
  (recall the pg-theorem) we don't actually have a "stationary distribution" to sample 
  from, as the probabilities that we might transition to some states under a certain
  policy might be really different as other agents act differently over the training period.

One approach to use would be to augment each agent with the observations (and potentially
actions as well) of other agents. This would be acceptable in some scenarios (like in ganes,
from which we can grab these non-local measurements as we please), but for some scenarios
it'd be unfeasible or not scale properly even if we had a centralized node that could act
as a global server.

The approach taken by the authors of [2] is to use only local information for the agents 
(decentralized execution), and to use non-local information during training (centralized
training). These two key considerations are combined in the following way using an actor-critic 
approach:

* During execution all agents act according to their own policies (actors). So, the agents' 
  policies receive only local information during testing and training (keep the same actor) 
  as well.

![rl-actor-representation][img_rl_actor_representation]

* However, for execution we don't actually need the critics, so we can actually train
  the critics using information from even other agents (non-local information). This is
  why during training (where we use the critics) we use critics that combine augmented
  versions of the actions and observations: combination of all actions and all observations
  from all agents.

![rl-critic-representation][img_rl_critic_representation]

We could implement this actor-critic framework for the multi-agent setup by using a 
single-agent actor-critic algorithm, and extending it using augmented information for
the critics. The authors of two chose DDPG as their actor-critic algorithm, and extended
it according to its framework, resulting in the MADDPG algorithm. 

Recall that in regular single-agent DDPG we make use of the following actor and critic:

* Actor: A deterministic policy **&mu;<sub>&theta;</sub>(o<sub>t</sub>)**, with its own
         target counterpart **&mu;<sub>&theta;<sup>-</sup></sub>(o<sub>t</sub>)**.

![rl-ddpg-actor][img_rl_ddpg_actor_representation]

* Critic: A parametrized Q-function **Q<sub>&phi;</sub>(o<sub>t</sub>,a<sub>t</sub>)**,
          with its own target counterpart **Q<sub>&phi;<sup>-</sup></sub>(o<sub>t</sub>,a<sub>t</sub>)**.

![rl-ddpg-critic][img_rl_ddpg_critic_representation]

* Training Critic: We trained the critic using fitted Q-learning as in DQN, using the
                   same features of DQN that helped stabilize learning (Exp. replay and
                   Target networks).

![rl-ddpg-critic-training][img_rl_ddpg_critic_training]

* Training Actor: We trained the actor using the Deterministic Policy Gradient theorem,
                  computing the gradients of the Q-function w.r.t. the actor weights and
                  applying the chain rule.

![rl-ddpg-actor-training][img_rl_ddpg_actor_training]

The modifications used for MADDPG are made mainly to the critics, as they now are using
non-local information (both augmented actions and observations). These would be the modifications
needed for MADDPG:

* Actor: It's still a deterministic policy, and it still uses only the local information 
         available to the agent.

![rl-maddpg-actor-representation][img_rl_maddpg_actor_representation]

* Critic: The Q-function accepts now both augmented observations and augmented actions as
          its inputs.

![rl-maddpg-critic-representation][img_rl_maddpg_critic_representation]

* Training Critic: We still train the critic as in DQN, although now we have to deal with
                   augmented observations and actions. This makes us modify our replay buffer
                   such that we store this kind of information.

![rl-maddpg-critic-training][img_rl_maddpg_critic_training]

* Training Actor: We train the actor using the DPG theorem as in the single-agent case, with
                  the slight modification that the actions used to evaluate the related Critic
                  must be augmented versions of the actions, so we have to query all agents
                  for this information beforehand.

![rl-mappdpg-actor-training][img_rl_maddpg_actor_training]

Combining all these modification we end up with the MADDPG algorithm, shown below:

![rl-maddpg-algorithm][img_rl_maddpg_algorithm]

### 1.2 Models architecture

As starting point we decided to use similar architectures to the ones presented used
in the previous project. The only modifications had to be made to de critic networks,
which have to handle both augmented observations and actions.

### **Actor-network architecture**

Our actor is a deteministic policy parametrized using a Multi-layer perceptron with
ReLU activations and Batch Normalization. This MLP receives as inputs local observations
from each actor.

As we saw in the [README.md][url_readme], our environment consists of an observation space of vector 
observations of 24 dimensions (rank-1 tensor of size 24), and an action space of vector actions 
of 2 dimensions (rank-1 tensor of size 2). Hence, our network has as inputs vectors of size 24, 
and outputs of size 2. In between there are fully connected layers with ReLU activations and Batch 
Normalization layers just before the activations (except at the inputs, where we use batchnorm right 
away on the inputs). Below we show an image of the architecture, and also a small summary of the 
network.

![rl-maddpg-actor-network][img_rl_maddpg_network_architecture_actor]

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
       BatchNorm1d-1                   [-1, 24]              48
            Linear-2                  [-1, 256]           6,400
       BatchNorm1d-3                  [-1, 256]             512
            Linear-4                  [-1, 128]          32,896
       BatchNorm1d-5                  [-1, 128]             256
            Linear-6                    [-1, 2]             258
================================================================
Total params: 40,370
Trainable params: 40,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.15
Estimated Total Size (MB): 0.16
----------------------------------------------------------------
PiNetwork(
  (bn0): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```

### **Critic-network architecture**

Our critic is a Q-function that accepts augmented versions of the observations and
actions. It is implemented using a Multi-layer perceptron as well, using both ReLU
activations and Batch Normalization.

Our Q-network receives as inputs joint observations combined in a `NUM_AGENTS x observations_shape`
vector, and joint actions combined in a `NUM_AGENTS x actions_shape` vector. For our case we have
**2** agents, **24** as size of the observations, and **2** as size of the actions. Our network
combines joint observations and joint actions in an intermediate concatenation step, instead of
combining them as both inputs in a single vector. The resulting architecture is shown below, as well
as a small summary of the network.

![rl-maddpg-critic-network][img_rl_maddpg_network_architecture_critic]

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
       BatchNorm1d-1                   [-1, 48]              96
            Linear-2                  [-1, 128]           6,272
            Linear-3                  [-1, 128]          17,024
            Linear-4                    [-1, 1]             129
================================================================
Total params: 23,521
Trainable params: 23,521
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.09
Estimated Total Size (MB): 0.09
----------------------------------------------------------------
Qnetwork(
  (bn0): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=48, out_features=128, bias=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

## 2. Implementation

Our implementation is self-contained minimal implementation (unlike our previous projects), which
consists of a single documented file with most of the details contained in that file (network models,
utils, trainer, etc.). We'll abstract away the functionality (as in previous projects) in later updates
to the project. The following are the main components of the implementation we will explain in more 
detail:

* **Models**: Pytorch implementation of both decentralized actor and centralized critic networks.
* **Utils**: Some helper classes, like replay buffers and noise generators.
* **Trainer**: Training loop implementation, with the implementation of the training
               of both critics (like DQN) and actors (DPG-theorem).

### 2.1 **Models**

The model architectures are the ones described in the previuos section (section 1.2), and
consist of MLPs. Below there's the Pytorch implementation of the actor-network (deterministic
policy), which can be found in between lines [63-107][url_impl_model_actor] in the **maddpg_tennis.py** 
file.

```python
class PiNetwork( nn.Module ) :
    r"""A simple deterministic policy network class to be used for the actor

    Args:
        observationShape (tuple): shape of the observations given to the network
        actionShape (tuple): shape of the actions to be computed by the network

    """
    def __init__( self, observationShape, actionShape, seed ) :
        super( PiNetwork, self ).__init__()

        self.seed = torch.manual_seed( seed )
        self.bn0 = nn.BatchNorm1d( observationShape[0] )
        self.fc1 = nn.Linear( observationShape[0], 256 )
        self.bn1 = nn.BatchNorm1d( 256 )
        self.fc2 = nn.Linear( 256, 128 )
        self.bn2 = nn.BatchNorm1d( 128 )
        self.fc3 = nn.Linear( 128, actionShape[0] )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *lecunishUniformInitializer( self.fc1 ) )
        self.fc2.weight.data.uniform_( *lecunishUniformInitializer( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, observation ) :
        r"""Forward pass for this deterministic policy

        Args:
            observation (torch.tensor): observation used to decide the action

        """
        x = self.bn0( observation )
        x = F.relu( self.bn1( self.fc1( x ) ) )
        x = F.relu( self.bn2( self.fc2( x ) ) )
        x = F.tanh( self.fc3( x ) )

        return x
```

Below there's the Pytorch implementation of the critic-network, which can be found
in between lines [110-155][url_impl_model_critic] in the **maddpg_tennis.py** file.
Notice that the network accepts joint observations and joint actions to evaluate q-values,
as shown previously in the description of the critic architecture.

```python
class Qnetwork( nn.Module ) :
    r"""A simple Q-network class to be used for the centralized critics

    Args:
        jointObservationShape (tuple): shape of the augmented state representation [o1,o2,...on]
        jointActionShape (tuple): shape of the augmented action representation [a1,a2,...,an]

    """
    def __init__( self, jointObservationShape, jointActionShape, seed ) :
        super( Qnetwork, self ).__init__()

        self.seed = torch.manual_seed( seed )

        self.bn0 = nn.BatchNorm1d( jointObservationShape[0] )
        self.fc1 = nn.Linear( jointObservationShape[0], 128 )
        self.fc2 = nn.Linear( 128 + jointActionShape[0], 128 )
        self.fc3 = nn.Linear( 128, 1 )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *lecunishUniformInitializer( self.fc1 ) )
        self.fc2.weight.data.uniform_( *lecunishUniformInitializer( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, jointObservation, jointAction ) :
        r"""Forward pass for this critic at a given (x=[o1,...,an],aa=[a1...an]) pair

        Args:
            jointObservation (torch.tensor): augmented observation [o1,o2,...,on]
            jointAction (torch.tensor): augmented action [a1,a2,...,an]

        """
        _h = self.bn0( jointObservation )
        _h = F.relu( self.fc1( _h ) )
        _h = torch.cat( [_h, jointAction], dim = 1 )
        _h = F.relu( self.fc2( _h ) )
        _h = self.fc3( _h )

        return _h
```

### 2.2 **Utils**

As in the DDPG algorithm, we make use of a replay buffer to train both our critics (like
in DQN) and actors (using DPG-theorem). However, this replay buffer is implemented with
some variations for the multiagent case:

* It receives transitions to be stores consisting of packed version of observations, actions, 
  rewards, next observations, and termination flags. All these are received as packed numpy 
  arrays with first dimension equal to the number of agents, and stored in that way (see **store**
  method). So, it effectively stores tensors of the form `[NUM-AGENTS | DATA]` of shapes `(2,24)`,
  `(2,2)`, `(2,)`, `(2,24)`, `(2,)` respectively (w.r.t. measurements earlier).

* It provides (sampling) minibatches of components packed with an extra batch-dimension (minibatch-
  size). So, it effectively returns tensors of the form `[BATCH-SIZE | NUM-AGETS | DATA]` of shapes
  `(bsize,2,24)`, `(bsize,2,2)`, `(bsize,2,1)`, `(bsize,2,24)`, `(bsize,2,1)` for the observations,
  actions, rewards, next-observations, and termination-flags respectively (all batches). Later, the
  information per agent can be easily extracted from these tensors via indexing (with agent index),
  and can also be reshaped into joint-measurements (for critics to consume).

```python
class ReplayBuffer( object ) :
    r"""Replay buffer class used to train centralized critics.

    This replay buffer is the same as our old friend the replay-buffer from
    the vanilla dqn for a single agent, with some slight variations as the 
    tuples stored now consist in some cases in augmentations of the observations
    and action spaces:

    ([o1,...,on],[a1,...,an],[r1,...,rn],[o1',...,on'],[d1,...,dn])
          x                                    x'

    The usage depends on the network that will consume this data in its forward
    pass, which could be either a decentralized actor or a centralized critic.

    For a decentralized actor:

        u    ( oi ) requires the local observation for that actor
         theta-i

    For a centralized critic:

        Q     ( [o1,...,on], [a1,...,an] ) requires both the augmented observation
         phi-i   ----------  -----------   and the joint action from the actors
                     |            |
                     x        joint-action

    So, to make things simpler, as the environment is already returning packed
    numpy ndarrays with first dimension equal to the num-agents, we will store
    these as packed tensors with first dimention equal to num-agents, and return an
    even more packed version, which would include a batch dimension on top of the other
    dimensions (n-agents,variable-shape), so we would have something like:
    
    e.g. storing:

        store( ( [obs1(24,),obs2(24,)], [a1(2,),a2(2,)], ... ) )
                 --------------------   ---------------
                    ndarray(2,24)         ndarray(2,2)

    e.g. sampling:
        batch -> ( batchObservations, batchActions, ... )
                   -----------------  ------------
                    tensor(128,2,24)   tensor(128,2,2)

    Args:
        bufferSize (int): max. number of experience tuples this buffer will hold
                          until it starts throwing away old experiences in a FIFO
                          way.
        numAgents (int): number of agents used during learning (for sanity-checks)

    """
    def __init__( self, bufferSize, numAgents ) :
        super( ReplayBuffer, self ).__init__()

        self._memory = deque( maxlen = bufferSize )
        self._numAgents = numAgents


    def store( self, transition ) :
        r"""Stores a transition tuple in memory

        The transition tuples to be stored must come in the form:

        ( [o1,...,on], [a1,...,an], [r1,...,rn], [o1',...,on'], [done1,...,donen] )

        Args:
            transition (tuple): a transition tuple to be stored in memory

        """
        # sanity-check: ensure first dimension of each transition component has the right size
        assert len( transition[0] ) == self._numAgents, 'ERROR> group observation size mismatch'
        assert len( transition[1] ) == self._numAgents, 'ERROR> group actions size mismatch'
        assert len( transition[2] ) == self._numAgents, 'ERROR> group rewards size mismatch'
        assert len( transition[3] ) == self._numAgents, 'ERROR> group next observations size mismatch'
        assert len( transition[4] ) == self._numAgents, 'ERROR> group dones size mismatch'

        self._memory.append( transition )


    def sample( self, batchSize ) :
        _batch = random.sample( self._memory, batchSize )

        _observations       = torch.tensor( [ _transition[0] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _actions            = torch.tensor( [ _transition[1] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _rewards            = torch.tensor( [ _transition[2] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )
        _observationsNext   = torch.tensor( [ _transition[3] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _dones              = torch.tensor( [ _transition[4] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )

        return _observations, _actions, _rewards, _observationsNext, _dones


    def __len__( self ) :
        return len( self._memory )
```

For the noise-generator implementations, there are two types of noise generators 
that we used in our experiments: Ornstein-Uhlenbeck noise-generator, and a simple
Normal noise-generator. Both implementations can be found in the [**OUNoise**][url_impl_replay_noisegen_ouproc] 
and [**NormalNoise**][url_impl_replay_noisegen_normal] classes.

### 2.3 **Trainer**

Finally, the trainer in our case is just a [function][url_impl_trainer] that implements the 
MADDPG algorithm, using the models and utilities explained earlier. We split the detailed
explanation of the implementation below into various important sections. Notice we have two
implementations, each in a separate file and each different is just a few lines:

* Our initial implementation, found in the [maddpg_tennis.py][url_maddpg_main] file, is just
  slightly different to the original MADDPG algorithm from the paper in the sense that it uses
  the same sampled minibatch for all agents, and it evaluates the actions for training the actor
  using the most recent versions of the actor-networks from all agents (not just its own).

* The original implementation, found in the [maddpg_tennis_original.py][url_maddpg_original] file,
  uses the same structure as in the paper, in which they sampled a different minibatch per agent,
  and also used only the actor-network of the agent being trained to predict the actions used to
  compose with the critics in the DPG-theorem update. For all other entries of the actions in the
  joint-action, it uses the actions from the minibatch for all other agents instead of computing
  them again using the respective actor-networks.

First, we create the actors and critics for all agents we will be using. Notice we use
the joint observation size for the critics as explained in previous sections.

```python
    ##------------- Create actor network (+its target counterpart)------------##
    actorsNetsLocal = [ PiNetwork( env.observation_space.shape,
                                   env.action_space.shape,
                                   seed ) for _ in range( NUM_AGENTS ) ]
    actorsNetsTarget = [ PiNetwork( env.observation_space.shape,
                                    env.action_space.shape,
                                    seed ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( actorsNetsLocal, actorsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    optimsActors = [ opt.Adam( _actorNet.parameters(), lr = LEARNING_RATE_ACTOR ) \
                        for _actorNet in actorsNetsLocal ]

    ##----------- Create critic network (+its target counterpart)-------------##
    criticsNetsLocal = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                   (NUM_AGENTS * env.action_space.shape[0],),
                                   seed ) for _ in range( NUM_AGENTS ) ]
    criticsNetsTarget = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                    (NUM_AGENTS * env.action_space.shape[0],),
                                    seed ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( criticsNetsLocal, criticsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    optimsCritics = [ opt.Adam( _criticNet.parameters(), lr = LEARNING_RATE_CRITIC ) \
                        for _criticNet in criticsNetsLocal ]
```

We then create some of the utilities we will need, like the replay buffer and the
noise generator. We used both available generators in two different experiments we
ran, as we will explain in the results section. Notice we also declare some boiler
plate code logging purposes, and the epsilon scale factor for the noise-scaling
with the appropriate schedule. Then, we have the training loop implementation, which
we will shown and explain in detail later.

```python
    # Circular Replay buffer
    rbuffer = ReplayBuffer( REPLAY_BUFFER_SIZE, NUM_AGENTS )
    # Noise process
    noise = OUNoise( env.action_space.shape, seed )
    ## noise = NormalNoise( env.action_space.shape, seed )
    # Noise scaler factor (annealed with a schedule)
    epsilon = 1.0

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( os.path.join( SESSION_FOLDER, 'tensorboard_summary' ) )
    istep = 0

    for iepisode in progressbar :

        noise.reset()
        _oo = env.reset()
        _scoreAgents = np.zeros( NUM_AGENTS )

        for i in range( MAX_STEPS_IN_EPISODE ) :
            # Training Step Implementation:
            #
            # 1. All agents take actions
            # 2. Take step in env
            # 3. Take a learning step
            #   3.0 Compute/Assemble joint quantities
            #   3.1 Train critic
            #   3.2 Train actor
```

The first part of the training step code consists of grabbing the actions that each 
agent has to take in the environment, which is shown in the snippet below. We get the 
actions using the appropriate actor networks, and add some noise for exploration. As
we don't want any gradient calculations be made in this step, we ask Pytorch to deactivate
this feature momentarily, and also make sure we are in eval mode (because of batchnorm usage).

```python
            # take full-random actions during these many steps
            if istep < TRAINING_STARTING_STEP :
                _aa = np.clip( np.random.randn( *((NUM_AGENTS,) + env.action_space.shape) ), -1., 1. )
            # take actions from exploratory policy
            else :
                # eval-mode (in case batchnorm is used)
                for _actorNet in actorsNetsLocal :
                    _actorNet.eval()

                # choose an action for each agent using its own actor network
                with torch.no_grad() :
                    _aa = []
                    for iactor, _actorNet in enumerate( actorsNetsLocal ) :
                        # evaluate action to take from each actor policy
                        _a = _actorNet( torch.from_numpy( _oo[iactor] ).unsqueeze( 0 ).float().to( DEVICE ) ).cpu().data.numpy().squeeze()
                        _aa.append( _a )
                    _aa = np.array( _aa )
                    # add some noise sampled from the noise process (each agent gets different sample)
                    _nn = np.array( [ epsilon * noise.sample() for _ in range( NUM_AGENTS ) ] ).reshape( _aa.shape )
                    _aa += _nn
                    # actions are speed-factors (range (-1,1)) in both x and y
                    _aa = np.clip( _aa, -1., 1. )

                # back to train-mode (in case batchnorm is used)
                for _actorNet in actorsNetsLocal :
                    _actorNet.train()
```

The next part of the training step is to take those actions in the environment, and
grab the information received and stored it into the replay buffer. This is shown in
the snippet below. Notice we store the packed versions of the measurements required
for training, as expected for our replay buffer implementation.

```python
            # take action in the environment and grab bounty
            _oonext, _rr, _dd, _ = env.step( _aa )
            # store joint information (form (NAGENTS,) + MEASUREMENT-SHAPE)
            if i == MAX_STEPS_IN_EPISODE - 1 :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, np.ones_like( _dd ) ) )
            else :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, _dd ) )
```

The following section is in charge of checking when to take a learning step. This
part is a bit different between our two implementations

```python
            if len( rbuffer ) > BATCH_SIZE and istep % TRAIN_FREQUENCY_STEPS == 0 and \
               istep >= TRAINING_STARTING_STEP :
                for _ in range( TRAIN_NUM_UPDATES ) :
                    # Learning Step
                    # ...
```

This next part is a bit different from our initial implementation, and the original implementation
of the paper. Firstly, for the initial implementation, we sample a minibatch from the replay buffer,
and then used it for all agents (as you can see the sampling being outside the for-loop over the actors).
Also, we assemble the joint-observations, joint-actions, joint-next-observations and joint-next-actions
outside that for-loop from the common minibatch. Notice also that the joint-next-actions are computed with
the other agents' target actor-networks, which might differ from the original implementation as the target
networks might have received a soft-update from their related-agent update step.

```python
                    # grab a batch of data from the replay buffer
                    _observations, _actions, _rewards, _observationsNext, _dones = rbuffer.sample( BATCH_SIZE )

                    # compute joint observations and actions to be passed ...
                    # to the critic, which basically consists of keep the ...
                    # batch dimension and vectorize everything else into one ...
                    # single dimension [o1,...,on] and [a1,...,an]
                    _batchJointObservations = _observations.reshape( _observations.shape[0], -1 )
                    _batchJointObservationsNext = _observationsNext.reshape( _observationsNext.shape[0], -1 )
                    _batchJointActions = _actions.reshape( _actions.shape[0], -1 )

                    # compute the joint next actions required for the centralized ...
                    # critics q-target computation
                    with torch.no_grad() :
                        _batchJointActionsNext = torch.stack( [ actorsNetsTarget[iactor]( _observationsNext[:,iactor,:] )  \
                                                                for iactor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsNext = _batchJointActionsNext.reshape( _batchJointActionsNext.shape[0], -1 )

                    for iactor in range( NUM_AGENTS ) :
                        # extract local observations to be fed to the actors, ...
                        # as well as local rewards and dones to be used for local 
                        # q-targets computation using critics
                        _batchLocalObservations = _observations[:,iactor,:]
                        _batchLocalRewards = _rewards[:,iactor,:]
                        _batchLocalDones = _dones[:,iactor,:]

                        # 3.1 Train critic
                        # 3.2 Train actor
```

The original implementation uses different batches per agent, which is shown in the snippet
below. Notice how we just swapped the order of the for-loop over the agents, and then grabbed
all information needed (as in the previous snippet) using the information of the specific
minibatch sampled per agent.

```python
                    for iactor in range( NUM_AGENTS ) :
                        # grab a batch of data from the replay buffer
                        _observations, _actions, _rewards, _observationsNext, _dones = rbuffer.sample( BATCH_SIZE )
    
                        # compute joint observations and actions to be passed ...
                        # to the critic, which basically consists of keep the ...
                        # batch dimension and vectorize everything else into one ...
                        # single dimension [o1,...,on] and [a1,...,an]
                        _batchJointObservations = _observations.reshape( _observations.shape[0], -1 )
                        _batchJointObservationsNext = _observationsNext.reshape( _observationsNext.shape[0], -1 )
                        _batchJointActions = _actions.reshape( _actions.shape[0], -1 )
    
                        # compute the joint next actions required for the centralized ...
                        # critics q-target computation
                        with torch.no_grad() :
                            _batchJointActionsNext = torch.stack( [ actorsNetsTarget[iactorIndx]( _observationsNext[:,iactorIndx,:] )  \
                                                                    for iactorIndx in range( NUM_AGENTS ) ], dim = 1 )
                            _batchJointActionsNext = _batchJointActionsNext.reshape( _batchJointActionsNext.shape[0], -1 )

                        # extract local observations to be fed to the actors, ...
                        # as well as local rewards and dones to be used for local 
                        # q-targets computation using critics
                        _batchLocalObservations = _observations[:,iactor,:]
                        _batchLocalRewards = _rewards[:,iactor,:]
                        _batchLocalDones = _dones[:,iactor,:]

                        # 3.1 Train critic
                        # 3.2 Train actor
```

The next part is similar for both implementations, and is in charge of training the
critic in a similar way to DQN (fitting q-values to TD-targets). The only difference
is the usage of joint-observations and joint-actions to evaluate the q-values and
the TD-targets (joint-next-observations and joint-next-actions)s.

```python
                        #---------------------- TRAIN CRITICS  --------------------#

                        # compute current q-values for the joint-actions taken ...
                        # at joint-observations using the critic, as explained ...
                        # in the MADDPG algorithm:
                        #
                        # Q(x,a1,a2,...,an) -> Q( [o1,o2,...,on], [a1,a2,...,an] )
                        #                       phi-i
                        _qvalues = criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActions )
                        # compute target q-values using both decentralized ...
                        # target actor and centralized target critic for this ...
                        # current actor, as explained in the MADDPG algorithm:
                        #
                        # Q-targets  = r  + ( 1 - done ) * gamma * Q  ( [o1',...,on'], [a1',...,an'] )
                        #          i    i             i             phi-target-i
                        # 
                        # 
                        with torch.no_grad() :
                            _qvaluesTarget = _batchLocalRewards + ( 1. - _batchLocalDones ) \
                                                * GAMMA * criticsNetsTarget[iactor]( _batchJointObservationsNext, 
                                                                                     _batchJointActionsNext )
        
                        # compute loss for the critic
                        optimsCritics[iactor].zero_grad()
                        _lossCritic = F.mse_loss( _qvalues, _qvaluesTarget )
                        _lossCritic.backward()
                        torch.nn.utils.clip_grad_norm( criticsNetsLocal[iactor].parameters(), 1 )
                        optimsCritics[iactor].step()
```

We then have to train the actor, so we used the DPG-theorem in a similar way to the
single-agent DDPG case, with the small difference that the critic from which we take
gradients from use joint-observations and joint-actions instead. The snippet below 
shows the implementation of our initial variant that uses all most recent actor networks
from all other agents (as well as the current actor-network for the agent being trained), 
to compute the actions used to compose the joint-action for the critic.

```python
                        #---------------------- TRAIN ACTORS  ---------------------#
    
                        # compute loss for the actor, from the objective to "maximize":
                        #
                        # dJ / dtheta = E [ dQ / du * du / dtheta ]
                        #
                        # where:
                        #   * theta: weights of the actor
                        #   * dQ / du : gradient of Q w.r.t. u (actions taken)
                        #   * du / dtheta : gradient of the Actor's weights
        
                        optimsActors[iactor].zero_grad()

                        # compute predicted actions for current local observations ...
                        # as we will need them for computing the gradients of the ...
                        # actor. Recall that these gradients depend on the gradients ...
                        # of its own related centralized critic, which need the joint ...
                        # actions to work. Keep with grads here as we have to build ...
                        # the computation graph with these operations

                        _batchJointActionsPred = torch.stack( [ actorsNetsLocal[indexActor]( _observations[:,indexActor,:] )  \
                                                                    for indexActor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsPred = _batchJointActionsPred.reshape( _batchJointActionsPred.shape[0], -1 )

                        # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
                        _lossActor = -criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActionsPred ).mean()
                        _lossActor.backward()
                        optimsActors[iactor].step()
```

The original implementation of the training step for the actor-networks of the agents is
shown below. Notice that we only use the actor-network of the current actor being updated,
and to compose the joint-action we use all actions from the replay buffer to fill the
actions from the other agents.

```python
                        #---------------------- TRAIN ACTORS  ---------------------#
    
                        # compute loss for the actor, from the objective to "maximize":
                        #
                        # dJ / dtheta = E [ dQ / du * du / dtheta ]
                        #
                        # where:
                        #   * theta: weights of the actor
                        #   * dQ / du : gradient of Q w.r.t. u (actions taken)
                        #   * du / dtheta : gradient of the Actor's weights
        
                        optimsActors[iactor].zero_grad()

                        # compute predicted actions for current local observations ...
                        # as we will need them for computing the gradients of the ...
                        # actor. Recall that these gradients depend on the gradients ...
                        # of its own related centralized critic, which need the joint ...
                        # actions to work. Keep with grads here as we have to build ...
                        # the computation graph with these operations
                        _batchJointActionsPred = torch.stack( [ actorsNetsLocal[indexActor]( _observations[:,indexActor,:] )  \
                                                                  if indexActor == iactor else _actions[:,indexActor,:] \
                                                                    for indexActor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsPred = _batchJointActionsPred.reshape( _batchJointActionsPred.shape[0], -1 )

                        # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
                        _lossActor = -criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActionsPred ).mean()
                        _lossActor.backward()
                        optimsActors[iactor].step()
```

Finally, just before exiting the update step for each agent, we make a soft-update using
Polyak-averaging for both actor and critic networks. Once all learning steps are made for
all agents, we update the noise-scaler factor using the schedule requested.

```python
                        # update target networks
                        actorsNetsTarget[iactor].copy( actorsNetsLocal[iactor], TAU )
                        criticsNetsTarget[iactor].copy( criticsNetsLocal[iactor], TAU )
    
                    # update epsilon using schedule
                    if EPSILON_SCHEDULE == 'linear' :
                        epsilon = max( 0.1, epsilon - EPSILON_DECAY_LINEAR )
                    else :
                        epsilon = max( 0.1, epsilon * EPSILON_DECAY_FACTOR )
```

There's some more boiler plate code used for logging and saving training results, which
can be found at the end of the train function, but all parts explained above are the main
implementation of the MADDPG algorithm.

## 3. Results

In this section we give a more detailed description of the results, which include
our choice of hyperparameters, results provided for the project submission, and some
test over different random seeds to check variability.

### 3.0 Running a pretrained agent

We provide pre-trained agents that used the initial implementation (trained with the 
hyperparameters for the submission). To test these agents just run the following in 
your terminal:

```bash
python maddpg_tennis.py test --sessionId=session_submission
```

There are also pre-trained agents that were trained using the original-paper implementation.
To test these just run the following in your terminal:

```bash
python maddpg_tennis_original.py test --sessionId=session_submission
```

The submission results consist of the following:

* The weights of the trained agent are provided in the **results/session_submission** 
  folder, and are saved in the **checkpoint_actor_0.pth**, **checkpoint_actor_1.pth**,
  **checkpoint_critic_0.pth** and **checkpoint_critic_1.pth** files.

* The tensorboard logs of the training sessions, which can be found in the folder
  called **results/session_submission/tensorboard_summary**. You can check the training
  results invoking tensorboard using the command below, and then open the tensorboard
  client in your browser using the url given in the terminal:

```bash
tensorboard --logdir=./results/session_submission/tensorboard_summary
```

Notice that there are also results for the submission using the original implementation,
which can be checked in a similar way as follows:

```bash
tensorboard --logdir=./results/session_submission_original/tensorboard_summary
```

Also, you can check [this](https://youtu.be/jxE4OoPmtzM) video of the pre-trained 
agent solving the required task.

### 3.1 Hyperparameters

The hyperparameters available are the same as in the single-agent case of DDPG. We
started with the same hyperparameters as in the previous project (for the batch-norm
case), and then started tweaking them until we got a working solution. The way we approached
this tuning process was mostly heuristic (based on intuition of the training dynamics), as 
we couldn't afford running either grid-search nor random-search due to compute constraints.
Below we mention the steps taken to arrive at the hyperparameters that best work for us so far.

* We let the agents train with that configuration at first, and noticed that the episodes
  termiate very quickly, and that they did not learn much at first as the noise forced the
  agents at first to take too extreme actions. To ammend this, we decided to give the agents
  a period of random exploration before starting taking learning steps.

* Also, as at first we don't usually receive that many good episodes, we decided to avoid taking
  too many learning steps per update, and reduce it to a sweetspot of 4:2 (update every 4 timesteps,
  take two learning step per update).

* After only those two changes we started getting good results (almost reaching the objective of +0.5),
  so we just adjusted the epsilon factor by looking at the tensorboard plots and identifying a sweetspot
  for this factor (as at first we had a too small decay factor).

The final hyperparameters used for the submission using the initial MADDPG implementation are shown in
the table below.

Hyperparameter              |     Value     |   Description
----------------------------|---------------|-----------------
gamma                       | 0.99          | Discount factor
tau                         | 0.001         | Polyak averaging factor
replay-buffer size          | 1000000       | Size of the replay buffer
learning rate - Actor       | 0.001         | Learning rate for the actor (using Adam)
learning rate - Critic      | 0.001         | Learning rate for the actor (using Adam)
batch-Size                  | 256           | Batch size used for a learning step
train-Frequency-Steps       | 4             | How often to request learning from the agent
train-NumLearning-Steps     | 2             | Number of learning steps per learning request of the agent
noise-type                  | 'ounoise'     | Type of noise used, either ounoise='Ornstein-Uhlenbeck', or normal="Sample-gaussian"
noise-OU-Mu                 | 0.0           | Mu-param &mu; of the Ornstein-Uhlenbeck noise process
noise-OU-Theta              | 0.15          | Theta param &theta; of the Ornstein-Uhlenbeck noise process
noise-OU-Sigma              | 0.2           | Sigma param &sigma; of the Ornstein-Uhlenbeck noise process
noise-Normal-Stddev         | 0.2           | Standard deviation of the normal noise
epsilon-schedule            | 'linear'      | Type of schedule used to decay the noise factor &epsilon;,either 'linear' or 'geometric'
epsilon-factor-Geom         | 0.999         | Decay factor for the geometric schedule
epsilon-factor-Linear       | 2e-5          | Decay factor for the linear schedule
training-starting-step      | 50000         | At which step to start learning

As we mentioned in previous sections, our initial implementation was different from the original
algorithm from the paper. We noticed this while writing section 1, and decided to ammend the implementation
to have a more suitable comparison (for future experiments) and to check if the original provides
better training performance. After updating the implementation we started training and got somewhat
similar *starting* results with the same set of hyperparameters. However, the performance never got
better than +0.5, and training was a bit unstable. To ammend this, we started doing the following
tweaks to the hyperparameters until we got a working solution:

* Because learning was a bit more unstable than the previous implementation, we decided to use
  a bit more conservative/smaller learning rates. This started showing promise as we almost got
  to the required +0.5 objective. We then just let it train for more episodes and it eventually
  got to a point similar to the early-timesteps results of the previous implementation, which
  we thought might benefit from training from even more episodes.

* We then decided to take a slightly different approach by not using a starting exploration phase,
  for which we decided to change to choose a different noise-generator than the Ornstein-Uhlenbeck
  process. We noticed from the previous implementation that non-correlated noise seemed to work
  well for exploration at the beginning of training, as the agent might have more chances to touch
  the ball and pass it thatn with the previous noise-generator. This worked fine and we started to
  get results similar to the ones shown in the project description in the Udacity Platform.

* After just adjusting the noise-scaler factor and changing the learning frequency and number of steps
  per update, we ended up with an implementation in accordance to the original (from the paper) that
  worked quite well and did not needed a lot of exploration at first.

The final hyperparameters used for this implementation are shown in the table below.

Hyperparameter              |     Value     |   Description
----------------------------|---------------|-----------------
gamma                       | 0.999         | Discount factor
tau                         | 0.001         | Polyak averaging factor
replay-buffer size          | 1000000       | Size of the replay buffer
learning rate - Actor       | 0.0004        | Learning rate for the actor (using Adam)
learning rate - Critic      | 0.0008        | Learning rate for the actor (using Adam)
batch-Size                  | 256           | Batch size used for a learning step
train-Frequency-Steps       | 1             | How often to request learning from the agent
train-NumLearning-Steps     | 1             | Number of learning steps per learning request of the agent
noise-type                  | 'normal'      | Type of noise used, either ounoise='Ornstein-Uhlenbeck', or normal="Sample-gaussian"
noise-OU-Mu                 | 0.0           | Mu-param &mu; of the Ornstein-Uhlenbeck noise process
noise-OU-Theta              | 0.15          | Theta param &theta; of the Ornstein-Uhlenbeck noise process
noise-OU-Sigma              | 0.2           | Sigma param &sigma; of the Ornstein-Uhlenbeck noise process
noise-Normal-Stddev         | 0.2           | Standard deviation of the normal noise
epsilon-schedule            | 'linear'      | Type of schedule used to decay the noise factor &epsilon;,either 'linear' or 'geometric'
epsilon-factor-Geom         | 0.999         | Decay factor for the geometric schedule
epsilon-factor-Linear       | 5e-6          | Decay factor for the linear schedule
training-starting-step      | 0             | At which step to start learning

### 3.2 Submission results

Using the configurations mentioned earlier we ended up having working solutions that could
maintain an average reward of +0.5 over 100 episodes. Below there are some plots of the results
of training sessions with such configurations. 

* **Initial implementation**. Notice that **we reach an average score of +0.5 starting at episode 5200**, 
  and then maintain it over the remaining episodes.

![maddpg-results-passing-submission-1][img_results_submission_1]

* **Paper implementation**. Notice that **we reach an average score of +0.5 starting at episode 1500**, 
  and then maintain it over the remaining episodes.

![maddpg-results-passing-submission-2][img_results_submission_2]

Below we show the agent's performance during evaluation. Notice that the agent can smoothly
follow the goals, missing it just in very few situations for just a small time before recovering.

![maddpg-results-agent-test][gif_project_3_tennis_agent]

### 3.3 Experiments results

We did some experiments to check both the variability of the results obtained using both implementations
by running the two versions of the algorithm with various random seeds. Below we show the results of such
runs:

* **Initial implementation**: Sub-figure (a) shows average scores over a window of 100, and sub-figure 
  (b) shows an std-plot over all runs.

![maddpg-results-initial-implementation][img_results_initial_implementation]

* **Original paper implementation**: Sub-figure (a) shows average scores over a window of 100, and sub-figure 
  (b) shows an std-plot over all runs.

![maddpg-results-original-implementation][img_results_original_implementation]

Notice that in both cases all runs successfully reach the objective, with some degree of variability in between
random seeds. It'd be interesting to run both implementations for longer in order to check if the performance 
crashes, as in the solution plot shown as baseline by Udacity.

## 4. Future Work

Finally, below we mention some of the improvements we consider making in following
updates to this post:

* **Run both implementations over the same conditions of no-initial exploration**: As mentioned
  in previous sections, for the initial implementation we decided to go for some fixed number of
  exploration steps, whereas for the original implementation we just started learning from the beginning.
  This issue was caused mainly because for the initial submission runs we used a Ornstein-Uhlenbeck based
  noise generator, which seem to not gave us sufficient exploration over the whole space, as the
  environment resets quite quickly and rewards are a bit more sparse than the previous project.

* **Use PPO and SAC instead of DDPG as actor-critic algorithm**: The algorithm studied here is based
  on a general actor-critic framework in which critics are centralized and actors are decentralized, so
  we could use a different actor-critic algorithm in a similar way. It'd be interesting to see how much
  more stable these other actor-critic algorithms would perform.

* **Run the implementations for longer**: By running it for longer we could test if the algorithm
  crashes as the baseline does.

* **Solve the optional soccer task**: Tune all baselines and try to solve the optional Soccer Task (soccer 
  env. from ml-agents).

* **Implement recurrent versions**: Finally, we'd like to implement recurrent variants of these agents. It 
  might take way longer to train, but it'd be interesting to see if they give better results.

## References

* [1] [Sutton, Richard & Barto, Andrew. *Reinforcement Learning: An introduction.*](http://incompleteideas.net/book/RLbook2018.pdf)
* [2] [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* paper by Lowe et. al.](https://arxiv.org/pdf/1706.02275.pdf)
* [3] [*Continuous control through deep reinforcement learning* paper by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf)
* [4] [*Deterministic Policy Gradients Algorithms* paper by Silver et. al.](http://proceedings.mlr.press/v32/silver14.pdf)
* [5] [DDPG implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
* [6] [Post on *Deep Deterministic Policy Gradients* from OpenAI's **Spinning Up in RL**](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [7] [Post on *Policy Gradient Algorithms* by **Lilian Weng**](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
* [8] [*Lecture 8: Advanced Q-Learning Algorithms* from Berkeley's cs294 DeepRL course](https://youtu.be/hP1UHU_1xEQ?t=4365)
* [9] [Udacity DeepRL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)