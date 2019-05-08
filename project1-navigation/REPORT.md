# Using DQN to solve the Banana environment from ML-Agents

This is an accompanying post for the submission of the **Project 1: navigation**
from the [**Udacity Deep Reinforcement Learning Nanodegree**](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893),
which consisted on building a DQN-based agent to navigate and collect bananas
from the *Banana Collector* environment from [**Unity ML-Agents**](https://github.com/Unity-Technologies/ml-agents).

{{<figure src="/imgs/gif_banana_agent.gif" alt="fig-banana-agent" position="center" 
    caption="Figure 1. DQN agent collecting bananas" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

The following are the topics to be covered in this post:

1. Description of the *Banana Collector Environment*.
2. Setting up the dependencies to run the accompanying code.
3. An overview of the DQN algorithm.
4. The chosen hyperparameters and some discussion.
5. The results obtained and some discussion.
6. An overview of the improvements: Double DQN.
7. An overview of the improvements: Prioritized Experience Replay.
8. Some ablation tests and some discussion.
9. Final remarks and future improvements.

## 1. Description of the Banana Collector Environment

The environment chosen for the project was a **modified version of the Banana 
Collector Environment** from the Unity ML-Agents toolkit. The original version
can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector),
and our version consists of a custom build provided by Udacity with the following
description:

* A single agent that can move in a planar arena, with **observations** given by
  a set of distance-based sensors and some intrinsic measurements, and **actions** 
  consisting of 4 discrete commands.
* A set of NPCs consisting of bananas of two categories: **yellow bananas**, which give the
  agent a **reward of +1**, and **purple banans**, which give the agent a **reward of -1**.
* The task is **episodic**, with a maximum of 300 steps per episode.

### 1.1 Agent observations

The observations the agent gets from the environment come from the **agent's linear velocity**
in the plane (2 entries), and a **set of 7 ray perceptions**. These ray perceptions consist 
of rays shot in certain fixed directions from the agent. Each of these perceptions 
returns a vector of **5 entries each**, whose values are explained below:

* The first 4 entries consist of a one-hot encoding of the type of object the ray hit, and
  these could either be a yellow banana, a purple banana, a wall or nothing at all.
* The last entry consist of the percent of the ray length at which the object was found. If
  no object is found at least at this maximum length, then the 4th entry is set to 1, and this
  entry is set to 0.0.

Below there are two separate cases that show the ray-perceptions. The first one
to the left shows all rays reaching at least one object (either purple banana, yellow
banana or a wall) and also the 7 sensor reading in array form (see the encodings in the
4 first entries do not contain the *none found* case). The second one to the right
shows all but one ray reaching an object and also the 7 sensor readings in array form (see
the encodings in the 4 first entrines do include the *none found* case for the 4th perception).

{{<figure src="/imgs/img_banana_env_observations.png" alt="fig-banana-agent-ray-observations" position="center" 
    caption="Figure 2. Agent ray-perceptions. a) 7 rays reaching at least one object (banana or wall). b) One rayreaching the max. length before reaching any object" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

All these measurements account for an observation consisting of a vector with
37 elements. This vector observation will be the representation of the state of 
the agent in the environment that we will use as input to the Q-network, which
will be discussed later.

#### **Note**

This representation is applicable only to our custom build (provided
by Udacity), as the original Banana Collector from ML-Agents consists of a vector
observation of 53 entries. The rays in the original environment give extra information
about the current state of the agent (the agent in the original environment can shoot and 
be shot), which give 2 extra measurements, and the rays give also information about 
neighbouring agents (2 extra measurements per ray).

If curious, you can take a look at the C# implementation in the UnitySDK folder
of the ML-Agents repository. See the *'CollectObservations'* method (shown below) 
in the [BananaAgent.cs](https://github.com/Unity-Technologies/ml-agents/blob/37d139af636e4a2351751fbf0f2fca5a9ed7457f/UnitySDK/Assets/ML-Agents/Examples/BananaCollectors/Scripts/BananaAgent.cs#L44)
file. This is helpfull in case you want to create a variation of the environment
for other purposes other than this project.

```Csharp
    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            float rayDistance = 50f;
            float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
            string[] detectableObjects = { "banana", "agent", "wall", "badBanana", "frozenAgent" };
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
            Vector3 localVelocity = transform.InverseTransformDirection(agentRb.velocity);
            AddVectorObs(localVelocity.x);
            AddVectorObs(localVelocity.z);
            AddVectorObs(System.Convert.ToInt32(frozen));
            AddVectorObs(System.Convert.ToInt32(shoot));
        }
    }
```

### 1.2 Agent actions

The actions that the agent can take consist of 4 discrete actions that serve as
commands for the movement of the agent in the plane. The indices for each of these
actions are the following :

* **Action 0**: Move forward.
* **Action 1**: Move backward.
* **Action 2**: Turn left.
* **Action 3**: Turn right.

Figure 3 shows these four actions that conform the discrete action space of the
agent.

{{<figure src="/imgs/img_banana_env_actions.png" alt="fig-banana-agent-actions" position="center" 
    caption="Figure 3. Agent actions." captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

#### **Note**

This actions are applicable again only for our custom build, as the original
environment from ML-Agents has even more actions, using action tables (newer API).
This newer API accepts in most of the cases a tuple or list for the actions, with
each entry representing corresponding to a specific action table (a nested set of
actions) that the agent can take. For example, for the original banana collector
environment the actions passed should be:

```python
# actions are of the form: [actInTable1, actInTable2, actINTable3, actInTable4]

# move forward
action = [ 1, 0, 0, 0 ]
# move backward
action = [ 2, 0, 0, 0 ]
# mode sideways left
action = [ 0, 1, 0, 0 ]
# mode sideways right
action = [ 0, 2, 0, 0 ]
# turn left
action = [ 0, 0, 1, 0 ]
# turn right
action = [ 0, 0, 2, 0 ]
```

### 1.3 Environment dynamics and rewards

The agent spawns randomly in the plane of the arena, which is limited by walls. Upon
contact with a banana (either yellow or purple) the agents receives the appropriate reward
(+1 or -1 depending on the banana). The task is considered solved once the agent
can consistently get an awerage reward of **+13** over 100 episodes.

## 2. Accompanying code and setup

The code for our submission is hosted on [github](https://github.com/wpumacay/DeeprlND-projects/tree/master/project1-navigation). 
The [README.md](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/README.md) 
file already contains the instruction of how to setup the environment, but we repeat them
here for completeness (and to save you a tab in your browser :laughing:).

### 2.1 Custom environment build

The environment provided is a custom build of the ml-agents Banana Collector
environment, with the features described in the earlier section. The environment
is provided as an executable which we can download from the links below according
to our platform:

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Keep the .zip file where you downloaded it for now. We will later explain where
to extract its contents when finishing the setup.

### **Note**

The executables provided are compatible only with an older version of the ML-Agents
toolkit (version 0.4.0). The setup below will take care of this, but keep in mind
if you want to use the executable on your own.

### 2.2 Dependencies

As already mentioned, the environment provided is an instance of an ml-agents
environment and as such it requires the **appropriate** ml-agents python API to be
installed in order to interact with it. There are also some other dependencies, so
to facilitate installation all this setup is done by installing the **navigation**
from the accompanying code, which we will discuss later. Also, there is no need
to download the Unity Editor for this project (unless you want to do a custom build
for your platform) because the builds for the appropriate platforms are already
provided for us.

### 2.3 Downloading accompanying code and finishing setup

* Grab the accompanying code from the github repo.

```bash
# clone the repo
git clone https://github.com/wpumacay/DeeprlND-projects
# go to the project1-navigation folder
cd DeeprlND-projects/project1-navigation
```

* (Suggested) Create an environment using a virtual environment manager like 
  [pipenv](https://docs.pipenv.org/en/latest/) or [conda](https://conda.io/en/latest/).

```bash
# create a virtual env. using conda
conda create -n deeprl_navigation python=3.6
# activate the environment
source activate deeprl_navigation
```

* Install [pytorch](https://pytorch.org/get-started/locally/#start-locally).

```bash
# Option 1: install pytorch (along with cuda). No torchvision, as we are not using it yet.
conda install pytorch cudatoolkit=9.0 -c pytorch
# Option 2: install using pip. No torchvision, as we are not using it yet.
pip install torch
```

* (Optional) Our implementation decouples the requirement of the function approximator
  (model) from the actual DQN core implementation, so we have also an implementation
  based on tensorflow in case you want to try that out.

```bash
# install tensorflow (1.12.0)
pip install tensorflow==1.12.0
# (Optional) In case you want to train it using a GPU
pip install tensorflow-gpu==1.12.0
```

* Finally, install the navigation package using pip and the provided 
  [setup.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/setup.py) 
  file (make sure you are in the folder where the *setup.py* file is located).

```bash
# install the navigation package and its dependencies using pip (dev mode to allow changes)
pip install -e .
```

## 3. An overview of the DQN algorithm

As mentioned in the title, our agent is based on the DQN agent introduced in the 
[**Human-level control through deep reinforcement learning**](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
paper by Mnih, et. al. We will give a brief description of the algorithm in this section,
which is heavily based on Sutton and Barto's [book](http://incompleteideas.net/book/RLbook2018.pdf).
For completeness we will start with a brief introduction to reinforcement learning, and
then give the full description of the DQN algorithm.

### 3.1 RL concepts

Reinforcement Learning (RL) is a learning approach in which an **Agent** learns by
**trial and error** while interacting with an **Environment**. The core setup is shown 
in Figure 1, where we have an agent in a certain **state** \\( S_{t} \\)
interacting with an environment by applying some **action** \\( A_{t} \\). 
Because of this interaction, the agent receives a **reward** \\( R_{t+1} \\) from 
the environment and it also ends up in a **new state** \\( S_{t+1} \\).

{{<figure src="/imgs/img_rl_loop.png" alt="img-rl-loop" position="center" 
    caption="Figure 4. RL interaction loop" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

This can be further formalize using the framework of Markov Decision Proceses (MDPs).
Using this framework we can define our RL problem as follows:

> A Markov Decision Process (MDP) is defined as a tuple of the following components:
>
> * **A state space** \\( \mathbb{S} \\) of configurations \\( s_{t} \\) for an agent.
> * **An action space** \\( \mathbb{A} \\) of actions \\( a_{t} \\) that the agent can take in the environment.
> * **A transition model** \\( p(s',r | s,a) \\) that defines the distribution of states
>   that an agent can land on and rewards it can obtain \\( (s',r) \\) by taking an
>   action \\( a \\) in a state \\( s \\).

The objective of the agent is to maximize the **total sum of rewards** that it can
get from its interactions, and because the environment can potentially be stochastic
(recall that the transition model defines a probability distribution) the objective
is usually formulated as an **expectation** ( \\( \mathbb{E}  \\) ) over the random 
variable defined by the total sum of rewards. Mathematically, this objective is 
described in as follows.

$$
\mathbb{E} \left \{ r_{t+1} + r_{t+2} + r_{t+3} + \dots \right \}
$$

Notice the expectation is over the sum of a sequence of rewards. This sequence comes
from a **trajectory** (\\( \tau \\)) that the agent defines by interacting with the environment
in a sequential manner.

$$
\tau = \left \{ (s_{0},a_{0},r_{1}),(s_{1},a_{1},r_{2}),\dots,(s_{t},a_{t},r_{t+1}),\dots \right \}
$$

Tasks that always give finite-size trajectories can be defined as **episodic** (like games), 
whereas tasks that go on forever are defined as **continuous** (like life itself). The
task we are dealing in this post is episodic, and the length of an episode (max. length
of any trajectory) is 300.

There is a slight addition to the objective defined earlier that is often used: **the discount factor**
\\( \gamma \\). This factor tries to take into account the effect that a same ammount 
of reward in the far future should be less interesting than the same amount now (kind of
like interest rates when dealing with money). We introduce this by multiplying each
reward by a power of this factor to the number of steps into the future.

$$
\mathbb{E} \left \{ r_{t+1} + \gamma r_{t+2} + \gamma^{2} r_{t+3} + \dots \right \}
$$

**Sidenote**: *The most important reason we use discounting is for mathematical 
convenience, as it allows to keep our objective from exploding in the non-episodic 
case (to derive this, just replace each \\( r_{t} \\) for the maximum reward \\( r_{max} \\), 
and sum up the geometric series). There is another approach which deals with the [undiscounted 
average reward setting](https://link.springer.com/content/pdf/10.1007%2FBF00114727.pdf). 
This approach leads to a different set of bellman equations from the ones we will
study.*

A solution to the RL problem consists of a **Policy** \\( \pi \\), which is a mapping
from the current state we are ( \\( s_{t} \\) ) to an appropriate action ( \\( a_{t} \\) )
from the action space. Such mapping is basically a function, and we can define it as follows:

> A **deterministic** policy is a mapping \\( \pi : \mathbb{S} \rightarrow \mathbb{A} \\)
> that returns an action to take \\( a_{t} \\) from a given state \\( s_{t} \\).
>
> $$
> a_{t} = \pi(s_{t})
> $$

We could also define an stochastic policy, which instead of returning an action
\\( a_{t} \\) in a certain situation given by the state \\( s_{t} \\) it returns
a distribution over all possible actions that can be taken in that situation 
\\( a_{t} \sim \pi(.|s_{t}) \\).

> A **stochastic** policy is a mapping \\( \pi : \mathbb{S} \times \mathbb{A} \rightarrow \mathbb{R} \\)
> that returns a distribution over actions to take \\( a_{t} \\) from a given state \\( s_{t} \\).
>
> $$
> a_{t} \sim \pi(.|s_{t})
> $$

Our objective can then be formulated as follows: **find a policy \\( \pi \\) that maximizes
the expected discounted sum of rewards**.

$$
\max_{\pi} \mathbb{E}_{\pi} \left \{ r_{t+1} + \gamma r_{t+2} + \gamma^{2} r_{t+3} + \dots \right \}
$$

To wrap up this section we will introduce three more concepts that we will use 
throughout the rest of this post: **returns** \\( G \\), **state-value functions** 
\\( V(s) \\) and **action-value functions** \\( Q(s,a) \\).

> The return \\( G_{t} \\) is defined as the discounted sum of rewards obtained
> over a trajectory from time step \\( t \\) onwards.
>
> $$
> G_{t} = r_{t+1} + \gamma r_{t+2} + \gamma^{2} r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+1}
> $$

By using the return we can write the objective of the agent in a more compact way,
as follows:

$$
\max_{\pi} \mathbb{E}_{\pi} \left \{ G_{t} \right \}
$$

> The state-value function \\( V_{\pi}(s) \\) is defined as the expected return that
> an agent can get if it starts at state \\( s_{t} = s \\) and then follows policy
> \\( \pi \\) onwards.
>
> $$
> V_{\pi}(s) = \mathbb{E} \left \{ G_{t} | s_{t} = s; \pi \right \}
> $$

This function \\( V_{\pi}(s) \\) serves as a kind of **intuition of how well a certain
state is if we are following a specific policy**. The figure below (taken from the DQN paper [2])
illustrates this more clearly with the game of breakout as an example. The agent's 
state-value function on the bottom part of the figure shows that in the state in which 
the agent makes a hole in the bricks its estimation of the value greatly increases 
(section labeled with *4*) in the graph.

{{<figure src="/imgs/img_rl_vfunction_intuition.png" alt="fig-rl-vfunction-intuition" position="center" 
    caption="Figure 6. State-value function in the game of breakout. Top: states of the agent. Bottom: estimate of the return from this state via state-value function. Taken from [2]" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

> The action-value function \\( Q_{\pi}(s,a) \\) is defined as the expected return that
> an agent can get if it starts at state \\( s_{t} = s \\), take an action \\( a_{t} = a \\)
> and then follows the policy onwards.
>
> $$
> Q_{\pi}(s,a) = \mathbb{E} \left \{ G_{t} | s_{t} = s, a_{t} = a; \pi \right \}
> $$

This function \\( Q_{\pi}(s,a) \\) serves also as a kind of **intuition of how well
a certain action is if we apply it in a certation state if we are following a specific policy**.
The figure below illustrates this more clearly (again, taken from the DQN paper [2])
with the game of pong as an example. The agent's action-value function tell us how
well is a certain action in a certain situation, and as you can see in the states labeled
with (2) and (3) the function estimates that action Up will give a greater return than
the other two actions.

{{<figure src="/imgs/img_rl_qfunction_intuition.png" alt="fig-rl-qfunction-intuition" position="center" 
    caption="Figure 7. Action-value function in the game of pong. Top: states of the agent. Bottom: estimate of the return from this state for each action via action-value function. Taken from [2]" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

## 3.2 RL solution methods

There are various methods that we can use to solve this problem. The figure below (from [3])
shows a taxonomy of the available approaches and methods within each approach. We will
be following the Value-based approach, in which we will try to obtain the optimal
action-value function \\( Q^{\star} \\).

{{<figure src="/imgs/img_rl_algs_taxonomy.png" alt="fig-rl-algs-taxonomy" position="center" 
    caption="Figure 8. A non-exhaustive taxonomy of algorithms in modern RL. Taken from [3]" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

## 3.3 Tabular Q-learning

The method we will use is called Q-learning, which is a model-free method that
recovers \\( Q^{\star} \\) from experiences using the following update rule:

$$
Q(s,a) := \overbrace{Q(s,a)}^{\text{Current estimate}} + \alpha ( \overbrace{r + \gamma \max_{a'} Q(s',a')}^{\text{"Better" estimate}} - Q(s,a) )
$$

This update rule is used in the **tabular case**, which is used when dealing discrete state
and action spaces. These cases allow to easily represent the action-value function in
a table (numpy array or dictionary), and update each entry of this table separately.

For example, consider a simple MDP with \\( \mathbb{S}=0,1\\) and \\( \mathbb{A}=0,1\\). 
The action-value function could be represented with the following table.

State (s)   | Action (a)    | Q-value Q(s,a)
------------|---------------|---------------
0  | 0 | Q(0,0)
0  | 1 | Q(0,1)
1  | 0 | Q(1,0)
1  | 1 | Q(1,1)

In python we could just use :

```Python
# define a Q-table initialized with zeros (using numpy)
import numpy as np
Q = np.zeros( (nStates, nActions), dtype = np.float32 )

# define a Q-table initialized with zeros (using dictionaries)
from collections import defaultdict
Q = defaultdict( lambda : np.zeros( nActions ) )
```

The Q-learning algorithm for the tabular case is shown below, and it basically 
consists of updating the estimate of the q-value \\( Q(s,a) \\) for the state-action 
pair \\( (s,a) \\) from another estimate of the true q-value of the optimal policy given by 
\\( r + \gamma \max_{a'} Q(s',a') \\) called the **TD-Target**.

> **Q-learning (off-policy TD control** for estimating \\( \pi \approx \pi^{\star}\\)
> * Algorithm parameters: step size \\( \alpha \in [0,1] \\), small \\( \epsilon \gt 0 \\)
> * Initialize q-table \\( Q(s,a) \\) for all \\( s \in \mathbb{S}, a \in \mathbb{A} \\)
>
> * For each episode:
>     * Sample initial state \\( s_{0} \\) from the starting distribution.
>     * For each step \\( t \\) in the episode :
>         * Select \\( a_{t} \\) from \\( s_{t} \\) using e-greedy from \\( Q \\)
>         * Execute action \\( a_{t} \\) in the environment, and receive reward \\( r_{t+1} \\) and next state \\( s_{t+1} \\)
>         * Update entry in q-table corresponding to \\( (s,a) \\):
>
> $$
> Q(s,a) := Q(s,a) + \alpha ( r + \gamma \max_{a'} Q(s',a') - Q(s,a) )
> $$

In python we would have the following :

```python

def qlearning( env, Q, eps, alpha, gamma, numEpisodes, maxStepsPerEpisode ) :
  """Run q-learning to estimate the optimal Q* for the given environment
  
  Args:
    env                 : environment to be solved
    Q                   : action value function represented as a table
    eps                 : epsilon value for e-greedy heuristic (exploration)
    alpha               : learning rate
    gamma               : discount factor
    numEpisodes         : number of episodes to run the algorithm
    maxStepsPerEpisode  : maximum number of steps in an episode
  """

  for iepisode in range( numEpisodes ) :
    # sample initial state from the starting distribution (given by env. impl.)
    _s = env.reset()

    for istep in range( maxStepsPerEpisode ) :
      # select action using e-greedy policy
      _a = np.random.randint( env.nA ) if np.random.random() < eps else np.argmax( Q[_s] )
      # execute action in the environment and receive reward and next state
      _snext, _r, _finished, _ = env.step( _a )
      # compute target to update
      if _finished :
        _tdTarget = _r
      else :
        _tdTarget = _r + gamma * np.max( Q[_snext] )
      # update entry for (_s,_a) using q-learning update rule
      Q[_s][_a] = Q[_s][_a] + alpha * ( _tdTarget - Q[_s][_a] )

      # cache info for next step
      _s = _snext
```

For further information about Q-learning you can check resources from [4,5,6]

## 3.4 Q-learning with function approximation

The tabular case provides a nice way to solve our problem, but at the cost of storing
a big table with one entry per possible \\( (s,a) \\) pair. This is not scalable to
larger state spaces, which is the case for continous spaces. One approach would be
to discretize the state space into bins for various \\( (s,a) \\), treat each bin as
an entry for the table and then apply solve as in the tabular case. However this is not
practical for various reasons :

* As the discretization gets more precise we end up with more bins and our table
  explodes in size. This is an exponential explosion due to the **curse of dimensionality**.
* Each possible \\( (s,a) \\) is stored separately and updated separately, which
  doesn't take into account the fact that nearby \\( (s,a) \\) pairs should have 
  similar \\( Q(s,a) \\) values. This means we are not generalizing our knowledge of
  one pair to nearby pairs.



## References

* [1] Sutton, Richard & Barto, Andrew. [*Reinforcement Learning: An introduction.*](http://incompleteideas.net/book/RLbook2018.pdf)
* [2] Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David, et. al.. [*Human-level control through deep-reinforcement learning*](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [3] Achiam, Josh. [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
* [4] Simonini, Thomas. [*A Free course in Deep Reinforcement Learning from beginner to expert*](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
* [5] [*Stanford RL course by Emma Brunskill*](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
* [6] [*UCL RL course, by David Silver*](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)