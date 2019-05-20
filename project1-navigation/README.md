[//]: # (References)

<!-- IMAGES -->
[gif_project_1_banana_agent]: imgs/gif_project_1_banana_agent.gif
[img_banana_env_observations]: https://wpumacay.github.io/research_blog/imgs/img_banana_env_observations.png
[img_banana_env_actions]: https://wpumacay.github.io/research_blog/imgs/img_banana_env_actions.png

<!-- URLS -->
[url_project_1_post]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-navigation/
[url_project_1_post_part_1]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part1/
[url_project_1_post_part_2]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part2/
[url_project_1_post_part_3]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part3/

[url_trainer_script]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/trainer.py

# Project 1: Navigation

This project consists on the implementation of a banana-lover DQN-based agent to collect
as many bananas as possible from a simulated environment. Our implementation is based on
the DQN algorithm from the paper [**Human-level control through deep-reinforcement learning**](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by Mnih et. al.. Below
we show a trained agent successfully collecting yellow bananas and avoiding purple ones.

![project-1-banana-agent][gif_project_1_banana_agent]

There is an accompaying post to this project, which you can find as a blog post [here](url_project_1_post),
which you can also check as three different parts [here](url_project_1_post_part_1), [here](url_project_1_post_part_2),
and [here](url_project_1_post_part_3) (I think I kind of wrote too much for a single post). Most of the
content here and in the report is based on the post, with just some slight modifications.

## 1. Environment description

The environment chosen for the project was a **modified version of the Banana 
Collector Environment** from the Unity ML-Agents toolkit. The original version
can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector),
and our version consists of a custom build provided by Udacity with the following
description:

* A single agent that can move in a planar arena, with **observations** given by
  a set of distance-based sensors and some intrinsic measurements, and **actions** 
  consisting of 4 discrete commands.
* A set of NPCs consisting of bananas of two categories: **yellow bananas**, which give the
  agent a **reward of +1**, and **purple bananas**, which give the agent a **reward of -1**.
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

In the figure elow we show two separate cases that show the ray-perceptions. The first one
to the left shows all rays reaching at least one object (either purple banana, yellow
banana or a wall) and also the 7 sensor readings in array form (see the encodings in the
4 first entries do not contain the *none found* case). The second one to the right
shows all but one ray reaching an object and also the 7 sensor readings in array form (see
the encodings in the 4 first entries do include the *none found* case for the 4th perception).

![banana-env-observations][img_banana_env_observations]

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

![banana-env-actions][img_banana_env_actions]

#### **Note**

These actions are applicable again only for our custom build, as the original
environment from ML-Agents has even more actions, using action tables (newer API).
This newer API accepts in most of the cases a tuple or list for the actions, with
each entry corresponding to a specific action table (a nested set of actions) 
that the agent can take. For example, for the original banana collector environment 
the actions passed should be:

```python
# actions are of the form: [actInTable1, actInTable2, actInTable3, actInTable4]

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

## 2. Setting up this package

In this section we give instructions about how to configure the package provided in this repo for
the DQN project.

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
toolkit (version 0.4.0). The setup below will take care of this, but keep this in mind
if you want to use the executable on your own.

### 2.2 Dependencies

As already mentioned, the environment provided is an instance of an ml-agents
environment and as such it requires the **appropriate** ml-agents python API to be
installed in order to interact with it. There are also some other dependencies, so
to facilitate installation all this setup is done by installing the **navigation**
package from the accompanying code, which we will discuss later. Also, there is no need
to download the Unity Editor for this project (unless you want to do a custom build
for your platform) because the builds for the appropriate platforms are already
provided for us.

### 2.3 Downloading the accompanying code and finishing setup

* Grab the accompanying code from this repo.

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

* Install the navigation package using pip and the provided 
  [setup.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/setup.py) 
  file (make sure you are in the folder where the *setup.py* file is located).

```bash
# install the navigation package and its dependencies using pip (dev mode to allow changes)
pip install -e .
```

* Uncompress the executable downloaded previously into the executables folder in
  the repository

```bash
cd executables/
# copy the executable into the executables folder in the repository
cp {PATH_TO_ZIPPED_EXECUTABLE}/Banana_Linux.zip ./
# unzip it
unzip Banana_Linux.zip
```

* (Update|Optional) If you want to use the tensorflow implementation, you might run into a little problem
  when setting up tensorflow. The issue comes from the *unityagents* pip package, because it requires us
  to install tensorflow 1.7.0, which overwrites the version we installed earlier. This will cause various
  problems even if we want to install again our tensorflow version. The workaround we found was to just install
  the unityagents package for version 0.4.0, which is provided by udacity in its repo, with a slight modification
  that removes the tensorflow 1.7.0 dependency. So, instead of using the installation steps from before, follow
  the steps from below.

```bash
# clone the udacity repo
git clone https://github.com/udacity/deep-reinforcement-learning.git

# go to the python folder of the repo
cd deep-reinforcement-learning/python

# remove the tensorflow dependency from the requirements.txt file with your favourite editor
vim requirements.txt # remove the tensorflow dependency

# install the unityagents package from this folder
pip install -e .

# install the requirements from our package
cd PATH_TO_OUR_PACKAGE
pip install -r requirements.txt

# install the appropriate tensorflow version after that
# either the gpu version
pip install tensorflow-gpu==1.12.0
# or the cpu version (it still can train with cpu in a moderate amount of time)
pip install tensorflow==1.12.0
```

## 3. Training and testing

To train the agent we provided a trainer script ([trainer.py][url_trainer_script]) which you can use in the following way:

```bash
python trainer.py train --sessionId=banana_train_1
```

This requests the trainer to run a training session with id "banana_train_1", name that will be used later for
storing the training results (like weights and scores during training). Once training is finished, the results
will be stored in the folder **./results/banana_train_1**.

To test the trained agent just run the trainer script in test mode as follows:

```bash
python trainer.py test --sessionId=banana_train_1
```

There are also some bash scripts we made to run some experiments (which are discussed in more detail in the report and the
blog post), and these can be runned as follows:

```bash
# run all three experiments
./train.sh

# run experiment 1
./training_tests_1.sh

# run experiment 2
./training_tests_2.sh

# run experiment 3
./training_tests_3.sh

# make runs to get the results of the submission
./training_submission.sh
```

To run a pretrained banana-agent just run the following:

```bash
python trainer.py test --sessionId=banana_submission
```

To run a pretrained lunarlander-agent just run the following:

```bash
python trainer_full.py --sessionId=gym_lunarlander --gym=LunarLander-v2
```

## References

These are some code-references I used while implementing my DQN agent:

* [Original DQN implementation](https://sites.google.com/a/deepmind.com/dqn/)
* [DQN lunar-lander implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution)
* [OpenAI baselines PER implementation](https://github.com/openai/baselines/blob/master/baselines/deepq/)
* [@Jaromiru PER implementation](https://github.com/jaromiru/AI-blog/blob/master/SumTree.py)
* [@MorvanZhou PER implementation](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py)