[//]: # (References)

<!-- IMAGES -->
[gif_project_2_reacher_multi_agent]: imgs/gif_project_2_reacher_multi_agent.gif
[img_reacher_environment_observations]: imgs/img_reacher_environment_observations.png
[img_reacher_environment_actions]: imgs/img_reacher_environment_actions.png
[img_reacher_environment_unity_editor]: imgs/img_reacher_environment_unity_editor.png
[img_tensorboard_sample]: imgs/img_tensorboard_sample.png

[url_report]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/REPORT.md
[url_trainer_script]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/trainer.py
[url_gin_config]: https://github.com/google/gin-config
[url_gin_config_default_file]: https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/configs/ddpg_reacher_multi_default.gin
[url_tensorboardX]: https://github.com/lanpa/tensorboardX

# Project 2: Continuous Control

This project consists of an implementation of an RL-based agent to control various
simulated 2-linked arms and track various given targets. Our agent implementation
is based on the DDPG algorithm from the paper [**Continuous control with deep reinforcement learning**](https://arxiv.org/pdf/1509.02971.pdf)
by Lillicrap, et. al.. Below we show a trained policy that successully controls
each arm enabling it to track the desired goals defined by the green spheres.

![project-2-reacher-agent][gif_project_2_reacher_multi_agent]

For further details about the algorithm, agent implementation and some results please 
refer to the accompanying report ([REPORT.md][url_report]).

## 1. Environment description

The environment chosen for the project is a **modified version of the Reacher Environment** 
from the Unity ML-Agents toolkit. The original version can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher),
and the version we will work with consists of a custom build provided by Udacity 
with the following description:

* A group of 20 agents consisting of 2-link arms with 2 torque-actuated joints. The
  objective of all agents is to reach a goal, defined by a green sphere, and track
  it with the end effector of the arm, defined by a small blue sphere at the end of
  the last link.

* Each agent receives an **observation** consisting of a 33-dimensional vector
  with measurements like relative position and orientations of the links, relative
  position of the goal and its speed, etc..

* Each agent moves its arm around by applying **actions** consisting of 4 torques
  applied to each of the 2 actuated joints (2 torques per joint).

* Each agents get a **reward** of +0.1 each step its end effector is within the limits
  of the goal. The environment is considered solved once the agent gets an average
  reward of +30 over 100 episodes.

* The task is **episodic**, with a maximum of 1000 steps per episode.

### 1.1 Observation-space

Each agent receives an observation consisting of a 33d vector, with features defined
by the following measurements:

* Relative position of the two links (3d vector) and their orientation (4d quaternion).
* Linear velocity (3d-vector) and angular velocity (3d-vector) of each link.
* Relative positions of the end effector (3d-vector) and the goal (3d-vector).
* Speed (scalar) of the moving goal.

These measurements are shown in the figure below, in which we describe the reference
frames at hand and all measurements from the 33d observation. 

![reacher-env-observations][img_reacher_environment_observations]

These measurements can be double-checked with the definition of the agent in the 
ml-agents toolkit, which can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/9b1a39982fd03de8f40f85d61f903e6d972fd2cc/UnitySDK/Assets/ML-Agents/Examples/Reacher/Scripts/ReacherAgent.cs#L32).
Below we show a small snippet from their C# agent implementation.

```Csharp
    public override void CollectObservations()
    {
        AddVectorObs(pendulumA.transform.localPosition);
        AddVectorObs(pendulumA.transform.rotation);
        AddVectorObs(rbA.angularVelocity);
        AddVectorObs(rbA.velocity);

        AddVectorObs(pendulumB.transform.localPosition);
        AddVectorObs(pendulumB.transform.rotation);
        AddVectorObs(rbB.angularVelocity);
        AddVectorObs(rbB.velocity);

        AddVectorObs(goal.transform.localPosition);
        AddVectorObs(hand.transform.localPosition);
        
        AddVectorObs(goalSpeed);
    }
```

Note that although the local position of the goal and hand are used, these depend 
on the parent transform, which could be the world transform or a transform from the 
2-linked arm. To further check the reference frames relationships we can open the 
scene of the environment in the Unity Editor and check how objects are assembled
into a game object, which is shown in the figure below. Notice that all objects are
related to the parent game object, thus we end up with the observations as described
earlier.

![reacher-env-unity-editor][img_reacher_environment_unity_editor]

### 1.2 Actions-space

The actions that the agent can take consist of a 4d vector of torques, defined as
follows:

* Two torques along the x and z axes of the actuated joint of link 1.
* Two torques along the x and z axes of the actuated joint of link 2.

The figure below shows these actions a bit more clearly:

![reacher-env-actions][img_reacher_environment_actions]

Also, we can double check the meaning of these actions from the ml-agents implementation 
in the Reacher agent in C#, which you can find [here](https://github.com/Unity-Technologies/ml-agents/blob/9b1a39982fd03de8f40f85d61f903e6d972fd2cc/UnitySDK/Assets/ML-Agents/Examples/Reacher/Scripts/ReacherAgent.cs#L53). 
Below we show a snippet of the implementation:

```CSharp
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        goalDegree += goalSpeed;
        UpdateGoalPosition();

        var torqueX = Mathf.Clamp(vectorAction[0], -1f, 1f) * 150f;
        var torqueZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * 150f;
        rbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));

        torqueX = Mathf.Clamp(vectorAction[2], -1f, 1f) * 150f;
        torqueZ = Mathf.Clamp(vectorAction[3], -1f, 1f) * 150f;
        rbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));
    }
```

### 1.3 Environment dynamics and rewards

All 20 agents spawn at the same time at their corresponding positions (base positions) 
and the same zero-configuration (joint angles at zero). The goals spawn as well at 
the same time and at slightly different positions around their corresponding 
agent. The goals switch behaviours from moving-targets to static-targets in between 
episodes, and they give a reward of +0.1 per step to its corresponding agent if their 
end effector keeps inside of their bounding sphere. The environment is considered 
solved once the agents can consistently get an average return of +30 over 100 episodes. 
Note that the averaging is made over all agents.

## 2. Setting up this package

In this section we give instructions about how to use the code of this implementation,
which is distributed as a small package that can be easily installed using *pip*.

### 2.1 Custom environment build

The environment provided is a custom build of the ml-agents Reacher environment, 
with the features described in the earlier section. The environment is provided 
as an executable which we can download from the links below according to our platform:

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Keep the .zip file where you downloaded it for now. We will later explain where
to extract its contents when finishing the setup.

### **Note**

The executables provided are compatible only with an older version of the ML-Agents
toolkit (version 0.4.0). The setup below will take care of this and install the
appropriate version of the ml-agents API, but keep this in mind if you want to use 
an executable on your own (either a custom build or the original ml-agents reacher 
environment).

### 2.2 Dependencies

As already mentioned, the environment provided is an instance of an ml-agents
environment and as such it requires the **appropriate** ml-agents python API to be
installed in order to interact with it. There are also some other required dependencies, 
so to facilitate installation all this setup is done by installing the **ccontrol**
package from the accompanying code, which we will discuss later. Also, there is no need
to download the Unity Editor for this project (unless you want to do a custom build
for your platform) because the builds for the appropriate platforms are already
provided for us.

### 2.3 Downloading the accompanying code and finishing setup

* Grab the accompanying code from this repo and navigate to the folder of project-2.

```bash
# clone the repo
git clone https://github.com/wpumacay/DeeprlND-projects
# go to the project2-continuous-control folder
cd DeeprlND-projects/project2-continuous-control
```

* (Suggested) Create an environment using a virtual environment manager like 
  [pipenv](https://docs.pipenv.org/en/latest/) or [conda](https://conda.io/en/latest/).

```bash
# create a virtual env. using conda
conda create -n deeprl_ccontrol python=3.6
# activate the environment
source activate deeprl_ccontrol
```

* Install [pytorch](https://pytorch.org/get-started/locally/#start-locally).

```bash
# Option 1: install pytorch (along with cuda). No torchvision, as we do not require it for this project.
conda install pytorch cudatoolkit=9.0 -c pytorch
# Option 2: install using pip. No torchvision, as we do not require it for this project.
pip install torch
```

* "Install" the **ccontrol** package using pip and the provided 
  [setup.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project2-continuous-control/setup.py) 
  file (make sure you are in the same folder where this file is located).

```bash
# install the ccontrol package and its dependencies using pip (dev mode to allow changes)
pip install -e .
```

* Uncompress the executable downloaded previously into the executables folder in
  the repository

```bash
cd executables/
# copy the executable into the executables folder in the repository
cp {PATH_TO_ZIPPED_EXECUTABLE}/Reacher_Linux.zip ./
# unzip it
unzip Reacher_Linux.zip
```

## 3. Training and testing

To train the agent we provided a trainer script ([trainer.py][url_trainer_script]) 
which you can use in the following way:

```bash
python trainer.py train --config=./configs/GIN-CONFIGURATION-FILE.gin
```

This requests the trainer to run a training session using a specific **gin-config** 
file, located in the *./configs* folder. We make use of the [gin-config][url_gin_config] 
package to enable easy usage of the parameters of the trainer. A default .gin config 
file is shown below (see [ddpg_reacher_multi_default.gin][url_gin_config_default_file]):

```

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

The training results are stored in the **results** folder, and these are further
stored inside a folder created by the trainer whose name is given by the **sessionID**
parameter passed in the gin-config file. The results saved consist of the following :

* The weights of the trained models: both actor and critic model are saved in the 
  folder created for the session and with names **checkpoint_actor.pth** and **checkpoint_critic.pth**
  respectively (e.g. *./results/session_default/checkpoint_actor.pth*
  and *./results/session_default/checkpoint_critic.pth*).

* The training logs are stored in a **tensorboard** file (using [tensorboardX][url_tensorboardX]
  as library for interoperatibility with tensorboard) located inside a folder with
  name *tensorboard_summary_SESSION_ID*, e.g. *./results/session_default/tensorboard_summary_session_default/events.out.tfevents.1563313949.labpc*

To test the trained agent just run the trainer script in test mode as follows:

```bash
python trainer.py test --config=./configs/ddpg_reacher_multi_default.gin
```

To run the pretrained agent provided as part of the project submission just run the 
following:

```bash
python trainer.py test --config=./configs/ddpg_reacher_multi_submission.gin
```

Finally, to check the training logs using tensorboard just invoque tensorboard pointing
to the folder where the training logs are:

```bash
# invoque tensorboard there
tensorboard --logdir=./results/SOME_SESSION_NAME/tensorboard_summary_SOME_SESSION_NAME
# or, invoque it with a specific port, in case you have multiple tensorboard sessions running
tensorboard --logdir=./results/SOME_SESSION_NAME/tensorboard_summary_SOME_SESSION_NAME --port=SOME_PORT
```

For example, to check the training logs of the agent trained for the project submission, 
just do the following:

```bash
tensorboard --logdir=./results/session_submission/tensorboard_summary_session_submission/
```

## References

* [*Continuous control through deep reinforcement learning* paper by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf)
* [*Deterministic Policy Gradients Algorithms* paper by Silver et. al.](http://proceedings.mlr.press/v32/silver14.pdf)
* [DDPG implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
* [Post on *Deep Deterministic Policy Gradients* from OpenAI's **Spinning Up in RL**](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [Post on *Policy Gradient Algorithms* by **Lilian Weng**](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
* [*Lecture 8: Advanced Q-Learning Algorithms* from Berkeley's cs294 DeepRL course](https://youtu.be/hP1UHU_1xEQ?t=4365)