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
[img_reulsts_submission_all_runs_tensorflow_std]: https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs_tensorflow_std.png

<!-- URLS -->
[url_project_1_post]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-navigation/
[url_project_1_post_part_1]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part1/
[url_project_1_post_part_2]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part2/
[url_project_1_post_part_3]: https://wpumacay.github.io/research_blog/posts/deeprlnd-project1-part3/

# Using DQN to solve the Banana environment from ML-Agents

@Modifying intro from [post][url_project_1_post]

## 1. Agent description

@Modifying agent description from [post][url_project_1_post]

### 1.1 Algorithm (DQN)


### 1.2 Model architecture


### 1.3 Hyperparameters

## 2. Implementation

@Modifying implementation description from [post][url_project_1_post]

### 2.1 Agent


### 2.2 Model


### 2.3 Replay buffer


### 2.4 Trainer


## 3. Results

In this section we show the results of our submission, which were obtained through
various runs and with various seeds. The results are presented in the form of *time 
series plots* each over a single run, and *standard deviation* plots over a set of 
similar runs. We will also show the results of one of three experiments we made with 
different configurations. This experiment did not include the improvements (DDQN, PER), 
which are going to be explained in a later section. The results can be found also
in the [results_analysis.ipynb](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/results_analysis.ipynb).
However, due to the size of the training sessions, we decided not to upload most
of the results (apart from a single run for the submission) due to the file size
of the combined training sessions. Still, to reproduce the same experiments just
run the provided bash scripts:

* **training_submission.sh** : to run the base experiments for the submissions
* **training_tests_1.sh** : to run the first experiment, related to exploration tests
* **training_tests_2.sh** : to run the second experiment, related to how much the improvements (DDQN and PER) to DQN help.
* **training_tests_3.sh** : to run the third experiment, to check if our implementations of DDQN and PER actually helps in some setups with little exploration.


### 3.1 Submission results

* **Single runs**: We choose one of the results of the various runs, plotted as time
  series plots. The x-axis represents episodes (during training) and the y-axis represents
  the score obtained at that episode. Also, the noisy blue plot is the actual score per episode,
  whereas the red curve is a smoothed (running average) curve from the previous one with a
  window of size 100. Below we show one random run from the episode (not cherry-picked) 
  for both our pytorch and tensorflow implementations. As we can see, the agent 
  successfully solves the environment at around episode 900. Note that this submission 
  configuration is not the best configuration (hyperparameters tuned to squeeze the 
  most score out of the environment), but a configuration we considered appropriate 
  (moderate exploration). We found that for more aggressive exploration schedules 
  (fast decay over around 300 episodes) and a moderate sized replay buffer the task 
  could be solved in around 300 steps (see experiment 1 later).

![results-submission-single-pytorch][img_results_submission_single_pytorch]

![results-submission-single-tensorflow][img_results_submission_single_tensorflow]

* **All runs**: Below we shows graphs of all runs in a single plot for three different
  random seeds. We again present one graph for each backend (pytorch and tensorflow).
  We recorded 5 training runs per seed. Recall again that all runs correspond to the
  same set of hyperparameters given by the **config_submission.json** file.
  <!--The results are pretty uniform along random seeds. Of course, this might be caused
  by the nature of the algorithm itself. I was expecting some variability, as mentioned
  in [this](https://youtu.be/Vh4H0gOwdIg?t=1133) lecture on reproducibility. Perhaps
  we don't find that much variability because of the type of methods we are using,
  namely Q-learning which is off-policy. Most of the algorithms studied in the
  lecture were on-policy and based on policy gradients, which depend on the policy
  for the distribution of data they see. This might cause the effect of exploring
  and finding completely different regions due to various variabilities (different
  seeds, different implementations, etc.). Perhaps this might be caused as well-->

![results-submission-all-the-runs][img_results_submission_all_runs]

* **Std-plots**: Finally, we show the previous plots in the form of std-plots,
  which try to give a sense of the deviation of the various runs. We collected
  5 runs per seed, and each std-plot is given for a specific seed. Recall again 
  that all runs correspond to the same set of hyperparameters given by the 
  **config_submission.json** file.

![results-submission-all-the-runs-std-pytorch][img_reulsts_submission_all_runs_pytorch_std]

![results-submission-all-the-runs-std-tensorflow][img_reulsts_submission_all_runs_tensorflow_std]

### 3.2 Experiments

In this section we show some preleminary results obtained with the improvements.
These results come from the following two extra experiments, which we made in order 
to evaluate if PER and DDQN helped during training:

* **Experiment 1** : Test exploration schedules, comparing the use of moderate exploration v.s.
                     the use of little exploration.
* **Experiment 2** : Test vanilla DQN against each of the improvements (DDQN only, PER only and
                     DDQN + PER).
* **Experiment 3** : Test if DDQN + PER help in situations with too little exploration.
                     Our hypothesis was that too little exploration might run into unstable
                     learning, and DDQN + PER could help stabilize this process (or even
                     help solve it if the agent couldn't solve the task using only the baseline).

**Note**: the hyperparameters related to PER were not exposed through the .json file
(sorry,I will fix it in later tests), but instead hard-coded in the priority-buffer
implementation (see the [prioritybuffer.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/dqn/utils/prioritybuffer.py) 
file). These parameters were set to the following default values:

    eps   : 0.01    # small value added to the absolute value of the bellman error
    alpha : 0.6     # power to raise the priority in order to further control it
    beta  : 0.4     # importance sampling annealed factor
    dbeta : 0.00001 # linear increment per sample added to the "beta" parameter

**Spoiler alert**: we did not find much improvement in the task at hand by using PER and
DDQN. However, these results are preliminary as we did not tune the hyperparameters
of PER (nor the other hyperparameters), and we still haven't made test cases for all
details of the implementations of the algorithm. We tested each separate component
of the improvements (extra data structures, consistency with other sources' implementations,
etc.), but still we could have miss some detail in the implementation. Also, perhaps
the structure of the task at hand is not too complicated for our vanilla DQN to require
these improvements. We plan to make further tests and experiments in this and more
complicated environments (visual banana environment, atari, etc.) to see if our improvements
actually work and help during training.

### Experiment 1: tweaking exploration

In this experiment we tried to double-check if decreasing the amount of exploration
would allow the agent to solve the task in fewer episodes. Because the amount of exploration
is small, the agent would be forced to start trusting more its own estimates early on
and, if the setup is right (big enough replay buffer, etc.), this might have the
effect of making the agent solve the task quickly. The configurations used are the
following:

* [**config_agent_1_1.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_1_1.json) : 
  This configuration has a moderate exploration schedule. We decay epsilon
  from 1.0 to 0.01 using a multiplicative decay factor of 0.9975 applied per episode.
  This schedule makes epsilon decay from 1.0 to approx. 0.1 over 1000 episodes,
  and to the final value of 0.01 over approx. 1600 episodes.

* [**config_agent_1_2.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_1_2.json) : 
  This configuration has a little more aggresive exploration schedule. We decay epsilon
  from 1.0 to 0.01 using a multiplicative decay factor of 0.98 applied per episode.
  This schedule makes epsilon decay from 1.0 to approx. 0.1 over 100 episodes,
  and to the final value of 0.01 over approx. 200 episodes.

Below we show the training results from 5 runs over 2 random seeds using these
configurations. These plots reveal a trend which suggests that with the second
configuration (less exploration) we get to solve the task faster (around 400 episodes) 
than the first configuration (around 900 episodes).

![results-exp-1-all-runs][img_results_experiment_1_all_runs]

![results-exp-1-all-runs-std][img_results_experiment_1_all_runs_std]

### Experiment 2

This experiment consisted of testing what improvements do DDQN and PER offer to our
vanilla DQN implementation. The configurations for these can be found in the following
files:

* [**config_agent_2_1.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_2_1.json) : Configuration with only DDQN active.
* [**config_agent_2_2.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_2_2.json) : Configuration with only PER active.
* [**config_agent_2_3.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_2_3.json) : Configuration with DDQN + PER active.
* [**config_agent_1_1.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_1_1.json) : The baseline, with the same hyperparameters as the ones above.

We used 5 runs and 2 different seeds for each configuration of the experiment. The
preliminary results (shown below) look very similar for all configurations, and we
can't conclude if there is any improvement using the variations to vanilla DQN.

![results-exp-2-all-runs-std][img_results_experiment_2_all_runs_std]

### Experiment 3

This experiment consisted on testing if DDQN + PER would help in situations where
there is too little exploration. The configurations used for these experiment can 
be found in the following files:

* [**config_agent_3_1.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_3_1.json) : Configuration without DDQN nor PER.
* [**config_agent_3_2.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_3_2.json) : Configuration with DDQN + PER active.

The exploration schedule was such that in just 60 episodes we would have reached 
the fixed 0.01 minimum value for epsilon, and have just 18000 experience 
tuples in a buffer of size of 131072, which consisted just of a approx. 13% of the 
replay buffer size. All other interactions would add only experience tuples taken 
from a practically greedy policy. Also, after just 510 episodes, the replay buffer 
would have just experience tuples sampled from a greedy policy.

We considered that this case would require clever use of the data (PER) to squeeze
the most out of the experiences in the replay buffer, and also to not to overestimate
the q-values, as we would be following a pure greedy policy after just some steps.
The preliminary results don't show much evidence of improvement, except in a run
with a different seed, in which the learning curves were less jumpy, indicating
more stable learning. The results are shown in the image below, which were created
using 5 runs with 2 seeds for both configurations.

![results-exp-3-all-runs-std][img_results_experiment_3_all_runs_std]

## 4. Future Work

Finally, below we mention some of the improvements we consider making in following
updates to this post:

* Make more tests of the improvements implemented, namely DDQN and PER, as we 
  did not see much improvement in the given task. We could run the implementation
  in different environments from gym and atari, make "unit tests" for parts of the
  implementation, and  tune the hyperparameters (PER) to see if there are any improvements.

* Get the visual-based agent working for the banana environment. We had various issues
  with the executable provided, namely memory leaks of the executable that used 
  up all memory in my machine. I first thought they were issues with the replay buffer
  being to big (as the experiences now stored images), but after some debugging that
  consisted on checking the system-monitor and even only testing the bare unity 
  environment in the most simple setting possible, i.e. just instantiating it and running
  a random policy in a very simple script, we still got leaks that did not let us
  fully test our implementation. I was considering sending a PR, but the leaks only
  occur using the old ml-agent API (called unityagents, which is version 0.4.0).
  We made a custom build in the latest version of unity ml-agents and the unity editor,
  and got it working without leaks, but still could not get the agent to learn, which
  might be caused by our DQN agent implementation, our model, or the custom build
  we made.

* Finish the implementation of a generic model for tensorflow and pytorch (see the 
  incomplete implementations in the model_pytorch.py and model_tensorflow.py files), 
  in order to be able to just send a configuration easily via a .json file or similar, 
  and instantiate the requested model without having to write any pytorch nor tensorflow
  specific model by ourselves.

* Implement the remaining improvements from rainbow, namely Dueling DQN, Noisy
  DQN, A3C, Distributional DQN, and try to reproduce a similar ablation test in
  benchmarks like gym and ml-agents.

* Implement recurent versions of DQN and test the implementation in various environments.
