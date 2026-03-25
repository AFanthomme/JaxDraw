# In-Context Reinforcement Learning for hidden-rule stroke-reproduction 

This project is still under active development, see roadmap and first results below.

<p float="left">
  <img src="illustrations/onpolicy_noisy_oracle.gif" width="49%" />
  <img src="illustrations/offpolicy_agent_random_teacher_oracle.gif" width="49%" />
</p>


## Key project features:
* Visual observation based RL environment.
* Discrete (coarse) timesteps, continuous space for actions.
* End-to-end hardware acceleration using Jax.
* Composable rules : "draw each line left-to-right, starting with lines at the top".
* Reward delivered upon completing a line while repecting the (hidden) rules.
* Stable context: rule maintained between trials in a "block".
* Extensive baselines and ablation studies.

**In-Context Learning :** rewards from one trial inform the agent's world model on the hidden rule for the block. 

Studying agents that can solve this environment, as well as the failure modes that appear after ablations, can provide information on the following questions:
* How can an agent integrate information across trials to build a "world model" ?
* How can an agent leverage an internal world model for efficient In-Context learning ?
* How can the agent perform credit assignment between trials in a block ?
* What would happen if we increase time resolution (and therefore the context length) by a few orders of magnitude ?
* What are the tradeoffs between recurrence and attention mechanism in this context ?

# Running the project
This project is intended to be ran via Docker on a CUDA-enabled machine, see the [Nvidia official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for help setting it up.

TPU acceleration potential will be assessed in the near future, but is not supported at the moment. 


We recommend beginning by building the image and running all tests on the target machine using:
```
make build-gpu
bash run_gpu.sh pytest tests
```
ensuring that the correct hardware is detected and that all tests are passed.

Then, experiments can be launched using:
```
bash run_gpu.sh python experiments/XX.py
```
which will bind the results folder and use it to store the outputs of the experiment.

# Roadmap 

We expect the complete task to be quite challenging, and will therefore benefit from an an iterative development process in which the environment is progressively made more difficult as the agent's design and training procedure is adapted as needed to restore preformance. 

The broad roadmap is as follows:
1) Perform Behavior Cloning of an Oracle policy on a single-rule (hence, fully observable) environment.
2) Introduce multiple rules, while maintaining full observability and oracle supervision.
3) Learn both earlier variants using On-Policy Reinforcement Learning instead of Behavior Cloning.
4) Switch to block-design, perform Behavior Cloning with "Decision Transformer"-like setup.
5) Perform full RL learning from scratch on the partially observable block-trial task.




# v0.1: First baselines
* 4 strokes per canvas
* No hidden rules, any well-drawn stroke is rewarded (+1), bad strokes penalized (-0.1) 
* Oracle policy (reads env internals) : move to closest line-end, then push the pen and move to the other.
* Noisy versions of the oracle add random gaussian noise on the movement 
* Measure cumulative reward within one trial (over 64*512 trials) 

<p align="center">
  <img src="illustrations/cumulated_rewards.png" width="65%" />
</p>

# Next steps:
Scaling laws for Behavior Cloning on the oracle policy

Initial tests using basic CNN + MLP showed very poor performance, so we decoupled image -> lines from lines -> action
Both can be trained separately to environment precision, and some sanity checks and ablation studies will need to be performed before moving on to end-to-end training. 

