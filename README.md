# In-Context Reinforcement Learning for hidden-rule stroke-reproduction 

This project is still under active development, see roadmap and first results below.

<p float="left">
  <img src="illustrations/onpolicy_network_agent.gif" width="49%" />
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




# v0.2: Policy representability and Behavior Cloning
Before learning through Reinforcement, we want to ensure policies are representable.

First tests showed very poor results using basic CNN + MLP networks.

We investigated this failure by splitting the problem in two parts:
1) Decision: Can the action be extracted from the line coordinates ?
2) Perception: Can the line coordinates be extracted from the observation ? 

In both cases, attention seemed very natural due to the unordered nature of the set of lines.

Perception was heavily influenced by DeTr and relied on Hungarian matching, while Decision was implemented using standard self-attention and was significantly easier to train due to the lower memory / compute requirements compared to image analysis.

Both parts being able to reach "environment precision" (here defined as the inverse of the 128 pixels resolution) when trained separately, we combined them into a single image -> action network which was trained end-to-end to reproduce the oracle policy:

<p align="center">
  <img src="illustrations/cumulated_rewards.png" width="65%" />
</p>

Note that the reward has been modified to no-longer be binary, but instead ramping up from 0 to 1 as the "drawn line" endpoints get closer to the "target line" endpoints, which should help with finer convergence once we introduce RL training.

Now that we obtained this first sign-of-life, immediate next steps are:
* Ablation studies: quantifying the contribution of the attention mechanism in the decoder.
* RL fine-tuning: starting from a partially-trained BC network, can PPO reach a valid solution?

# Roadmap 

While exact next steps will be strongly influenced by intermediate results, the broad roadmap is as follows:
1) Introduce multiple rules, while maintaining full observability and oracle supervision.
2) Learn both earlier variants using On-Policy Reinforcement Learning instead of Behavior Cloning.
3) Switch to block-design, perform Behavior Cloning with "Decision Transformer"-like setup.
4) Perform full RL learning from scratch on the partially observable block-trial task.

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

