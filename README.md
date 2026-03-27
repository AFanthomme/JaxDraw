# In-Context Reinforcement Learning for hidden-rule stroke-reproduction 

This project is still under active development, see roadmap and first results below.

<p float="left">
  <img src="illustrations/onpolicy_network_agent.gif" width="49%" />
  <img src="illustrations/offpolicy_agent_random_teacher_oracle.gif" width="49%" />
</p>


## Key project features:
* Visual observation based RL environment.
* End-to-end hardware acceleration using Jax.
* Discrete timesteps, continuous actions.
* Composable rules : "draw each line left-to-right, starting with lines at the top", etc...
* Reward delivered upon completing a line while repecting the (hidden) rules.
* Stable context: rule maintained between trials in a "block".
* Extensive baselines and ablation studies.
* Fast to simulate, easy for humans, hard for machines.

**In-Context Learning :** rewards from one trial inform the agent's world model on the hidden rule for the block. 

Given the task's complexity, many state-of-the-art methods can be leveraged:
* Self-supervised, action-conditioned World Model for planning (JEPA).
* Reward-aware SSL for cross-trial credit assignment (Decision Transformer).
* Long-Horizon In-Context learning using temporal aggregation (Q-Chunking).
* Expressive Transformer architectures to represent multimodal stochastic policies (TRM).


# v0.2: Policy representability and Behavior Cloning
Before introducing PPO, we validate our architectures by ensuring policies are learnable via Behavior Cloning.\
We begin with a "rule-less" environment, where all drawing orders are accepted.\
The cloned policy is a noisy (at inverse of the resolution) oracle, always drawing the closest line.


For interpretability purposes, we can split the problem into two parts:
1) Perception: Can the line coordinates be extracted from the observation ? 
2) Decision: Can the action be extracted from the line coordinates ?


Both parts are readily solved by standard networks (DeTr, Self-attention respectively).\
End-to-end observation to action cloning is also possible, with the CNN doing the heavy lifting.\
We expect the decoder Transformer to become more limiting as task complexity increases.


Below, we present performance of trained networks compared to increasingly perturbed versions of the oracle:

<p align="center">
  <img src="illustrations/cumulated_rewards.png" width="65%" />
</p>

While some covariate shift can be observed, trained networks manage to solve the task at very satisfying levels.\
Qualitative analysis suggests gap to oracle mostly driven by ambiguous cases with partially overlapping lines.

# Roadmap 

While exact next steps will be strongly influenced by intermediate results, the broad roadmap is as follows:
1) Introduce multiple rules, while maintaining full observability and oracle supervision.
2) Train using On-Policy RL (PPO) (starting from a partially-trained BC network at first)
3) Introduce partial observability and trial-block design, repeat BC / RL steps. 
4) Increase temporal resolution by 1-2 orders of magnitude to force action chunking.

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

