# In-Context Reinforcement Learning for hidden-rule stroke-reproduction 


<p float="left">
  <img src="illustrations/onpolicy_noisy_oracle.gif" width="49%" />
  <img src="illustrations/offpolicy_agent_random_teacher_oracle.gif" width="49%" />
</p>


The goal of this project is to build a minimalistic, highly efficient environment that enables study of In-Context learning mechanisms.

This environment is inspired by stroke reproduction tasks used in Behavioral Neuroscience, notably the one used in "Neural representation of action symbols in primate frontal cortex, Tian *et al.* (2025)"

The final environment will have the following characteristics:
* Rule-based: Depending on a (hidden) context, the agent has to reproduce strokes either left-to-right, top-to-bottom, *etc...*
* Block design: Trials will be arranged in blocks with shared hidden rule. 

These two features, combined, mean that rewards collected in one trial will update the agent's prior on the hidden rule, allowing and requiring In-Context Learning to perform optimally in a complete block.

We expect this task to be quite challenging, and will need to adress the following points :
* How to efficiently encode actions and environment state?
* How to integrate this information across trajectories to build a "world model" ?
* How to leverage this knowledge into actionable "planning-like" behavior (akin to the Dreamer series of models) ?
* How to perform efficient credit assignment inside trial blocks?

Therefore, we will adopt an iterative development process, of which a tentative roadmap follows:
1) Perform Behavior Cloning of an Oracle policy on a single-rule (hence, fully observable) environment.
2) Introduce multiple rules, while maintaining full observability and oracle supervision.
3) Learn both earlier variants using On-Policy Reinforcement Learning instead of Behavior Cloning.
4) Switch to block-design, perform Behavior Cloning with "Decision Transformer"-like setup.
5) Perform full RL learning from scratch on the partially observable block-trial task.


At every step, our goal will be to reduce the complexity of the setup as much as possible before moving on, and provide extensive baselines to promote critical evaluation of all agent performance.
We will introduce complexities (*eg.* stateful agents, non-deterministic policies, advanced network architectures) only as the previous solution becomes incapable of solving the task (as validated via ablation studies).
We will also emphasize mechanistic interpretability of each of the components in our proposed architectures, guided by the results of ablation studies.


# Running the project
This project relies heavily on Jax, so we recommend using GPU to reproduce results and CPU only has not been tested.
We provide gpu and tpu docker options, for now only GPU option has been tested

We recommend beginning by running all tests on the target machine using:
```
make build-gpu
bash run_gpu.sh pytest tests
```
ensuring that the correct hardware is detected and that all tests are passed.

Then, experiments can be launched as follows, which will bind mount the results folder and use it to store the outputs of the experiment, in that case some 
trajectories rolled out from baseline policies: 
```
bash run_gpu.sh python experiments/01_baseline_policies_analysis.py
```
