# In-Context Reinforcement Learning for hidden-rule stroke-reproduction 

This project is still under active development, see roadmap and first results below.

<p float="left">
  <img src="illustrations/onpolicy_network_agent.gif" width="49%" />
  <img src="illustrations/offpolicy_agent_random_teacher_oracle.gif" width="49%" />
</p>


## Key project features:
* Rule-based, Partially Observable RL environment.
* Abstract state for fast prototyping, visual for JEPA. 
* Fast to simulate, easy for humans, hard for machines.
* End-to-end hardware acceleration using Jax, highly scalable.
* Strict typing and containerization for reliability and maintainability.

**In-Context Learning :** rewards from one trial inform the agent's world model on the hidden rule for the next. 

# v0.4: Behavior Cloning for multiple (parametric) rules
We implemented 5 different rules (draw along axis, draw by proximity to reference, draw by stroke length, draw by internal / global angular position), each with increasing / decreasing variants.

Through Behavior Cloning, high-quality policies can be trained using small networks (<10 MB) in a few hours on consumer GPUs:

<p float="left">
  <img src="illustrations/parametric_rule.png" width="90%" />
</p>


# General roadmap 

While exact next steps will be strongly influenced by intermediate results, the broad roadmap is as follows:
1) Fine-tune BC using On-Policy Reinforcement Learning (PPO).
2) Integrate JEPA to help tackle Visual observations.

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

