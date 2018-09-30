# CADRL
Implementation of paper [Decentralized Non-communicating Multiagent
Collision Avoidance with Deep Reinforcement Learning](https://arxiv.org/abs/1609.07845)
by Yu Fan Chen, Miao Liu, Michael Everett and Jonathan P. How

This library is no longer maintained. CADRL and GA3C-CADRL is also implemented in our newest library [CrowdNav](https://github.com/vita-epfl/CrowdNav),
which is easier to use and can be better used for benchmarking RL navigation algorithms.

## Usage
Training with specified configuration on CPU:
```
python train.py --config=configs/model.config
```

Training with specified configuration on GPU:
```
python train.py --config=configs/model.config --gpu
```

Visualize the trained agent:
```
python visualize.py --output_dir={OUTPUT_DIR}
```


## Implementation details
* All the training data for imitation learning/initialization of the model
is generated with [RVO2](git@github.com:vita-epfl/RVO2.git). Both evaluation
and test are performed on the crossing scenarios.


* The kinematics can be toggled in the env.config, which gives
a hard rotation constraint.