# 3d-map-rl-localization

## Introduction

This repository presents the research conducted as part of the Undergraduate Research Opportunities Programme (UROP) at Imperial College London. The project was carried out during a Research Internship at the [Deepwok Lab](https://deepwok.github.io/) led by Dr. Aaron (Yiren) Zhao in the Department of Electrical and Electronic Engineering.

#### *Project Title: '3D Map-based Deep Reinforcement Learning Localization'*

Supervised by: Dr. Aaron (Yiren) Zhao and Dr. Ilia Shumailov\
Authored By: Anthony Bolton


## Overview
`3d-map-rl-localization` is a Deep Reinforcement Learning (DRL) model designed to navigate through large satellite images to locate specific target locations. The model works with high-resolution 3D bird's-eye view imagery from Google Earth Engine API, capturing depth and spatial relationships in the environment. It uses the Proximal Policy Optimization (PPO) algorithm and operates within a custom environment based on the Gymnasium framework. The system serves as a baseline for a novel approach to navigating and analyzing 3D aerial imagery using reinforcement learning techniques, aiming to estimate coordinates of event occurrences from image inputs shared across digital platforms.

 By leveraging 3D aerial imagery, the model can potentially be applied to various fields such as urban planning, emergency response, and geographical analysis. The use of reinforcement learning allows the system to adapt to different environments and improve its performance over time, making it a versatile tool for spatial analysis and location estimation tasks.

## Technical Implementation
### Envvironment (`SatelliteEnv`)
- Custom Gymnasium (previously OpenAI Gym) environment simulating navigation in large 3D satellite ariel images
- **State Space**: dictionary with two 224x224x3 RGB images (target and current view)
- **Action Space**: continuous 2D movement, Box(2) with range [-1, 1], scaled by action_step (default 0.05)
- **Reward Function**: 
    - *Primary*: sosine similarity between current view and target image
    - *Scaling Bonus*: Increases as the agent approaches the target, providing a smoother reward signal (e.g. success threshold when testing the best baseline model was 0.95 cosine similarity, reward starts to scale once it reaches 0.90)
    - *Success Reward*: +100 when similarity exceeds threshold (default 0.90)
    - *Time Penalty*: -0.01 per step for efficiency
- **step(action)**: updates position, extracts new view, calculates reward
- **reset()**: randomizes agent position, selects new target
- **get_image(coordinate)**: extracts current view of agent, image at same scale as target image is cropped based on the agent's current position as the center from environment image, ensuring consistency in navigation size
- **Agent Movement**: new position calculated as (x + ax * action_step, y + ay * action_step), clipped to environment bounds
- **Episode Termination**: success (similarity > threshold) or max steps reached (default 1000)

### NN Architecture (`MapRLNetwork`)
- Feature extractor inheriting from `BaseFeaturesExtractor`
- **Core**: [MobileNetV2](https://arxiv.org/abs/1801.04381) (pretrained on ImageNet, classifier removed)
    - Processes target and current state images independently
    - Output: 1280-dimensional feature vector per image
- Agent Observation (Feature Combination):
    - Concatenation of target and current state features (2560-dimensional vector)

```
-> Linear(2560, 128) followed by ReLU activation 
-> Linear(128, 64) 
-> Final output: 64-dimensional feature vector
```

### Policy(`MapRLPolicy`)
- Custom `ActorCriticPolicy` using MapRLNetwork as feature extractor
- **Actor (Policy) Network**:
    - Maps 64-dimensional state features to action probabilities
    - Output: Mean and standard deviation for Gaussian action distribution
- **Critic (Value) Network**:
    - Estimates state-value function
    - Maps 64-dimensional features to single value
- Implements PPO algorithm from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for stable training in continuous action space

### Training Process
1. [Optuna](https://optuna.org/) framework for hyperparameter optimization with custom `SimulatedAnnealingSampler`
2. Environment Setup:
    - Training, evaluation and testing manually cropped from environment image at `data`, with a 80/20 split
    - Applied Wrappers: TimeLimit and Monitor for episode management and logging
3. Model initialization: PPO with optimized hyperparameters
4. Training Loop:
    - `model.learn()` called with specified total timesteps
    - Callbacks:
        - `LoggerRewardCallback`: Detailed logging of rewards and episode info
        - `PeriodicEvalCallback`: Evaluation on validation set every 10,000 steps
        - `TestCallback`: Testing on separate test set every 50,000 steps
5. PPO implementation with adaptive learning rate (`get_linear_fn`)
6. Final Evaluation:
    - Performed on train/eval set and test set with number of episodes











## Results
### Environment Image
![Alt text](data/env/env_image_exibition_road.jpg
)
## Experiment 0: Best Baseline Run on 1 Image
### Target Image

<div align="center">
  <img src="data/train_eval/target_003_statue.jpg" alt="Alt text" width="200"/>
</div>

Directory: `data/train_eval/target_003_statue.jpg`

- Rollout Length(n_steps): 1,024
- Optuna Trials: 2
- Optuna Timesteps per Trial: 50,000


- Average Cosine Similarity: 0.9036
- Highest Cosine Similarity: 0.9154
- Highest Episode(2000 steps) Reward: 3738.2632
- Highest Step Reward: 2.4582

### Training
![Alt text](results/train1_100000steps/diagrams/baseline_cos_sim_and_reward.svg)

### [PPO Parameters]((https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters)) From SB3 During Training

![Alt text](results/train1_100000steps/diagrams/baseline_sb3_PPO.png)

### Evaluation

![Alt text](results/train1_100000steps/diagrams/baseline_eval.png)


## Experiment 1: Train and Evaluate on 27 Images, Test on 7 Images



