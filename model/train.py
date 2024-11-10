import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from environment.custom_classes import CustomObservationFunction, CustomRewardFunction

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from environment.env import SumoEnvironment


if __name__ == "__main__":
    min_green = 5
    max_green = 60
    epoches = 3000  # no of files
    seconds_per_episode = 3000
    steps = seconds_per_episode // (max_green - 10)
    total_timesteps = epoches * steps

    reward = CustomRewardFunction(
        max_green_duration=max_green, min_green_duration=min_green
    )
    env = SumoEnvironment(
        net_file="./2way-single-intersection/single-intersection.net.xml",
        route_file="./2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/PPO",
        single_agent=True,
        max_steps=steps,
        max_green=max_green,
        min_green=min_green,
        reward_fn=reward,
        observation_class=CustomObservationFunction,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=256,  # Number of steps to rusn for each environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs to optimize the surrogate loss
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2,  # Clipping parameter, it can be a function
        ent_coef=0.01,  # Entropy coefficient for the loss calculation
        vf_coef=0.5,  # Value function coefficient for the loss calculation
        max_grad_norm=0.5,  # The maximum value for the gradient clipping
        verbose=0,  # Verbosity level: 0 no output, 1 info, 2 debug
    )
    # steps to train on
    model.learn(
        total_timesteps=total_timesteps,
    )  # Increased total timesteps
    env.reset()
    model.save("ppo_sumo_model")
