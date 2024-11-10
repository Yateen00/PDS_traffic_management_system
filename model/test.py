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

signal = 0


def dummy_action_function(env):
    """Dummy action function that takes uses fixed signal."""
    global signal
    action = [signal % 4, 55]
    signal += 1
    return action


def smart_action_function(env):
    """selects lane with most cars, for cars*1.8 seconds"""
    counts = env.traffic_signals[env.ts_ids[0]].get_lanes_count()
    counts = [counts[i] + counts[i + 1] for i in range(0, len(counts), 2)]
    max_count = max(counts)
    max_index = counts.index(max_count)
    time = max_count * 1.8 if max_count * 1.8 <= 60 else 60
    time = time if time >= 6 else 6
    print(max_index, int(time - 5))
    return [max_index, int(time - 5)]


def evaluate_model(model, env, num_steps):
    """Evaluate the trained model."""
    obs, info = env.reset()  # Gym API
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation from the tuple
    total_reward = 0
    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)  # Adjusted unpacking
        total_reward += reward
        if done or truncated:
            obs, info = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    return total_reward


def evaluate_dummy(env, num_steps):
    """Evaluate the dummy action function."""
    obs, info = env.reset()  # Gym API
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation from the tuple
    total_reward = 0
    for _ in range(num_steps):
        action = dummy_action_function(env)
        obs, reward, done, truncated, info = env.step(action)  # Adjusted unpacking
        total_reward += reward
        if done or truncated:
            obs, info = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    return total_reward


def evaluate_smart(env, num_steps):
    """Evaluate the smart action function."""
    obs, info = env.reset()  # Gym API
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation from the tuple
    total_reward = 0
    for _ in range(num_steps):
        action = smart_action_function(env)
        obs, reward, done, truncated, info = env.step(action)  # Adjusted unpacking
        total_reward += reward
        if done or truncated:
            obs, info = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    env.reset()
    return total_reward


if __name__ == "__main__":
    min_green = 5
    max_green = 60
    seconds_per_episode = 3000
    steps = seconds_per_episode // (max_green - 10)

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
        sumo_seed=42,
    )

    # Load the trained model
    model = PPO.load("ppo_sumo_model")

    # Evaluate the trained model
    trained_model_reward = evaluate_model(model, env, steps)
    print(f"Total reward for trained model: {trained_model_reward}")

    # Evaluate the dummy action function
    dummy_model_reward = evaluate_dummy(env, steps)
    print(f"Total reward for dummy action function: {dummy_model_reward}")
    smart_model_reward = evaluate_smart(env, steps)
    print(f"Total reward for smart action function: {smart_model_reward}")

# mode,dummy,smart
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/model/ -start 1 -end 1
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/dummy/ -start 2 -end 2
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/smart/ -start 3 -end 3
