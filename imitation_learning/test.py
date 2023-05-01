from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import argparse

from agent.bc_agent import BCAgent
from utils import *
from imitation_learning.dataset import stack_histories
import torch


def run_episode(env, agent, histories, rendering=True, max_timesteps=1000, device="cpu"):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    # print(state.shape)
    # fix bug of corrupted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    history_length = 1
    num_episode = 0
    # print("hi")
    count = 0 
    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        # state = 
        state = rgb2gray(state)
        # print(state)
        state = state[None, ..., None]
        state = stack_histories(state, history_length=histories)

        # if agent.normalize != None:
        #     state /= 255.0
        #     state -= 0.5
        # print(state)
        # history length : to add
        # state = np.zeros((*state.squeeze(axis=-1).shape, history_length))

        # for i in range(len(state)):
        #     if i < history_length:
        #         state[i, ..., :i+1] = np.transpose(state[:i+1].squeeze(axis=-1), (1, 2, 0))
        #     else:
        #         state[i] = np.transpose(state[i-history_length+1:i+1].squeeze(axis=-1), (1, 2, 0))
        # print(state.shape)
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        
        if count < 30:
            a = np.array([0.0, 0.1, 0.0])
            count += 1
        else:
            state = torch.from_numpy(state).float()
            a, actions = agent.predict(state.to(device=device))
            a = id_to_action(actions, max_speed=0.4)

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    parser = argparse.ArgumentParser("Imitation Learning")
    # parser.add_argument("--load_model", type=str, default="", help="load the given model, and resume the training")
    # parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # parser.add_argument("--run_name", type=str, default="", help="for tensorboard")
    parser.add_argument("--model_name", type=str, default="", help="for model to use")
    parser.add_argument("--num_histories", type=int, default= 1 , help="histories")
    args = parser.parse_args()

    if args.model_name == "":
        parser.error("input valid model name e.g) agent1.pt")

    # TODO: load agent
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    agent = BCAgent(device=device, histories=args.num_histories)
    agent.load(f"models/{args.model_name}")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, histories=args.num_histories, rendering=rendering, device=device)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
