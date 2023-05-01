from __future__ import print_function

import gym
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np
import os
from datetime import datetime
import json
import argparse

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0
    cmdline_parser = argparse.ArgumentParser('Carracing - Testing')
    cmdline_parser.add_argument('-n', '--name',
                                default="",
                                help='name of model',
                                type=str)
    args, unknowns = cmdline_parser.parse_known_args()

    if args.name == "":
        cmdline_parser.error("please input name of model you want to save.")

    #TODO: Define networks and load agent
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    num_actions = 5
    max_timesteps = 1000
    skip_frames = 1
    Q = CNN(n_classes=num_actions)
    Q_target = CNN(n_classes=num_actions)
    agent = DQNAgent(Q=Q, Q_target=Q_target, num_actions=num_actions, task="car", device=device)
    agent.eval_mode()
    agent.load_checkpoint(os.path.join("./models_carracing", args.name))
    # agent.load(os.path.join("./models_carracing", "dqn_agent.ckpt"))

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

