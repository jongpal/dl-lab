import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np
import argparse

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    cmdline_parser = argparse.ArgumentParser('Cartpole - Testing')
    cmdline_parser.add_argument('-n', '--name',
                                default="",
                                help='name of model',
                                type=str)
    args, unknowns = cmdline_parser.parse_known_args()

    if args.name == "":
        cmdline_parser.error("please input name of model you want to test.")
    

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # TODO: load DQN agent
    Q_current = MLP(state_dim=4, action_dim=2)
    Q_target = MLP(state_dim=4, action_dim=2)
    agent = DQNAgent(Q=Q_current, Q_target=Q_target, num_actions=num_actions, device=device)
    agent.load(f"models_cartpole/{args.name}")

    # n_test_episodes = 15
    n_test_episodes = 30

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

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

