import sys
sys.path.append("../") 

import argparse
import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
import torch
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods run one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()
    # print(state.shape) # => 4 : See websites for more info https://gymnasium.farama.org/environments/classic_control/cart_pole/
    step = 0
    while True:
        # print(state)
        # explore
        action_id = agent.act(state=state, deterministic=deterministic)

        next_state, reward, terminal, info = env.step(action_id)

        # stats.episode_reward = reward

        if do_training:  
            loss = agent.train(state, action_id, next_state, reward, terminal)
        
        # print(reward)
        stats.step(reward, action_id)
        # print("Hi ", stats.episode_reward)
        # print(stats.get_action_usage(0))
        # print(stats.get_action_usage(1))

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1
    # print(stats.episode_reward)
    return stats

def train_online(env, agent, num_episodes, eval_cycle, model_dir="./models_cartpole", tensorboard_dir="./tensorboard", name="dqn_agent.pt"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name = ["episode_reward", "a_0", "a_1"], stats=["episode_reward", "a_0", "a_1"])
    num_eval_episodes = 10
    # training
    for i in range(num_episodes):
        # print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        # stats = run_episode(env, agent, deterministic=False, do_training=True)
        # print(stats.episode_reward)
        # print(stats.get_action_usage(0))
        # print(stats.get_action_usage(1))
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # agent.train

        if i % eval_cycle == 0:
           r = 0
           count = 0
           for j in range(num_eval_episodes):
               stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
               r += stats.episode_reward
               count += 1
           print(f"episode {i}\n episode reward {r / count}")
            #   loss = agent.train()
            #   print(f"eval cycle {eval_cycle}, num_eval_episodes {j}, loss {loss}")

        
        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, name))
   
    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 250   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes
    # num_eval_episodes = 5   # evaluate on 5 episodes
    # eval_cycle = 20         # evaluate every 10 episodes

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    cmdline_parser = argparse.ArgumentParser('Cartpole - Training')
    cmdline_parser.add_argument('-n', '--name',
                                default="",
                                help='name of model',
                                type=str)
    args, unknowns = cmdline_parser.parse_known_args()

    if args.name == "":
        cmdline_parser.error("please input name of model you want to save.")

    env = gym.make("CartPole-v0").unwrapped

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = "cpu"
    print("device ", device)
    state_dim = 4
    num_actions = 2

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
    Q_current = MLP(state_dim=4, action_dim=2)
    Q_target = MLP(state_dim=4, action_dim=2)
    ag = DQNAgent(Q=Q_current, Q_target=Q_target, num_actions=num_actions, device=device)
    train_online(env, agent=ag, num_episodes=num_eval_episodes , eval_cycle=eval_cycle, name=args.name)