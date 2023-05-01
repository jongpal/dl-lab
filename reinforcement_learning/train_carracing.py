# export DISPLAY=:0 

import sys
sys.path.append("../") 

import argparse
import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks import CNN


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        action_id = agent.act(state, deterministic=deterministic, step=step)
        action = id_to_action(action_id)
        # action = your_id_to_action_method(...)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, load_model=False, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard", max_timesteps=1000, skip_frames=0, \
                 name="dqn_agent.ckpt"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
    epoch = 0
    if load_model:
        try:
            epoch = agent.load_checkpoint(os.path.join("./models_carracing", name))
        except FileNotFoundError:
            epoch = 0
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name="carracng-dqn"+name, stats=["episode_reward", "straight", "left", "right", "accel", "brake"])
    
    num_eval_episodes = 10

    for i in range(epoch, num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        if i < 20:
            stats = run_episode(env, agent, max_timesteps=500, deterministic=False, do_training=True, skip_frames=skip_frames)
        else:
            stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True, skip_frames=skip_frames)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if i % eval_cycle == 0:
           for j in range(num_eval_episodes):
               stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
               # greedy_reward = stats.episode_reward

        # store model.
        if i % 20 == 0 or (i >= num_episodes - 1):
            # agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt")) 
            # agent.save(os.path.join(model_dir, "dqn_agent.ckpt")) 
            print("saving models . . .")
            agent.save_checkpoint(file_name=os.path.join(model_dir, name), num_episodes=i)

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 100
    eval_cycle = 50

    env = gym.make('CarRacing-v0').unwrapped

    cmdline_parser = argparse.ArgumentParser('Carracing - Training')
    cmdline_parser.add_argument('-n', '--name',
                                default="",
                                help='name of model',
                                type=str)
    args, unknowns = cmdline_parser.parse_known_args()

    if args.name == "":
        cmdline_parser.error("please input name of model you want to save.")

    
    # TODO: Define Q network, target network and DQN agent
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # device = "cpu"
    print(device)
    num_actions = 5
    max_timesteps = 1000
    skip_frames = 1
    Q = CNN(n_classes=num_actions)
    Q_target = CNN(n_classes=num_actions)
    agent = DQNAgent(Q=Q, Q_target=Q_target, num_actions=num_actions, task="car", device=device)

    load_model = True

    
    train_online(env, agent, load_model=load_model, num_episodes=200, skip_frames=skip_frames, history_length=0, model_dir="./models_carracing", max_timesteps=max_timesteps,\
                 name=args.name)

