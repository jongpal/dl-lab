from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import time


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    # if k == key.UP:    a[1] = +0.1
    if k == key.UP:    a[1] = +0.1
    if k == key.DOWN:  a[2] = +0.1

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    # data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    # data_file = os.path.join(datasets_dir, 'data3.pkl.gzip')
    data_file = os.path.join(datasets_dir, 'alldata.pkl.gzip')
    # try:
    #     with gzip.open(data_file, "rb") as f_r:
    #         ori_data = pickle.load(f_r)
    # except EOFError:
    #     ori_data = {}
        
    # f_r = gzip.open(data_file,'rb')
    # print(ori_data)

    # comb_data = {**data, **ori_data}
    # with gzip.open(data_file, "wb") as f:
    #     pickle.dump(comb_data, f)

    # f = gzip.open(data_file,'ab')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    # if not os.path.exists('./data'):
    #     os.mkdir('./data')
    # data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    # data_file = os.path.join('./data', 'data2.pkl.gzip')
    # try:
    #     with gzip.open(data_file, "rb") as f:
    #         ori_data = pickle.load(f)
    # except EOFError:
    #     ori_data = {}

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }
    env = gym.make('CarRacing-v0').unwrapped
    # env = gym.make('CarRacing-v2').unwrapped

    env.reset()
    print(env)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0.0, 0.0, 0.0]).astype('float32')
    
    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        num_episode = 0
        while True:

            time.sleep(0.05)
            # for first 40 episodes, always accelerate
            if num_episode < 40:
                
                next_state, r, done, info = env.step(np.array([0.0, 0.1, 0.0]))
                episode_reward += r
                # samples["state"].append(state)            # state has shape (96, 96, 3)
                # samples["action"].append(np.array([0.0, 0.2, 0.0]))     # action has shape (1, 3)
                # samples["next_state"].append(next_state)
                # samples["reward"].append(r)
                # samples["terminal"].append(done)
            else:
                # print(a)
                next_state, r, done, info = env.step(a)
                episode_reward += r
                samples["state"].append(state)            # state has shape (96, 96, 3)
                samples["action"].append(np.array(a))     # action has shape (1, 3)
                samples["next_state"].append(next_state)
                samples["reward"].append(r)
                samples["terminal"].append(done)
                
            state = next_state
            steps += 1
            num_episode += 1

            if steps % 1000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if args.collect_data and steps % 5000 == 0:
                print('... saving data')
                store_data(samples, "./data")
                save_results(episode_rewards, "./results")

            env.render()
            if done: 
                break
        
        episode_rewards.append(episode_reward)

    env.close()

    

   
