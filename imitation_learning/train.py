from __future__ import print_function
import argparse
import sys

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from dataset import CarDataset
from torchinfo import summary
import logging

from agent.bc_agent import BCAgent
sys.path.append("../") 
from utils import *
from tensorboard_evaluation import Evaluation
import torch
from data_augmentation import baseline_augment
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    # data_file = os.path.join(datasets_dir, 'data2.pkl.gzip')
    # data_file = os.path.join(datasets_dir, 'alldata.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')
    # print(np.sum(y == -1.0))
    # print(np.sum(y == 1.0))
    # print(np.sum(y == 0.0)) # this data is too much. remove this data? or give penalty?
    # print(np.sum(y == 0.2))

    # print(y[90:110])

    # get also rewards, to approximate Q values? Or just approximate actions, not using Q ? FIrst try latter one, and then
    # compare it to the former one.

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # not mutating the data
    X_train_gray, X_valid_gray = rgb2gray(X_train), rgb2gray(X_valid)
    X_train_gray, X_valid_gray = X_train_gray[..., None], X_valid_gray[..., None]
    assert X_train_gray.shape[-1] == 1, X_valid_gray.shape[-1] == 1

    # discrete action space? (should consider also using continuous action space later)
    y_train_discrete = np.zeros((len(y_train),))
    y_valid_discrete = np.zeros((len(y_valid),))
    for i in range(len(y_train)):
        y_train_discrete[i] = action_to_id(y_train[i])
    for i in range(len(y_valid)):
        y_valid_discrete[i] = action_to_id(y_valid[i])

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    # How to handle boundary cases? : e.g. first image has no last N images => repeat
    # mutate
    X_train_history = np.zeros((*X_train_gray.squeeze().shape, history_length))
    y_train_history = np.zeros((*y_train_discrete.shape, history_length))
    X_valid_history = np.zeros((*X_valid_gray.squeeze().shape, history_length))
    y_valid_history = np.zeros((*y_valid_discrete.shape, history_length))

    for i in range(len(y_train)):
        if i < history_length:
            X_train_history[i, ..., :i+1] = np.transpose(X_train_gray[:i+1].squeeze(axis=-1), (1, 2, 0))
            y_train_history[i] = y_train_discrete[:i+1].reshape(1, -1)
        else:
            X_train_history[i] = np.transpose(X_train_gray[i-history_length+1:i+1].squeeze(axis=-1), (1, 2, 0))
            y_train_history[i] = y_train_discrete[i-history_length+1:i+1].reshape(1, -1)
        # manual version
        # X_train_gray[i] = np.concatenate((X_train_gray[i - history_length], X_train_gray[i - history_length + 1], ...), axis=-1) 

    for i in range(len(y_valid)):
        if i < history_length:
            X_valid_history[i, ..., :i+1] = np.transpose(X_valid_gray[:i+1].squeeze(axis=-1), (1, 2, 0))
            y_valid_history[i] = y_valid_discrete[:i+1].reshape(1, -1)
        else:
            X_valid_history[i] = np.transpose(X_valid_gray[i-history_length+1:i+1].squeeze(axis=-1), (1, 2, 0))
            y_valid_history[i] = y_valid_discrete[i-history_length+1:i+1].reshape(1, -1)
    # Should test : 1.history = 1, 2. when the history is bigger than 1. 

    # print(y_valid_history[:15])
        # y_valid_history[i] = np.transpose(y_valid_discrete[i-4:i+1], (1, 2, 0))
    
    return torch.tensor(X_train_history), torch.tensor(y_train_history), torch.tensor(X_valid_history), torch.tensor(y_valid_history)


def stack_histories(X, y, history_length = 1):
    X_history = torch.zeros((*X.squeeze().shape, history_length))
    y_history = torch.zeros((*y.shape, history_length))
    # print(X_history.shape)
    # print(y_history.shape) # this is fine
    # print(y.shape)
    count = 0
    for i in range(len(y)):
        if i < history_length - 1:
            # print(X[:i+1].shape)
            # print(X[:i+1].squeeze(axis=-1).shape)
            # print(X[:i+1])
            # print(torch.permute(X[:i+1].squeeze(axis=-1), (1, 2, 0)))
            # add logic to duplicate one.
            to_copy = torch.permute(X[:i+1].squeeze(axis=-1), (1, 2, 0)).clone()


            X_history[i, ..., history_length - 1 - i: history_length] = to_copy
            # copy the first frame
            X_history[i, ..., :history_length - 1 - i] = X[0, ..., None].repeat(1, 1, 1, history_length - i - 1)
            # # problem
            # # print(y[:i+1].reshape(1, -1).shape)
            # # print(y_history[i].shape)
            to_copy = y[:i+1].reshape(1, -1).clone()
            y_history[i, history_length - 1 - i:history_length] = to_copy
            y_history[i, :history_length - 1 - i] = y[0].repeat(1, history_length - i - 1)
            if count < 1:
                # print(X[:i+1]) # 1, 96, 96
                # print(to_copy.shape)
                # print(to_copy)
                # print(X_history[i, ..., :i+1].shape)
                # print( X_history[i, ..., history_length - 1 - i: history_length])
                # print(X[0, ..., None].repeat(1, 1, 1, history_length - i - 1))
                # print(X_history[i, ..., :history_length - 1 - i])
                # print(y_history[i, history_length - 1 - i:history_length])
                # print(to_copy)
                # print(y_history[i, :history_length - 1 - i])
                # print(y[0].repeat(1, history_length - i - 1))
                # print(X_history[0])
                count += 1
        else:
            # torch.permute not allocate new one.
            to_copy_X = torch.permute(X[i-history_length+1:i+1].squeeze(axis=-1), (1, 2, 0)).clone()
            X_history[i] = to_copy_X
            to_copy = y[i-history_length+1:i+1].reshape(1, -1).clone()
            y_history[i] = to_copy
            if count == 1:
                print(to_copy_X)
                print(X_history[1])
                count += 1
        # manual version
        # X_train_gray[i] = np.concatenate((X_train_gray[i - history_length], X_train_gray[i - history_length + 1], ...), axis=-1) 
    # Should test : 1.history = 1, 2. when the history is bigger than 1. 

    # print(y_valid_history[:15])
        # y_valid_history[i] = np.transpose(y_valid_discrete[i-4:i+1], (1, 2, 0))
    
    # return torch.tensor(X_history), torch.tensor(y_history)
    return X_history, y_history

# def eval(agent, val_loader):



def train_model(train_loader, val_loader, histories, n_minibatches, batch_size, lr,  model_dir="./models", tensorboard_dir="./tensorboard", load_model="", run_name="run-1", model_name="agent.pt"):
    # what is n_minibatches ??
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # Dataset

 
    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(device=device, lr=lr, histories=histories)
    logging.info('Model being trained:')
    summary(agent.net, [(1, 96, 96, histories)], device=device)
    
    tensorboard = Evaluation(name=run_name, store_dir=tensorboard_dir, stats=["train_loss", "train_acc", "val_acc"])

    # load the model
    if load_model != "":
        print(f"Load model weights from {model_dir}/{load_model}")
        agent.load(f'{model_dir}/{load_model}')

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 

    
    # num minbatches
    for epoch in range(100):
        agent.train_mode() 
        t = tqdm(train_loader)

        count,count_val = 0, 0
        losses, accs = 0, 0
        loss_val, acc_val = 0, 0
        for i, (states, actions) in enumerate(t):
            states = states.to(torch.float32).to(device)
            actions = actions.to(torch.float32).to(device)
            # print(states.shape)
            # print(actions.shape) # why this is [16, 3], not [16] : not discretized?
            # should add history.
            # First , let's do it without history as the code is not yet finished
            # states, actions = stack_histories(states, actions, history_length=2)
            
            # print(states[:5])
            # print(actions[:5]) # why it is still outputing bunch of zeros => maybe in stack_histories, bunch of zeros? Yes.
            # print(states[:2]) # why it is still outputing bunch of zeros => maybe in stack_histories, bunch of zeros? Yes.
            # loss = agent.update(states[..., None], actions[..., None])
            loss = agent.update(states, actions)
            # _, predicted_actions = agent.predict(states[..., None])
            _, predicted_actions = agent.predict(states)
            acc = torch.sum(predicted_actions == actions)
            losses += loss
            accs += acc
            count += 1
        t_v = tqdm(val_loader)
        for i, (states, actions) in enumerate(t_v):
            agent.eval_mode()
            states = states.to(torch.float32).to(device)
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            # _, predicted_actions = agent.predict(states[..., None])
            _, predicted_actions = agent.predict(states)
            acc = torch.sum(predicted_actions == actions)
            acc_val += acc
            count_val += 1

        
        # eval
        # if epoch % 5 == 0:
            # acc = eval(agent, val_loader)
        # print(f"epoch{epoch} \n loss {losses / count} \n acc {accs / count} ")
        print(f"epoch{epoch} \n loss {losses / count}")

        if epoch % 1 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            tensorboard.write_episode_data(epoch, eval_dict={ 
                                                         "train_loss" : losses / count,
                                                         "train_acc" : accs / count,
                                                         "val_acc" : acc_val / count_val
                                                      })
    tensorboard.close_session()


    # TODO: save your agent
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  

    model_dir = agent.save(os.path.join(model_dir, model_name))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Imitation Learning")
    parser.add_argument("--load_model", type=str, default="", help="load the given model, and resume the training")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--run_name", type=str, default="", help="for tensorboard")
    parser.add_argument("--model_name", type=str, default="", help="for model save")
    parser.add_argument("--num_histories", type=int, default= 1 , help="histories")
    args = parser.parse_args()

    if args.run_name == "":
        parser.error("input run name for tensorboard")
    elif args.model_name == "":
        parser.error("input valid model name e.g) agent1.pt")

    # Dataset
    # train_dataset = CarDataset(data_dir="./data", augmentation=baseline_augment, data_type="train")
    # val_dataset = CarDataset(data_dir="./data", augmentation=baseline_augment, data_type="val")
    train_dataset = CarDataset(data_dir="./data", data_type="train", histories=args.num_histories)
    val_dataset = CarDataset(data_dir="./data", data_type="val", histories=args.num_histories)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    '''
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
    '''
    # train model (you can change the parameters!)
    train_model(train_loader, val_loader, n_minibatches=1000, histories = args.num_histories, \
                batch_size=args.batch_size, lr=1e-4, load_model=args.load_model, \
                    run_name=args.run_name, model_name=args.model_name)
    # train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4, load_model=args.load_model)
 
