import torch.utils.data as data
import torch
import os
import gzip
import pickle
import numpy as np
import albumentations
import sys
import copy
import matplotlib.pyplot as plt
# from einops import rearrange
from imblearn.over_sampling import RandomOverSampler
sys.path.append("../") 
from utils import *

def stack_histories(X, history_length=1):
    # print(X[:2])
    if len(X.shape) == 4:
        X = X.squeeze(axis=-1)
    X_history = np.zeros((*X.shape, history_length))
    for i in range(len(X)):
        if i < history_length - 1:
            to_copy = copy.deepcopy(np.transpose(X[:i+1], (1, 2, 0)))
            X_history[i, ..., history_length - 1 - i: history_length] = to_copy
            X_history[i, ..., :history_length - 1 - i] = np.tile(X[0, ..., None], (1, 1, 1, history_length - i - 1))
        else:
            # torch.permute not allocate new one.
            to_copy_X = copy.deepcopy(np.transpose(X[i-history_length+1:i+1], (1, 2, 0)))
            X_history[i] = to_copy_X
    # print(X_history[0])
    return X_history


class CarDataset(data.Dataset):
    def __init__(self, data_dir, histories, frac=0.1,  data_type="train", normalize=None, augmentation=None):
        super(CarDataset).__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        
        # self.normalize = albumentations.Normalize(mean=normalize, std=normalize)
        self.normalize = normalize
        self.num_histories = histories
        self.data_type = data_type
        self.X, self.y = None, None
        self.frac = frac

        X, y = self.read_data()

        n_samples = len(X)
        print("n_samples ", n_samples)
        if self.data_type == "train":
            self.X, self.y = X[:int((1-self.frac) * n_samples)], y[:int((1-self.frac) * n_samples)]
        else:
            self.X, self.y = X[int((1-self.frac) * n_samples):], y[int((1-self.frac) * n_samples):]
        print(f"{data_type} n_samples {len(self.X)}")
    
        
    def read_data(self):
        print("... read data")
        data_file = os.path.join(self.data_dir, 'data.pkl.gzip')
        # data_file = os.path.join(self.data_dir, 'data2.pkl.gzip')
        # data_file = os.path.join(self.data_dir, 'alldata.pkl.gzip')
    

        f = gzip.open(data_file,'rb')
        data = pickle.load(f)
        # with gzip.open(data_file,'rb') as f:
        #     data = pickle.load(f)
        # get images as features and actions as targets
        X = np.array(data["state"]).astype('float32')
        y = np.array(data["action"]).astype('float32')

        X = rgb2gray(X)
        # discretize
        y_ = np.zeros((len(y), 1))
        for i in range(len(y)):
            y_[i] = action_to_id(y[i])
 
        X_histories = stack_histories(X, history_length=self.num_histories)
        # X_histories, y_histories = stack_histories(X, y_, history_length=2)

        # balance the dataset
        # first, identify the desired amount by taking a look at distribution

        # print(np.sum(y_ == 0))
        # print(np.sum(y_ == 1))
        # print(np.sum(y_ == 2))
        # print(np.sum(y_ == 3))
        # print(np.sum(y_ == 4))
        print(np.unique(y_,return_counts=True))
  
        # then, loop til it gets the right amount
        # or, get idx of each class and concatenate them with desired amount
        
        
        idx = []
        i = 0
        y_ = y_.squeeze()
        # Let's sample data randomly so it doesn't have a meaning of sequence. (But is it? => If markov assumption?)
        
        # while i < len(X):
        #     if y_[i] == 0:
        #         # with probability 50 %, drop it.
        #         if np.random.uniform() > 0.5:
        #             idx.append(i)
        #     else:
        #         idx.append(i)
        #     i += 1
        
        # print(X.shape) # (size, 96, 96)
        '''
        new_X = np.zeros((len(y), X.shape[1], X.shape[2])).astype('float32')
        new_y = np.zeros((len(y),)).astype('float32')

        # i = 0
        # for j in idx:
        #     new_X[i] = X[j]
        #     new_y[i] = y_[j]
        #     i += 1

        print(new_X.shape) # (0, 96, 96)
        print(new_y.shape) # (0)

        # ros = RandomOverSampler(sampling_strategy={3 : 4000})
        # ros2 = RandomOverSampler(sampling_strategy={2 : 6000})
        # ros3 = RandomOverSampler(sampling_strategy={1 : 6000})
        # new_X_fit = rearrange(new_X, "b h w -> b (h w)")
        new_X_fit = new_X.reshape((new_X.shape[0], -1))
        new_y_fit = new_y
        # new_X_fit, new_y_fit = ros.fit_resample(new_X_fit, new_y_fit) # new_X's dimension => use rearrange
        # new_X_fit, new_y_fit = ros2.fit_resample(new_X_fit, new_y_fit) # new_X's dimension => use rearrange
        # new_X_fit, new_y_fit = ros3.fit_resample(new_X_fit, new_y_fit) # new_X's dimension => use rearrange
        # the b has increased here, so we need to specify h, w?
        # new_X_fit = rearrange(new_X_fit, "b (h w) -> b h w")
        new_X_fit = new_X_fit.reshape((-1, 96, 96))
        # print(new_X_fit.shape) # (0, 96, 96)
        # print(new_y_fit.shape) # (0)

        # print(np.sum(new_y_fit == 0))
        # print(np.sum(new_y_fit == 1))
        # print(np.sum(new_y_fit == 2))
        # print(np.sum(new_y_fit == 3))
        # print(np.sum(new_y_fit == 4))
        # return new_X, new_y
        
        return new_X_fit[:10000], new_y_fit[:10000]
        '''
        # return X[:15000], y_[:15000]
        return X_histories[:15000], y_[:15000]
    
        # return X, y_
 
    def normalize_image(self, x):
        x /= 255.0
        x -= 0.5
        return x


    def __getitem__(self, index):
    
        if self.augmentation:
            x, y = self.augmentation(image = self.X[index])['image'], self.y[index]
        else:
            x, y = self.X[index], self.y[index]
        # if self.normalize:
        #     x = self.normalize(image = x)['image']
        # elif self.normalize == None:
            # x = self.normalize_image(x)
        if self.normalize != None:
            x = self.normalize_image(x)
        # print(x[:3])
        # return torch.from_numpy(x), torch.from_numpy(y)
        return torch.from_numpy(x), y

    def __len__(self):
        return len(self.X)