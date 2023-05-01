import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

# s->100->400->200->a : "mean": 22.513333333333332, "std": 3.2654487525538594 epsilon = 0.1
# s 200 400 200 a : "mean": 45.92666666666667, "std": 15.413023353284354 epsilon=0.1
# s 100 400 400 200 a, epoch 450, lr :  def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.2, tau=0.01, lr=1e-4, history = 0
# "mean": 166.43333333333334, "std": 29.495404538485126
# s 100 400 400 200 a, epoch 450, lr :  def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.2, tau=0.01, lr=1e-4, history = 0
# "mean": 166.43333333333334, "std": 29.495404538485126, optim_grad = 0

# class MLP(nn.Module):
#   def __init__(self, state_dim, action_dim, hidden_dim=400):
#     super(MLP, self).__init__()
#     self.fc1 = nn.Linear(state_dim, 200)
#     self.fc2 = nn.Linear(200, hidden_dim)
#     self.fc3 = nn.Linear(hidden_dim, 200)
#     self.fc4 = nn.Linear(200, action_dim)

#   def forward(self, x):
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = F.relu(self.fc3(x))
#     return self.fc4(x)
  
# class MLP(nn.Module):
#   def __init__(self, state_dim, action_dim, hidden_dim=400):
#     super(MLP, self).__init__()
#     self.fc1 = nn.Linear(state_dim, 100)
#     self.fc2 = nn.Linear(100, hidden_dim)
#     self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#     self.fc4 = nn.Linear(hidden_dim, 200)
#     self.fc5 = nn.Linear(200, action_dim)

#   def forward(self, x):
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = F.relu(self.fc3(x))
#     x = F.relu(self.fc4(x))
#     return self.fc5(x)

# class MLP(nn.Module):
#   def __init__(self, state_dim, action_dim, hidden_dim=400):
#     super(MLP, self).__init__()
#     self.fc1 = nn.Linear(state_dim, 100)
#     self.fc2 = nn.Linear(100, 200)
#     self.fc3 = nn.Linear(200, hidden_dim)
#     self.fc4 = nn.Linear(hidden_dim, hidden_dim)
#     self.fc5 = nn.Linear(hidden_dim, 200)
#     self.fc6 = nn.Linear(200, action_dim)

#   def forward(self, x):
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = F.relu(self.fc3(x))
#     x = F.relu(self.fc4(x))
#     x = F.relu(self.fc5(x))
#     return self.fc6(x)

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=3): 
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.linear1 = nn.Linear(5 * 5 * 128, n_classes)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # x = torch.transpose(x, (0, 3, 1, 2))
        if len(x.shape) != 4:
          x = x[None, ...]
        x = rearrange(x, "b h w c -> b c h w")
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        # reshape always copies memory. view never copies memory.
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return x



