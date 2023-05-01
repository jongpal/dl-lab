import torch
import torch.nn as nn
from agent.networks import CNN, CNN2, CNN3, CNN4

class BCAgent:
    
    def __init__(self, histories, device="cuda:0", lr=1e-4):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(n_classes=5).to(device).float()
        # self.net = CNN2(n_classes=5).to(device).float()
        self.net = CNN4(n_classes=5, history_length=histories).to(device).float()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=lr)

        # self.optim = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.device = device
        self.lr = lr
        # pass
    # def loss(self, output, target, eps = 1e-3):
        # cross entropy with regularization.
    # def add_loss_to_zero(self, out):
    #     print(y.shape)
    #     return len(y) - torch.sum(torch.sum(y == 0.0, axis=1))

    def update(self, X_batch, y_batch):
        # eps = 1e-3
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        self.optim.zero_grad()
        X_batch = X_batch.to(self.device).float()
        # X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device).long()
        # add noise to y? label smoothing?
        out = self.net(X_batch)
        
        # print(y_batch.squeeze().shape)
        loss = self.loss_fn(out, y_batch.squeeze())
        # this does not went through gradient?
        # loss += torch.sum(torch.argmax(out, axis=1) == 0) * 10
        # loss += torch.sum(torch.exp(out[:, 0]) / torch.sum(torch.exp(out), axis=1))
        # loss += torch.sum(out[:, 0]) / 100

        loss.backward()
        self.optim.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        # outputs = self.net(X)
        self.eval_mode()
        with torch.no_grad():
            out = self.net(X)
            # print(out)
            # actions = torch.argmax(out, dim=1).item()
            actions = torch.argmax(out, dim=1)
            # acc = torch.sum(actions == y.squeeze()) / len(y)     

        # return out, acc
        return out, actions
        
    
    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

