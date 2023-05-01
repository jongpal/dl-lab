import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.2, tau=0.01, lr=1e-4,
                 history_length=0, device="cuda:0", task="pole"):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        # self.Q = Q.cuda()
        # self.Q_target = Q_target.cuda()
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.task = task

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions
        self.device = device

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)

        # how to check if the state is terminal? is it binary?
        self.train_mode()
        self.optimizer.zero_grad()

        self.replay_buffer.add_transition(state, action, next_state, reward, done=terminal)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        # from fixed Q networks
        # batch_actions ?
        with torch.no_grad():
            Q_values = self.Q_target(batch_next_states.to(self.device))
            a_prime = torch.argmax(Q_values, dim=1)
            # Q_max_values = torch.max(Q_values, dim=1)[0]

        ii = torch.arange(0, len(a_prime), 1)
        Q_max_fixed = Q_values[ii, a_prime]


        y_target = batch_rewards.to(torch.float32).to(self.device) +\
              self.gamma * Q_max_fixed * (torch.tensor(1) - batch_dones.to(torch.float32)).to(self.device) # If batch_dones is binary, then 

        # from online Q networks
        Q_values = self.Q(batch_states.to(self.device))
        # a = torch.argmax(Q_values)[0]
        # batch actions should index this Q_values

        ''' One way
        x_index = torch.arange(0, len(batch_states), 1)
        y_index = batch_actions
        y_predict = Q_values[x_index, y_index]
        '''
        x_index = torch.arange(0, len(batch_states), 1)
        y_index = batch_actions
        y_predict = Q_values[x_index, y_index]
        # The other way
        # y_predict = torch.gather(Q_values, 1, batch_actions[..., None].to(self.device))
        # print(torch.gather(Q_values, 1, batch_actions[..., None].to(self.device)).shape)
        # print(y_predict == torch.gather(Q_values, 1, batch_actions[..., None].to(self.device)))
        loss = self.loss_function(y_predict, y_target.detach())

        loss.backward()
        self.optimizer.step()
        soft_update(self.Q_target, self.Q, self.tau)

        return loss
    
    def train_mode(self):
        self.Q.train()
        self.Q_target.eval()


    def eval_mode(self):
        self.Q.eval()
        self.Q_target.eval()

    def act(self, state, deterministic, step=None):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # pass
            # TODO: take greedy action (argmax)
            state = torch.tensor(state).to(self.device)
            # print(self.Q(state).shape)
            # action_id = torch.argmax(self.Q(state), dim=1)
            self.Q.eval()
            with torch.no_grad():
                action_id = torch.argmax(self.Q(state)).squeeze()
            # action_id = self.Q(state)
    
            return action_id.detach().cpu().numpy()
        else:
            # pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            
            # first, for carpole : uniform
            if self.task == "pole":
                if np.random.uniform() > 0.5:
                    action_id = 0
                else:
                    action_id = 1
                return action_id
            else:
                if step and step < 30:
                    action_id = 3
                    # print(action_id)
                    return action_id

                # for carracing,
                # first, accelerate : 0.4, straight : 0.3, decelerate : 0.1 go left : 0.1 go right : 0.1
                action_id = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.1, 0.1, 0.4, 0.1])
            
            return action_id

    def save_checkpoint(self, file_name, num_episodes):
        
        dict = {'num_episodes': num_episodes,
                'Q_model_state_dict': self.Q.state_dict(),
                'Q_target_model_state_dict' : self.Q_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
               }
        torch.save(dict, file_name)

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load_checkpoint(self, file_name):
        checkpoint = torch.load(file_name)
        self.Q.load_state_dict(checkpoint['Q_model_state_dict'])
        self.Q_target.load_state_dict(checkpoint['Q_target_model_state_dict'])
        num_episodes = checkpoint["num_episodes"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return num_episodes

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
