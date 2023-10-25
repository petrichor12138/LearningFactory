import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# hyper parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
# confirm the shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    
    
class DQN(object):
    def __init__(self) -> None:
        self.eval_net, self.target_net = Net().to(DEVICE), Net().to(DEVICE)
        
        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(DEVICE)
        # input one sample
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            # get the max value index
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
              
        return action
    
    def store_transition(self, state, action, reward, n_state):
        transition = np.hstack((state, [action, reward], n_state))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        
        
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        # sample batch samples
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_status = torch.FloatTensor(batch_memory[ : , :N_STATES]).to(DEVICE)
        batch_action = torch.LongTensor(batch_memory[ : , N_STATES:N_STATES+1].astype(int)).to(DEVICE)
        batch_reward = torch.FloatTensor(batch_memory[: , N_STATES+1:N_STATES+2]).to(DEVICE)
        batch_n_status = torch.FloatTensor(batch_memory[: , -N_STATES:]).to(DEVICE)
        
        q_eval = self.eval_net(batch_status).gather(1, batch_action)
        q_next = self.target_net(batch_n_status).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
dqn = DQN()

print('\nCollecting experience...')

reward_list = []
for i_episode in range(400):
    s, info = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        
        # take action
        s_, r, done, truncated, info = env.step(a)
        
        # reward function
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        r = r1 + r2
        
        # store transition
        dqn.store_transition(s, a, r, s_)
        
        ep_r += r
        
        dqn.learn()
            
        s = s_
        
        if done:
            break
    
    print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
    reward_list.append(ep_r)
        
import matplotlib.pyplot as plt
plt.plot(np.arange(len(reward_list)), reward_list)
plt.ylabel('Reward')
plt.xlabel('training steps')
plt.show()
plt.savefig("./DRL/myreward.png")