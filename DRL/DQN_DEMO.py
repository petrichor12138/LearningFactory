import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# 创建DQN模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, gamma=0.99, lr=0.001, buffer_capacity=100000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_network.fc3.out_features)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
            lambda x: torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32).to(self.device),
            batch
        )

        q_values = self.q_network(state_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0]

        expected_q_values = (1 - done_batch) * self.gamma * next_q_values + reward_batch

        loss = nn.functional.mse_loss(q_values.gather(1, action_batch.long().unsqueeze(1)), expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# 训练DQN代理
def train_dqn(agent, env, num_episodes, max_steps):
    def plot_reward(rewards):
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8")
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        plt.savefig("./DRL/reward.png")

    rewards = []
    max_reward = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            if reward > max_reward*0.8:
                agent.buffer.push(state, action, reward, next_state, done)
                max_reward = max(max_reward, reward)
            state = next_state
            agent.update()
            total_reward += reward 
            if done:
                break
        agent.update_target_network()
        rewards.append([total_reward])
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    plot_reward(rewards)
    with open("./DRL/reward.txt", "w") as f:
        f.write(str(rewards) + '\n')

    


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 200
    max_steps = 1000

    train_dqn(agent, env, num_episodes, max_steps)
