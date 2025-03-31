import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt # Import matplotlib
import time


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, episodes, batch_size, n_actions, renew_eps, max_mem_size=100000, epsilon_end=0.01, epsilon_dec=1e-4):
        self.gamma = gamma
        self.episodes = episodes
        self.renew_eps = renew_eps
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        state = np.array(state)
        state_ = np.array(state_)

        if state.shape != self.state_memory.shape[1:]:
            state = state.reshape(self.state_memory.shape[1:])

        if state_.shape != self.new_state_memory.shape[1:]:
            state_ = state_.reshape(self.new_state_memory.shape[1:])

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Convert observation to a NumPy array first
            observation_array = np.array([observation])
            state = T.tensor(observation_array).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Move tensors to the correct device immediately
        state_batch = T.tensor(self.state_memory[batch], device=self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], device=self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch], device=self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool, device=self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()


        '''if self.epsilon == self.eps_min and [scores[i] < 0 for i in range(int(self.episodes/10))]:
            self.epsilon = self.renew_eps
        else:'''
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec) 


start_time = time.time()

if __name__ == '__main__':
    n_games = 500
    env = gym.make('LunarLander-v3')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, epsilon_end=0.15, input_dims=[8], episodes=n_games, lr=0.003, renew_eps=0.5)
    scores, eps_history = [], []
    
    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print('episode ', i + 1, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i + 1 for i in range(n_games)] 

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time

    print(f"Execution time: {execution_time:.2f} seconds")  # Print the execution time

    # Plot scores with line of best fit
    plt.plot(x, scores, label='Scores')
    z = np.polyfit(x, scores, 1)  # 1 for linear fit
    p = np.poly1d(z)

    plt.plot(x, p(x), "r--", label='Line of best fit')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Lunar Lander Scores')
    plt.legend()
    plt.savefig('lunar_lander_scores.png')
    plt.show()

    '''Plot epsilon with line of best fit
    plt.plot(x, eps_history, label='Epsilon')
    z = np.polyfit(x, eps_history, 1)
    p = np.poly1d(z)
    
    plt.plot(x, p(x), "r--", label='Line of best fit')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon History')
    plt.legend()
    plt.savefig('lunar_lander_epsilon.png')
    plt.show()'''
