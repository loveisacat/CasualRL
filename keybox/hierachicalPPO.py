from torch.distributions.multivariate_normal import MultivariateNormal
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from gym import make

import gym_examples
import gym
import time
import os

#os.environ["SDL_VIDEODRIVER"] = "dummy"

import matplotlib.pyplot as plt

net_dim = 64

def transform(x):
    return torch.Tensor([x])


def transform_state(x):
    return torch.tensor(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.transition = namedtuple('Transition', ('state', 'next_state', 'reward', 'action', 'done', 'goal'))

    def add(self, *args):
        new_transition = self.transition(*args)
        self.buffer.append(new_transition)

    def sample(self, batch_size):
        ind = np.random.choice(np.arange(self.max_size), batch_size, replace=False)
        buff = self.transition(*zip(*self.buffer))
        states = torch.cat(buff.state)[ind]
        next_states = torch.cat(buff.next_state)[ind]
        rewards = torch.cat(buff.reward).unsqueeze(1)[ind]
        actions = torch.cat(buff.action).unsqueeze(1)[ind]
        done = torch.cat(buff.done).unsqueeze(1)[ind]
        return states, actions, next_states, rewards, done

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, net_dim)
        self.fc2 = nn.Linear(net_dim, net_dim)
        self.mu_linear = nn.Linear(net_dim, action_dim)
        self.sigma_linear = nn.Linear(net_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_linear(x))
        sigma = F.softplus(self.sigma_linear(x))
        return mu, sigma.squeeze() + 1e-5


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, net_dim)
        self.fc2 = nn.Linear(net_dim, net_dim)
        self.fc3 = nn.Linear(net_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(self, state_dim, action_dim, optimizer, gamma, clip, trajectory_size, epochs, batch_size, lr_actor,
                 lr_critic):
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.trajectory_size = trajectory_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=trajectory_size)

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.actor_old = Actor(state_dim=state_dim, action_dim=action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim=state_dim)

        self.actor_optimizer = optimizer(self.actor.parameters(),  lr_actor)
        self.critic_optimizer = optimizer(self.critic.parameters(),  lr_critic)

    def act(self, state):
        with torch.no_grad():
            mu, sigma = self.actor(state)
        multivatiate_normal = MultivariateNormal(mu, scale_tril=torch.diag(sigma))
        action = multivatiate_normal.sample()
        #Hard Code for action_space
        low = -1
        high = 1
        action.clamp_(low, high)
        #action.clamp_(env.action_space.low[0], env.action_space.high[0])
        return action.numpy()[0]

    def get_value(self, state, action):
        mu, sigma = self.actor(state)
        multivatiate_normal_new = MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))
        log_prob = multivatiate_normal_new.log_prob(action)

        with torch.no_grad():
            mu_old, sigma_old = self.actor_old(state)
        multivatiate_normal_old = MultivariateNormal(mu_old, scale_tril=torch.diag_embed(sigma_old))
        old_log_prob = multivatiate_normal_old.log_prob(action)

        return log_prob.unsqueeze(1), old_log_prob.unsqueeze(1),

    def update(self, meta):

        for _ in range(self.epochs):
            states, actions, next_states, rewards, done = self.buffer.sample(self.batch_size)

            V_cur = algo.critic(states)
            with torch.no_grad():
                V_next = algo.critic(next_states)
            if meta == True:
                 V_target = rewards + (1 - True) * self.gamma * V_next
            else:
                 V_target = rewards + (1 - done) * self.gamma * V_next
            advantage = (V_target - V_cur).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std())

            new_log_prob, old_log_prob = self.get_value(states, actions)
            ratio = torch.exp(new_log_prob - old_log_prob)
            s1 = ratio * advantage
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
            L_actor = (-torch.min(s1, s2)).mean()

            self.actor_optimizer.zero_grad()
            L_actor.backward()
            for param in self.actor.parameters():
                param.grad.data.clamp_(-10, 10)
            self.actor_optimizer.step()

            L_critic = F.mse_loss(V_target, V_cur)
            self.critic_optimizer.zero_grad()
            L_critic.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-10, 10)
            self.critic_optimizer.step()
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.reset()

    def save(self):
        torch.save(self.actor.state_dict(), "./models/agent_ppo_9d2.pkl")


def intrinsic_reward(state, action, state_next, goal):
        return 0.1 if (state_next.tolist()[0] == goal).all() else 0.0

def epsGreedy(state):
     return 0


if __name__ == "__main__":
    # Initialize the "maze" environment
    # Pre-generated mazes: maze-sample-3x3,maze-sample-5x5,maze-sample-10x10, maze-sample-100x100
    # Randomly generated mazes (same maze every epoch): maze-random-3x3, maze-random-5x5, maze-random-10x10, maze-random-100x100
    # Randomly generated mazes with portals and loops:maze-random-10x10-plus, maze-random-20x20-plus, maze-random-30x30-plus
    env = gym.make('gym_examples/GridWorld-v0', render_mode='human', size=9)
    parameters = {'state_dim': 2,
                  'action_dim': 4,
                  'optimizer': optim.Adam,
                  'lr_actor': 1e-5,
                  'lr_critic': 1e-4,
                  'gamma': 0.99,
                  'clip': 0.1,
                  'trajectory_size': 2000,
                  'epochs': 128,
                  'batch_size': 64}

    algo = PPO(**parameters)
    algo_g = PPO(**parameters)
    episodes = 1000
    #episodes = 300
    best = -1500
    rewards = deque(maxlen=50)

    start = time.time()
    #MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    MAZE_SIZE = tuple(np.array([env.size,env.size]))
    
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    #obv = env.reset()
    #env.render()
    #time.sleep(1000)

    if os.path.exists('./models/bihPPO.pkl'):
      algo.actor.state_dict(torch.load("./models/bihPPO.pkl"))
      algo_g.actor.state_dict(torch.load("./models/goal.pkl"))
      count = steps = allreward = maxstep = 0
      minstep = 99999
      test_runs = 100
      for i in range(test_runs):
         ori_state = env.reset()['agent']
         state = transform(ori_state)
         total_reward = 0
         done = False
         for j in range(MAX_T):
            action = algo.act(state)
            action_pro = np.argmax(action)
            action_env = action_pro.item()
            #action = np.array([action])
            #action = tuple(action) #Conversion
            next_state, reward, done, _ = env.step(action_env)
            if done == False and reward == 0:
                reward = - 1.0/MAX_T
          
            state = transform(next_state['agent'])
            total_reward += reward
            if done:
                break
         if done == True and total_reward >= 0:
            count += 1
            steps += j
            allreward += total_reward
            if j < minstep:
               minstep = j
            if j > maxstep:
               maxstep = j
         print("Test:", i," Step:",j," Reward:",total_reward, " count:",count)

      print("WinRate:", count/test_runs, " MeanStep:", steps/count, " MaxStep:",maxstep, " MinStep:",minstep," Mean Reward:", allreward/count)    
      exit(0)
    else:
      g = []
      g.append(np.array([2,2]))
      g.append(np.array([2,6]))
      g.append(np.array([6,2]))
      g.append(np.array([6,6]))
      for i in range(episodes):
        ori_state = env.reset()['agent']
        state = transform(ori_state)
        total_reward = 0
        steps = 0
        done = False
        while (not done and steps < MAX_T):
          goal = g[np.argmax(algo_g.act(state)).item()]
          r = 0
          while (not done and steps < MAX_T and r<=0):
            action = algo.act(state)
            s0 = state[0][0].item()
            s1 = state[0][1].item()
            action_pro = np.argmax(action)
            action_env = action_pro.item()
            #action = np.array([action])
            #action = tuple(action) #Conversion
            next_state, reward, done, _ = env.step(action_env)
            if done == False:
                reward = - 1.0/MAX_T
           
            next_state = transform(next_state['agent'])
            steps += 1
            r = intrinsic_reward(state, action, next_state, goal)
            R = r + reward
            total_reward += R
            algo.buffer.add(state, next_state, transform(R), transform(action), transform(done), goal)
            #algo.buffer.add(state, next_state, transform(reward), transform(action), transform(done), r)
            

            if len(algo.buffer) == algo.trajectory_size:
                algo.update(False)
            state = next_state

        rewards.append(total_reward)
        algo_g.buffer.add(state, next_state, transform(total_reward), transform(action), transform(done), r)
        if len(algo_g.buffer) >= 64:
           algo_g.update(True) 
        if len(rewards) == 50 and np.mean(rewards) > best:
            best = np.mean(rewards)
            print(f'NEW BEST {best}')
            algo.save()

        print(f'episode # {i} reward: {total_reward} intrisic_reward: {r} steps: {steps}')
    end = time.time()
    print("All Done:", end - start)




