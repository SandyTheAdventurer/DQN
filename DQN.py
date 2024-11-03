import torch
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from main import Base, printb, Buffer

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)


class DQN(Base):
    def __init__(self, env: gym.Env, gamma=0.99, logdir="logs/DQN", epsilon=0.99, batch_size=100) -> None:
        self.env = env
        self.logdir = logdir
        self.epsilon = epsilon
        self.actionlen = self.env.action_space.n
        self.writer = SummaryWriter(self.logdir)
        self.gamma = gamma
        self.replay_buffer= Buffer(max_size=10000)
        self.batch_size = batch_size
        super().__init__(input_size=self.env.observation_space.shape[0] + 1, output_size=1)

    def act(self, obs: np.array, pred_only=False):
        if not pred_only and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionlen)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float64, device='cuda').repeat(self.actionlen, 1)
            actions = torch.arange(self.actionlen, dtype=torch.float64, device='cuda').unsqueeze(1)
            input = torch.cat((obs_tensor, actions), dim=1)
            q_values = self(input).squeeze()
            return torch.argmax(q_values).item()

    def infer(self):
        batch = random.sample(self.replay_buffer.data, self.batch_size)
        obs_batch, action_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        inputs = torch.stack([torch.tensor(list(obs) + [action], dtype=torch.float64, device='cuda')
                              for obs, action in zip(obs_batch, action_batch)])

        next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float64, device='cuda').repeat_interleave(self.actionlen, dim=0)
        next_actions_tensor = torch.arange(self.actionlen, dtype=torch.float64, device='cuda').repeat(len(next_obs_batch)).unsqueeze(1)
        next_inputs = torch.cat((next_obs_tensor, next_actions_tensor), dim=1)

        with torch.no_grad():
            next_q_values = self(next_inputs).view(len(next_obs_batch), self.actionlen)
        max_next_q_values = next_q_values.max(dim=1).values

        targets = torch.tensor(rew_batch, dtype=torch.float64, device='cuda') + self.gamma * max_next_q_values * torch.tensor([1 - d for d in done_batch], dtype=torch.float64, device='cuda')

        return self.update(inputs, targets)



    def learn(self, timesteps: int):
        self.epsilon = 1.0
        rew_list = []
        loss_list = []
        steps_list = []

        for i in tqdm(range(timesteps)):
            obs, _ = self.env.reset()
            terminated = False
            total_reward = 0
            steps = 0

            while not terminated:
                action = self.act(obs)
                prev_obs = obs.copy()
                obs, rew, terminated, truncated, _ = self.env.step(action)
                total_reward += rew

                self.replay_buffer.append((prev_obs, action, rew, obs, terminated))

                if len(self.replay_buffer) % self.batch_size==0:
                    loss = self.infer()
                    loss_list.append(loss)

                steps += 1
                if truncated:
                    break

            self.epsilon = max(0.05, self.epsilon - (1 / (timesteps * (1 - 0.1))))
            rew_list.append(total_reward)
            steps_list.append(steps)

            self.writer.add_scalar("Reward/avg", np.mean(rew_list), i)
            self.writer.add_scalar("Reward/max", np.max(rew_list), i)
            self.writer.add_scalar("Reward/min", np.min(rew_list), i)
            self.writer.add_scalar("Loss/avg", np.mean(loss_list), i)
            if(len(loss_list)>0):
                self.writer.add_scalar("Loss/max", np.max(loss_list), i)
                self.writer.add_scalar("Loss/min", np.min(loss_list), i)
                self.writer.add_scalar("Steps/avg", np.mean(steps_list), i)
            self.writer.add_scalar("Steps/max", np.max(steps_list), i)
            self.writer.add_scalar("Steps/min", np.min(steps_list), i)
            self.writer.add_scalar("Epsilon", self.epsilon, i)

            if i % 1000 == 0 and i > 0:
                printb(f"Avg Reward for {i}th iteration: {np.mean(rew_list):.2f}",
                       f"Max Reward for {i}th iteration: {np.max(rew_list):.2f}",
                       f"Min Reward for {i}th iteration: {np.min(rew_list):.2f}",
                       f"Avg Loss for {i}th iteration: {np.mean(loss_list):.4f}",
                       f"Max Loss for {i}th iteration: {np.max(loss_list):.4f}",
                       f"Min Loss for {i}th iteration: {np.min(loss_list):.4f}",
                       f"Avg Steps for {i}th iteration: {np.mean(steps_list):.2f}",
                       f"Max Steps for {i}th iteration: {np.max(steps_list):.2f}",
                       f"Min Steps for {i}th iteration: {np.min(steps_list):.2f}",
                       f"Epsilon: {self.epsilon:.4f}")
                rew_list.clear()
                loss_list.clear()
                steps_list.clear()
        self.writer.close()
