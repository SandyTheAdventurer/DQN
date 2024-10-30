import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from tqdm import tqdm

class BaseDQN(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters())
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor):
        logits = self.linear_relu_stack(X)
        return logits
    
    def update(self, X: torch.Tensor, y: torch.Tensor):
        self.train()
        y_pred=self(X)
        loss = self.loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class DQN(BaseDQN):
    def __init__(self, env: gym.Env, gamma=0.99) -> None:
        self.env = env
        self.actionlen = self.env.action_space.n
        self.gamma = gamma
        super().__init__(input_size=self.env.observation_space.shape[0] + 1)
        self.scripted_model = torch.jit.script(self)

    def act(self, obs: np.array, pred_only=False):
        if not pred_only and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionlen)
        with torch.no_grad():
            input = torch.tensor(obs, dtype=torch.float32, device=self.device).repeat(self.actionlen, 1)
            action_tensor = torch.arange(self.actionlen, dtype=torch.float32, device=self.device).unsqueeze(1).view(-1, 1)
            input = torch.cat((input, action_tensor), dim=1)
            rew = self.scripted_model(input).squeeze()
            return torch.argmax(rew).item()

    def infer(self, obs, action, rew, next_obs, done):
        input = torch.tensor(list(obs) + [action], dtype=torch.float32, device=self.device)
        
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).repeat(self.actionlen, 1)
        next_actions_tensor = torch.arange(self.actionlen, dtype=torch.float32, device=self.device).view(-1, 1)
        next_inputs = torch.cat((next_obs_tensor, next_actions_tensor), dim=1)
        
        with torch.no_grad():
            next_q_values = self.scripted_model(next_inputs).squeeze()
        max_next_q_value = next_q_values.max().item()

        target = rew if done else rew + self.gamma * max_next_q_value
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)
        
        return self.update(input, target_tensor)

    def learn(self, timesteps: int):
        self.epsilon = 1
        rew_list = []
        loss_list = []
        steps_list = []

        for i in tqdm(range(timesteps)):
            obs, _ = self.env.reset()
            terminated = False
            total_reward = 0
            total_loss = 0
            steps = 0

            while not terminated:
                action = self.act(obs)
                prev_obs = obs.copy()
                obs, rew, terminated, truncated, _ = self.env.step(action)
                total_reward += rew

                if steps % 10 == 0:
                    loss = self.infer(prev_obs, action, rew, obs, terminated)
                    total_loss += loss
                steps += 1

                if truncated:
                    break
                
            self.epsilon = max(0.1, self.epsilon - (0.9 / (timesteps * (1 - 0.25))))
            loss_list.append(total_loss)
            rew_list.append(total_reward)
            steps_list.append(steps)

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

    def save_model(self, path: str):
        torch.jit.save(self.scripted_model, path)
        print(f"Model saved to {path}")


def printb(*messages):
    width = max(len(message) for message in messages) + 4
    print("+" + "-" * width + "+")
    for message in messages:
        print("| " + message.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")