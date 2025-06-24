# filepath: /skrl_f1tenth_pipeline/skrl_f1tenth_pipeline/src/skrl_qler/train.py

#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch.random import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

class SpatioTempDuelingTransformerNet(nn.Module):
    def __init__(self, seq_len, num_ranges, num_actions):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 5, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 48, 4, 2)
        self.bn3 = nn.BatchNorm1d(48)
        self.conv4 = nn.Conv1d(48, 64, 3, 1)
        self.bn4 = nn.BatchNorm1d(64)
        
        with torch.no_grad():
            d = torch.zeros(1, 2, num_ranges)
            x = F.relu(self.bn1(self.conv1(d)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            self.conv_feat_size = x.flatten(1).shape[1]
        
        self.proj = nn.Linear(self.conv_feat_size, 128)
        enc = nn.TransformerEncoderLayer(128, 4, 256, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc, 2)
        self.ln = nn.LayerNorm(128)
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_actions))

    def forward(self, x):
        B, T, R, _ = x.shape
        x = x.view(B * T, R, 2).permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.flatten(1).view(B, T, -1)
        x = self.proj(x)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln(x)
        ctx = x.mean(1)
        v = self.value(ctx)
        a = self.adv(ctx)
        return v + a - a.mean(1, keepdim=True)

def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
    X, y = [], []
    N = lidar.shape[1]
    for i in range(len(lidar) - seq_len):
        f = lidar[i:i + seq_len]
        dt = np.diff(ts[i:i + seq_len + 1]).reshape(seq_len, 1)
        dt = np.repeat(dt, N, axis=1)
        X.append(np.stack([f, dt], 2))
        y.append([servo[i + seq_len], speed[i + seq_len]])
    return np.array(X, np.float32), np.array(y, np.float32)

def load_all(npz_paths, seq_len):
    all_l, all_s, all_sp, all_t = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_l.append(d['lidar'])
        all_s.append(d['servo'])
        all_sp.append(d['speed'])
        all_t.append(d['ts'])
    L, S, P, T = map(lambda lst: np.concatenate(lst, 0), (all_l, all_s, all_sp, all_t))
    return create_lidar_sequences(L, S, P, T, seq_len)

class LidarEnv(gym.Env):
    def __init__(self, X, y, seq_len, servo_bins, speed_bins, sup_model_path,
                 eta_jerk=1.0, eta_drive=1.0, eta_collision=10.0, v_star=None):
        super().__init__()
        self.X, self.y = X, y
        self.seq_len, self.idx = seq_len, 0
        self.servo_bins = servo_bins
        self.speed_bins = speed_bins
        self.eta_jerk = eta_jerk
        self.eta_drive = eta_drive
        self.eta_collision = eta_collision
        self.v_star = v_star if v_star is not None else y[:, 1].mean()
        self.v_max = y[:, 1].max()
        self.prev_speed = None
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (seq_len, X.shape[2], 2), np.float32)
        self.action_space = gym.spaces.Discrete(len(servo_bins) * len(speed_bins))
        self.sup_model = load_model(sup_model_path, compile=False)

    def reset(self, seed=None, options=None):
        self.idx = 0
        self.prev_speed = None
        return self.X[0], {}

    def step(self, action):
        i, j = divmod(action, len(self.speed_bins))
        pred = np.array([self.servo_bins[i], self.speed_bins[j]], np.float32)
        speed = pred[1]

        loss = np.mean((pred - self.y[self.idx]) ** 2)
        R_base = -loss

        if self.prev_speed is None:
            R_jerk = 0.0
        else:
            R_jerk = -self.eta_jerk * abs(speed - self.prev_speed)
        self.prev_speed = speed

        if speed <= self.v_star:
            R_drive = self.eta_drive * (speed / self.v_star)
        else:
            R_drive = self.eta_drive * ((self.v_max - speed) / (self.v_max - self.v_star))

        R_coll = 0.0

        reward = R_base + R_jerk + R_drive + R_coll

        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros_like(self.X[0])
        return obs, reward, done, False, {}

class CumulativeRewardTrainer(SequentialTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_reward = 0.0
        self.episode_reward = 0.0

    def on_step(self, timestep, timesteps):
        r = float(self._reward[0])
        self.total_reward += r
        self.episode_reward += r
        return super().on_step(timestep, timesteps)

    def on_episode_end(self):
        print(f"[Episode Completed] Return = {self.episode_reward:.4f}")
        self.episode_reward = 0.0
        return super().on_episode_end()

    def train(self):
        super().train()
        print(f"[Training Finished] Cumulative reward = {self.total_reward:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', nargs='+', required=True)
    p.add_argument('--sup_model', required=True)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--servo_bins', type=int, default=7)
    p.add_argument('--speed_bins', type=int, default=5)
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--capacity', type=int, default=100_000)
    args = p.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    X, y = load_all(args.npz, args.seq_len)
    sb = np.linspace(y[:, 0].min(), y[:, 0].max(), args.servo_bins)
    sp = np.linspace(y[:, 1].min(), y[:, 1].max(), args.speed_bins)
    raw_env = LidarEnv(X, y, args.seq_len, sb, sp, args.sup_model)

    skrl_env = wrap_env(raw_env)
    class DeviceEnvWrapper(gym.Env):
        def __init__(self, sk_env, raw_env, device):
            self.env = sk_env
            self.device = device
            self.observation_space = raw_env.observation_space
            self.action_space = raw_env.action_space
            self.num_agents = 1
            self.num_envs = 1

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return torch.as_tensor(obs, device=self.device), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return torch.as_tensor(obs, device=self.device), reward, terminated, truncated, info

        def render(self, *args, **kwargs): return self.env.render(*args, **kwargs)
        def close(self): return self.env.close()

    env = DeviceEnvWrapper(skrl_env, raw_env, device)

    obs_space, act_space = raw_env.observation_space, raw_env.action_space
    class QNetwork(DeterministicMixin, Model):
        def __init__(self):
            Model.__init__(self, obs_space, act_space, device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = SpatioTempDuelingTransformerNet(args.seq_len, X.shape[2], act_space.n).to(device)
            self.obs_shape = obs_space.shape

        def compute(self, inputs, role):
            s = inputs["states"]
            if s.dim() == 2:
                s = s.view(s.size(0), *self.obs_shape)
            elif s.dim() == 5:
                s = s.squeeze(0)
            return self.net(s), {}

    q_net, target_net = QNetwork(), QNetwork()
    memory = RandomMemory(memory_size=args.capacity, num_envs=1, device=device)

    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg.update({
        "learning_rate": 5e-5,
        "discount_factor": 0.99,
        "batch_size": args.batch_size,
        "update_frequency": 4,
        "device": str(device),
        "target_q_network": {"update_interval": 1_000}
    })

    agent = DQN(models={"q_network": q_net, "target_q_network": target_net},
                memory=memory, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    trainer = CumulativeRewardTrainer(
        cfg={"timesteps": args.timesteps, "headless": True},
        env=env, agents=agent
    )
    trainer.train()

    os.makedirs("Models", exist_ok=True)
    agent.save("Models/dqrn_mps")
    memory.save("Models/dqrn_memory")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()