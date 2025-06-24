# filepath: /skrl_f1tenth_pipeline/skrl_f1tenth_pipeline/src/skrl_qler/skrl_f1tenth_pipeline.py

import os
import numpy as np
import torch
import gymnasium as gym
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch.random import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from train import SpatioTempDuelingTransformerNet, load_all
from gym_interface import LidarEnv, DirectFlatObs, FrictionRand, ProgressReward

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

def main():
    npz_paths = ["data/pure_pursuit_data.npz"]  # Update with actual paths
    seq_len = 5
    X, y = load_all(npz_paths, seq_len)

    servo_bins = np.linspace(y[:, 0].min(), y[:, 0].max(), 7)
    speed_bins = np.linspace(y[:, 1].min(), y[:, 1].max(), 5)

    raw_env = LidarEnv(X, y, seq_len, servo_bins, speed_bins, "path/to/sup_model")  # Update with actual model path
    skrl_env = wrap_env(raw_env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DeviceEnvWrapper(skrl_env, raw_env, device)

    obs_space, act_space = raw_env.observation_space, raw_env.action_space

    class QNetwork(DeterministicMixin, Model):
        def __init__(self):
            Model.__init__(self, obs_space, act_space, device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = SpatioTempDuelingTransformerNet(seq_len, X.shape[2], act_space.n).to(device)

        def compute(self, inputs, role):
            s = inputs["states"]
            if s.dim() == 2:
                s = s.view(s.size(0), *obs_space.shape)
            elif s.dim() == 5:
                s = s.squeeze(0)
            return self.net(s), {}

    q_net, target_net = QNetwork(), QNetwork()
    memory = RandomMemory(memory_size=100000, num_envs=1, device=device)

    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg.update({
        "learning_rate": 5e-5,
        "discount_factor": 0.99,
        "batch_size": 64,
        "update_frequency": 4,
        "device": str(device),
        "target_q_network": {"update_interval": 1000}
    })

    agent = DQN(models={"q_network": q_net, "target_q_network": target_net},
                memory=memory, cfg=cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    trainer = SequentialTrainer(
        cfg={"timesteps": 200000, "headless": True},
        env=env, agents=agent
    )
    trainer.train()

    os.makedirs("Models", exist_ok=True)
    agent.save("Models/dqn_model")
    memory.save("Models/dqn_memory")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()