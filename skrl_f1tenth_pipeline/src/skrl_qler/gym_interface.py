"""
This file implements an end-to-end pipeline for training a policy using the SKRL framework with the F1Tenth gym environment.
"""

from pathlib import Path
import os
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from skrl.models.torch import Model, DeterministicMixin
from skrl.agents.torch.dqn import DQN
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch.random import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from train import SpatioTempDuelingTransformerNet, load_all, LidarEnv

# Configuration
DATA_NPZ = "data/pure_pursuit_data.npz"
SEQ_LEN = 5
SERVO_BINS = 7
SPEED_BINS = 5
TIMESTEPS = 200_000
BATCH_SIZE = 64
CAPACITY = 100_000

# Load data
X, y = load_all([DATA_NPZ], SEQ_LEN)
sb = np.linspace(y[:, 0].min(), y[:, 0].max(), SERVO_BINS)
sp = np.linspace(y[:, 1].min(), y[:, 1].max(), SPEED_BINS)

# Create environment
raw_env = LidarEnv(X, y, SEQ_LEN, sb, sp, "path/to/sup_model")
skrl_env = wrap_env(raw_env)

# Define device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Define QNetwork
class QNetwork(DeterministicMixin, Model):
    def __init__(self):
        Model.__init__(self, raw_env.observation_space, raw_env.action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = SpatioTempDuelingTransformerNet(SEQ_LEN, X.shape[2], raw_env.action_space.n).to(device)

    def compute(self, inputs, role):
        s = inputs["states"]
        return self.net(s), {}

# Initialize models and memory
q_net = QNetwork()
target_net = QNetwork()
memory = RandomMemory(memory_size=CAPACITY, num_envs=1, device=device)

# DQN configuration
cfg = {
    "learning_rate": 5e-5,
    "discount_factor": 0.99,
    "batch_size": BATCH_SIZE,
    "update_frequency": 4,
    "device": str(device),
    "target_q_network": {"update_interval": 1000}
}

# Create agent
agent = DQN(models={"q_network": q_net, "target_q_network": target_net},
            memory=memory, cfg=cfg,
            observation_space=raw_env.observation_space,
            action_space=raw_env.action_space, device=device)

# Trainer
trainer = SequentialTrainer(cfg={"timesteps": TIMESTEPS, "headless": True}, env=skrl_env, agents=agent)
trainer.train()

# Save models
os.makedirs("Models", exist_ok=True)
agent.save("Models/dqn_model")
memory.save("Models/dqn_memory")
print("Training complete. Models saved.")