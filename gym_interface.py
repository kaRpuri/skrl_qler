"""
train.py – end‑to‑end pipeline with plotting


 • Converts ROS2 bag => .npz if needed
 • trains Recurrent Lidar Net (PyTorch)
 • trains PPO (SB3) with friction‑randomisation + progress reward
 • logs and plots RLN loss, episode reward/length, friction histogram
"""
################ CONFIG BLOCK ################################
DATA_NPZ          = "data/pure_pursuit_data.npz"
BAG_DIR           = "data/pure_pursuit_data/"
LIDAR_BEAMS       = 1080
TOTAL_RLN_EPOCHS  = 25
FREEZE_RLN        = True
TOTAL_PPO_STEPS   = 4_000_000
F1_MAP_YAML       = "maps/Levine/levine_2nd_floor.yaml"
TIMESTEP          = 0.01
N_ENVS            = 8
PLOT_DIR          = "plots"
################## END CONFIG BLOCK ############################

from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np, torch as th, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import trange
import matplotlib.pyplot as plt                              

############### 0. dataset preparation ###############
if not Path(DATA_NPZ).exists():
    import importlib, sys
    sys.path.append(".")
    bag2npz = importlib.import_module("bag2npz")             
    bag2npz.BAG_DIR, bag2npz.OUT_FILE = Path(BAG_DIR), Path(DATA_NPZ)
    bag2npz.convert()

d = np.load(DATA_NPZ)
scan, speed, steer = d["lidar"], d["speed"], d["steer"]

####################### 1. RLN #########################
class RLN(nn.Module):
    def __init__(self, beams=LIDAR_BEAMS, latent=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1,16,5,padding=2), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.ReLU(),
            nn.Flatten())
        self.gru = nn.GRU(beams*32+2, latent, num_layers=2)
        self.dec = nn.Linear(latent, beams)
          
    def forward(self, s, v, d, h=None, teacher=None, ss_p=1.):
        x = self.enc(s.unsqueeze(1));  u = th.stack([v,d],1)
        y,h = self.gru(th.cat([x,u],1).unsqueeze(0), h);  pred = self.dec(y.squeeze(0))
        next_in = (teacher if self.training and teacher is not None and
                   th.rand(())<ss_p else pred.detach())
        return pred, next_in, h

device = "cuda" if th.cuda.is_available() else "cpu"
rln   = RLN().to(device)
opt   = th.optim.Adam(rln.parameters(), 3e-4)
loss_fn = nn.MSELoss()
ds = TensorDataset(th.tensor(scan[:-1]), th.tensor(speed[:-1]),
                   th.tensor(steer[:-1]), th.tensor(scan[1:]))
dl = DataLoader(ds, 256, True)

epoch_loss = []
for ep in trange(TOTAL_RLN_EPOCHS, desc="Train RLN"):
    rln.train();  running = 0;  ss_p = max(.5, 1-ep/TOTAL_RLN_EPOCHS)
    for s,v,d_,s_next in dl:
        s,v,d_,s_next = [t.float().to(device) for t in (s,v,d_,s_next)]
        pred,_,_ = rln(s,v,d_,teacher=s_next,ss_p=ss_p)
        loss = loss_fn(pred,s_next);  opt.zero_grad();  loss.backward()
        nn.utils.clip_grad_norm_(rln.parameters(),5);  opt.step()
        running += loss.item()
    epoch_loss.append(running/len(dl))
th.save(rln.state_dict(),"rln.pt")

########################### 2. env wrappers ###########################

# ObservationWrapper to strip agent ID for single-agent use
class DirectFlatObs(ObservationWrapper):
    """
    f1tenth_gym  ⋄  obs_type='direct'
    Dict(agent_0 = Dict(scan, std_state, ...))  =>  Dict(scan, linear_vel, steering_angle)
    """

    def __init__(self, env):
        super().__init__(env)
        agent_key  = next(iter(env.observation_space))          # 'agent_0'
        inner      = env.observation_space[agent_key]

        self._agent_key = agent_key
        self.observation_space = spaces.Dict({
            "scan":           inner["scan"],
            "linear_vel":     spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "steering_angle": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        })
    
    def observation(self, obs):
        ego       = obs[self._agent_key]
        std_state = ego["std_state"]           # plain np.ndarray, shape (7,)

        vx    = float(std_state[3])            # v_x  (forward speed, m/s)
        delta = float(std_state[6])            # steering angle, rad

        return {
            "scan":           np.asarray(ego["scan"], dtype=np.float32),
            "linear_vel":     np.asarray([vx],   dtype=np.float32),
            "steering_angle": np.asarray([delta],dtype=np.float32),
        }


class FrictionRand(gym.Wrapper):
    def __init__(self, env, lo=.7, hi=1.2): super().__init__(env); self.lo,self.hi=lo,hi
    def reset(self,*args,**kwargs):
        obs,info = self.env.reset(*args, **kwargs)
        self.mu = float(np.random.uniform(self.lo,self.hi)
                        )

        #self.env.simulator.update_params({'mu':self.mu})
        self.env.unwrapped.sim.params["mu"] = self.mu

        info["mu"]=self.mu

        return obs,info

class ProgressReward(gym.Wrapper):
    def __init__(self,env,w_p=250,w_v=2,w_st=0.2,crash=1000):
        super().__init__(env); self.w_p,self.w_v,self.w_st,self.crash = w_p,w_v,w_st,crash
    def _s(self):
        # get_frenet_s

        #x,y=self.env.simulator.agent_poses[0][:2]
        x, y = self.env.unwrapped.sim.agent_poses[0][:2]

        #return self.env.track.get_progress_along_centerline(np.array([x,y]))

        #return self.env.unwrapped.track.get_progress_along_centerline(np.array([x, y]))

        # centre-line spline => fast arclength estimator returns (s, ey)
        s, _ = (
            self.env.unwrapped.track.centerline.spline.calc_arclength_inaccurate(x, y)        # ← returns cumulative s
        )
        return float(s)

    def reset(self,**kw): obs,info=self.env.reset(**kw); self.prev=self._s(); return obs,info
    
    def step(self,act):
        obs,_,done,trunc,info=self.env.step(act)
        s = self._s(); ds= s-self.prev; self.prev=s
        speed = obs["linear_vel"][0]; steer = abs(act[0][0])
        rew = self.w_p*ds + self.w_v*speed - self.w_st*steer
        if info.get("collisions",0): rew -= self.crash
        return obs,rew,done,trunc,info

##################### 3. PPO feature extractor ######################

class RLNExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, freeze=FREEZE_RLN):
        super().__init__(obs_space, features_dim=LIDAR_BEAMS)   # 1080
        self.rln = RLN().to(device)
        self.rln.load_state_dict(th.load("rln.pt", map_location=device))
        # if freeze:
        #     for p in self.rln.parameters():
        #         p.requires_grad_(False)

    def forward(self, obs):
        scan  = obs["scan"]
        vel  = obs["linear_vel"].squeeze(-1)
        steering_angle  = obs["steering_angle"].squeeze(-1)
        pred, _, _ = self.rln(scan, vel, steering_angle)       # shape (B,1080)
        return pred

"""
class RLNExtractor(BaseFeaturesExtractor):
    def __init__(self,obs_space,latent=256,freeze=FREEZE_RLN):
        super().__init__(obs_space,latent)
        self.rln = RLN(latent=latent).to(device)
        self.rln.load_state_dict(th.load("rln.pt",map_location=device))
        if freeze:  [p.requires_grad_(False) for p in self.rln.parameters()]
    def forward(self,obs):
        s=obs["scan"]
        v=obs["linear_vel"].squeeze(-1)
        d_=obs["steering_angle"].squeeze(-1)

        pred,_,_ = self.rln(s,v,d_)
        return pred
"""

############## 4.callback #########################
class EpisodeStats(BaseCallback):
    def __init__(self): super().__init__(); self.rew,self.len,self.mu=[],[],[]
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.rew.append(info["episode"]["r"])
                self.len.append(info["episode"]["l"])
                self.mu.append(info.get("mu",np.nan))
        return True

# ##################### 5. make vector env #######################
def make_env(rank):
    def _init():
        #e=gym.make("f110_gym:f110-v0",map_path=F1_MAP_YAML,
        #           timestep=TIMESTEP,num_agents=1,render_mode=None)

        e = gym.make("f1tenth_gym:f1tenth-v0", map_path=F1_MAP_YAML,
                   timestep=TIMESTEP, num_agents=1, render_mode=None, obs_type="direct")    
        #e = StripAgentID(e);
        e = DirectFlatObs(e); 

        obs, _ = e.reset(seed=0)
        assert set(obs.keys()) == {"scan", "linear_vel", "steering_angle"}
        print("✓ flattened obs shapes:",
            obs["scan"].shape, obs["linear_vel"].shape, obs["steering_angle"].shape)
        
        e= FrictionRand(e); 
        e=ProgressReward(e); 
        e.reset(seed=rank); 
        return e
    return _init


#vec = VecMonitor(SubprocVecEnv([make_env(i) for i in range(N_ENVS)]))

from stable_baselines3.common.vec_env import DummyVecEnv
vec = VecMonitor(DummyVecEnv([make_env(0)]))


######################## 6. PPO ################################
agent = PPO("MultiInputPolicy", vec,
            learning_rate=1e-4, n_steps=1024//N_ENVS, batch_size=256,
            gamma=.99, gae_lambda=.9, clip_range=.2, max_grad_norm=.5,
            policy_kwargs=dict(features_extractor_class=RLNExtractor,
                               features_extractor_kwargs=dict(freeze=FREEZE_RLN),
                               net_arch=[256,128]),
            device=device, verbose=1,
            tensorboard_log="tb")                          # SB3 TB integration :contentReference[oaicite:3]{index=3}
ep_cb = EpisodeStats()
agent.learn(TOTAL_PPO_STEPS, callback=ep_cb)
agent.save("rln_ppo_agent")

#################### 7. plotting #########################
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_rln_loss():
    plt.figure(); plt.plot(epoch_loss); plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title("RLN training loss"); plt.grid(True); plt.savefig(f"{PLOT_DIR}/rln_loss.png")

def plot_ppo_stats():
    if not ep_cb.rew: return
    r=np.array(ep_cb.rew); l=np.array(ep_cb.len); m=np.array(ep_cb.mu)
    # reward curve
    plt.figure(); plt.plot(r,alpha=.3,label="episode"); 
    if len(r)>100: plt.plot(np.convolve(r,np.ones(100)/100,'valid'),label="100‑ep mean")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("PPO episode reward");
    plt.legend(); plt.grid(True); plt.savefig(f"{PLOT_DIR}/ppo_episode_reward.png")

    # length
    plt.figure(); plt.plot(l,alpha=.7); plt.xlabel("Episode"); plt.ylabel("Length");
    plt.title("Episode length"); plt.grid(True); plt.savefig(f"{PLOT_DIR}/ppo_episode_length.png")

    # friction histogram
    plt.figure(); plt.hist(m[~np.isnan(m)],bins=20); plt.xlabel("friction value");
    plt.title("Surface‑friction (friction) distribution"); plt.grid(True);
    plt.savefig(f"{PLOT_DIR}/ppo_friction_hist.png")

plot_rln_loss(); plot_ppo_stats()
print(f"✓ RLN + PPO training done.  Plots saved in {PLOT_DIR}/, model in rln_ppo_agent.zip")