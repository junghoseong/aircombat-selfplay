import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.model.baseline import PursueAgent, ManeuverAgent
import logging
logging.basicConfig(level=logging.DEBUG)

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class Args:
    def __init__(self):
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0
experiment_name = "vs_pursue_with_missile"

env = SingleCombatEnv("scenario1/scenario1_curriculum_vs_pursue")

env.seed(10)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.load_state_dict(torch.load("./checkpoint/actor_latest.pt"))

enm_policy = PursueAgent(agent_id=1)

for name, param in ego_policy.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
    
print("Start render")
for i in [0, 30, 60, 120, 150, 180]:
    obs = env.reset()
    env.reset_simulators_curriculum(i)
    if render:
        env.render(mode='txt', filepath=f'curriculum_{i}_vs_pursue.acmi')
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions = enm_policy.get_action(env, env.task)
        enm_actions = np.pad(enm_actions, (0, 3), 'constant', constant_values=0).reshape(1, -1)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        obs, rewards, dones, infos = env.step(actions)
        rewards = rewards[:num_agents // 2, ...]
        episode_rewards += rewards
        if render:
            env.render(mode='txt', filepath=f'curriculum_{i}_vs_pursue.acmi')
        if dones.all():
            print(infos)
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    print(episode_rewards)
