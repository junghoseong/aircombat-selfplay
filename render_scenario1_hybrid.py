import numpy as np
import torch
from envs.JSBSim.envs import HybridSingleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo_hybrid.ppo_actor import PPOActor
from envs.JSBSim.model.baseline import BaselineAgent, PursueAgent
from algorithms.utils.hybrid_action_embedder import Action_representation
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
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
        self.ppo_epoch = 5
        self.num_mini_batch = 32
        self.data_chunk_length = 60
    
def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 1
render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0
experiment_name = "Scenario1"

env = HybridSingleCombatEnv("scenario1/scenario1_hybrid")

env.seed(0)
args = Args()
action_rep = Action_representation(args,env.observation_space,env.rnn_actor_space,env.discrete_action_space,env.continuous_action_space,env.continuous_embedding_space,device = torch.device("cuda"))
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
# enm_policy = PursueAgent(agent_id=1)
vae_loaded_state_dict = torch.load('./checkpoint/vae_latest.pt')

# 키 변경 예시
new_state_dict = {}
for key in vae_loaded_state_dict.keys():
    new_key = 'vae.' + key  # 'vae.' 접두사 추가
    new_state_dict[new_key] = vae_loaded_state_dict[key]
action_rep.load_state_dict(new_state_dict)
ego_policy.load_state_dict(torch.load("./checkpoint/actor_latest.pt"))
ego_policy.eval()

for name, param in ego_policy.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
    
print("Start render")
obs = env.reset()
if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents, 1))
while True:
    ego_actions, _, ego_rnn_states = ego_policy(obs, ego_rnn_states, masks, deterministic=True)
    # print(ego_actions)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    # enm_actions = enm_policy.get_action(env,env.task,0)
    # Obser reward and next obs
    obs, rewards, dones,_,_, infos = env.step(obs, ego_actions,ego_rnn_states,action_rep)
    rewards = rewards[:num_agents, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    #enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

print(episode_rewards)