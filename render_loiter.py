from abc import ABC
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.model.baseline import BaselineAgent, PursueAgent, ManeuverAgent

env = SingleControlEnv(config_name='singlecontrol/approach')
obs = env.reset()
env.render(filepath="loiter.txt.acmi")
agent0 = ManeuverAgent(agent_id=0, maneuver='triangle')
agent1 = PursueAgent(agent_id=1)
reward_list = []
while True:
    action0 = agent0.get_action(env, env.task)
    # action1 = agent1.get_action(env, env.task)
    # actions = [action0, action1]
    actions = [action0]
    obs, reward, done, info = env.step(actions)
    env.render(filepath="loiter.txt.acmi")
    reward_list.append(reward[0])
    if np.array(done).all():
        print(info)
        break
# plt.plot(reward_list)
# plt.savefig('rewards.png')