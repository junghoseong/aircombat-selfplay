import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
FT2METERS = 1 / 3.28084

class GunWEZDOTReward(BaseRewardFunction):
    """
    Gun_WEZDOT_reward
    - Encourage getting close to the enemy fighter at a appropriate angle
    - Could be triggered by WEZ? 
    NOTE:
    - Only support one-to-one environments. because of the reward_item_name. 
    - TODO: by the specified episode number, use different reward trajectory names
    - 
    """
    def __init__(self, config):
        super().__init__(config)
        #related to 'render_items' in base class, constructing the reward trajectory
        # if epsode == '1v1':
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wezdot']] 

        # if epsode == '2v2':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wezdot1','_wezdot2']]

        # if epsode == '4v4':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wezdot1','_wezdot2','_wezdot3','_wezdot4']]

    def reset(self, task, env):
        self.prev_d_targets = []
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is to encourage constant aiming to the enemy
        Args:
            task: task instance
            env: environment instance
        Returns:
            (float): reward
        """
        #################
        hz = 60
        #################
        new_reward = 0
        wezdot_rewards = ()
        d_targets=[]
        i = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])


        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ###############
            if (R >= 500 * FT2METERS) & (R <= 3000 * FT2METERS):
                d_target = R * math.sin(AO) #using the arc
            else:
                d_target = math.sqrt(R**2 + (3000*FT2METERS)**2 - 2*R*(3000*FT2METERS)*math.cos(AO))  #using 2nd cosine law

            d_targets.append(d_target)
            if len(self.prev_d_targets) == 0:
                self.prev_d_targets.append(d_target)
            else:
                self.prev_d_targets.append(d_targets[i-1])

            wezdot_reward = -1/hz * math.tanh((d_targets[i] - self.prev_d_targets[i]) / math.sqrt(R))
            ###############
            wezdot_rewards = wezdot_rewards + (wezdot_reward,)
            new_reward += wezdot_reward

            i += 1

        return self._process(new_reward, agent_id, wezdot_rewards)