import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
FT2METERS = 1 / 3.28084

class GunWEZReward(BaseRewardFunction):
    """
    Gun_WEZ_reward
    - Encourage getting close to the enemy fighter at a appropriate angle
    NOTE:
    - Only support one-to-one environments. because of the reward_item_name. 
    - TODO: by the specified episode number, use different reward trajectory names
    - 
    """
    def __init__(self, config):
        super().__init__(config)
        #related to 'render_items' in base class, constructing the reward trajectory
        # if epsode == '1v1':
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wez']] 

        # if epsode == '2v2':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wez1','_wez2']]

        # if epsode == '4v4':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_wez1','_wez2','_wez3','_wez4']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built linear relative to the distance with enemy
        Args:
            task: task instance
            env: environment instance
        Returns:
            (float): reward
        """
        new_reward = 0
        wez_rewards = ()
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])


        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            if (R >= 500 * FT2METERS) & (R <= 3000 * FT2METERS) & (AO <= 1 * math.pi / 180):
                wez_reward = 5 + 5 * (3000 * FT2METERS - R) / (2500 * FT2METERS)
            else:
                wez_reward = 0
            wez_rewards = wez_rewards + (wez_reward,)
            new_reward += wez_reward

        return self._process(new_reward, agent_id, wez_rewards)