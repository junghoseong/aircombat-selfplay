import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
FT2METERS = 1 / 3.28084

class CombatGeometryReward(BaseRewardFunction):
    """
    Combat_Geometry_reward
    - Encourage getting close to enemy fighter's tail in the scope of angles
    NOTE:
    - Only support one-to-one environments. because of the reward_item_name. 
    - TODO: by the specified episode number, use different reward trajectory names
    - 
    """
    def __init__(self, config):
        super().__init__(config)
        #related to 'render_items' in base class, constructing the reward trajectory
        # if epsode == '1v1':
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_cg']] 

        # if epsode == '2v2':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_cg1','_cg2']]

        # if epsode == '4v4':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_cg1','_cg2','_cg3','_cg4']]

    def reset(self, task, env):
        self.prev_AOs = []
        self.prev_TAs = []
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is built linear to deviation of angles
        Args:
            task: task instance
            env: environment instance
        Returns:
            (float): reward
        """
        new_reward = 0
        cg_rewards = ()
        AOs = []
        TAs = []
        i = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            AOs.append(AO)
            TAs.append(TA)
            if len(self.prev_AOs) == 0:
                self.prev_AOs.append(AO)
                self.prev_TAs.append(TA)
            else:
                self.prev_AOs.append(AOs[i-1])
                self.prev_TAs.append(TAs[i-1])

            cg_reward = -(AOs[i] - self.prev_AOs[i]) - (TAs[i] - self.prev_TAs[i])
            cg_rewards = cg_rewards + (cg_reward,)
            new_reward += cg_reward

        return self._process(new_reward, agent_id, cg_rewards)