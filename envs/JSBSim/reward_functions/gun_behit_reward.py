import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
FT2METERS = 1 / 3.28084

class GunBEHITReward(BaseRewardFunction):
    """
    Gun_BEHIT_reward
    - Penalize being within enemy's gun range
    - fixed an error in ppt
    NOTE:
    - Only support one-to-one environments. -? by sungil
    """
    def __init__(self, config):
        super().__init__(config)
        #related to 'render_items' in base class, constructing the reward trajectory
        # if epsode == '1v1':
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_behit']] 

        # if epsode == '2v2':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_behit1','_behit2']]

        # if epsode == '4v4':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_behit1','_behit2','_behit3','_behit4']]

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
        behit_rewards = ()
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])


        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            if (R >= 500 * FT2METERS) & (R <= 3000 * FT2METERS) & (AO >= 179 * math.pi / 180):
                behit_reward = -5
            else:
                behit_reward = 0
            behit_rewards = behit_rewards + (behit_reward,)
            new_reward += behit_reward

        return self._process(new_reward, agent_id, behit_rewards)