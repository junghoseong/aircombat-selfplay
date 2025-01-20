import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
FT2METERS = 1 / 3.28084

class GunTargetTailReward(BaseRewardFunction):
    """
    Gun_TargetTail_reward
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
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_targettail']] 

        # if epsode == '2v2':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_targettail1','_targettail2']]

        # if epsode == '4v4':
        #     self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_targettail1','_targettail2','_targettail3','_targettail4']]

    def reset(self, task, env):
        self.prev_d_tails = []
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is to encourage catching tail of enemy(between 3000-5000ft)
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
        targettail_rewards = ()
        d_tails=[]
        i = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])


        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ##############
            if (R >= 3000 * FT2METERS) & (R <= 5000 * FT2METERS):
                d_tail=R * math.sin(TA)#using the arc
            elif (R <= 3000 * FT2METERS):
                d_tail=math.sqrt(R**2 + (3000*FT2METERS)**2 - 2*R*(3000*FT2METERS)*math.cos(TA)) #using 2nd cosine law
            else:
                d_tail=math.sqrt(R**2 + (5000*FT2METERS)**2 - 2*R*(5000*FT2METERS)*math.cos(TA))

            d_tails.append(d_tail)
            if len(self.prev_d_tails) == 0:
                self.prev_d_tails.append(d_tail)
            else:
                self.prev_d_tails.append(d_tails[i-1])

            targettail_reward = -1/hz * math.tanh((d_tails[i] - self.prev_d_tails[i]) / math.sqrt(R))
            ###############
            targettail_rewards = targettail_rewards + (targettail_reward,)
            new_reward += targettail_reward

            i += 1

        return self._process(new_reward, agent_id, targettail_rewards)