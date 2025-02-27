import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np


class HeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]
        self.last_roll_rad = {}
        self.last_roll_rate = {}
        self.last_pitch_rate = {}

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
            
        # target_altitude는 85,000이 한계
        """
        target_manner = 'roll_pitch_rate'  # 'roll_pitch_rate', 'roll_rad', 'roll_rate', 'roll_rad_rate'
        roll_lambda = 1.0

        if (target_manner in ['roll_rad', 'roll_rate', 'roll_rad_rate', 'roll_pitch_rate']) and (env.current_step > 1): # 이전 타임 스텝의 roll 값과 현재 타임 스텝의 롤 값의 차이만큼 페널티
            roll_rad_r = -np.abs(env.agents[agent_id].get_property_value(c.attitude_roll_rad) - self.last_roll_rad[agent_id])*roll_lambda
            roll_rate_r = -np.abs(env.agents[agent_id].get_property_value(c.velocities_p_rad_sec) - self.last_roll_rate[agent_id])*roll_lambda
            pitch_rate_r = -np.abs(env.agents[agent_id].get_property_value(c.velocities_q_rad_sec) - self.last_pitch_rate[agent_id])*roll_lambda
        
        heading_error_scale = 5.0  # degrees
        heading_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_heading) / heading_error_scale) ** 2))
        # sim.get_property_value(ExtraCatalog.target_heading_deg) - sim.get_property_value(JsbsimCatalog.attitude_psi_deg)

        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))
        # (sim.get_property_value(ExtraCatalog.target_altitude_ft) - sim.get_property_value(JsbsimCatalog.position_h_sl_ft)) * 0.3048

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_velocities_u) / speed_error_scale) ** 2))
        # (sim.get_property_value(ExtraCatalog.target_velocities_u_mps) - sim.get_property_value(ExtraCatalog.velocities_u_mps))

        if (target_manner == 'roll_rad') and (env.current_step > 1):
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + roll_rad_r
            # print(f'heading_r: {heading_r}, alt_r: {alt_r}, roll_r: {roll_r}, speed_r: {speed_r}, roll_rad_r: {roll_rad_r}')
        elif (target_manner == 'roll_rate') and (env.current_step > 1):
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + roll_rate_r
        elif (target_manner == 'roll_rad_rate') and (env.current_step > 1):
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + roll_rad_r + roll_rate_r
        elif (target_manner == 'roll_pitch_rate') and (env.current_step > 1):
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + roll_rate_r + pitch_rate_r
        else:
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)


        self.last_roll_rad[agent_id] = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        self.last_roll_rate[agent_id] = env.agents[agent_id].get_property_value(c.velocities_p_rad_sec)
        self.last_pitch_rate[agent_id] = env.agents[agent_id].get_property_value(c.velocities_q_rad_sec)
        
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))