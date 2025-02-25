import numpy as np
from gymnasium import spaces
from collections import deque

from .multiplecombat_with_missile_task import Scenario2, Scenario3
from ..reward_functions import AltitudeReward, CombatGeometryReward, EventDrivenReward, GunBEHITReward, GunTargetTailReward, \
    GunWEZReward, GunWEZDOTReward, PostureReward, RelativeAltitudeReward, HeadingReward, MissilePostureReward, ShootPenaltyReward

from ..utils.utils import LLA2NEU, get_AO_TA_R

class Scenario2_NvN(Scenario2):
    def __init__(self, config: str):
        Scenario2.__init__(self, config)
        self.reward_functions = [
            AltitudeReward(self.config),
            CombatGeometryReward(self.config),
            EventDrivenReward(self.config),
            GunBEHITReward(self.config),
            GunTargetTailReward(self.config),
            GunWEZDOTReward(self.config),
            GunWEZReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
            MissilePostureReward(self.config),
            ShootPenaltyReward(self.config),
        ]
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []

    def load_observation_space(self):
        num_ego_obs = 9
        num_partners = len(getattr(self.config, 'aircraft_configs', 4)) / 2
        num_enemies = len(getattr(self.config, 'aircraft_configs', 4)) / 2 # have to change... DEBUG
        num_partners_obs = 6 * num_partners
        num_enemies_obs = 6 * num_enemies
        num_missile_obs = 6
        self.obs_length = int(num_ego_obs + num_partners_obs + num_enemies_obs + num_missile_obs)
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def get_obs(self, env, agent_id):
        offset = 0
        norm_obs = np.zeros(self.obs_length)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        num_partner = len(env.agents[agent_id].partners)
        num_enemies = len(env.agents[agent_id].enemies)
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        offset = 8
        # (2) relative partner info        
        for i in range(num_partner):
            partner_obs_list = np.array(env.agents[agent_id].partners[i].get_property_values(self.state_var))
            partner_cur_ned = LLA2NEU(*partner_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            partner_feature = np.array([*partner_cur_ned, *partner_obs_list[6:9]])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, partner_feature, return_side=True)
            norm_obs[offset+1] = (partner_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (partner_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
            
        # (3) relative enemies info
        for i in range(num_enemies):
            enemies_obs_list = np.array(env.agents[agent_id].enemies[i].get_property_values(self.state_var))
            enemies_cur_ned = LLA2NEU(*enemies_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            enemies_feature = np.array([*enemies_cur_ned, *enemies_obs_list[6:9]])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enemies_feature, return_side=True)
            norm_obs[offset+1] = (enemies_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (enemies_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6

        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs
    
class Scenario3_NvN(Scenario3):
    def __init__(self, config: str):
        Scenario2.__init__(self, config)
        self.reward_functions = [
            AltitudeReward(self.config),
            CombatGeometryReward(self.config),
            EventDrivenReward(self.config),
            GunBEHITReward(self.config),
            GunTargetTailReward(self.config),
            GunWEZDOTReward(self.config),
            GunWEZReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
            MissilePostureReward(self.config),
            ShootPenaltyReward(self.config),
        ]
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []

    def load_observation_space(self):
        num_ego_obs = 9
        num_partners = len(getattr(self.config, 'aircraft_configs', 4)) / 2
        num_enemies = len(getattr(self.config, 'aircraft_configs', 4)) / 2 # have to change... DEBUG
        num_partners_obs = 6 * num_partners
        num_enemies_obs = 6 * num_enemies
        num_missile_obs = 6
        self.obs_length = int(num_ego_obs + num_partners_obs + num_enemies_obs + num_missile_obs)
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def get_obs(self, env, agent_id):
        offset = 0
        norm_obs = np.zeros(self.obs_length)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        num_partner = len(env.agents[agent_id].partners)
        num_enemies = len(env.agents[agent_id].enemies)
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        offset = 8
        # (2) relative partner info        
        for i in range(num_partner):
            partner_obs_list = np.array(env.agents[agent_id].partners[i].get_property_values(self.state_var))
            partner_cur_ned = LLA2NEU(*partner_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            partner_feature = np.array([*partner_cur_ned, *partner_obs_list[6:9]])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, partner_feature, return_side=True)
            norm_obs[offset+1] = (partner_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (partner_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
            
        # (3) relative enemies info
        for i in range(num_enemies):
            enemies_obs_list = np.array(env.agents[agent_id].enemies[i].get_property_values(self.state_var))
            enemies_cur_ned = LLA2NEU(*enemies_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            enemies_feature = np.array([*enemies_cur_ned, *enemies_obs_list[6:9]])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enemies_feature, return_side=True)
            norm_obs[offset+1] = (enemies_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (enemies_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6

        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs