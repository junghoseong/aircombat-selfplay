import numpy as np
from gymnasium import spaces
from collections import deque
import copy

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .multiplecombat_task import MultipleCombatTask, HierarchicalMultipleCombatTask
from .multiplecombat_with_missile_task import MultipleCombatShootMissileTask
from ..reward_functions import AltitudeReward, CombatGeometryReward, EventDrivenReward, GunBEHITReward, GunTargetTailReward, \
    GunWEZReward, GunWEZDOTReward, PostureReward, RelativeAltitudeReward, HeadingReward, MissilePostureReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator, AIM_9M, AIM_120B, ChaffSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R

class Scenario2(HierarchicalMultipleCombatTask, MultipleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalMultipleCombatTask.__init__(self, config)
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

    def load_observation_space(self):
        return MultipleCombatShootMissileTask.load_observation_space(self)

    @property
    def num_agents(self) -> int:
        return 4        
    
    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.MultiDiscrete([2, 2, 2, 2])])

    def get_obs(self, env, agent_id):
        return MultipleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            #self._shoot_action[agent_id] = action[-4:]
            idx = env.enm_ids.index(agent_id)
            agent = self.baseline_agent[idx]
            self._shoot_action[agent_id] = [0, 0, 0, 0]
            action = agent.get_action(env, env.task, idx)
            action = agent.normalize_action(env, agent_id, action)
            if self.use_artillery:
                self._shoot_action[agent_id] = [1, 1, 1, 1]
            return action
        elif agent_id in env.enm_ids:
            self._shoot_action[agent_id] = action[-4:]
        if agent_id in env.ego_ids:
            self._shoot_action[agent_id] = action[-4:] # ADD, check for enm ids!
        return HierarchicalMultipleCombatTask.normalize_action(self, env, agent_id, action[:-4].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.remaining_missiles_AIM_9M = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_missiles_AIM_120B = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}        
        self.remaining_gun = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_chaff_flare = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}

        MultipleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        MultipleCombatShootMissileTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag_gun = agent.is_alive and self._shoot_action[agent_id][0] and self.remaining_gun[agent_id] > 0
            shoot_flag_AIM_9M = agent.is_alive and self._shoot_action[agent_id][1] and self.remaining_missiles_AIM_9M[agent_id] > 0
            shoot_flag_AIM_120B = agent.is_alive and self._shoot_action[agent_id][2] and self.remaining_missiles_AIM_120B[agent_id] > 0
            shoot_flag_chaff_flare = agent.is_alive and self._shoot_action[agent_id][3] and self.remaining_chaff_flare[agent_id] > 0

            if shoot_flag_gun and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage gun duration
                avail, enemy = self.a2a_launch_available(agent, agent_id, env)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    #print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
                    self.remaining_gun[agent_id] -= 1
            
            if shoot_flag_AIM_120B and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage long-range missile duration
                avail, _ = self.a2a_launch_available(agent, agent_id, env)
                if avail[1]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_120B[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_120B.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-120B"))
                    self.remaining_missiles_AIM_120B[agent_id] -= 1

            if shoot_flag_AIM_9M and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage middle-range missile duration
                avail, _ = self.a2a_launch_available(agent, agent_id, env)
                if avail[2]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_9M[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_9M.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-9M"))
                    self.remaining_missiles_AIM_9M[agent_id] -= 1
            
            if shoot_flag_chaff_flare and (self.agent_last_shot_chaff[agent_id] == 0 or self.agent_last_shot_chaff[agent_id].is_done): # valid condition for chaff: can be bursted after the end of last chaff
                for missiles in env._tempsims.values():
                    if missiles.target_aircraft == agent and missiles.target_distance < 1000:
                        new_chaff_uid = agent_id + str(self.remaining_chaff_flare[agent_id] + 10)
                        self.agent_last_shot_chaff[agent_id] = env.add_chaff_simulator(
                            ChaffSimulator.create(parent=agent, uid=new_chaff_uid, chaff_model="CHF"))
                        self.remaining_chaff_flare[agent_id] -= 1
                        
    def a2a_launch_available(self, agent, agent_id, env):
        ret = [False, False, False]
        munition_info = {
            # KM / DEG
            "GUN": {"dist" : 3, "AO" : 5},
            "AIM-120B" : {"dist" : 37, "AO" : 90},
            "AIM-9M" : {"dist" : 7, "AO" : 90},
        }
        rad_missile_name_list = ["AIM-120B"]
        
        enemy = self.get_target(agent)
        if not enemy.is_alive:
            return ret, enemy
        target = enemy.get_position() - agent.get_position()
        heading = agent.get_velocity()
        distance = np.linalg.norm(target)
        attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
        
        if distance / 1000 < munition_info["GUN"]["dist"] and attack_angle < munition_info["GUN"]["AO"]:
            ret[0] = True 
        
        if distance / 1000 < munition_info["AIM-120B"]["dist"] and attack_angle < munition_info["AIM-120B"]["AO"]:
            ret[1] = True 
        
        if distance / 1000 < munition_info["AIM-9M"]["dist"] and attack_angle < munition_info["AIM-9M"]["AO"]:
            ret[2] = True
            
        if self.use_baseline == True and agent_id in env.enm_ids:
            ret[1] = False
            if distance / 1000 < munition_info["AIM-120B"]["dist"] and attack_angle < munition_info["AIM-120B"]["AO"]/2:
                ret[1] = True
        
        return ret, enemy
        
    def get_target(self, agent):
        target_distances = []
        for enemy in agent.enemies:
            target = enemy.get_position() - agent.get_position()
            distance = np.linalg.norm(target)
            target_distances.append(distance)
        return agent.enemies[np.argmax(target_distances)]     
    
class Scenario2_curriculum(Scenario2):
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

        self.done_other = False
        self.done_id = None
        self.success_other = False
        self.info_other = None
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []
        
    def reset(self, env):
        if self.winning_rate >= 0.6 and len(self.record) > 20:
            self.curriculum_angle += 1
            self.record = []
        env.reset_simulators_curriculum(self.curriculum_angle)
        self.success_other = False
        self.done_other = False
        self.done_id = None
        self.info_other = None
        Scenario2.reset(self, env)
    
    def get_termination(self, env, agent_id, info={}):
        # has to check whether both agents are done / success.
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d # if one termination condition is done, it is done
            success = success and s # all termination condition should mark success
            if done and agent_id in env.ego_ids:
                if agent_id != self.done_id and self.done_other == False:
                    self.done_other = True
                    self.done_id = agent_id
                    self.info_other = copy.deepcopy(info)
                    if success:
                        self.success_other = True
                    else:
                        self.success_other = False
                elif agent_id == self.done_id and self.done_other == True:
                    break
                else:
                    if self.success_other or success:
                        self.record.append(1)
                    else:
                        self.record.append(0)
                    if len(self.record) > 20:
                        self.record.pop(0) 
                    self.winning_rate = sum(self.record)/len(self.record)
                    print("current winning rate is {}/{}, curriculum is {}'th stage".format(sum(self.record), len(self.record), self.curriculum_angle))
                break
        return done, info
    
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
            norm_obs[offset+1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        return norm_obs

class Scenario2_NvN_curriculum(Scenario2_NvN):
    def __init__(self, config: str):
        Scenario2_NvN.__init__(self, config)
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

        self.done_other = False
        self.done_id = None
        self.success_other = False
        self.info_other = None
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []
        
    def reset(self, env):
        if self.winning_rate >= 0.6 and len(self.record) > 20:
            self.curriculum_angle += 1
            self.record = []
        env.reset_simulators_curriculum(self.curriculum_angle)
        self.success_other = False
        self.done_other = False
        self.done_id = None
        self.info_other = None
        Scenario2_NvN.reset(self, env)
    
    def get_termination(self, env, agent_id, info={}):
        # has to check whether both agents are done / success.
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d # if one termination condition is done, it is done
            success = success and s # all termination condition should mark success
            if done and agent_id in env.ego_ids:
                if agent_id != self.done_id and self.done_other == False:
                    self.done_other = True
                    self.done_id = agent_id
                    self.info_other = copy.deepcopy(info)
                    if success:
                        self.success_other = True
                    else:
                        self.success_other = False
                elif agent_id == self.done_id and self.done_other == True:
                    break
                else:
                    if self.success_other or success:
                        self.record.append(1)
                    else:
                        self.record.append(0)
                    if len(self.record) > 20:
                        self.record.pop(0) 
                    self.winning_rate = sum(self.record)/len(self.record)
                    print("current winning rate is {}/{}, curriculum is {}'th stage".format(sum(self.record), len(self.record), self.curriculum_angle))
                break
        return done, info   

class Scenario2_RWR(Scenario2_NvN):
    def __init__(self, config: str):
        Scenario2_NvN.__init__(self, config)
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

    def load_observation_space(self):
        num_ego_obs = 11
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
            norm_obs[offset+1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[offset+2] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[offset+3] = ego_AO
            norm_obs[offset+4] = ego_TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        
        norm_obs[offset+1] = 0
        norm_obs[offset+2] = 0
        return norm_obs
    
class Scenario2_RWR_curriculum(Scenario2_RWR):
    def __init__(self, config: str):
        Scenario2_RWR.__init__(self, config)
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

        self.done_other = False
        self.done_id = None
        self.success_other = False
        self.info_other = None
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []
        
    def reset(self, env):
        if self.winning_rate >= 0.6 and len(self.record) > 20:
            self.curriculum_angle += 1
            self.record = []
        env.reset_simulators_curriculum(self.curriculum_angle)
        self.success_other = False
        self.done_other = False
        self.done_id = None
        self.info_other = None
        Scenario2_RWR.reset(self, env)
    
    def get_termination(self, env, agent_id, info={}):
        # has to check whether both agents are done / success.
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d # if one termination condition is done, it is done
            success = success and s # all termination condition should mark success
            if done and agent_id in env.ego_ids:
                if agent_id != self.done_id and self.done_other == False:
                    self.done_other = True
                    self.done_id = agent_id
                    self.info_other = copy.deepcopy(info)
                    if success:
                        self.success_other = True
                    else:
                        self.success_other = False
                elif agent_id == self.done_id and self.done_other == True:
                    break
                else:
                    if self.success_other or success:
                        self.record.append(1)
                    else:
                        self.record.append(0)
                    if len(self.record) > 20:
                        self.record.pop(0) 
                    self.winning_rate = sum(self.record)/len(self.record)
                    print("current winning rate is {}/{}, curriculum is {}'th stage".format(sum(self.record), len(self.record), self.curriculum_angle))
                break
        return done, info
    
class Scenario2_Hybrid(Scenario2_NvN):
    def __init__(self, config: str):
        Scenario2_NvN.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return Scenario2_NvN.load_observation_space(self)

    @property
    def num_agents(self) -> int:
        return 4        
    
    def load_action_space(self):
        # altitude control[-0.1,0.1] + heading control[-pi/6,pi/6] + velocity control[-0.05,0.05] + shoot control
        #self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.MultiDiscrete([2, 2, 2, 2])])
        self.action_space = spaces.Box(low = np.array([-1,-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5]), \
                                        high = np.array([1,1,1,1,1,1,1.5,1.5,1.5,1.5]),\
                                        dtype = np.float64)
        self.discrete_action_space = spaces.MultiDiscrete([2,2,2,2])
        self.continuous_action_space = spaces.Box(low = np.array([-0.1,-np.pi/6,-0.05]),high = np.array([0.1,np.pi/6,0.05]),dtype = np.float64)
        self.discrete_embedding_space = spaces.MultiDiscrete([2,2,2,2])
        self.continuous_embedding_space = spaces.Box(low = -np.ones(6), high = np.ones(6),dtype = np.float64)

    def normalize_action(self, env, agent_id, obs, share_obs, action, action_representation): #must make actionrepresentation to come here
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action_representation.select_discrete_action(action[-4:])
        state = np.concatenate((obs, np.array(share_obs[0]).flatten()))


        return HierarchicalMultipleCombatTask.normalize_action(self, env, agent_id, \
                                                                action_representation.select_continuous_action(state,action[:-4],self._shoot_action[agent_id]).astype(np.float32))#action[:-4].astype(np.int32))type_changed

    def step(self, env):
        MultipleCombatShootMissileTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]            
            shoot_flag_gun = agent.is_alive and self._shoot_action[agent_id][0][0] and self.remaining_gun[agent_id] > 0
            shoot_flag_AIM_9M = agent.is_alive and self._shoot_action[agent_id][0][1] and self.remaining_missiles_AIM_9M[agent_id] > 0
            shoot_flag_AIM_120B = agent.is_alive and self._shoot_action[agent_id][0][2] and self.remaining_missiles_AIM_120B[agent_id] > 0
            shoot_flag_chaff_flare = agent.is_alive and self._shoot_action[agent_id][0][3] and self.remaining_chaff_flare[agent_id] > 0

            if shoot_flag_gun :#and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage gun duration
                avail, enemy = self.a2a_launch_available(agent)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    #print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
                    self.remaining_gun[agent_id] -= 1
            
            if shoot_flag_AIM_120B :#and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage long-range missile duration
                avail, _ = self.a2a_launch_available(agent)
                if avail[1]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_120B[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_120B.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-120B"))
                    self.remaining_missiles_AIM_120B[agent_id] -= 1

            if shoot_flag_AIM_9M :#and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage middle-range missile duration
                avail, _ = self.a2a_launch_available(agent)
                if avail[2]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_9M[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_9M.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-9M"))
                    self.remaining_missiles_AIM_9M[agent_id] -= 1
            
            if shoot_flag_chaff_flare :#and (self.agent_last_shot_chaff[agent_id] == 0 or self.agent_last_shot_chaff[agent_id].is_done): # valid condition for chaff: can be bursted after the end of last chaff
                for missiles in env._tempsims.values():
                    if missiles.target_aircraft == agent and missiles.target_distance < 1000:
                        new_chaff_uid = agent_id + str(self.remaining_chaff_flare[agent_id] + 10)
                        self.agent_last_shot_chaff[agent_id] = env.add_chaff_simulator(
                            ChaffSimulator.create(parent=agent, uid=new_chaff_uid, chaff_model="CHF"))
                        self.remaining_chaff_flare[agent_id] -= 1
                        
    def a2a_launch_available(self, agent):
        ret = [False, False, False]
        munition_info = {
            # KM / DEG
            "GUN": {"dist" : 3, "AO" : 5},
            "AIM-120B" : {"dist" : 37, "AO" : 90},
            "AIM-9M" : {"dist" : 7, "AO" : 90},
        }
        rad_missile_name_list = ["AIM-120B"]
        
        enemy = self.get_target(agent)
        target = enemy.get_position() - agent.get_position()
        heading = agent.get_velocity()
        distance = np.linalg.norm(target)
        attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
        
        if distance / 1000 < munition_info["GUN"]["dist"] and attack_angle < munition_info["GUN"]["AO"]:
            ret[0] = True 
        
        if distance / 1000 < munition_info["AIM-120B"]["dist"] and attack_angle < munition_info["AIM-120B"]["AO"]:
            ret[1] = True 
        
        if distance / 1000 < munition_info["AIM-9M"]["dist"] and attack_angle < munition_info["AIM-9M"]["AO"]:
            ret[2] = True
        
        return ret, enemy
        
    def get_target(self, agent):
        target_distances = []
        for enemy in agent.enemies:
            target = enemy.get_position() - agent.get_position()
            distance = np.linalg.norm(target)
            target_distances.append(distance)
        return agent.enemies[np.argmax(target_distances)]