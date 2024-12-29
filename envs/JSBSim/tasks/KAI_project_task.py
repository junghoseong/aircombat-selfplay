import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .multiplecombat_task import MultipleCombatTask, HierarchicalMultipleCombatTask

from .singlecombat_with_missile_task import SingleCombatShootMissileTask
from .multiplecombat_with_missile_task import MultipleCombatShootMissileTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator, AIM_9M, AIM_120B
from ..utils.utils import LLA2NEU, get_AO_TA_R

class Scenario1_for_KAI(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control (gun, AIM-9M, AIM-120B, Chaff/Flare)
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.MultiDiscrete([2, 2, 2, 2])])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-4:]
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            
            distance_target = []
            for i in range(len(env.agents[agent_id].enemies)):
                ac_lon, ac_lat, ac_alt = env.agents[agent_id].get_geodetic()
                tgt_lon, tgt_lat, tgt_alt = env.agents[agent_id].enemies[0].get_geodetic()
                ego_ned = LLA2NEU(ac_lon, ac_lat, ac_alt, env.agents[agent_id].lon0, env.agents[agent_id].lat0, env.agents[agent_id].alt0)
                tgt_ned = LLA2NEU(tgt_lon, tgt_lat, tgt_alt, env.agents[agent_id].enemies[i].lon0, env.agents[agent_id].enemies[i].lat0, env.agents[agent_id].enemies[i].alt0)
                distance = np.linalg.norm(ego_ned - tgt_ned)
                distance_target.append(distance)
            
            if env.current_step < 4 and env.agents[agent_id].color == 'Blue':
                x1, y1 = env.agents[agent_id].get_property_values(self.state_var)[0:2]
                x2, y2 = env.agents[agent_id].enemies[0].get_property_values(self.state_var)[0:2]
                self.initial_desired_heading = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2
                self.initial_desired_heading = (self.initial_desired_heading + np.pi) % (2 * np.pi) - np.pi
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]                
            elif self.initial_desired_heading is not None:
                if env.agents[agent_id].color == 'Blue':
                    curr_heading = env.agents[agent_id].get_property_values(self.state_var)[5]
                    if abs(min(curr_heading, abs(2*np.pi - curr_heading)) % (2 * np.pi)) > abs(self.initial_desired_heading):
                        self.initial_desired_heading = None 
                if self.initial_desired_heading is not None and self.initial_desired_heading > 0:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[1]
                    input_obs[2] = self.norm_delta_velocity[0]
                else:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[3]
                    input_obs[2] = self.norm_delta_velocity[0]
                                        
            elif any(component <= 120_000 for component in distance_target) or self.switch_policy == True:
                input_obs[0] = self.norm_delta_altitude[int(action[0])]
                input_obs[1] = self.norm_delta_heading[int(action[1])]
                input_obs[2] = self.norm_delta_velocity[int(action[2])]
                self.switch_policy = True
                print("now self play policy")
            else:
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]     
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4                   
        return norm_act
    
    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.remaining_missiles_AIM_9M = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_missiles_AIM_120B = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}        
        self.remaining_gun = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_chaff_flare = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.initial_desired_heading = None
        self.switch_policy = False
        
        init_states = [sim.init_state.copy() for sim in env.agents.values()]
        
        init_states[0].update({
            'ic_long_gc_deg': 127.87,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[1].update({
            'ic_long_gc_deg': 125.64,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        for idx, sim in enumerate(env.agents.values()):
            sim.reload(init_states[idx])
        
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]            
            shoot_flag_gun = agent.is_alive and self._shoot_action[agent_id][0] and self.remaining_gun[agent_id] > 0
            shoot_flag_AIM_9M = agent.is_alive and self._shoot_action[agent_id][1] and self.remaining_missiles_AIM_9M[agent_id] > 0
            shoot_flag_AIM_120B = agent.is_alive and self._shoot_action[agent_id][2] and self.remaining_missiles_AIM_120B[agent_id] > 0
            shoot_flag_chaff_flare = agent.is_alive and self._shoot_action[agent_id][3] and self.remaining_chaff_flare[agent_id] > 0

            if shoot_flag_gun and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage gun duration
                avail, enemy = self.a2a_launch_available(agent)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
                    self.remaining_gun[agent_id] -= 1
            
            if shoot_flag_AIM_120B and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage long-range missile duration
                avail, _ = self.a2a_launch_available(agent)
                if avail[1]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_120B[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_120B.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-120B"))
                    self.remaining_missiles_AIM_120B[agent_id] -= 1

            if shoot_flag_AIM_9M and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage middle-range missile duration
                avail, _ = self.a2a_launch_available(agent)
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
            
        print(distance, attack_angle, ret)       
        return ret, enemy
        
    def get_target(self, agent):
        target_distances = []
        for enemy in agent.enemies:
            target = enemy.get_position() - agent.get_position()
            distance = np.linalg.norm(target)
            target_distances.append(distance)
        return agent.enemies[np.argmax(target_distances)]          
    
class Scenario2_for_KAI(HierarchicalMultipleCombatTask, MultipleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalMultipleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
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
        self._shoot_action[agent_id] = action[-4:]
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            
            distance_target = []
            for i in range(len(env.agents[agent_id].enemies)):
                ac_lon, ac_lat, ac_alt = env.agents[agent_id].get_geodetic()
                tgt_lon, tgt_lat, tgt_alt = env.agents[agent_id].enemies[0].get_geodetic()
                ego_ned = LLA2NEU(ac_lon, ac_lat, ac_alt, env.agents[agent_id].lon0, env.agents[agent_id].lat0, env.agents[agent_id].alt0)
                tgt_ned = LLA2NEU(tgt_lon, tgt_lat, tgt_alt, env.agents[agent_id].enemies[i].lon0, env.agents[agent_id].enemies[i].lat0, env.agents[agent_id].enemies[i].alt0)
                distance = np.linalg.norm(ego_ned - tgt_ned)
                distance_target.append(distance)
            
            if env.current_step < 4 and env.agents[agent_id].color == 'Blue':
                x1, y1 = env.agents[agent_id].get_property_values(self.state_var)[0:2]
                x2, y2 = env.agents[agent_id].enemies[0].get_property_values(self.state_var)[0:2]
                self.initial_desired_heading = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2
                self.initial_desired_heading = (self.initial_desired_heading + np.pi) % (2 * np.pi) - np.pi
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]                
            elif self.initial_desired_heading is not None:
                if env.agents[agent_id].color == 'Blue':
                    curr_heading = env.agents[agent_id].get_property_values(self.state_var)[5]
                    if abs(min(curr_heading, abs(2*np.pi - curr_heading)) % (2 * np.pi)) > abs(self.initial_desired_heading):
                        self.initial_desired_heading = None 
                if self.initial_desired_heading is not None and self.initial_desired_heading > 0:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[1]
                    input_obs[2] = self.norm_delta_velocity[0]
                else:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[3]
                    input_obs[2] = self.norm_delta_velocity[0]
                                        
            elif any(component <= 120_000 for component in distance_target) or self.switch_policy == True:
                input_obs[0] = self.norm_delta_altitude[int(action[0])]
                input_obs[1] = self.norm_delta_heading[int(action[1])]
                input_obs[2] = self.norm_delta_velocity[int(action[2])]
                self.switch_policy = True
                print("now self play policy")
            else:
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]     
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4    
        return norm_act
    
    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.remaining_missiles_AIM_9M = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_missiles_AIM_120B = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}        
        self.remaining_gun = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_chaff_flare = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.initial_desired_heading = None
        self.switch_policy = False
        
        init_states = [sim.init_state.copy() for sim in env.agents.values()]

        init_states[0].update({
            'ic_long_gc_deg': 127.87,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[1].update({
            'ic_long_gc_deg': 127.88,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[2].update({
            'ic_long_gc_deg': 125.64,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[3].update({
            'ic_long_gc_deg': 125.65,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        for idx, sim in enumerate(env.agents.values()):
            sim.reload(init_states[idx])

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
                avail, enemy = self.a2a_launch_available(agent)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
                    self.remaining_gun[agent_id] -= 1
            
            if shoot_flag_AIM_120B and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage long-range missile duration
                avail, _ = self.a2a_launch_available(agent)
                if avail[1]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_120B[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_120B.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-120B"))
                    self.remaining_missiles_AIM_120B[agent_id] -= 1

            if shoot_flag_AIM_9M and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage middle-range missile duration
                avail, _ = self.a2a_launch_available(agent)
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
        
class Scenario3_for_KAI(HierarchicalMultipleCombatTask, MultipleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalMultipleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return MultipleCombatShootMissileTask.load_observation_space(self)

    @property
    def num_agents(self) -> int:
        return 8        
    
    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.MultiDiscrete([2, 2, 2, 2])])

    def get_obs(self, env, agent_id):
        return MultipleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-4:]
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            
            distance_target = []
            for i in range(len(env.agents[agent_id].enemies)):
                ac_lon, ac_lat, ac_alt = env.agents[agent_id].get_geodetic()
                tgt_lon, tgt_lat, tgt_alt = env.agents[agent_id].enemies[0].get_geodetic()
                ego_ned = LLA2NEU(ac_lon, ac_lat, ac_alt, env.agents[agent_id].lon0, env.agents[agent_id].lat0, env.agents[agent_id].alt0)
                tgt_ned = LLA2NEU(tgt_lon, tgt_lat, tgt_alt, env.agents[agent_id].enemies[i].lon0, env.agents[agent_id].enemies[i].lat0, env.agents[agent_id].enemies[i].alt0)
                distance = np.linalg.norm(ego_ned - tgt_ned)
                distance_target.append(distance)
            
            if env.current_step < 4 and env.agents[agent_id].color == 'Blue':
                x1, y1 = env.agents[agent_id].get_property_values(self.state_var)[0:2]
                x2, y2 = env.agents[agent_id].enemies[0].get_property_values(self.state_var)[0:2]
                self.initial_desired_heading = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2
                self.initial_desired_heading = (self.initial_desired_heading + np.pi) % (2 * np.pi) - np.pi
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]                
            elif self.initial_desired_heading is not None:
                if env.agents[agent_id].color == 'Blue':
                    curr_heading = env.agents[agent_id].get_property_values(self.state_var)[5]
                    if abs(min(curr_heading, abs(2*np.pi - curr_heading)) % (2 * np.pi)) > abs(self.initial_desired_heading):
                        self.initial_desired_heading = None 
                if self.initial_desired_heading is not None and self.initial_desired_heading > 0:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[1]
                    input_obs[2] = self.norm_delta_velocity[0]
                else:
                    input_obs[0] = self.norm_delta_altitude[1]
                    input_obs[1] = self.norm_delta_heading[3]
                    input_obs[2] = self.norm_delta_velocity[0]
                                        
            elif any(component <= 120_000 for component in distance_target) or self.switch_policy == True:
                input_obs[0] = self.norm_delta_altitude[int(action[0])]
                input_obs[1] = self.norm_delta_heading[int(action[1])]
                input_obs[2] = self.norm_delta_velocity[int(action[2])]
                self.switch_policy = True
                print("now self play policy")
            else:
                input_obs[0] = self.norm_delta_altitude[1]
                input_obs[1] = self.norm_delta_heading[2]
                input_obs[2] = self.norm_delta_velocity[0]     
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4                   
        return norm_act
    
    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.remaining_missiles_AIM_9M = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_missiles_AIM_120B = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}        
        self.remaining_gun = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_chaff_flare = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.initial_desired_heading = None
        self.switch_policy = False
        
        init_states = [sim.init_state.copy() for sim in env.agents.values()]

        init_states[0].update({
            'ic_long_gc_deg': 127.87,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[1].update({
            'ic_long_gc_deg': 127.88,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[2].update({
            'ic_long_gc_deg': 127.89,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[3].update({
            'ic_long_gc_deg': 127.90,
            'ic_lat_geod_deg': 37.03,
            'ic_psi_true_deg': 180,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[4].update({
            'ic_long_gc_deg': 125.64,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[5].update({
            'ic_long_gc_deg': 125.65,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })

        init_states[6].update({
            'ic_long_gc_deg': 125.66,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        init_states[7].update({
            'ic_long_gc_deg': 125.67,
            'ic_lat_geod_deg': 39.22,
            'ic_psi_true_deg': 0,
            'ic_h_sl_ft': 20000,
        })
        
        for idx, sim in enumerate(env.agents.values()):
            sim.reload(init_states[idx])
            
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
                avail, enemy = self.a2a_launch_available(agent)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
                    self.remaining_gun[agent_id] -= 1
            
            if shoot_flag_AIM_120B and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage long-range missile duration
                avail, _ = self.a2a_launch_available(agent)
                if avail[1]:
                    new_missile_uid = agent_id + str(self.remaining_missiles_AIM_120B[agent_id])
                    target = self.get_target(agent)
                    self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                        AIM_120B.create(parent=agent, target=target, uid=new_missile_uid, missile_model="AIM-120B"))
                    self.remaining_missiles_AIM_120B[agent_id] -= 1

            if shoot_flag_AIM_9M and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # manage middle-range missile duration
                avail, _ = self.a2a_launch_available(agent)
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