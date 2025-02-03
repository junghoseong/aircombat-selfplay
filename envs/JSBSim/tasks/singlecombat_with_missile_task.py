import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator, AIM_9M, AIM_120B, ChaffSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
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
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
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

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.agent_last_shot_missile = {agent_id: 0 for agent_id in env.agents.keys()} # To manage missile id (linking to agent id)
        self.agent_last_shot_chaff = {agent_id: 0 for agent_id in env.agents.keys()} # To manage chaffs
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
            if shoot_flag and (self.agent_last_shot_missile[agent_id] == 0 or self.agent_last_shot_missile[agent_id].is_done): # ADDED <- manage missile duration
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                self.agent_last_shot_missile[agent_id] = env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                
                self.remaining_missiles[agent_id] -= 1


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
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
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)

class Scenario1(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
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
        if self.use_baseline and agent_id in env.enm_ids:
            # self._shoot_action[agent_id] = action[-4:]
            self._shoot_action[agent_id] = [0,0,0,0]
            action = self.baseline_agent.get_action(env, env.task)
            action = self.baseline_agent.normalize_action(env, agent_id, action)
            if self.use_artillery:
                self._shoot_action[agent_id] = [1,1,1,1]
            
            return action
        if agent_id in env.ego_ids:
            self._shoot_action[agent_id] = action[-4:]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-4].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.remaining_missiles_AIM_9M = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_missiles_AIM_120B = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}        
        self.remaining_gun = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.remaining_chaff_flare = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        
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
                avail, enemy = self.a2a_launch_available(agent, agent_id, env)
                if avail[0]:
                    target = self.get_target(agent)
                    enemy.bloods -= 5
                    print(f"gun shot, blood = {enemy.bloods}") # Implement damage of gun to enemies
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

    
class Scenario1_curriculum(Scenario1):
    def __init__(self, config: str):
        Scenario1.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []
        
    def reset(self, env):
        if self.winning_rate >= 0.9 and len(self.record) > 20:
            self.curriculum_angle += 1
            self.record = []
        env.reset_simulators_curriculum(self.curriculum_angle)
        Scenario1.reset(self, env)
    
    def get_termination(self, env, agent_id, info={}):
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                if env.agents[agent_id].color == 'Blue':
                    print(success, s)
                    if success:
                        self.record.append(1)
                    else:
                        self.record.append(0)
                    self.winning_rate = sum(self.record)/len(self.record)   
                    print("current winning rate is {}/{}, curriculum is {}'th stage".format(sum(self.record), len(self.record), self.curriculum_angle))
                break
        return done, info
