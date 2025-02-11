import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
from ..tasks.multiplecombat_with_missile_task import HierarchicalMultipleCombatShootTask, Scenario2, Scenario3, Scenario2_curriculum, Scenario2_Hybrid
from ..tasks.KAI_project_task import Scenario2_for_KAI, Scenario3_for_KAI
from ..utils.utils import calculate_coordinates_heading_by_curriculum

class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.init_states = None

    @property
    def share_observation_space(self):
        return self.task.share_observation_space
    
    @property
    def discrete_action_space(self):
        return self.task.discrete_action_space
    
    @property
    def continuous_action_space(self):
        return self.task.continuous_action_space
    
    @property
    def continuous_embedding_space(self):
        return self.task.continuous_embedding_space
    
    @property
    def discrete_embedding_space(self):
        return self.task.discrete_embedding_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        elif taskname == 'scenario2':
            self.task = Scenario2(self.config)
        elif taskname == 'scenario3':
            self.task = Scenario3(self.config)
        elif taskname == 'scenario2_for_KAI':
            self.task = Scenario2_for_KAI(self.config)
        elif taskname == 'scenario3_for_KAI':
            self.task = Scenario3_for_KAI(self.config)
        elif taskname == 'scenario2_curriculum':
            self.task = Scenario2_curriculum(self.config)
        elif taskname == 'scenario2_hybrid':
            self.task = Scenario2_Hybrid(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        # Assign new initial condition here!
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
            
        # # enemy
        # self.init_states[0].update({
        #     'ic_long_gc_deg': 125.88,
        #     'ic_lat_geod_deg': 38.29,
        #     'ic_psi_true_deg': 180,
        #     'ic_h_sl_ft': 20000,
        # })
        
        # self.init_states[1].update({
        #     'ic_long_gc_deg': 125.89,
        #     'ic_lat_geod_deg': 38.29,
        #     'ic_psi_true_deg': 180,
        #     'ic_h_sl_ft': 20000,
        # })
        
        
        # # ego
        # self.init_states[2].update({
        #     'ic_long_gc_deg': 126.49,
        #     'ic_lat_geod_deg': 36.70,
        #     'ic_psi_true_deg': 0,
        #     'ic_h_sl_ft': 25000,
        # })
        
        # self.init_states[3].update({
        #     'ic_long_gc_deg': 126.49,
        #     'ic_lat_geod_deg': 36.70,
        #     'ic_psi_true_deg': 0,
        #     'ic_h_sl_ft': 25000,
        # })
        
        init_states = self.init_states.copy()
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()

    def step(self, obs:np.ndarray, share_obs:np.ndarray, action: np.ndarray,action_representation) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        continuous_actions={}
        discrete_actions={}
        for agent_id in self.agents.keys():
            a_action, cont_action= self.task.normalize_action(self, agent_id, self.task.get_obs(self,agent_id) ,share_obs, action[agent_id],action_representation)
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
            #print("cont_action", cont_action)
            continuous_actions[agent_id] = cont_action
            discrete_actions[agent_id] = self.task._shoot_action[agent_id][0]
            
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
            for sim in self._chaffsims.values(): # implement chaff
                sim.run()
            for missile in self._tempsims.values():
                if missile.is_done:
                    continue
                for chaff in self._chaffsims.values():
                    if chaff.is_done:
                        continue
                    if (np.linalg.norm(chaff.get_position() - missile.get_position()) <= chaff.effective_radius):
                        if (np.random.rand() < 0.85):
                            missile.missed() # Check if Chaff completely delete missile object
        self.task.step(self)

        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones),\
              self._pack(continuous_actions), self._pack(discrete_actions), info
    
    
    def reset_simulators_curriculum(self, angle):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        
        # 중심점과 반지름
        center_lat = 60.1  # 중심 위도
        center_lon = 120.0  # 중심 경도
        radius_km = 11.119  # 반지름 (위도 0.1도)

        # 각도 리스트 (0~360도)
        angles_deg = list(range(0, 181, 1))  # 1도 간격

        # 좌표 계산 (아래쪽이 0도, 오른쪽이 90도, 위쪽이 180도, 왼쪽이 270도)
        result_corrected = calculate_coordinates_heading_by_curriculum(center_lat, center_lon, radius_km, angles_deg)

        self.init_states[0].update({
            'ic_lat_geod_deg': result_corrected[angle][0],
            'ic_long_gc_deg': result_corrected[angle][1],
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': result_corrected[angle][2],
            'ic_u_fps': 800.0,
        })
        
        # 중심점과 반지름
        center_lat = 60.1  # 중심 위도
        center_lon = 120.01  # 중심 경도
        radius_km = 11.119  # 반지름 (위도 0.1도)

        # 각도 리스트 (0~360도)
        angles_deg = list(range(0, 181, 1))  # 1도 간격

        # 좌표 계산 (아래쪽이 0도, 오른쪽이 90도, 위쪽이 180도, 왼쪽이 270도)
        result_corrected = calculate_coordinates_heading_by_curriculum(center_lat, center_lon, radius_km, angles_deg)


        self.init_states[1].update({
            'ic_lat_geod_deg': result_corrected[angle][0],
            'ic_long_gc_deg': result_corrected[angle][1],
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': result_corrected[angle][2],
            'ic_u_fps': 800.0,
        })
        
        self.init_states[2].update({
            'ic_lat_geod_deg': 60.1,
            'ic_long_gc_deg': 120.0,
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': 0,
            'ic_u_fps': 800.0,
        })
        
        self.init_states[3].update({
            'ic_lat_geod_deg': 60.1,
            'ic_long_gc_deg': 120.01,
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': 0,
            'ic_u_fps': 800.0,
        })
                
        init_states = self.init_states.copy()
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
