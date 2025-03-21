import numpy as np
from typing import Tuple, Dict
from .env_base import BaseEnv
from ..tasks.singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask, Maneuver_curriculum
from ..tasks.singlecombat_with_missile_task import SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..tasks import Scenario1, Scenario1_curriculum, Scenario1_RWR, Scenario1_RWR_curriculum, WVRTask, Scenario1_Hybrid
from ..tasks.KAI_project_task import Scenario1_for_KAI
from ..utils.utils import calculate_coordinates_heading_by_curriculum

class HybridSingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
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
    
    @property
    def rnn_actor_space(self):
        return self.task.rnn_actor_space
    
    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == "scenario1_hybrid":
            self.task = Scenario1_Hybrid(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        
        center_lat = 60.1  # 중심 위도
        center_lon = 120.0  # 중심 경도
        radius_km = 11.119  # 반지름 (위도 0.1도)

        # 각도 리스트 (0~360도)
        angle = 0
        # 좌표 계산 (아래쪽이 0도, 오른쪽이 90도, 위쪽이 180도, 왼쪽이 270도)
        result_corrected = calculate_coordinates_heading_by_curriculum(center_lat, center_lon, radius_km, [angle])

        self.init_states[0].update({
            'ic_lat_geod_deg': result_corrected[angle][0],
            'ic_long_gc_deg': result_corrected[angle][1],
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': result_corrected[angle][2],
            'ic_u_fps': 800.0,
        })
        
        self.init_states[1].update({
            'ic_lat_geod_deg': 60.1,
            'ic_long_gc_deg': 120.0,
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': 0,
            'ic_u_fps': 800.0,
        })
                
        init_states = self.init_states.copy()
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()

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
        
        self.init_states[1].update({
            'ic_lat_geod_deg': 60.1,
            'ic_long_gc_deg': 120.0,
            'ic_h_sl_ft': 20000,
            'ic_psi_true_deg': 0,
            'ic_u_fps': 800.0,
        })
                
        init_states = self.init_states.copy()
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()

    def step(self, obs:np.ndarray, action: np.ndarray, rnn_states: np.ndarray, action_representation) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
        # print(info, self.agents[self.ego_ids[0]].get_position(), self.agents[self.enm_ids[0]].get_position())

        # apply actions
        action = self._unpack(action)
        continuous_actions={}
        discrete_actions={}
        rewards = {}
        dones = {}
        # if not self.config.use_baseline:
        for agent_id in self.agents.keys():
            #print(self.task.normalize_action(self, agent_id, self.task.get_obs(self,agent_id) ,rnn_states, action[agent_id],action_representation))
            a_action, cont_action= self.task.normalize_action(self, agent_id, self.task.get_obs(self,agent_id) ,rnn_states, action[agent_id],action_representation)
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
            #print("cont_action", cont_action)
            continuous_actions[agent_id] = cont_action
            discrete_actions[agent_id] = self.task._shoot_action[agent_id]

            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]
                
        # else:
        #     for agent_id in self.ego_ids:
        #         a_action, cont_action= self.task.normalize_action(self, agent_id, self.task.get_obs(self,agent_id) ,rnn_states, action[agent_id],action_representation)
        #         self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        #         continuous_actions[agent_id] = cont_action
        #         discrete_actions[agent_id] = self.task._shoot_action[agent_id][0]
        #         reward, info = self.task.get_reward(self, agent_id, info)
        #         rewards[agent_id] = [reward]

        #         done, info = self.task.get_termination(self, agent_id, info)
        #         dones[agent_id] = [done]


        #     for agent_id in self.enm_ids:
        #         a_action = self.task.normalize_action(self, agent_id, self.task.get_obs(self,agent_id) ,rnn_states, action[agent_id],action_representation)
        #         self.agents[agent_id].set_property_values(self.task.action_var, a_action)
            
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
            for sim in self._chaffsims.values(): 
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

        return self._pack(obs), self._pack(rewards), self._pack(dones),\
              self._pack(continuous_actions), self._pack(discrete_actions), info
