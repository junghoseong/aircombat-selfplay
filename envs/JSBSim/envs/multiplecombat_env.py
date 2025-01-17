import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
from ..tasks.multiplecombat_with_missile_task import HierarchicalMultipleCombatShootTask, Scenario2, Scenario3
from ..tasks.KAI_project_task import Scenario2_for_KAI, Scenario3_for_KAI

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
            print("init state is {}".format(self.init_states))
            
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
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

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info