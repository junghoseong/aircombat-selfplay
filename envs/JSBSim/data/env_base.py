import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import UnControlAircraftSimulator, AircraftSimulator, BaseSimulator, UnControlSAMSimulator
from ..tasks.task_base import BaseTask
from ..tasks.test_task import TestTask
from ..tasks.singlecombat_task import SingleCombatTask
from ..utils.utils import parse_config, LLA2ECEF, LLA2NEU, ECEF2vNED, KNOT2METER, DEG2RAD, MU_NAME_WEIGHT, get_AO_TA_R
from ..core.catalog import Catalog as c

import socket
import math
import traceback
import time
import re

class BaseEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str):
        # basic args
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self._create_records = False
        self.acmi_file_path = None
        self.render_step = 0

        self.dict_reset()

        self.first_exe_flag = True

        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        self._samsims = {}     # type: Dict[str, UnControlSAMSimulator]
        self.load()

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def num_ai_agents(self) -> int:
        return self.task.num_ai_agents

    @property
    def observation_space(self) -> gym.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def sams(self) -> Dict[str, UnControlSAMSimulator]:
        return self._samsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.seed()

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == "test":
            self.task = TestTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
        
    def load_simulator(self):
        self._jsbsims = {}
        self._samsims = {}
        
        init_lon = 125.7
        init_lat = 38.3
        init_alt = 20000
        init_roll = 30
        init_pitch = 60
        init_heading = 90
        init_velocities_u = 800
        init_fuel = 7300
        
        ego_id = 110100001
        ego_color = "Blue"
        enemy_id = 210100001
        enemy_color = "Red"
        self._jsbsims[ego_id] = AircraftSimulator(
            uid=ego_id,
            color=ego_color,
            model="c172p",
            init_state={
                "ic_h_sl_ft": init_alt * 1 / 0.3048,
                "ic_lat_geod_deg": init_lat,
                "ic_long_gc_deg": init_lon,
                "ic_psi_true_deg" : 0.0,
                "ic_u_fps": 800.0, # 초기속도
            },
            origin = [124.00, 37.00, 0.0],
            sim_freq = 60,
            num_missiles = 0,
            munition_info = {}
        )
        # self._jsbsims[enemy_id] = AircraftSimulator(
        #     uid=enemy_id,
        #     color=enemy_id,
        #     model="f16",
        #     init_state={
        #         "ic_h_sl_ft": init_alt * 1 / 0.3048,
        #         "ic_lat_geod_deg": init_lat,
        #         "ic_long_gc_deg": init_lon,
        #         "ic_psi_true_deg" : 0.0,
        #         "ic_u_fps": 800.0, # 초기속도
        #     },
        #     origin = [124.00, 37.00, 0.0],
        #     sim_freq = 60,
        #     num_missiles = 0,
        #     munition_info = {}
        # )
        
        self.ai_ids = [uid for uid, agent in self._jsbsims.items() if agent.mode == "AI"]
        self.rule_ids = [uid for uid, agent in self._jsbsims.items() if agent.mode == "Rule"]

    def dict_reset(self):
        # aircraft/munition id - state
        self.ac_id_init_state = {}
        self.mu_id_init_state = {}
        self.ac_id_state = {}
        self.mu_id_state = {}

        self.sam_id_state = {}

        self.red_tgt_id_state = {}
        self.blue_tgt_id_state = {}

        self.ed_id_state = {}
        self.hmd_id_state = {}

        # id - spec
        self.launcher_id_spec = {}
        self.rad_id_spec = {}
        self.jammer_id_spec = {}

        # 전자장비 id - state
        self.rad_upid_state = {}
        self.rwr_upid_state = {}
        self.mws_upid_state = {}

        # damage page
        self.mu_id_target_id_dmg = {}

        # lockon 
        self.lockon_id_target = {}
        ###

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        self.load_simulator()
        self.dict_reset()
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        
        self.task.reset(self)
        obs = self.get_obs()

        return self._pack(obs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        print("current timestep is {}".format(self.current_step))
        action = self._unpack(action)

        for agent_id, agent in self.agents.items():
            if (agent.mode == "AI"): # ONLY FOR AI
                a_action = self.task.normalize_action(self, agent_id, action[agent_id])
                self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for agent_id, sim in self._jsbsims.items():
                if (sim.mode == "AI"):
                    sim.run()
 
        self.task.step(self)
        obs = self.get_obs()
        
        rewards = {}
        for agent_name, agent in self.agents.items():
            if (agent.mode == "AI"):
                reward, info = self.task.get_reward(self, agent_name, info)
                rewards[agent_name] = [reward]
        ego_reward = np.mean([rewards[ai_id] for ai_id in self.ai_ids])
        
        for ai_id in self.ai_ids:
            rewards[ai_id] = [ego_reward]

        dones = {}
        for agent_name, agent in self.agents.items():
            if (agent.mode == "AI"):
                done, info = self.task.get_termination(self, agent_name, info)
                dones[agent_name] = [done]
                
        self.render()
        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id, agent in self.agents.items() if agent.mode == "AI"]) # ONLY FOR AI (need to train)

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id, agent in self.agents.items() if agent.mode == "AI"])


    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._samsims.values():
            sim.close()

        self._jsbsims.clear()
        self._samsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.acmi', aircraft_id = None):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if (self.acmi_file_path != filepath):
            # if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                # self._create_records = True
                self.acmi_file_path = filepath
                self.render_step = 0
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.render_step * self.time_interval
                self.render_step += 1
                f.write(f"#{timestamp:.3f}\n")

                # aircraft
                for air_id, sim in self._jsbsims.items():
                    if (air_id == aircraft_id or aircraft_id == None):    
                        log_msg = sim.log()
                        print(log_msg)
                        if log_msg is not None:
                            f.write(log_msg + "\n")
                
                # sam
                for sam_id, sam in self._samsims.items():
                    log_msg = sam.log()
                    if (log_msg is not None):
                        f.write(log_msg + "\n")

        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ai_data = np.array([data[uid] for uid in self.ai_ids])
        
        try:
            rule_data = np.array([data[uid] for uid in self.rule_ids])
        except:
            rule_data = np.array([])
        
        if rule_data.shape[0] > 0:
            data = np.concatenate((ai_data, rule_data))  # type: np.ndarray
        else:
            data = ai_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            data = np.nan_to_num(data)

            # import pdb
            # pdb.set_trace()
        # only return data that belongs to RL agents
        try:
            ret = data[:self.num_agents, ...]
        except Exception as e:
            print("e : " , e)

        return ret

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        # assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ai_ids + self.rule_ids)[:len(data)], data))
        # fill in None for other not-RL agents
        # for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
        #     unpack_data[agent_id] = None
        return unpack_data
