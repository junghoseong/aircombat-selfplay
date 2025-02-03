import torch
import numpy as np
from gymnasium import spaces
from typing import Literal
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward
from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..model.baseline import BaselineAgent, PursueAgent, ManeuverAgent, StraightFlyAgent, DodgeMissileAgent


class SingleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        print("use_artillery=",self.use_artillery)
        if self.use_baseline:
            for index, (key, value) in enumerate(self.config.aircraft_configs.items()):
                if value['color'] == 'Red':
                    agent_id = index
            self.baseline_agent = self.load_agent(self.config.baseline_type, agent_id)
            

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        # return 2 if not self.use_baseline else 1
        return 2

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

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
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20  - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._agent_die_flag = {}
        if self.use_baseline:
            self.baseline_agent.reset()
        return super().reset(env)

    def step(self, env):
        def _orientation_fn(AO):
            if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
                return 1 - AO / 0.5236
            elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
                return 1 + AO / 0.5236
            return 0
        def _distance_fn(R):
            if R <=1: # [0, 1km]
                return 1
            elif R > 1 and R <= 3: # [1km, 3km]
                return (3 - R) / 2.
            else:
                return 0
        if self.use_artillery:
            for agent_id in env.agents.keys():
                ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                        env.agents[agent_id].get_velocity()])
                for enm in env.agents[agent_id].enemies:
                    if enm.is_alive:
                        enm_feature = np.hstack([enm.get_position(),
                                                enm.get_velocity()])
                        AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
                        enm.bloods -= _orientation_fn(AO) * _distance_fn(R/1000)
                        # if agent_id == 'A0100' and enm.uid == 'B0100':
                        #     print(f"AO: {AO * 180 / np.pi}, {_orientation_fn(AO)}, dis:{R/1000}, {_distance_fn(R/1000)}")

    def get_reward(self, env, agent_id, info=...):
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info
        else:
            self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
            return super().get_reward(env, agent_id, info=info)

    def load_agent(self, name, agent_id):
        if name == 'pursue':
            return PursueAgent(agent_id=agent_id)
        elif name == 'maneuver':
            return ManeuverAgent(agent_id=agent_id, maneuver='triangle')
        elif name == 'dodge':
            return DodgeMissileAgent(agent_id=agent_id)
        elif name == 'straight':
            return StraightFlyAgent(agent_id=agent_id)
        else:
            raise NotImplementedError

class HierarchicalSingleCombatTask(SingleCombatTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cuda')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.normalize_action(env, agent_id, action)
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            # (1) delta altitude/heading/velocity
            input_obs[0] = self.norm_delta_altitude[action[0]]
            input_obs[1] = self.norm_delta_heading[action[1]]
            input_obs[2] = self.norm_delta_velocity[action[2]]
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
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)