import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask, Scenario1, Scenario1_curriculum, WVRTask, Maneuver_curriculum
from ..tasks.KAI_project_task import Scenario1_for_KAI
from ..utils.utils import calculate_coordinates_heading_by_curriculum

class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == "scenario1":
            self.task = Scenario1(self.config)        
        elif taskname == "scenario1_for_KAI":
            self.task = Scenario1_for_KAI(self.config) 
        elif taskname == "scenario1_curriculum":
            self.task = Scenario1_curriculum(self.config)     
        elif taskname == "wvr":
            self.task = WVRTask(self.config)     
        elif taskname == "maneuver_curriculum":
            self.task = Maneuver_curriculum(self.config)
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
        
        # # ego
        # self.init_states[0].update({
        #     'ic_long_gc_deg': 125.68,
        #     'ic_lat_geod_deg': 38.08,
        #     'ic_psi_true_deg': 180,
        #     'ic_h_sl_ft': 20000,
        # })
        
        # # enemy
        # self.init_states[1].update({
        #     'ic_long_gc_deg': 125.93,
        #     'ic_lat_geod_deg': 36.41,
        #     'ic_psi_true_deg': 0,
        #     'ic_h_sl_ft': 25000,
        # })
        
        # self.init_states[0].update({
        #     'ic_lat_geod_deg': 60.0,
        #     'ic_long_gc_deg': 120.0,
        #     'ic_h_sl_ft': 20000,
        #     'ic_psi_true_deg': 0,
        #     'ic_u_fps': 800.0,
        # })
        
        # self.init_states[1].update({
        #     'ic_lat_geod_deg': 60.1,
        #     'ic_long_gc_deg': 120.0,
        #     'ic_h_sl_ft': 20000,
        #     'ic_psi_true_deg': 0,
        #     'ic_u_fps': 800.0,
        # })
                
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
