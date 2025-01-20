import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from .singlecombat_task import HierarchicalSingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward, PostureReward, EventDrivenReward, CombatGeometryReward, GunBEHITReward, GunTargetTailReward, GunWEZReward, GunWEZDOTReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn


class WVRTask(HierarchicalSingleCombatTask):
    def __init__(self, config):
        super().__init__(config)
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            for index, (key, value) in enumerate(self.config.aircraft_configs.items()):
                if value['color'] == 'Red':
                    agent_id = index
            self.baseline_agent = self.load_agent(self.config.baseline_type, agent_id)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            CombatGeometryReward(self.config),
            GunBEHITReward(self.config),
            GunTargetTailReward(self.config),
            GunWEZReward(self.config),
            GunWEZDOTReward(self.config),
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            Timeout(self.config),
        ]
        
        self.curriculum_angle = 0
        self.winning_rate = 0
        self.record = []
        
    def reset(self, env):
        if self.winning_rate >= 0.9 and len(self.record) > 20:
            self.curriculum_angle += 1
            self.record = []
        env.reset_simulators_curriculum(self.curriculum_angle)
        HierarchicalSingleCombatTask.reset(self, env)
        
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

    def step(self, env):
        HierarchicalSingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            target_distances = []
            for enemy in agent.enemies:
                target = enemy.get_position() - agent.get_position()
                distance = np.linalg.norm(target)
                target_distances.append(distance)
            enemy = agent.enemies[np.argmax(target_distances)]
            distance = np.linalg.norm(enemy.get_position() - agent.get_position())
            heading = agent.get_velocity()
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            
            if distance / 1000 < 3 and attack_angle < 5:
                enemy.bloods -= 5

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env, env.task)
            action = self.baseline_agent.normalize_action(env, agent_id, action)
            return action
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action.astype(np.int32))