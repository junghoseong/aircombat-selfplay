from .reward_function_base import BaseRewardFunction

class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0

        # if env.agents[agent_id].is_shotdown:
        #     reward -= 200
        # elif env.agents[agent_id].is_crash:
        #     reward -= 200
        done = False
        success = True
        for condition in task.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                if env.agents[agent_id].color == 'Blue':
                    print(success, s)
                    if success:
                        reward += 500
                    else:
                        reward -= 500
                break

        # for missile in env.agents[agent_id].launch_missiles:
        #     if missile.is_success:
        #         reward += 200 
        return self._process(reward, agent_id)
