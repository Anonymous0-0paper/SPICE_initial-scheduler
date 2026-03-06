import math
from typing import Dict

import numpy as np

from salsa.core.actions.agentAction import ActionType, AgentAction
from salsa.core.states.systemState import SystemState
from salsa.entities.cluster import Cluster
from salsa.utils.typing import cluster_id


class SalsaRewardSystem:
    def __init__(self, state: SystemState, config: Dict):
        self.res_factor: float = config['scheduler']['rewardsystem']['res_factor']

        self.slo_factor: float = config['scheduler']['rewardsystem']['slo_factor']

        self.mig_factor: float = config['scheduler']['rewardsystem']['mig_factor']

        self.prevention_factor: float = config['scheduler']['rewardsystem']['prevention_factor']

        self.load_balancing_factor: float = config['scheduler']['rewardsystem']['load_balancing_factor']

        self.churn_factor: float = config['scheduler']['rewardsystem']['churn_factor']

        self.placement_factor: float = config['scheduler']['rewardsystem']['placement_factor']

        self.state = state

    def compute_global_reward(self, actions: Dict[cluster_id, AgentAction], global_costs: Dict[str, float], rnd: int) -> float:
        prevention_bonus = self.proactive_prevention_bonus_fn(rnd=rnd)
        churn_penalty = self.churn_penalty_fn(actions)
        load_balancing_penalty = self.load_balancing_penalty_fn()
        
        res = global_costs['res']
        max_res = global_costs['max_res']
        res_ratio = res / max_res if max_res > 0 else 0.0

        global_reward = (
                - self.res_factor * res_ratio
                - self.slo_factor * global_costs['slo']
                - self.mig_factor * global_costs['mig']
                - self.load_balancing_factor * load_balancing_penalty
                - self.churn_factor * churn_penalty
                + self.prevention_factor * prevention_bonus
        )
        return global_reward

    def compute_local_reward(self, cluster: Cluster, action: AgentAction) -> float:
        # 1. Normalize Resource Cost
        actual_cost = cluster.get_resource_cost()
        max_cost = cluster.get_max_potential_cost()
        resource_cost_ratio = actual_cost / max_cost if max_cost > 0 else 0.0

        # 2. Calculate SLO Cost
        slo_cost = 0
        for app in [a for a in self.state.get_all_applications() if a.id in cluster.applications]:
            beta_i = app.penalty_coefficient
            violation = app.sloTracker.get_latest()
            slo_cost += violation * beta_i if app.is_deployed and app.get_receives_work() else 0.0
        
        # 3. Calculate Placement Logic
        reward_placement = 0
        action_type = action.action_type
        
        # Bonus for placing the first microservice of an undeployed app
        if action_type != ActionType.NO_OP:
            msvc_id = action.service_id
            msvc = self.state.get_microservice(msvc_id)
            app = self.state.get_application(msvc.app_id)
            if app and not app.is_deployed and action_type == ActionType.PLACE:
                reward_placement += 1

        # Penalty for doing anything else while apps are still undeployed
        all_apps = self.state.get_all_applications()
        exists_undeployed = False
        for app in all_apps:
            if not app.is_deployed:
                exists_undeployed = True

        if exists_undeployed and action_type != ActionType.PLACE:
            reward_placement -= 5

        # 4. Apply Factors and Return
        weighted_res = resource_cost_ratio * self.res_factor 
        weighted_slo = slo_cost * self.slo_factor
        weighted_placement = reward_placement * self.placement_factor

        local_reward = - weighted_res - weighted_slo + weighted_placement
        
        print(f"({cluster.id}) Util Ratio: {resource_cost_ratio:.4f}, Res Cost: {weighted_res:.4f}, SLO Cost: {weighted_slo:.4f}, Placement: {weighted_placement:.2f}")
        return local_reward

    def proactive_prevention_bonus_fn(self, rnd: int) -> float:
        reward = 0

        for app in self.state.get_all_applications():
            violation = app.sloTracker.get_latest()
            if violation > 0: # Don't reward if a violation happened
                continue

            violation_probability = app.sloTracker.get_violation_prediction(rnd=rnd)
            if math.isnan(violation_probability) or math.isinf(violation_probability):
                violation_probability = 1.0

            beta_i = app.penalty_coefficient

            reward += violation_probability * beta_i
        return reward

    def load_balancing_penalty_fn(self) -> float:
        """
        Computes the load balancing penalty based on the variance of cpu utilization across nodes within a cluster.
        Currently, returns 0 as this version of the scheduler does not consider specific node placement.
        :param:
        :return load_balancing_penalty:
        """
        penalty = 0
        return penalty

    def churn_penalty_fn(self, actions: Dict[cluster_id, AgentAction]) -> float:
        penalty = 0
        for cid, action in actions.items():
            match action.action_type:
                case ActionType.NO_OP:
                    penalty += 0
                case ActionType.SCALE_OUT:
                    penalty += np.abs(action.magnitude)
                case ActionType.SCALE_IN:
                    penalty += np.abs(action.magnitude)
                case _:
                    penalty += 1
        return penalty
