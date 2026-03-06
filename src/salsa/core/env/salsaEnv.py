import time
from typing import Dict

import numpy as np

from salsa.core.actions.agentAction import AgentAction
from salsa.core.actions.kubernetesExecutor import KubernetesExecutor
from salsa.core.rewardSystems.salsaRewardSystem import SalsaRewardSystem
from salsa.core.states.agentObservation import AgentObservation
from salsa.core.states.monitor import MetricMonitor
from salsa.core.states.observationBuilder import ObservationBuilder
from salsa.core.states.systemState import SystemState
from salsa.externals.clock import EventClock
from salsa.externals.karmadaClient import KarmadaClient
from salsa.externals.karmadaEventProducer import MultiDeploymentMigrationMonitor
from salsa.sloViolationPredictor.base_predictor import BasePredictor
from salsa.utils.typing import cluster_id, application_id


class SalsaEnv:
    def __init__(self,
                 state: SystemState,
                 reward_system: SalsaRewardSystem,
                 monitor: MetricMonitor,
                 karmada_event_monitor: MultiDeploymentMigrationMonitor,
                 latency_predictors: Dict[application_id, BasePredictor],
                 throughput_predictors: Dict[application_id, BasePredictor],
                 clock: EventClock,
                 config: dict):
        self.state = state
        self.reward_system = reward_system

        self.monitor = monitor
        self.karmada_event_monitor = karmada_event_monitor
        self.latency_predictors = latency_predictors
        self.throughput_predictors = throughput_predictors

        self.observer = ObservationBuilder(monitor=monitor,
                                           karmada_event_monitor=karmada_event_monitor,
                                           clock=clock,
                                           latency_predictors=latency_predictors,
                                           throughput_predictors=throughput_predictors,
                                           state=state,
                                           config=config)

        kube_config = config["karmada"]["apiserver_kubeconfig"]
        k8s_client = KarmadaClient(kubeconfig_path=kube_config, context='karmada-apiserver')
        self.executor = KubernetesExecutor(k8s_client, state)
        self.step_interval = config['scheduler']['step_interval']
        self.rnd = 0

        self.lambda_local = config['scheduler']['global_coordinator']['lambda_local']

    def reset(self) -> Dict[cluster_id, AgentObservation]:
        print("Applying work for next Episode...")
        self.executor.apply_work()
        time.sleep(10)

        for app in self.state.get_all_applications():
            app.sloTracker.clear_history()
            app.is_deployed = False
            print(f"App ({app.id}) is now undeployed.")
            app.undeployed_microservices = [msvc.id for msvc in app.microservices]
            self.latency_predictors[app.id].reset()
            self.throughput_predictors[app.id].reset()
        self.monitor.clear_histories()

        self.rnd = 0
        self._punish_undeployed_applications()

        observations = self.observer.build_all_observations(target_cluster_ids=[c.id for c in self.state.get_all_clusters()], rnd=self.rnd)
        self.update_entities(observations)

        return observations

    def step(self, actions: Dict[cluster_id, AgentAction], rnd: int):
        self.executor.execute_actions(actions)

        time.sleep(self.step_interval)

        next_observation = self.observer.build_all_observations([c.id for c in self.state.get_all_clusters()], rnd=self.rnd)

        self.update_entities(next_observation)
        self._punish_undeployed_applications()

        rewards: Dict[cluster_id, float] = self._calculate_rewards(next_observation, actions=actions, rnd=rnd)

        self.rnd += 1

        done = False
        all_app_states = {
            app_id: app_state
            for obs in next_observation.values()
            for app_id, app_state in obs.applications.items()
        }
        for app in self.state.get_all_applications():
            state = all_app_states.get(app.id)
            if state:
                print(f"[{app.id}] Latency: {state.current_latency:.4f}, Throughput: {state.current_throughput:.4f} | SLO Violations: {app.sloTracker.get_latest()}")
        return next_observation, rewards, done, {}

    def update_entities(self, observation):
        # Update Entities based on Observations
        for cid, obs in observation.items():
            # Update based on Node State
            cluster = self.state.get_cluster(cid)
            cluster.mem_utilization = obs.node.cpu_utilization
            cluster.cpu_core_utilization = obs.node.cpu_utilization

            # Update based on Microservice States
            for ms_id, ms_state in obs.microservices.items():
                ms = self.state.get_microservice(ms_id)
                ms.current_replicas = int(ms_state.replicas_effective)
                ms.desired_replicas = ms_state.replicas_desired
                ms.is_migration_in_progress = ms_state.is_migration_in_progress

    def _calculate_rewards(self, observation: Dict[cluster_id, AgentObservation],
                           actions: Dict[cluster_id, AgentAction], rnd: int) -> Dict[cluster_id, float]:
        slo_cost = 0
        mig_cost = 0
        total_capacity_cost = 0.0
        for app in self.state.get_all_applications():
            beta_i = app.penalty_coefficient
            violation = app.sloTracker.get_latest()
            slo_cost += violation * beta_i if app.is_deployed and app.get_receives_work() else 0.0

            for ms in app.microservices:
                mig = 1 if ms.is_migration_in_progress else 0
                mig_cost += mig * ms.current_replicas * ms.migration_cost

        res_cost = 0
        for cluster in self.state.get_all_clusters():
            cluster.cpu_core_utilization = observation[cluster.id].node.cpu_utilization
            cluster.mem_utilization = observation[cluster.id].node.mem_utilization
            res_cost += cluster.get_resource_cost()
            total_capacity_cost += cluster.get_max_potential_cost()
        global_costs = {
            "slo": slo_cost,
            "res": res_cost,
            "max_res": total_capacity_cost,
            "mig": mig_cost,
        }

        raw_global_reward = self.reward_system.compute_global_reward(global_costs=global_costs, actions=actions,
                                                                     rnd=rnd)
        rewards = {}
        print("=== REWARDS ===")
        for cid, obs in observation.items():
            raw_local_reward = self.reward_system.compute_local_reward(self.state.get_cluster(cid), action=actions[cid])

            rewards[cid] = raw_global_reward + (self.lambda_local * raw_local_reward)
            rewards[cid] /= 20.0
            print(f"({cid}): Local Reward: {raw_local_reward:.2f}")
            print(f"({cid}): Global Reward: {raw_global_reward:.2f}")
            print(f"({cid}): Total: {rewards[cid]:.2f}")
            print("---------------")
        print("===============")
        return rewards

    def _punish_undeployed_applications(self):
        for app in self.state.get_all_applications():
            if not app.is_deployed:
                app.sloTracker.report_violation()
                app.sloTracker.flush_violations(rnd=self.rnd)
