import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch

from salsa.core.actions.actionMapper import ActionMapper
from salsa.core.actions.agentAction import AgentAction, ActionType
from salsa.core.env.salsaEnv import SalsaEnv
from salsa.core.agents.salsaAgent import SalsaAgent
from salsa.core.globalCoordinator.sharedReplayBuffer import SharedReplayBuffer
from salsa.entities.cluster import Cluster
from salsa.entities.microservice import Microservice
from salsa.utils.typing import agent_id, microservice_id


class GlobalCoordinator:
    def __init__(self,
                 env: SalsaEnv,
                 agents: Dict[str, SalsaAgent],
                 buffer: SharedReplayBuffer,
                 config: dict):
        self.env: SalsaEnv = env
        self.agents = agents
        self.buffer: SharedReplayBuffer = buffer
        self.microservices: Dict[microservice_id, Microservice] = {ms.id: ms for app in env.state.get_all_applications() for ms in app.microservices}
        self.action_mapper = ActionMapper(
            microservices_list=[ms.id for mid, ms in self.microservices.items()],
            num_clusters=len(env.state.get_all_clusters()),
            state=self.env.state,
            clock=self.env.observer.clock
        )
        self.start_training_after = config['scheduler']['start_training_after']

        self.max_timesteps = config['scheduler']['max_timesteps']

        self.placed_microservices: List[microservice_id] = []
        self.current_episode = 0
        self.train: bool = config['scheduler']['train']

        self.gather_metrics: bool = config['scheduler']['gather_metrics']

        if self.gather_metrics:
            # System Metrics
            self.f_sys = open("metrics_system.csv", "w", newline='')
            self.writer_sys = csv.writer(self.f_sys)
            self.writer_sys.writerow(["episode", "step", "avg_cpu", "avg_mem", "cluster_id", "cpu_util", "mem_util"])

            # Application Metrics
            self.f_app = open("metrics_applications.csv", "w", newline='')
            self.writer_app = csv.writer(self.f_app)
            self.writer_app.writerow(
                ["episode", "step", "app_id", "throughput", "latency", "is_violating", "violation_prob", "d_lat", "d_thr"])

            # Microservice Metrics
            self.f_msvc = open("metrics_microservices.csv", "w", newline='')
            self.writer_msvc = csv.writer(self.f_msvc)
            self.writer_msvc.writerow(
                ["episode", "step", "msvc_id", "replicas_desired", "replicas_effective", "replicas_starting",
                 "is_migration", "response_time", "request_rate"])

            # Rewards and Action Counters
            self.f_summary = open("metrics_summary.csv", "w", newline='')
            self.writer_summary = csv.writer(self.f_summary)
            action_keys = ["NO_OP", "PLACE", "SCALE_OUT", "SCALE_IN", "MIGRATE"]
            self.writer_summary.writerow(["episode", "step"] + [f"act_{k}" for k in action_keys] + ["reward_mean"])

            # Training Metrics
            self.f_train = open("metrics_training.csv", "w", newline='')
            self.writer_train = csv.writer(self.f_train)
            self.writer_train.writerow(
                ["episode", "step", "agent_id", "loss_critic", "loss_actor", "value_mean_q", "policy_entropy",
                 "value_target_mean"])


    def run_episode(self):

        save_dir = "./checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        print("Stopping Monitors ...")
        self.env.monitor.stop()
        self.env.karmada_event_monitor.stop()
        time.sleep(2)

        self.env.executor.delete_work()
        time.sleep(10)
        self.env.executor.delete_placement_rules()
        time.sleep(10)
        self.placed_microservices = []
        for c in self.env.state.get_all_clusters():
            c.microservices = []
        observations = self.env.reset()
        all_clusters = self.env.state.get_all_clusters()

        print("Starting Monitors ...")
        self.env.monitor.start()
        self.env.karmada_event_monitor.start()
        time.sleep(5)

        masks = self.action_mapper.compute_action_mask([c.id for c in self.env.state.get_all_clusters()],
                                                       self.placed_microservices)
        huge_neg = torch.tensor(-1e9)
        done = False
        while not done and self.env.rnd < self.max_timesteps:
            try:
                self.placed_microservices = []
                for c in self.env.state.get_all_clusters():
                    self.placed_microservices.extend(c.microservices)
                
                print(f"=========== [ ROUND {self.env.rnd} ] ===========")
                action_counts = {"NO_OP": 0, "PLACE": 0, "SCALE_OUT": 0, "SCALE_IN": 0, "MIGRATE": 0}
                per_cluster_actions = {}
                action_idxs: Dict[agent_id, int] = {}
                agent_logits: Dict[agent_id, torch.Tensor] = {}


                for aid, agent in self.agents.items():
                    local_obs = observations[agent.cluster.id]
                    agent_logits[aid] = agent.forward(local_obs).squeeze(0)

                masked_agent_logits = {}
                for aid, logits in agent_logits.items():
                    agent = self.agents[aid]
                    cluster_id = self.agents[aid].cluster.id
                    bool_mask_list = masks[cluster_id]

                    mask_tensor = torch.tensor(bool_mask_list, device=agent.device, dtype=torch.bool)
                    masked_logits = logits.masked_fill(~mask_tensor, -1e9)
                    if torch.all(masked_logits <= -1e8):
                        masked_logits[0] = 0.0
                        print(f"WARNING: Agent {aid} had 0 feasible actions. Forcing NO_OP.")

                    masked_agent_logits[aid] = masked_logits

                agents_to_be_checked = self.agents.copy() # for action feasibility check
                agent_logits_copy = masked_agent_logits.copy()
                action_idxs_copy = action_idxs.copy()
                chosen_actions_idx = {}
                conflicts_solved = False
                while not conflicts_solved:
                    for aid, agent in agents_to_be_checked.items():
                        action_idxs_copy[aid] = agent.select_action(agent_logits_copy[aid], eval=not self.train)
                    conflict_map = self.check_action_feasibility(action_idxs_copy)

                    for aid, conflict in conflict_map.items():
                        action_idx = action_idxs_copy[aid]
                        if conflict:
                            logits = agent_logits_copy[aid]
                            logits[action_idx] = -1e9

                            if torch.max(logits) <= -1e8:
                                logits[0] = 10.0
                        else:
                            cluster = self.agents[aid].cluster
                            cid = cluster.id
                            per_cluster_actions[cid] = self.action_mapper.decode_network_output(action_idx, nbr_list=[c.id for c in all_clusters if c.id != cluster.id])
                            chosen_actions_idx[aid] = action_idxs_copy.pop(aid)
                            agents_to_be_checked.pop(aid)
                            agent_logits_copy.pop(aid)
                    conflicts_solved = len(agents_to_be_checked) == 0
                for cid, action in per_cluster_actions.items():
                    decoded_action = per_cluster_actions[cid]
                    action_counts[decoded_action.action_type.name] += 1

                next_observations, rewards, done, _ = self.env.step(per_cluster_actions, self.env.rnd)

                next_masks = self.action_mapper.compute_action_mask([c.id for c in self.env.state.get_all_clusters()],
                                                               self.placed_microservices)

                timeout = self.env.rnd >= self.max_timesteps
                stop_episode = done or timeout
                if self.train:
                    self.buffer.push(
                        state=observations,
                        actions=chosen_actions_idx,
                        rewards=rewards,
                        next_state=next_observations,
                        done=done,  # Keep this as False if it's just a timeout!
                        slo_violation_occurred=_slo_violations_occurred(observations),
                        action_masks=masks,
                        next_action_masks=next_masks
                    )
                    if self.env.rnd > self.start_training_after and self.buffer.buffer.__len__() >= self.buffer.batch_size:
                        self._train_agents()
                        for agent in self.agents.values():
                            filename = f"episode_{self.current_episode}_{self.env.rnd}_agent_{agent.aid}.pt"
                            full_path = os.path.join(save_dir, filename)
                            if self.env.rnd % 200 == 0:
                                agent.save_checkpoint(full_path)

                self.gather_metrics_fn(observations=observations, action_counts=action_counts, rewards=rewards)

                print(f"Placements: { {c.id: c.microservices for c in self.env.state.get_all_clusters()} }")
                print(f"DONE: {done}")
                observations = next_observations
                masks = next_masks

                if stop_episode:
                    print(f"Episode finished (Timeout: {timeout}, Done: {done})")
                    break
            except KeyboardInterrupt as e:
                print("Stopping current episode...")
                print("Press CTRL + C again to quit the program.")
                time.sleep(5)

        print(f"Finished Episode {self.current_episode}.")

        self.current_episode += 1

    def check_action_feasibility(self, action_idxs: Dict[agent_id, int]) -> Dict[agent_id, bool]:
        agent_actions: Dict[agent_id, AgentAction] = {}
        for aid, action_idx in action_idxs.items():
            all_clusters = self.env.state.get_all_clusters()
            cluster = self.agents[aid].cluster
            nbr_list = [c.id for c in all_clusters if c.id != cluster.id]
            agent_actions[aid] = self.action_mapper.decode_network_output(action_idx=action_idx, nbr_list=nbr_list)

        conflict_map: Dict[agent_id, bool] = {}
        placement_groups: Dict[microservice_id, List[agent_id]] = {}

        for aid, action in agent_actions.items():
            if action.action_type == ActionType.PLACE:
                msvc = self.microservices[action.service_id]
                if msvc.id not in placement_groups:
                    placement_groups[msvc.id] = []
                placement_groups[msvc.id].append(aid)

            elif action.action_type == ActionType.MIGRATE:
                msvc = self.microservices[action.service_id]
                target_cluster = self.env.state.get_cluster(action.target_cluster)
                
                eff_replicas = max(1, msvc.desired_replicas) 
                req_cpu = msvc.cpu_core_demands * eff_replicas
                req_mem = msvc.mem_demands_bytes * eff_replicas
                
                avail_cpu = (1 - target_cluster.cpu_core_utilization) * target_cluster.cpu_cores
                avail_mem = (1 - target_cluster.mem_utilization) * target_cluster.mem_gb * 1024**3
                
                conflict_map[aid] = not (req_cpu < avail_cpu and req_mem < avail_mem)
            else:
                conflict_map[aid] = False

        for mid, agents in placement_groups.items():
            msvc = self.microservices[mid]
            candidates = []
            scores = []

            for aid in agents:
                cluster = self.agents[aid].cluster
                
                eff_replicas = max(1, msvc.desired_replicas)
                r_cpu = msvc.cpu_core_demands * eff_replicas
                r_mem = msvc.mem_demands_bytes * eff_replicas
                
                a_cpu = cluster.cpu_cores * (1 - cluster.cpu_core_utilization)
                a_mem = cluster.mem_gb * 1024**3 * (1 - cluster.mem_utilization)

                if r_cpu < a_cpu and r_mem < a_mem:
                    cpu_frac = r_cpu / a_cpu if a_cpu > 0 else 0
                    mem_frac = r_mem / a_mem if a_mem > 0 else 0
                    # Fix: Add epsilon to ensure score > 0
                    score = np.sqrt(cpu_frac * mem_frac) + 1e-6
                    candidates.append(aid)
                    scores.append(score)
                else:
                    conflict_map[aid] = True

            if not candidates:
                for aid in agents: conflict_map[aid] = True
            else:
                # Fix: Probabilistic Winner Selection
                total = sum(scores)
                probs = [s / total for s in scores]
                winner_aid = np.random.choice(candidates, p=probs)
                
                for aid in agents:
                    conflict_map[aid] = (aid != winner_aid)

        return conflict_map

    def _train_agents(self):
        sample_result = self.buffer.sample()
        if sample_result is None: return

        s, a, r, s_next, d, masks, next_masks, indices = sample_result

        aggregate_errors = np.zeros(len(indices))

        for aid, agent in self.agents.items():
            agent_errors, metrics = agent.update(state_dict=s, actions_dict=a,
                                                 rewards_dict=r, next_state_dict=s_next,
                                                 dones_batch=d, masks=masks,
                                                 next_masks=next_masks,
                                                 neighbor_agents=self.agents)
            agent_errors = np.abs(agent_errors).flatten()

            agent_errors = np.nan_to_num(agent_errors, nan=0.0, posinf=100.0, neginf=0.0)

            aggregate_errors = np.maximum(aggregate_errors, agent_errors)

            if self.gather_metrics:
                self.writer_train.writerow([
                    self.current_episode,
                    self.env.rnd,
                    aid,
                    metrics.get("loss_critic", 0),
                    metrics.get("loss_actor", 0),
                    metrics.get("value_mean_q", 0),
                    metrics.get("policy_entropy", 0),
                    metrics.get("value_target_mean", 0)
                ])
                self.f_train.flush()

        self.buffer.update_priorities(indices, aggregate_errors)

    def log_metrics_to_file(self, episode, step, aid: agent_id, metrics, filename="training_logs.csv"):
        file_exists = os.path.isfile(filename)

        with (open(filename, mode='a', newline='') as f):
            writer = csv.writer(f)

            if not file_exists:
                headers = ["episode", "step", "agent_id"] + list(metrics.keys())
                writer.writerow(headers)

            row = [episode, step, aid] + list(metrics.values())
            writer.writerow(row)

    def gather_metrics_fn(self, observations, action_counts, rewards):
        if not self.gather_metrics:
            return

        ep = self.current_episode
        step = self.env.rnd

        all_clusters = self.env.state.get_all_clusters()
        total_cpu_util = np.mean([c.cpu_core_utilization for c in all_clusters])
        total_mem_util = np.mean([c.mem_utilization for c in all_clusters])

        for c in all_clusters:
            self.writer_sys.writerow([
                ep, step,
                total_cpu_util, total_mem_util,
                c.id, c.cpu_core_utilization, c.mem_utilization
            ])

        processed_apps = set()

        for cid, obs in observations.items():
            for app_state in obs.applications.values():
                if app_state.app_id in processed_apps: continue

                self.writer_app.writerow([
                    ep, step,
                    app_state.app_id,
                    app_state.current_throughput,
                    app_state.current_latency,
                    app_state.is_violating,
                    app_state.predicted_violation_prob,
                    app_state.dist_latency,
                    app_state.dist_throughput
                ])
                processed_apps.add(app_state.app_id)

        processed_msvcs = set()

        for cid, obs in observations.items():
            for ms_state in obs.microservices.values():
                if ms_state.service_id in processed_msvcs: continue

                self.writer_msvc.writerow([
                    ep, step,
                    ms_state.service_id,
                    ms_state.replicas_desired,
                    ms_state.replicas_effective,
                    ms_state.replicas_starting,
                    ms_state.is_migration_in_progress,
                    ms_state.response_time,
                    ms_state.request_rate
                ])
                processed_msvcs.add(ms_state.service_id)

        if isinstance(rewards, dict):
            avg_reward = np.mean(list(rewards.values()))
        else:
            avg_reward = rewards

        action_keys = ["NO_OP", "PLACE", "SCALE_OUT", "SCALE_IN", "MIGRATE"]
        action_values = [action_counts.get(k, 0) for k in action_keys]

        self.writer_summary.writerow([ep, step] + action_values + [avg_reward])

        if step % 100 == 0:
            self.f_sys.flush()
            self.f_app.flush()
            self.f_msvc.flush()
            self.f_summary.flush()

    def close_loggers(self):
        if self.gather_metrics:
            self.f_sys.close()
            self.f_app.close()
            self.f_msvc.close()
            self.f_summary.close()
            self.f_train.close()

def _slo_violations_occurred(observations):
    for cid, obs in observations.items():
        for aid, slo_state in obs.applications.items():
            if slo_state.is_violating:
                return True
    return False
