from typing import List

from salsa.core.actions.agentAction import AgentAction, ActionType
from salsa.core.states.systemState import SystemState
from salsa.entities.cluster import Cluster
from salsa.externals.clock import EventClock
from salsa.utils.typing import cluster_id, microservice_id


class ActionMapper:
    def __init__(self, microservices_list: List[microservice_id], num_clusters, state: SystemState, clock: EventClock):
        self.ms_list = microservices_list

        self.num_place_actions = len(self.ms_list)

        self.num_scale_actions = len(self.ms_list) * 2 # in and out

        self.num_mig_actions = len(self.ms_list) * (num_clusters - 1)

        self.total_dim = 1 + self.num_place_actions + self.num_scale_actions + self.num_mig_actions

        self.state = state

        self.clock = clock

    def compute_action_mask(self, cluster_ids: List[cluster_id], placed_microservices: List[microservice_id]):
        masks = {}

        for cid in cluster_ids:
            masks[cid] = [True] * self.total_dim
            nbr_list = [c for c in cluster_ids if c != cid]
            len_nbr_list = len(cluster_ids) - 1

            cluster = self.state.get_cluster(cid)
            
            any_undeployed = any(not app.is_deployed for app in self.state.get_all_applications())
            if any_undeployed:
                masks[cluster.id][0] = False

            for i, mid in enumerate(self.ms_list):
                msvc = self.state.get_microservice(mid)
                app = self.state.get_application(msvc.app_id)
                
                idx_place = 1 + i
                idx_scale_out = 1 + self.num_place_actions + i * 2
                idx_scale_in = 1 + self.num_place_actions + i * 2 + 1
                idx_mig_start = 1 + self.num_place_actions + self.num_scale_actions + i * len_nbr_list

                if mid not in cluster.microservices:
                    masks[cluster.id][idx_scale_out] = False
                    masks[cluster.id][idx_scale_in] = False
                    masks[cluster.id][idx_mig_start: idx_mig_start + len_nbr_list] = [False] * len_nbr_list

                if mid in placed_microservices:
                    masks[cluster.id][idx_place] = False

                if msvc.desired_replicas == 1:
                    masks[cluster.id][idx_scale_in] = False

                is_replica_limit_exceeded = msvc.desired_replicas >= msvc.max_tolerated_replicas
                req_mem_bytes = msvc.mem_demands_bytes
                req_cpu_cores = msvc.cpu_core_demands
                available_cpu_cores = (1 - cluster.cpu_core_utilization) * cluster.cpu_cores
                available_mem_gb = (1 - cluster.mem_utilization) * cluster.mem_gb
                available_mem_bytes = available_mem_gb * 1024 ** 3
                has_enough_resources = req_mem_bytes < available_mem_bytes and req_cpu_cores < available_cpu_cores
                
                if is_replica_limit_exceeded or not has_enough_resources:
                    masks[cluster.id][idx_scale_out] = False
                    reason = "Max Replicas" if is_replica_limit_exceeded else "No Resources"

                time_since_last_scale = self.clock.get_time_since("scale_" + msvc.id)
                if not app.is_deployed or time_since_last_scale < app.scaling_interval:
                    masks[cluster.id][idx_scale_out] = False
                    masks[cluster.id][idx_scale_in] = False
                    reason = "App Not Deployed" if not app.is_deployed else f"Cooldown ({time_since_last_scale:.1f}s)"

                time_since_last_migrate = self.clock.get_time_since("migrate_" + app.id)
                if not app.is_deployed or time_since_last_migrate < app.migration_interval or msvc.is_migration_in_progress:
                    masks[cluster.id][idx_mig_start: idx_mig_start + len_nbr_list] = [False] * len_nbr_list
                    reason = "App Not Deployed" if not app.is_deployed else ("In Progress" if msvc.is_migration_in_progress else "Cooldown")

                invalid_migration_idxs = []
                for j, nbr_cluster_id in enumerate(nbr_list):
                    c_nbr = self.state.get_cluster(nbr_cluster_id)
                    req_mem = msvc.mem_demands_bytes
                    req_cpu = msvc.cpu_core_demands
                    avail_cpu = (1 - c_nbr.cpu_core_utilization) * c_nbr.cpu_cores
                    avail_mem = (1 - c_nbr.mem_utilization) * c_nbr.mem_gb * 1024**3
                    
                    if req_mem >= avail_mem or req_cpu >= avail_cpu:
                        abs_idx = idx_mig_start + j
                        masks[cluster.id][abs_idx] = False

        return masks

    def decode_network_output(self, action_idx: int, nbr_list: List[cluster_id]) -> AgentAction:
        """
        Converts the Neural Network output vector (logits) into a concrete Action.
        """
        if action_idx == 0:
            return AgentAction(action_type=ActionType.NO_OP)

        current_idx = action_idx - 1

        if current_idx < self.num_place_actions:
            service_id = self.ms_list[current_idx]

            return AgentAction(
                action_type=ActionType.PLACE,
                service_id=service_id,
            )

        current_idx -= self.num_place_actions

        if current_idx < self.num_scale_actions:
            ms_index = current_idx // 2
            is_scale_out = (current_idx % 2 == 0)
            service_id = self.ms_list[ms_index]

            return AgentAction(
                action_type=ActionType.SCALE_OUT if is_scale_out else ActionType.SCALE_IN,
                service_id=service_id,
                magnitude=1
            )

        current_idx -= self.num_scale_actions

        len_nbr_list = len(nbr_list)
        if current_idx < self.num_mig_actions:
            ms_index = current_idx // len_nbr_list
            nbr_index = current_idx % len_nbr_list

            return AgentAction(
                action_type=ActionType.MIGRATE,
                service_id=self.ms_list[ms_index],
                target_cluster=nbr_list[nbr_index]
            )

        return AgentAction(action_type=ActionType.NO_OP)
