import time
import threading
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from salsa.core.states.systemState import SystemState
from salsa.externals.clock import EventClock
from salsa.externals.karmadaEventProducer import MultiDeploymentMigrationMonitor, EventType
from salsa.core.states.SLOState import SLOState
from salsa.core.states.agentObservation import AgentObservation
from salsa.core.states.microServiceState import MicroserviceState
from salsa.core.states.monitor import MetricMonitor
from salsa.core.states.neighborClusterState import NeighborClusterState
from salsa.core.states.clusterState import ClusterState
from salsa.sloViolationPredictor.base_predictor import BasePredictor
from salsa.utils.typing import cluster_id, application_id


# ==========================================
# Math Helpers
# ==========================================

def compute_distance_metrics(
        observed_lat: float, slo_lat: float,
        observed_thr: float, slo_thr: float
) -> Tuple[float, float]:
    d_lat = (slo_lat - observed_lat) / slo_lat if slo_lat != 0 else 0.0
    d_thr = (observed_thr - slo_thr) / slo_thr if slo_thr != 0 else 0.0

    if observed_lat == 0:
        d_lat = -1.0

    return d_lat, d_thr

def compute_resource_utilization(history: List[Dict[str, float | str]]) -> Tuple[float, float]:
    if not history:
        return 0.0, 0.0

    cpu = float(history[-1]['utilized_cpu']) / float(history[-1]['allocatable_cpu'])
    mem = float(history[-1]['utilized_mem'].removesuffix('Ki')) / float(history[-1]['allocatable_mem'].removesuffix('Ki'))

    return cpu, mem

# ==========================================
# Data Snapshot
# ==========================================

class MonitorSnapshot:
    def __init__(self, monitor: MetricMonitor):
        with monitor.metrics_lock:
            self.cluster_resources = {k: list(v) for k, v in monitor.per_cluster_resources_history.items()}
            self.cluster_replicas = {k: list(v) for k, v in monitor.per_cluster_replication_history.items()}

            self.svc_rates = {k: list(v) for k, v in monitor.per_svc_request_rate_history.items()}
            self.svc_delays = {k: list(v) for k, v in monitor.per_svc_delay_history.items()}
            self.svc_internal_delays = {k: list(v) for k, v in monitor.per_svc_internal_delay_history.items()}

            self.app_latency = {k: list(v) for k, v in monitor.per_app_latency_history.items()}
            self.app_throughput = {k: list(v) for k, v in monitor.per_app_throughput_history.items()}


# ==========================================
# Concurrent Builder
# ==========================================

class ObservationBuilder:
    def __init__(self, monitor: MetricMonitor,
                 karmada_event_monitor: MultiDeploymentMigrationMonitor,
                 clock: EventClock,
                 latency_predictors: Dict[application_id, BasePredictor],
                 throughput_predictors: Dict[application_id, BasePredictor],
                 state: SystemState,
                 config: Dict[str, Any]):
        self.config = config
        self.monitor = monitor
        self.state = state
        self.clock: EventClock = clock

        self.karmada_event_monitor = karmada_event_monitor

        self.latency_predictors: Dict[application_id, BasePredictor] = latency_predictors
        self.throughput_predictors: Dict[application_id, BasePredictor] = throughput_predictors

        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.horizon = config["slo_predictor"]["horizon_in_seconds"] / config["scheduler"]["step_interval"]

    def update_entities(self, karmada_event_list):
        for event in karmada_event_list:
            # Handle Migration Events
            if event.event_type == EventType.MIGRATION:
                svc_name = event.resource_name
                app_name = event.namespace
                msvc_id = f"{svc_name}_{app_name}"

                from_clusters: List[cluster_id] = event.payload["from"]
                to_clusters: List[cluster_id] = event.payload["to"]
                msvc = self.state.get_microservice(msvc_id)
                msvc.is_migration_in_progress = True

                for cid in from_clusters:
                    c = self.state.get_cluster(cid)

                    # Remove Microservice from src_cluster
                    if msvc_id in c.microservices:
                        c.microservices.remove(msvc_id)
                        app_id = msvc.app_id

                        # check if microservice of the same app still exists in the src_cluster
                        exists = False
                        for mid in c.microservices:
                            c_ms = self.state.get_microservice(mid)
                            if c_ms.app_id == app_id:
                                exists = True
                                break

                        # remove application from src_cluster if no such microservice exists
                        if not exists:
                            c.applications.remove(app_id)

                for cid in to_clusters:
                    c = self.state.get_cluster(cid)

                    # Add Microservice to dst_cluster
                    if msvc_id not in c.microservices:
                        c.microservices.append(msvc_id)
                        app_id = msvc.app_id

                        # Add microservice application to dst_cluster if not present
                        if app_id not in c.applications:
                            c.applications.append(app_id)

                self.clock.touch("scale_" + msvc_id)
                self.clock.touch("migrate_" + msvc.app_id)

            # Handle Placement Events
            elif event.event_type == EventType.PLACEMENT:
                svc_name = event.resource_name
                app_name = event.namespace
                msvc_id = f"{svc_name}_{app_name}"

                to_clusters: List[cluster_id] = event.payload["to"]
                msvc = self.state.get_microservice(msvc_id)
                for cid in to_clusters:
                    c = self.state.get_cluster(cid)

                    # Add Microservice to dst_cluster
                    if msvc_id not in c.microservices:
                        c.microservices.append(msvc_id)
                        app_id = msvc.app_id
                        if app_id not in c.applications:
                            c.applications.append(app_id)

                app = self.state.get_application(app_name)
                if msvc_id in app.undeployed_microservices:
                    app.undeployed_microservices.remove(msvc_id)
                app.is_deployed = len(app.undeployed_microservices) == 0

                self.clock.touch("scale_" + msvc_id)
                self.clock.touch("migrate_" + msvc.app_id)

            # Handle Scaling Events
            elif event.event_type == EventType.SCALING:
                svc_name = event.resource_name
                app_name = event.namespace
                msvc_id = f"{svc_name}_{app_name}"
                self.clock.touch("scale_" + msvc_id)

    def build_all_observations(self, target_cluster_ids: List[str], rnd: int) -> Dict[cluster_id, AgentObservation]:
        snapshot = MonitorSnapshot(self.monitor)
        karmada_event_list = self.karmada_event_monitor.consume_events()

        # Update Entities based on events
        self.update_entities(karmada_event_list)


        slo_cache: Dict[str, SLOState] = {}
        slo_cache_lock = threading.Lock()

        # Compute Local State and Self-Risk (Parallel)
        futures = {}
        for cid in target_cluster_ids:
            futures[cid] = self.thread_pool.submit(
                self._compute_partial_observation,
                cid,
                snapshot,
                slo_cache,
                slo_cache_lock,
                rnd=rnd,
            )

        # {cluster_id: (PartialObservationObject, max_violation_prob)}
        partial_results = {}
        for cid, future in futures.items():
            partial_results[cid] = future.result()

        final_observations = {}

        # res[1] -> max_violation_prob from _compute_partial_observation
        cluster_risk_map = {cid: res[1] for cid, res in partial_results.items()}

        for cid, (obs, _) in partial_results.items():
            for neighbor_id, neighbor_state in obs.neighbors.items():
                if neighbor_id in cluster_risk_map:
                    neighbor_state.global_slo_risk_score = cluster_risk_map[neighbor_id]
                else:
                    neighbor_state.global_slo_risk_score = 0.0
                    print(f"WARNING: Missing risk score for neighbor {neighbor_id} in cluster {cid}")

            final_observations[cid] = obs

        return final_observations

    def _compute_partial_observation(
            self,
            target_cluster_id: str,
            snapshot: MonitorSnapshot,
            slo_cache: Dict[str, SLOState],
            slo_cache_lock: threading.Lock,
            rnd: int
    ) -> Tuple[AgentObservation, float]:

        target_cluster = self.state.clusters.get(target_cluster_id)
        if not target_cluster:
            raise ValueError(f"Cluster {target_cluster_id} not found in SystemState")

        # --- A. Cluster State ---
        raw_res = list(snapshot.cluster_resources.get(target_cluster_id, []))
        cpu, mem = compute_resource_utilization(raw_res)

        cluster_state = ClusterState(
            cluster_id=target_cluster_id,
            cpu_utilization=cpu,
            mem_utilization=mem
        )

        # --- B. Microservice States ---
        microservices_map = {}
        relevant_svcs = [s for s in snapshot.svc_rates.keys() if s in target_cluster.microservices]

        for svc_id in relevant_svcs:
            ms_state = self._build_microservice_state(svc_id, target_cluster_id, snapshot)
            if ms_state:
                microservices_map[svc_id] = ms_state

        # --- C. SLO States (with Memoization) ---
        applications_map = {}
        relevant_apps = [a for a in snapshot.app_latency.keys() if a in target_cluster.applications]

        max_violation_prob = 0.0
        for app_id in relevant_apps:
            with slo_cache_lock:

                cached_slo = slo_cache.get(app_id)

                if cached_slo:
                    slo_obj = copy.deepcopy(cached_slo)
                else:
                    slo_obj = self._compute_slo_state(app_id, snapshot, rnd=rnd)

                if app_id not in slo_cache:
                    slo_cache[app_id] = slo_obj
                else:
                    slo_obj = copy.deepcopy(slo_cache[app_id])

            applications_map[app_id] = slo_obj
            if slo_obj.predicted_violation_prob > max_violation_prob:
                max_violation_prob = slo_obj.predicted_violation_prob

        # --- D. Neighbor States (Incomplete) ---
        neighbors_map = {}
        all_known_clusters = snapshot.cluster_resources.keys()
        neighbor_ids = [nid for nid in all_known_clusters if nid != target_cluster_id]

        for nid in neighbor_ids:
            hist = snapshot.cluster_resources.get(nid, [])
            n_cpu, n_mem = compute_resource_utilization(hist)

            # Retrieve static network latency
            neighbor_obj = self.state.get_cluster(nid)
            neighbor_tier = neighbor_obj.tier_type.name.lower()
            latency_val = target_cluster.get_network_latency(neighbor_tier)

            neighbors_map[nid] = NeighborClusterState(
                cluster_id=nid,
                cpu_utilization=n_cpu,
                mem_utilization=n_mem,
                global_slo_risk_score=0.0, # Placeholder, filled in at a later step
                network_latency_to_neighbor=latency_val
            )

        observation = AgentObservation(
            cluster_id=target_cluster_id,
            timestamp=time.time(),
            node=cluster_state,
            microservices=microservices_map,
            applications=applications_map,
            neighbors=neighbors_map
        )

        return observation, max_violation_prob

    def _build_microservice_state(self, svc_id: str, cid: str, snapshot: MonitorSnapshot) -> Optional[
        MicroserviceState]:
        app_id = self.state.get_microservice(svc_id).app_id
        app = self.state.get_application(app_id)
        if not app:
            return None

        r_hist = snapshot.svc_rates.get(svc_id, [])
        d_hist = snapshot.svc_delays.get(svc_id, [])

        cluster_rep_hist = snapshot.cluster_replicas.get(cid, [])
        rep_info = {}
        if cluster_rep_hist:
            rep_info = cluster_rep_hist[-1].get(svc_id, {})
        replicas_desired = rep_info.get('desired_replicas', 0)
        replicas_effective = rep_info.get('ready_replicas', 0)

        # Migration Status
        ms_def = next((m for m in app.microservices if m.id == svc_id), None)

        is_migrating = ms_def.is_migration_in_progress and not (replicas_desired == replicas_effective) if ms_def else False

        return MicroserviceState(
            service_id=svc_id,
            app_id=app.id,
            request_rate=r_hist[-1] if r_hist else 0.0,
            queue_length=0,
            response_time=d_hist[-1] if d_hist else 0.0,
            replicas_desired=replicas_desired,
            replicas_effective=replicas_effective,
            replicas_starting=max(0, replicas_desired - replicas_effective),
            seconds_since_last_scale=self.clock.get_time_since(f"scale_{svc_id}"),
            seconds_since_last_migrate=self.clock.get_time_since(f"migrate_{svc_id}"),
            is_migration_in_progress=is_migrating,
        )

    def _compute_slo_state(self, app_id: str, snapshot: MonitorSnapshot, rnd: int) -> SLOState:
        app = self.state.applications.get(app_id)

        latency_threshold = app.slos["latency"]
        throughput_threshold = app.slos["throughput"]

        l_hist = snapshot.app_latency.get(app_id, [])
        t_hist = snapshot.app_throughput.get(app_id, [])

        current_lat = l_hist[-1] if l_hist else 0.0
        current_thr = t_hist[-1] if t_hist else 0.0

        is_violating = (current_lat >= latency_threshold) or (current_thr < throughput_threshold)
        with app.lock:
            if is_violating and app.is_deployed and app.receives_work:
                app.sloTracker.report_violation()
            app.sloTracker.flush_violations(rnd=rnd)
            history_window = list(app.sloTracker.violation_history.values())

        # Inputs: DataFrame construction from Snapshot dicts
        last_lats = {k: v[-1] if v[-1] else 0 for k, v in snapshot.svc_delays.items() if v}
        last_thrs = {k: 1000/v[-1] if v[-1] and v[-1] != 0 else 0 for k, v in snapshot.svc_internal_delays.items() if v}
        app_lat = snapshot.app_latency.get(app_id, [0.0])[-1]
        app_thr = snapshot.app_throughput.get(app_id, [0.0])[-1]

        lat_prob = self.latency_predictors[app_id].predict(last_lats, context={"app_val": app_lat})
        thr_prob = self.throughput_predictors[app_id].predict(last_thrs, context={"app_val": app_thr})


        total_prob = lat_prob + thr_prob - (lat_prob * thr_prob)
        app.sloTracker.report_violation_prediction(total_prob)
        app.sloTracker.flush_violation_prediction(rnd=rnd + self.horizon)

        d_lat, d_thr = compute_distance_metrics(current_lat, latency_threshold, current_thr, throughput_threshold)

        return SLOState(
            app_id=app_id,
            current_latency=current_lat if app.is_deployed else 0.0,
            current_throughput=current_thr if app.is_deployed else 0.0,
            current_availability=1.0 if app.is_deployed else 0.0,
            is_violating=is_violating if app.is_deployed and app.get_receives_work() else False,
            violation_history_window=history_window,
            predicted_violation_prob=total_prob,
            dist_latency=d_lat if app.is_deployed else -1.0,
            dist_throughput=d_thr if app.is_deployed else -1.0,
        )
