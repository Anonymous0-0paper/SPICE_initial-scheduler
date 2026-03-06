import math
import time
import threading
from collections import deque, defaultdict
from typing import Deque, Dict, Tuple

from salsa.core.states.systemState import SystemState
from salsa.externals.karmadaInfo import KarmadaInfo
from salsa.externals.thanosQuery import ThanosQuery
from salsa.utils.typing import microservice_id


class MetricMonitor:
    def __init__(self, system_state: SystemState, config):
        self.state = system_state
        self.interval = config['scheduler']['step_interval']
        self._running = False
        self._threads = []

        self.thanos_query = ThanosQuery(config)
        self.karmada_info = KarmadaInfo(config["karmada"]["apiserver_kubeconfig"])

        self.per_app_latency_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.per_app_throughput_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.per_svc_request_rate_history: Dict[microservice_id, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.per_svc_delay_history: Dict[microservice_id, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.per_svc_internal_delay_history: Dict[microservice_id, Deque] = defaultdict(lambda: deque(maxlen=100))

        # Cluster specific dicts
        self.per_cluster_resources_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.per_cluster_replication_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))

        self.queries_to_run = ["svc_request_rate", "svc_delay", "svc_internal_delay"]
        self.query_metrics_map = {
            "svc_request_rate": self.per_svc_request_rate_history,
            "svc_delay": self.per_svc_delay_history,
            "svc_internal_delay": self.per_svc_internal_delay_history,
        }

        self.metrics_lock = threading.Lock()

    def start(self):
        self._running = True
        print("Starting Monitoring Threads...")

        t_global = threading.Thread(target=self._global_monitor_loop, daemon=True, name="Monitor-Global-Thanos")
        t_global.start()
        self._threads.append(t_global)

        for cluster in self.state.get_all_clusters():
            cid = cluster.id
            t_cluster = threading.Thread(target=self._cluster_monitor_loop, args=(cid,), daemon=True,
                                         name=f"Monitor-Cluster-{cid}")
            t_cluster.start()
            self._threads.append(t_cluster)

    def stop(self):
        self._running = False
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._threads.clear()
        print("Monitoring Threads Stopped")

    def _global_monitor_loop(self):


        while self._running:
            try:
                thanos_metrics = self.thanos_query.run_all_queries(self.queries_to_run)

                with self.metrics_lock:
                    if "svc_request_rate" in thanos_metrics:
                        for app_id, ms_dict in thanos_metrics["svc_request_rate"].items():
                            for ms_id, val in ms_dict.items():
                                if math.isnan(val) or math.isinf(val):
                                    val = 0.0
                                combined_id = f"{ms_id}_{app_id}"
                                self.per_svc_request_rate_history[combined_id].append(val)

                    if "svc_delay" in thanos_metrics:
                        for app_id, ms_dict in thanos_metrics["svc_delay"].items():
                            for ms_id, val in ms_dict.items():
                                if math.isnan(val) or math.isinf(val):
                                    val = 0.0
                                combined_id = f"{ms_id}_{app_id}"
                                self.per_svc_delay_history[combined_id].append(val)

                    if "svc_internal_delay" in thanos_metrics:
                        for app_id, ms_dict in thanos_metrics["svc_internal_delay"].items():
                            for ms_id, val in ms_dict.items():
                                if math.isnan(val) or math.isinf(val):
                                    val = 0.0
                                combined_id = f"{ms_id}_{app_id}"
                                self.per_svc_internal_delay_history[combined_id].append(val)

                    # Handle Microservices not returning metrics (yet)
                    for msvc in self.state.get_all_microservices():
                        for query_id, ms_dict in thanos_metrics.items():
                            mid = msvc.id
                            deployment_id = mid.removesuffix("_" + msvc.app_id)
                            if deployment_id not in ms_dict.get(msvc.app_id, {}).keys():
                                self.query_metrics_map[query_id][mid].append(0)

                    for app in self.state.get_all_applications():
                        aid = app.id
                        app_latency, app_throughput = self.compute_app_latency_and_throughput(aid)
                        self.per_app_latency_history[aid].append(app_latency)
                        self.per_app_throughput_history[aid].append(app_throughput)
                        if app_latency != 0.0 and app_throughput != 0.0:
                            app.set_receives_work(True)
                        else:
                            app.set_receives_work(False)

            except Exception as e:
                print(f"[Global Monitor] Error: {e}")

                time.sleep(self.interval)

    def _cluster_monitor_loop(self, cid: str):
        while self._running:
            try:
                resources = self.karmada_info.get_cluster_resource_util(cid)
                apps = [c for c in self.state.get_all_clusters() if c.id == cid][0].applications
                replication = self.karmada_info.get_microservice_replication(cid, namespaces=apps)

                with self.metrics_lock:
                    self.per_cluster_resources_history[cid].append(resources)
                    self.per_cluster_replication_history[cid].append(replication)

            except Exception as e:
                print(f"[Cluster Monitor {cid}] Error: {e}")

            time.sleep(self.interval)

    def compute_app_latency_and_throughput(self, aid: str) -> Tuple[float, float]:
        app_obj = self.state.get_application(aid)

        per_svc_current_delay = {msvc: dq[-1] for msvc, dq in self.per_svc_delay_history.items() if msvc.endswith(f"_{aid}")}

        critical_path_ids = app_obj.dependency_graph.reevaluate_critical_path(per_svc_current_delay)

        latency_sum = 0.0
        for ms_id in critical_path_ids:
            history = self.per_svc_delay_history[ms_id]
            if history:
                latency_sum += history[-1]

        most_internal_latency = 0.0
        for ms_id in critical_path_ids:
            history = self.per_svc_internal_delay_history[ms_id]
            if history:
                most_internal_latency = max(most_internal_latency, history[-1])
        if most_internal_latency > 0:
            throughput = 1000 / most_internal_latency
        else:
            throughput = 0
        return latency_sum, throughput

    def clear_histories(self):
        with self.metrics_lock:
            self.per_app_latency_history.clear()
            self.per_app_throughput_history.clear()
            self.per_svc_request_rate_history.clear()
            self.per_svc_delay_history.clear()
            self.per_svc_internal_delay_history.clear()
            self.per_cluster_resources_history.clear()
            self.per_cluster_replication_history.clear()
