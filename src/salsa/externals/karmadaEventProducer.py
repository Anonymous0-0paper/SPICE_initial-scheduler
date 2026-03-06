import threading
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from kubernetes import client, config, watch


class EventType(Enum):
    MIGRATION = auto()
    PLACEMENT = auto()
    SCALING = auto()

@dataclass
class ClusterEvent:
    event_type: EventType
    timestamp: float
    namespace: str
    resource_name: str
    resource_kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"<Event {self.event_type.name} | {self.namespace}/{self.resource_name}>"


class KarmadaMonitorEngine:
    def __init__(self, namespaces: List[str], cnfg, kube_context="karmada-apiserver"):
        self.namespaces = list(set(namespaces))  # Deduplicate
        self.context = kube_context

        self._buffer_lock = threading.Lock()
        self._event_buffer: List[ClusterEvent] = []

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        try:
            config.load_kube_config(config_file=cnfg['karmada']['apiserver_kubeconfig'], context=self.context)
            self.api_client = client.CustomObjectsApi()
        except Exception as e:
            print(f"Error loading kubeconfig: {e}")
            raise

    def start(self):
        if self._threads:
            return

        self._stop_event.clear()
        print(f"--- Starting Monitor for namespaces: {self.namespaces} ---")

        for ns in self.namespaces:
            t = threading.Thread(
                target=self._namespace_watch_loop,
                args=(ns,),
                name=f"Watch-{ns}",
                daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._threads.clear()

    def consume_events(self) -> List[ClusterEvent]:
        with self._buffer_lock:
            if not self._event_buffer:
                return []
            events = list(self._event_buffer)
            self._event_buffer.clear()
            return events

    def _commit_event(self, event: ClusterEvent):
        with self._buffer_lock:
            self._event_buffer.append(event)

    def _namespace_watch_loop(self, namespace: str):
        raise NotImplementedError


class MultiDeploymentMigrationMonitor(KarmadaMonitorEngine):

    def __init__(self,
                 target_deployments: List[str],
                 namespaces: List[str],
                 target_kind: str = "Deployment",
                 **kwargs):

        super().__init__(namespaces=namespaces, **kwargs)

        self.target_names: Set[str] = set(target_deployments)
        self.target_kind = target_kind

        self._state_lock = threading.Lock()
        self._cluster_state: Dict[str, List[str]] = {}
        self._replica_state: Dict[str, int] = {}

    def _get_clusters(self, obj) -> List[str]:
        return sorted([c['name'] for c in obj.get('spec', {}).get('clusters', [])])

    def _get_state_key(self, namespace, name):
        return f"{namespace}/{name}"

    def _get_replicas(self, obj) -> int:
        return obj.get('spec', {}).get('replicas', 0)

    def _namespace_watch_loop(self, namespace: str):
        print(f"[{namespace}] Thread started.")

        try:
            response = self.api_client.list_namespaced_custom_object(
                group="work.karmada.io",
                version="v1alpha2",
                namespace=namespace,
                plural="resourcebindings"
            )

            latest_rv = response['metadata']['resourceVersion']

            with self._state_lock:
                for item in response.get('items', []):
                    ref = item.get('spec', {}).get('resource', {})
                    name = ref.get('name')
                    kind = ref.get('kind')

                    if kind == self.target_kind and name in self.target_names:
                        key = self._get_state_key(namespace, name)
                        clusters = self._get_clusters(item)
                        self._cluster_state[key] = clusters
                        self._replica_state[key] = self._get_replicas(item)

        except Exception as e:
            print(f"[{namespace}] Init failed: {e}")
            return

        w = watch.Watch()
        stream = w.stream(
            self.api_client.list_namespaced_custom_object,
            group="work.karmada.io",
            version="v1alpha2",
            namespace=namespace,
            plural="resourcebindings",
            resource_version=latest_rv
        )

        try:
            for event in stream:
                if self._stop_event.is_set():
                    break

                obj = event['object']
                event_type = event['type']
                ref = obj.get('spec', {}).get('resource', {})
                name = ref.get('name')
                kind = ref.get('kind')

                if kind != self.target_kind or name not in self.target_names:
                    continue

                new_clusters = self._get_clusters(obj)
                new_replicas = self._get_replicas(obj)
                key = self._get_state_key(namespace, name)

                if event_type == 'DELETED':
                    with self._state_lock:
                        self._cluster_state.pop(key, None)
                        self._replica_state.pop(key, None)
                    continue

                with self._state_lock:
                    previous_clusters = self._cluster_state.get(key)

                    if previous_clusters != new_clusters:
                        if previous_clusters:
                            evt = ClusterEvent(
                                event_type=EventType.MIGRATION,
                                timestamp=time.time(),
                                namespace=namespace,
                                resource_name=name,
                                resource_kind=kind,
                                payload={
                                    "from": previous_clusters,
                                    "to": new_clusters
                                }
                            )
                            self._commit_event(evt)

                        elif not previous_clusters and new_clusters:
                            evt = ClusterEvent(
                                event_type=EventType.PLACEMENT,
                                timestamp=time.time(),
                                namespace=namespace,
                                resource_name=name,
                                resource_kind=kind,
                                payload={
                                    "from": [],
                                    "to": new_clusters
                                }
                            )
                            self._commit_event(evt)
                        self._cluster_state[key] = new_clusters

                    previous_replicas = self._replica_state.get(key)
                    if previous_replicas is not None and previous_replicas != new_replicas:
                        direction = "OUT" if new_replicas > previous_replicas else "IN"

                        evt = ClusterEvent(
                            event_type=EventType.SCALING,
                            timestamp=time.time(),
                            namespace=namespace,
                            resource_name=name,
                            resource_kind=kind,
                            payload={
                                "from": previous_replicas,
                                "to": new_replicas,
                                "delta": new_replicas - previous_replicas,
                                "direction": direction
                            }
                        )
                        self._commit_event(evt)

                    self._replica_state[key] = new_replicas
        except Exception as e:
            print(f"[{namespace}] Watcher crashed: {e}")
