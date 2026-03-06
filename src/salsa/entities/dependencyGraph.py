import json
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional

from salsa.utils.typing import microservice_id, dependency_graph_id, application_id


@dataclass
class DependencyGraph:
    id: dependency_graph_id
    app_id: application_id
    edges: Dict[microservice_id, Dict[microservice_id, float]]
    critical_path: Optional[List[microservice_id]]

    def find_consumers(self, mid: microservice_id) -> List[microservice_id]:
        consumers = []
        for src, targets in self.edges.items():
            if mid in targets:
                consumers.append(src)
        return consumers

    def find_consumees(self, mid: microservice_id) -> List[microservice_id]:
        return list(self.edges[mid].keys())

    def get_critical_path(self) -> List[microservice_id]:
        if self.critical_path is None:
            path = self.find_critical_path()
            self.critical_path = path
        return self.critical_path

    def reevaluate_critical_path(self, latencies_per_service: Dict[microservice_id, float]) -> List[microservice_id]:
        for svc, target_svcs in self.edges.items():
            for target_svc, weight in target_svcs.items():
                self.edges[svc][target_svc] = latencies_per_service[target_svc]

        self.critical_path = None
        return self.get_critical_path()

    def find_critical_path(self) -> List[microservice_id]:
        input_degree = {ms: 0 for ms in self.edges}
        all_ms = set(self.edges.keys())
        for src, targets in self.edges.items():
            for target, weight in targets.items():
                all_ms.add(target)
                input_degree[target] = input_degree.get(target, 0) + 1

        dist: Dict[microservice_id, float] = {ms: -float('inf') for ms in all_ms}
        predecessor: Dict[microservice_id, Optional[microservice_id]] = {node: None for node in all_ms}

        queue = deque([ms for ms in all_ms if input_degree.get(ms, 0) == 0])

        for ms in queue:
            dist[ms] = 0.0

        topo_order = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)

            if u in self.edges.keys():
                for v, weight in self.edges[u].items():
                    if dist[u] + weight > dist[v]:
                        dist[v] = dist[u] + weight
                        predecessor[v] = u

                    input_degree[v] -= 1
                    if input_degree[v] == 0:
                        queue.append(v)

        if not dist:
            return []

        end_node = max(dist, key=dist.get)

        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = predecessor[curr]

        return path[::-1]