import json

import numpy as np
from typing import Dict
from pydantic import BaseModel

from salsa.core.states.SLOState import SLOState
from salsa.core.states.microServiceState import MicroserviceState
from salsa.core.states.neighborClusterState import NeighborClusterState
from salsa.core.states.clusterState import ClusterState
from salsa.core.states.constants import (
    MAX_MICROSERVICES,
    MAX_NEIGHBORS,
    MAX_APPLICATIONS,
    DIM_MICROSERVICE,
    DIM_APPLICATION,
    DIM_NEIGHBOR
)


class AgentObservation(BaseModel):
    cluster_id: str
    timestamp: float
    node: ClusterState
    microservices: Dict[str, MicroserviceState]
    applications: Dict[str, SLOState]
    neighbors: Dict[str, NeighborClusterState]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Converts the observation into a dictionary of fixed-size numpy arrays 
        ready for the DeepSet Actor. Uses Zero-Padding for missing slots.
        """

        # 1. Cluster State (Static, Shape: [DIM_CLUSTER])
        cluster_arr = np.array([
            self.node.cpu_utilization,
            self.node.mem_utilization
        ], dtype=np.float32)

        # 2. Microservices (Set, Shape: [MAX_MICROSERVICES, DIM_MICROSERVICE])
        ms_arr = self._pad_and_stack(
            items=self.microservices,
            max_len=MAX_MICROSERVICES,
            feature_dim=DIM_MICROSERVICE
        )

        # 3. Applications (Set, Shape: [MAX_APPLICATIONS, DIM_APPLICATION])
        app_arr = self._pad_and_stack(
            items=self.applications,
            max_len=MAX_APPLICATIONS,
            feature_dim=DIM_APPLICATION
        )

        # 4. Neighbors (Set, Shape: [MAX_NEIGHBORS, DIM_NEIGHBOR])
        nbr_arr = self._pad_and_stack(
            items=self.neighbors,
            max_len=MAX_NEIGHBORS,
            feature_dim=DIM_NEIGHBOR
        )

        return {
            "cluster": cluster_arr,
            "microservices": ms_arr,
            "slos": app_arr,
            "neighbors": nbr_arr
        }

    def _pad_and_stack(self, items: Dict, max_len: int, feature_dim: int) -> np.ndarray:
        # A. Sort keys for deterministic input
        sorted_ids = sorted(items.keys())

        # B. Collect valid vectors up to MAX_LEN
        vectors = []
        for item_id in sorted_ids[:max_len]:
            obj = items[item_id]
            # Assumes substates implement get_as_list() returning List[float]
            vectors.append(obj.get_as_list())

        # C. Convert to Numpy
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
        else:
            # Handle empty case (e.g. no neighbors)
            arr = np.zeros((0, feature_dim), dtype=np.float32)

        # D. Zero-Padding
        rows_present = arr.shape[0]
        rows_needed = max_len - rows_present

        if rows_needed > 0:
            padding = np.zeros((rows_needed, feature_dim), dtype=np.float32)
            arr = np.vstack([arr, padding]) if rows_present > 0 else padding

        return arr

    def to_json_str(self) -> str:
        """Serializes the object to a JSON string."""
        return json.dumps(
            self.__dict__,
            default=lambda o: o.__dict__,
            indent=2
        )

    @classmethod
    def from_json_str(cls, json_str: str):
        """Deserializes a JSON string back into an object."""
        data = json.loads(json_str)

        return cls(
            cluster_id=data["cluster_id"],
            timestamp=data["timestamp"],
            node=ClusterState(**data["node"]),
            microservices={k: MicroserviceState(**v) for k, v in data["microservices"].items()},
            applications={k: SLOState(**v) for k, v in data["applications"].items()},
            neighbors={k: NeighborClusterState(**v) for k, v in data["neighbors"].items()}
        )