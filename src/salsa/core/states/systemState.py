from threading import RLock
from typing import Dict, List, Optional

from salsa.entities.cluster import Cluster
from salsa.entities.application import Application
from salsa.entities.microservice import Microservice
from salsa.utils.typing import microservice_id, application_id, cluster_id


class SystemState:
    def __init__(self):
        self.lock = RLock()

        self.clusters: Dict[cluster_id, Cluster] = {}
        self.applications: Dict[application_id, Application] = {}
        self.microservices: Dict[microservice_id, Microservice] = {}

    def add_cluster(self, cluster: Cluster):
        with self.lock:
            self.clusters[cluster.id] = cluster

    def get_cluster(self, cid: cluster_id) -> Optional[Cluster]:
        with self.lock:
            return self.clusters.get(cid)

    def get_all_clusters(self) -> List[Cluster]:
        with self.lock:
            return list(self.clusters.values())

    def add_application(self, app: Application):
        with self.lock:
            self.applications[app.id] = app

    def get_application(self, aid: application_id) -> Optional[Application]:
        with self.lock:
            return self.applications.get(aid)

    def get_all_applications(self) -> List[Application]:
        with self.lock:
            return list(self.applications.values())

    def find_application_by_microservice(self, mid: microservice_id) -> Optional[Application]:
        for app in self.applications.values():
            if mid in [ms.id for ms in app.microservices]:
                return app
        return None

    def add_microservice(self, microservice: Microservice):
        with self.lock:
            self.microservices[microservice.id] = microservice

    def get_microservice(self, mid: microservice_id) -> Optional[Microservice]:
        with self.lock:
            return self.microservices.get(mid)

    def get_all_microservices(self) -> List[Microservice]:
        with self.lock:
            return list(self.microservices.values())