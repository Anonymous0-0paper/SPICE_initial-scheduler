from typing import Dict, List

from kubernetes import client, config

from salsa.utils.sizeParser import parse_cpu, parse_memory_bytes
from salsa.utils.typing import cluster_id


class KarmadaInfo:
    def __init__(self, kube_api):
        config.load_kube_config(kube_api, context="karmada-apiserver")
        self.custom_api = client.CustomObjectsApi()
        self.api_client = client.ApiClient()
        self.apps_api = client.AppsV1Api()

    def get_cluster_utilization(self, cluster_name):
        """
        :param cluster_name:
        :return dictionary of cpu utilization in #cores (may be decimal), memory utilization in Kibibytes:
        """
        path = f"/apis/cluster.karmada.io/v1alpha1/clusters/{cluster_name}/proxy/apis/metrics.k8s.io/v1beta1/pods"
        resp = self.api_client.call_api(path, "GET", response_type="object", _preload_content=True)

        total_cpu = 0
        total_mem = 0

        pods = resp[0]
        for pod in pods.get("items", []):
            for container in pod['containers']:
                total_cpu += parse_cpu(container['usage']['cpu'])
                total_mem += parse_memory_bytes(container['usage']['memory'])

        return {"cpu": total_cpu, "mem": f"{int(total_mem / 1024)}Ki"}

    def get_cluster_resource_util(self, cluster: cluster_id) -> Dict[str, str | int]:
        clusters = self.custom_api.list_cluster_custom_object(
            group="cluster.karmada.io",
            version="v1alpha1",
            plural="clusters"
        )

        for c in clusters["items"]:
            if c["metadata"]["name"] == cluster:
                summary = c.get("status", {}).get("resourceSummary", {})
                capacity_summary = self.get_cluster_utilization(cluster)
                return {
                    "allocatable_cpu": summary.get("allocatable", {}).get("cpu"),
                    "allocatable_mem": summary.get("allocatable", {}).get("memory"),
                    "utilized_cpu": capacity_summary.get("cpu", {}),
                    "utilized_mem": capacity_summary.get("mem", {}),
                }
        return {}

    def get_microservice_replication(self, cluster: str, namespaces: List[str] = None) -> Dict[str, Dict]:
        if namespaces is None:
            namespaces = []

        clusters_list = self.custom_api.list_cluster_custom_object(
            group="cluster.karmada.io", version="v1alpha1", plural="clusters"
        )
        if not any(c["metadata"]["name"] == cluster for c in clusters_list["items"]):
            return {}

        microservice_replication_info = {}

        for ns in namespaces:
            try:
                deployments = self.apps_api.list_namespaced_deployment(namespace=ns)
                bindings_json = self.custom_api.list_namespaced_custom_object(
                    group="work.karmada.io",
                    version="v1alpha1",
                    namespace=ns,
                    plural="resourcebindings"
                )

                deployment_to_binding = {}
                for b in bindings_json.get('items', []):
                    res = b.get('spec', {}).get('resource', {})

                    if res.get('kind') == 'Deployment':
                        deploy_name = res.get('name')
                        deployment_to_binding[deploy_name] = b

                for d in deployments.items:
                    name = d.metadata.name

                    binding = deployment_to_binding.get(name)

                    if not binding:
                        continue

                    target_clusters = [t['name'] for t in binding.get('spec', {}).get('clusters', [])]
                    if cluster not in target_clusters:
                        continue

                    cluster_stats = {}
                    aggr_status_list = binding.get('status', {}).get('aggregatedStatus', [])

                    for status_item in aggr_status_list:
                        if status_item.get('clusterName') == cluster:
                            cluster_stats = status_item.get('status', {})
                            break

                    ready_replicas = cluster_stats.get('readyReplicas', 0)
                    available_replicas = cluster_stats.get('availableReplicas', 0)
                    microservice_replication_info[name + "_" + ns] = {
                        "name": name,
                        "namespace": ns,
                        "labels": d.metadata.labels,
                        "desired_replicas": d.spec.replicas,
                        "ready_replicas": ready_replicas,
                        "available_replicas": available_replicas,
                        "unavailable_replicas": d.spec.replicas - ready_replicas
                    }

            except Exception as e:
                print(f"Warning: Failed to process namespace '{ns}': {e}")
                continue

        return microservice_replication_info
