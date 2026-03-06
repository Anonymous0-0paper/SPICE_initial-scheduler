import threading
import time
from kubernetes.dynamic.exceptions import UnprocessibleEntityError
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from salsa.core.actions.agentAction import AgentAction, ActionType
from salsa.core.states.systemState import SystemState
from salsa.externals.karmadaClient import KarmadaClient
from salsa.utils.yaml_io import load_manifests
from salsa.utils.typing import cluster_id

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent  # parent of src/

work_path = BASE_DIR / "work"


class KubernetesExecutor:
    def __init__(self, k8s_client: KarmadaClient , state: SystemState):
        self.k8s_client = k8s_client
        self.state = state
        self._mcs_lock = threading.Lock()

    def execute_actions(self, actions: Dict[cluster_id, AgentAction]) -> None:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for source_cluster, action in actions.items():
                futures.append(executor.submit(self.execute_single_action, action, source_cluster))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error executing action: {e}")

    def execute_single_action(self, action: AgentAction, source_cluster: cluster_id):
        print(f"({source_cluster}) PICKED ACTION: {action.action_type}")
        if action.action_type == ActionType.NO_OP:
            return

        msvc = self.state.get_microservice(action.service_id)
        namespace = msvc.app_id
        is_entrypoint = self.state.get_application(namespace).entrypoint == msvc.id

        deployment_id = msvc.id.removesuffix("_" + msvc.app_id)
        dependency_graph = self.state.get_application(msvc.app_id).dependency_graph
        dependency_providers = [mid.removesuffix("_" + msvc.app_id) for mid in dependency_graph.find_consumees(msvc.id)]

        if is_entrypoint:
            consumer_cids = [c.id for c in self.state.get_all_clusters()]
        else:
            consumer_mids = [mid.removesuffix("_" + msvc.app_id) for mid in dependency_graph.find_consumers(msvc.id)]
            consumer_cids = [c.id for c in self.state.get_all_clusters() for mid in consumer_mids if mid in c.microservices]

        if action.action_type == ActionType.PLACE:
            self.place(deployment_id=deployment_id,
                       namespace=namespace,
                       cid=source_cluster,
                       consumer_ids=consumer_cids,
                       dependency_providers=dependency_providers
            )
        elif action.action_type == ActionType.SCALE_OUT or action.action_type == ActionType.SCALE_IN:
            self.scale(deployment_id=deployment_id,
                       namespace=namespace,
                       magnitude=action.magnitude * (1 if action.action_type == ActionType.SCALE_OUT else -1)
            )
        elif action.action_type == ActionType.MIGRATE:
            self.migrate(deployment_id=deployment_id,
                         namespace=namespace,
                         src_cluster_id=source_cluster,
                         dst_cluster_id=action.target_cluster,
                         consumer_ids=consumer_cids,
                         dependency_providers=dependency_providers
            )
        else:
            pass

    def place(self, deployment_id, namespace, cid, consumer_ids, dependency_providers):
        # Prepare Deployment Propagation
        deploy_pp_manifest, = load_manifests("propagation_policy.yaml", context={
            "policy_name": f"{deployment_id}-propagation",
            "api_version": "apps/v1",
            "resource": "Deployment",
            "resource_id": deployment_id,
            "target_member": [cid],
            "namespace_name": namespace
        })
        self.k8s_client.apply(deploy_pp_manifest)

        # Prepare MultiClusterService (MCS)
        mcs_manifest, = load_manifests("multicluster_service.yaml", context={
            "service_name": f"{deployment_id}",
            "provider_clusters": [{"name": cid}],
            "consumer_clusters": [{"name": cons_id} for cons_id in consumer_ids],
            "namespace_name": namespace
        })

        self.k8s_client.apply(mcs_manifest)

        for dep_id in dependency_providers:
            if "_" in dep_id:
                dep_namespace = dep_id.split("_")[-1]
                dep_name = dep_id.rpartition('_')[0]
            else:
                dep_namespace = namespace
                dep_name = dep_id
            try:
                self.update_mcs_consumers(dep_name, namespace=dep_namespace, add=[cid], remove=[])
            except Exception as e:
                print(f"Warning: Could not register as consumer of dependency {dep_name} ({dep_namespace}). You can safely ignore this warning if the service has not been placed yet.")
        print(f"Placed {deployment_id} ({namespace}) on {cid}")

    def scale(self, deployment_id, namespace, magnitude):
        # Fetch current state
        deployment = self.k8s_client.get("Deployment", name=deployment_id, namespace=namespace)
        current_replicas = deployment.spec.replicas

        # Calculate new state
        new_replicas = current_replicas + magnitude
        if new_replicas < 1:
            new_replicas = 1

        # Patch the deployment
        patch_payload = {"spec": {"replicas": new_replicas}}
        self.k8s_client.patch("Deployment", name=deployment_id, body=patch_payload, namespace=namespace)

        print(f"Scaled {deployment_id} ({namespace}) from {current_replicas} to {new_replicas}")

    def migrate(self, deployment_id, namespace, src_cluster_id, dst_cluster_id, consumer_ids, dependency_providers):
        mcs_name = f"{deployment_id}"
        deploy_pp_name = f"{deployment_id}-propagation"

        # Update Dependencies (OUTBOUND Traffic)
        for dep_id in dependency_providers:
            try:
                self.update_mcs_consumers(f"{dep_id}", namespace=namespace, add=[dst_cluster_id], remove=[])
            except Exception as e:
                print(f"Warning: Could not register as consumer of dependency {dep_id} ({namespace}). You can safely ignore this warning if the service has not been placed yet.")
        # Update Own MCS (OUTBOUND Traffic)
        self.update_mcs_providers(mcs_name, namespace=namespace, add=[dst_cluster_id], remove=[], consumer_ids=consumer_ids)

        # Update Propagation Policy (Move Workload)
        # Instead of patching placement, we re-render and APPLY the policy targeting the new cluster
        deploy_pp_manifest, = load_manifests("propagation_policy.yaml", context={
            "policy_name": deploy_pp_name,
            "namespace_name": namespace,
            "api_version": "apps/v1",
            "resource": "Deployment",
            "resource_id": deployment_id,
            "target_member": [dst_cluster_id]
        })
        self.k8s_client.apply(deploy_pp_manifest)

        # Update Dependencies (Remove Old Access)
        for dep_id in dependency_providers:
            try:
                src_cluster_microservices = [self.state.get_microservice(mid) for mid in self.state.get_cluster(src_cluster_id).microservices if mid != deployment_id + '_' + namespace]
                # for each microservice check if it consumes dep_id
                still_needed = False
                for msvc in src_cluster_microservices:
                    if msvc.id in self.state.get_application(msvc.app_id).dependency_graph.find_consumers(dep_id):
                        still_needed = True
                        break
                if not still_needed:
                    self.update_mcs_consumers(f"{dep_id}", namespace=namespace, add=[], remove=[src_cluster_id])
            except Exception as e:
                print(f"Warning: Could not register as consumer of dependency {dep_id} ({namespace}). You can safely ignore this warning if the service has not been placed yet.")

        # Update Own MCS (Remove Old Provider)
        self.update_mcs_providers(mcs_name, namespace=namespace, add=[], remove=[src_cluster_id], consumer_ids=consumer_ids)

        print(f"Migrated {deployment_id} ({namespace}) from {src_cluster_id} to {dst_cluster_id}")

    def update_mcs_consumers(self, mcs_name, namespace, add, remove):
        with self._mcs_lock:
            mcs = self.k8s_client.get("MultiClusterService", name=mcs_name, namespace=namespace)

            existing_consumers = {c['name'] for c in mcs.spec.consumerClusters} if mcs.spec.consumerClusters else set()

            updated_set = existing_consumers.union(set(add)) - set(remove)
            if updated_set == existing_consumers:
                return

            new_consumer_objs = [{"name": name} for name in list(updated_set)]

            current_provider_objs = [{"name": entry.name}for entry in mcs.spec.providerClusters] if mcs.spec.providerClusters else []

            mcs_manifest, = load_manifests("multicluster_service.yaml", context={
                "service_name": mcs_name,
                "namespace_name": namespace,
                "provider_clusters": current_provider_objs,
                "consumer_clusters": new_consumer_objs
            })
            self.k8s_client.apply(mcs_manifest)

    def update_mcs_providers(self, mcs_name, namespace, add, remove, consumer_ids):
        with self._mcs_lock:
            mcs = self.k8s_client.get("MultiClusterService", name=mcs_name, namespace=namespace)

            existing_providers = {p['name'] for p in mcs.spec.providerClusters} if mcs.spec.providerClusters else set()
            new_provider_names = list(existing_providers.union(set(add)) - set(remove))
            new_provider_objs = [{"name": name} for name in new_provider_names]

            if consumer_ids is not None:
                consumer_objs = [{"name": name} for name in consumer_ids]
            else:
                consumer_objs = mcs.spec.consumerClusters if mcs.spec.consumerClusters else []

            mcs_manifest, = load_manifests("multicluster_service.yaml", context={
                "service_name": mcs_name,
                "namespace_name": namespace,
                "provider_clusters": new_provider_objs,
                "consumer_clusters": consumer_objs
            })
            self.k8s_client.apply(mcs_manifest)

    def delete_work(self):
        """
        Deletes all resources defined in YAML files inside `work_path`.
        Waits until they are fully removed from the cluster.
        """
        print(f"--- Deleting Base Workload from {work_path} ---")
        manifests = self._load_manifests_from_dir(work_path)

        resources_to_track = []
        services_deleted = False
        for manifest in manifests:
            kind = manifest.get("kind")
            name = manifest["metadata"]["name"]
            namespace = manifest["metadata"].get("namespace", "default")
            
            if kind == "Service":
                services_deleted = True

            was_found = self.k8s_client.delete(
                    kind=kind,
                    name=name,
                    namespace=namespace,
                    propagation_policy='Foreground'
            )
            if was_found:
                resources_to_track.append((kind, name, namespace))

        self._wait_for_resources(resources_to_track, condition="deleted", timeout=60)
        if services_deleted:
            print("DEBUG: Services detected. Waiting 15s for NodePort/LoadBalancer release...")
            time.sleep(15)

        print("--- All Workloads Deleted ---")

    def apply_work(self):
        """
        Applies all resources defined in YAML files inside `work_path`.
        Waits until Deployments are Ready.
        """
        print("DEBUG: Resetting network connection pool...")
        self.k8s_client.reset_connection_pool()
        print(f"--- Applying Base Workload from {work_path} ---")

        print("DEBUG: Loading manifests from disk...")
        manifests = self._load_manifests_from_dir(work_path)
        print(f"DEBUG: Loaded {len(manifests)} manifests. Starting application...")

        resources_to_track = []

        for i, manifest in enumerate(manifests):
            kind = manifest.get("kind")
            name = manifest["metadata"]["name"]
            namespace = manifest["metadata"].get("namespace", "default")

            print(f"DEBUG [{i + 1}/{len(manifests)}]: Applying {kind} '{name}' in '{namespace}'...", end="", flush=True)

            max_retries = 10
            for attempt in range(max_retries):
                try:
                    self.k8s_client.apply(manifest)
                    print(" Done.")
                    break
                except UnprocessibleEntityError as e:
                    if hasattr(e, 'body') and e.body:
                        error_body = e.body.decode('utf-8') if isinstance(e.body, bytes) else str(e.body)
                    else:
                        error_body = str(e)
                    if "already allocated" in error_body and attempt < max_retries - 1:
                        print(f"\nWARNING: Port allocation collision for {name}. Retrying in 5s...")
                        time.sleep(5)
                    else:
                        print(f"\nERROR: Failed to apply {name}: {e}")
                        raise e
                except Exception as e:
                    error_msg = str(e)
                    is_network_error = (
                            "Remote end closed connection" in error_msg or
                            "Connection refused" in error_msg or
                            "Connection aborted" in error_msg or
                            "Broken pipe" in error_msg
                    )

                    if is_network_error and attempt < max_retries - 1:
                        print(
                            f"\nWARNING: Network flake applying {name} ({error_msg}). Retrying in 2s... ({attempt + 1}/{max_retries})")
                        time.sleep(2)
                        if attempt > 2:
                            self.k8s_client.reset_connection_pool()
                    else:
                        print(f"\nCRITICAL: Unknown or persistent error applying {name}: {e}")
                        raise e

            if kind in ["Deployment", "StatefulSet", "DaemonSet"]:
                resources_to_track.append((kind, name, namespace))

        print(f"DEBUG: Waiting for {len(resources_to_track)} resources to register in API...")
        self._wait_for_resources(resources_to_track, condition="exists", timeout=120)
        print("--- All Workloads Applied & Ready ---")

    def delete_placement_rules(self):
        """
        Deletes all PropagationPolicies and MultiClusterServices generated by the Agent.
        Waits until they are gone.
        """
        print("--- Cleaning up Placement Rules (PP & MCS) ---")
        resources_to_track = []

        for app in self.state.get_all_applications():
            namespace = app.id
            for msvc in app.microservices:
                base_name = msvc.id.removesuffix("_" + app.id)

                pp_name = f"{base_name}-propagation"
                mcs_name = base_name

                # Delete PropagationPolicy
                self.k8s_client.delete(
                        "PropagationPolicy",
                        name=pp_name,
                        namespace=namespace,
                        api_version="policy.karmada.io/v1alpha1",
                        propagation_policy='Foreground'
                )
                resources_to_track.append(("PropagationPolicy", pp_name, namespace))

                # Delete MultiClusterService
                self.k8s_client.delete(
                        "MultiClusterService",
                        name=mcs_name,
                        namespace=namespace,
                        api_version="networking.karmada.io/v1alpha1",
                        propagation_policy='Foreground'
                )
                resources_to_track.append(("MultiClusterService", mcs_name, namespace))

        self._wait_for_resources(resources_to_track, condition="deleted", timeout=60)
        print("--- Placement Rules Cleaned ---")

    def _load_manifests_from_dir(self, directory: Path) -> List[dict]:
        manifests = []
        if not directory.exists():
            print(f"Warning: Work path {directory} does not exist.")
            return []

        for file_path in directory.glob("**/*.yaml"):
            with open(file_path, 'r') as f:
                try:
                    docs = yaml.safe_load_all(f)
                    for doc in docs:
                        if doc: manifests.append(doc)
                except yaml.YAMLError as exc:
                    print(f"Error parsing {file_path}: {exc}")
        return manifests

    def _wait_for_resources(self, resources: List[Tuple[str, str, str]], condition: str, timeout: int):
        """
        Blocks execution until the condition is met for all resources or timeout occurs.
        resources: List of (kind, name, namespace)
        condition: 'deleted' | 'ready'
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_met = True

            for kind, name, namespace in resources:
                obj = self.k8s_client.get(kind, name, namespace)

                if condition == "deleted":
                    # If obj is not None, it still exists
                    if obj is not None:
                        all_met = False
                        break

                elif condition == "ready":
                    # If obj is None, it's not created yet -> wait
                    if obj is None:
                        all_met = False
                        break

                    # Check Readiness (Logic works for Deployments)
                    status = obj.get("status", {})
                    spec = obj.get("spec", {})

                    replicas_desired = spec.get("replicas", 1)
                    replicas_ready = status.get("readyReplicas", 0)

                    if replicas_ready != replicas_desired:
                        all_met = False
                        break

                elif condition == "exists":
                    if obj is None:
                        all_met = False
                        break

            if all_met:
                return

            time.sleep(2)

        print(f"WARNING: Timeout reached while waiting for resources to be {condition}.")
