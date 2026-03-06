from kubernetes import config, dynamic
from kubernetes.client import api_client
from kubernetes.dynamic.exceptions import ResourceNotFoundError, NotFoundError


class KarmadaClient:
    def __init__(self, kubeconfig_path=None, context=None):
        try:
            config.load_kube_config(config_file=kubeconfig_path, context=context)
        except config.ConfigException:
            config.load_incluster_config()

        self.client = dynamic.DynamicClient(api_client.ApiClient())
        self._api_registry = {}

    def _get_resource_api(self, kind: str, api_version: str = None):
        key = (kind, api_version)
        if key in self._api_registry:
            return self._api_registry[key]

        try:
            api_resource = self.client.resources.get(kind=kind, api_version=api_version)
            self._api_registry[key] = api_resource
            return api_resource
        except (ResourceNotFoundError, NotFoundError):
            raise ValueError(f"CRD or Resource Kind '{kind}' not found in the cluster.")

    def get(self, kind: str, name: str, namespace: str = "default", api_version: str = None):
        """
        Returns the resource object or None if it doesn't exist.
        """
        api = self._get_resource_api(kind, api_version)
        try:
            return api.get(name=name, namespace=namespace)
        except (ResourceNotFoundError, NotFoundError):
            return None

    def apply(self, manifest: dict):
        kind = manifest.get("kind")
        name = manifest["metadata"]["name"]
        namespace = manifest["metadata"].get("namespace", "default")
        api_version = manifest.get("apiVersion")

        api = self._get_resource_api(kind, api_version)

        return api.patch(
            name=name,
            namespace=namespace,
            body=manifest,
            field_manager="salsa-agent",
            content_type="application/apply-patch+yaml",
            force=True
        )

    def delete(self, kind: str, name: str, namespace: str = "default", api_version: str = None, **kwargs):
        """
        Deletes a resource. Returns True if deleted/triggered, False if it didn't exist.
        """
        api = self._get_resource_api(kind, api_version)
        try:
            api.delete(name=name, namespace=namespace, **kwargs)
            return True
        except (ResourceNotFoundError, NotFoundError):
            return False

    def patch(self, kind: str, name: str, body: dict, namespace: str = "default"):
        target_version = "apps/v1" if kind == "Deployment" else None

        api = self._get_resource_api(kind, api_version=target_version)
        
        return api.patch(
            name=name,
            namespace=namespace,
            body=body,
            content_type="application/strategic-merge-patch+json",
        )

    def reset_connection_pool(self):
        """
        Forcefully clears the underlying urllib3 connection pool to release
        stuck sockets from previous episodes.
        """
        try:
            if hasattr(self.client, "client"):
                rest_client = self.client.client.rest_client
                if hasattr(rest_client, "pool_manager"):
                    rest_client.pool_manager.clear()
                    print("DEBUG: KarmadaClient connection pool cleared successfully.")
        except Exception as e:
            print(f"WARNING: Failed to reset connection pool: {e}")
