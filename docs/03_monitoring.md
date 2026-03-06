# Monitoring Stack

The scheduler relies on real-time metrics for decision-making. The architecture consists of a central Thanos Receiver on the Karmada Host and Prometheus instances with Kepler on member clusters.

## 1. Central Thanos Setup (Karmada Host)

The Thanos Receiver aggregates metrics from all member clusters.

### Configuration

1.1.  **Switch Context:** Ensure `KUBECONFIG` points to the K3s host config, not the Karmada API config.

1.2.  **Create Hashring Config:**
```bash
cat <<EOF > hashring.json
[
  {
    "hashring": "default",
    "tenants": [],
    "endpoints": ["thanos-receive-0.thanos-receive.default.svc.cluster.local:10901"]
  }
]
EOF
kubectl create configmap hashring-config --from-file=hashring.json
```

1.3.  **Deploy Thanos Components:**
Apply the manifests for `thanos-receive` and `thanos-query`.
*See `manifests/monitoring/thanos-receive.yaml` and `thanos-query.yaml`.*

1.4.  **Verify Service:**
Identify the NodePort for remote write (internal port 19291):
```bash
kubectl get svc thanos-receive
# Expected NodePort: 31218
```

## 2. Member Cluster Setup

Perform the following on **each** member cluster.

Install helm first if you don't have it yet.

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4 | bash
```

### Install Prometheus & Connect to Thanos

Use the provided script `scripts/monitoring-install.sh` to install the `kube-prometheus-stack` and configure the remote write target.

**Usage:**

```bash
./scripts/monitoring-install.sh \
  -c <CLUSTER_NAME> \
  -i <KARMADA_HOST_IP> \
  -p <THANOS_NODE_PORT>
```


### Install Kepler

Kepler is used to estimate energy consumption of pods.

```bash
helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
helm repo update

helm install kepler kepler/kepler \
    --namespace kepler \
    --create-namespace \
    --set serviceMonitor.enabled=true \
    --set serviceMonitor.labels.release=prometheus

kubectl label namespace kepler prometheus-scrape="true"
```

### Configure Pod Monitors

To monitor specific namespaces (e.g., for benchmarking), apply the `PodMonitor` configuration:

1. Edit `mub-monitor.yaml` to set the target `namespace`.
2. Apply the configuration:
```bash
kubectl apply -f mub-monitor.yaml
```

### Verify Thanos connectivity on Karmada

Finally we can verify the connectivity of the Thanos Query instance to the Kubernetes Clusters.

```bash
curl -s "http://<THANOS-QUERY-CLUSTER-IP>:9090/api/v1/query?query=count(up==1)by(cluster)"
```
