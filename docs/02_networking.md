# Networking and Service Discovery

To enable cross-cluster communication and service discovery, the `MultiClusterService` feature gate must be enabled, and a networking overlay (Submariner) must be deployed.

## 1. Enable MultiClusterService

Update the Karmada Controller Manager to enable the required feature gate.

1.1.  Edit the deployment:  
```bash
kubectl edit deploy karmada-controller-manager -n karmada-system
```  
1.2.  Locate `spec.template.spec.containers[0].command` and append:  
    
```bash
- --feature-gates=MultiClusterService=true
```
    
1.3.  Verify the rollout:  
```bash
kubectl get pods -n karmada-system -l app=karmada-controller-manager
```  
*Note: If the new pod remains pending, delete the old replica manually.*

## 2. Submariner Deployment

Submariner is required for direct pod-to-pod and service networking across clusters.

### Installation

2.1.  **Install `subctl`:**
```bash
curl -Ls https://get.submariner.io | bash
# Ensure subctl is added to your PATH
```

2.2.  **Deploy Broker (Karmada Host):**
Ensure your `~/.kube/config` points to the internal network address.
```bash
subctl deploy-broker --kubeconfig ~/.kube/config
```
This generates a `broker-info.subm` file.

2.3.  **Join Member Clusters:**
Transfer the `broker-info.subm` file to all member clusters. Run the join command on each member:
```bash
subctl join broker-info.subm --clusterid <CLUSTER_NAME> --kubeconfig $KUBECONFIG --natt=false
```

### Troubleshooting: MTU Configuration
In specific environments (e.g., Chameleon Cloud), you may need to lower the MTU on the CNI interfaces if timeouts occur.

```bash
sudo ip link set cni0 mtu 1350
sudo ip link set flannel.1 mtu 1350
```
