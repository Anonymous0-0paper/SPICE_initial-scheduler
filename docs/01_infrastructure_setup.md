# Infrastructure Setup

This guide details the provisioning of the Karmada Control Plane and member clusters.

## 1. Hardware Provisioning (Chameleon Cloud)

Requirement: **5 Devices**
* 1 Karmada Host (Control Plane)
* 2 Kubernetes Member Clusters (Each consisting of 1 Headnode + Worker nodes)

**Provisioning Steps:**
1.  Create instances using the image `CC-ubuntu22.04`.
2.  Ensure all instances reside within the same virtual network.
3.  Assign a Floating IP to the designated **Karmada Host**.
4.  Configure SSH keys for the `cc` user to enable inter-node communication.

---

## 2. K3s Cluster Initialization

This process applies to both the Karmada Host and Member Cluster Headnodes.

### Firewall Configuration
Open the required ports for Kubernetes communication:

```bash
sudo firewall-cmd --add-port=6443/tcp --permanent
sudo firewall-cmd --reload
sudo systemctl stop firewalld
sudo systemctl disable firewalld
```

### Installation

Install K3s with unique CIDRs to prevent IP conflicts between clusters.

**Note:** It is recommended to use `/16` subnets (e.g., `10.41.0.0/16` for Cluster CIDR). Ensure each cluster utilizes a unique non-overlapping range.

```bash
curl -sfL https://get.k3s.io | \
  INSTALL_K3S_EXEC="\
    --cluster-cidr=<UNIQUE-CLUSTER-RANGE> \
    --service-cidr=<UNIQUE-SERVICE-RANGE>" \
  sh -
```

### Retrieve Join Token

Retrieve the node token required for worker nodes to join this cluster:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

### Configure Kubeconfig

Export the configuration for local usage:

```bash
sudo mkdir -p $HOME/.kube
sudo cp /etc/rancher/k3s/k3s.yaml $HOME/.kube/config
sudo chown $USER:$USER $HOME/.kube/config
# Optional: Add to .bashrc
echo "export KUBECONFIG=\$HOME/.kube/config" >> ~/.bashrc
```

---

## 3. Worker Node Setup

To attach worker nodes to a K3s cluster, use the provided helper script `scripts/kube-worker.sh` or perform the following manual steps on the worker node.

3.1. **Prepare Firewall:** Run the firewall commands listed in Section 2.  
3.2. **Join Cluster:**

```bash
export TOKEN=<K3S_NODE_TOKEN>
export K3S_SERVER=<HEADNODE_INTERNAL_IP>
curl -sfL https://get.k3s.io | K3S_URL=https://$K3S_SERVER:6443 K3S_TOKEN=$TOKEN sh -
```

### Verification

On the headnode, verify the node status:

```bash
sudo kubectl get nodes
```

---

## 4. Karmada Control Plane Setup

The Karmada API Server is hosted as a workload on the "Karmada Host" K3s cluster created in Section 2.

### 4.1 Prerequisites

Install the required CLI tools on the Karmada Host:

**4.1.1. Install `kubectl`**

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
# Validate SHA256
echo "$(curl -sL "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256")  kubectl" | sha256sum --check
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**4.1.2. Install `kubectl-karmada` and `karmadactl`**

```bash
curl -s https://raw.githubusercontent.com/karmada-io/karmada/master/hack/install-cli.sh | sudo bash -s kubectl-karmada
curl -s https://raw.githubusercontent.com/karmada-io/karmada/master/hack/install-cli.sh | sudo bash
```

### 4.2. Karmada Initialization

4.2.1. **Prepare Configuration:**
Ensure the Karmada configuration points to the external IP, not localhost.
```bash
# Prepare K3s config
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config

# Initialize Karmada
sudo kubectl karmada init --kubeconfig /etc/rancher/k3s/k3s.yaml

# Prepare Karmada config
sudo cp /etc/karmada/karmada-apiserver.config ~/.kube/karmada.config
sudo chown $(id -u):$(id -g) ~/.kube/karmada.config

# Update Server IP
sed -i 's/127.0.0.1/<HOST_INTERNAL_IP>/g' $HOME/.kube/karmada.config
```

4.2.3. **Verification:**
```bash
kubectl api-resources | grep karmada
kubectl get pods -n karmada-system
```

### 4.3 Registering Member Clusters (Push Mode)

4.3.1. **Retrieve Kubeconfigs:**
Copy the `kubeconfig` files from the member clusters (kube1, kube2) to the Karmada Host.
```bash
ssh -i "~/.ssh/cc" <USER@KUBE1-IP> "sudo cat /home/cc/.kube/config" > $HOME/kube1.yaml
ssh -i "~/.ssh/cc" <USER@KUBE2-IP> "sudo cat /home/cc/.kube/config" > $HOME/kube2.yaml
```


4.3.2. **Sanitize Configs:**
Edit the YAML files to replace `127.0.0.1` with the respective member cluster's internal IP.

4.3.3. **Join Clusters:**
```bash
kubectl karmada join kube1 --cluster-kubeconfig ~/kube1.yaml
kubectl karmada join kube2 --cluster-kubeconfig ~/kube2.yaml
```

4.3.4. **Verify Registration:**
```bash
kubectl get clusters
```

