# Benchmarking with µBench

µBench is used to deploy microservice workloads to validate the scheduler's performance.

## 1. Prerequisites

Install required system tools and Docker on the Karmada Host.

```bash
# System Tools
sudo apt update && sudo apt install -y apache2-utils build-essential cmake libffi-dev libcairo2-dev

# Docker Engine
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 2. Configuring µBench for Karmada

### Start Container

Run the µBench container, mounting the **Karmada** configuration (not the host cluster config).

```bash
sudo docker run -it -id \
  --name mubench \
  -v $HOME/.kube/karmada.config:/root/.kube/config \
  msvcbench/mubench
```

### Label Injection

The scheduler identifies applications via the `application_name` label. You must modify the µBench templates to inject the namespace as the application name.

1. Enter the container.
2. Edit `/root/muBench/Deployers/K8sDeployer/Templates/DeploymentTemplate.yaml`.
3. Add the following to both `metadata.labels` and `spec.template.metadata.labels`:
```yaml
application_name: {{NAMESPACE}}
```

## 3. Execution

1. Generate the application YAMLs using µBench.
2. **Restriction:** Do not allow µBench to deploy Nginx ingress controllers automatically, as they are not migration-safe. Apply Nginx services manually with a static PropagationPolicy.
3. Place the generated application YAMLs into the `work/` directory of the SALSA project.
4. Start the SALSA scheduler.
5. Start the benchmark load generation within the µBench container.

### Monitoring Progress

To view the distribution of workloads:

```bash
kubectl get work -A
```
