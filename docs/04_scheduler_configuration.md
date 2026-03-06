# Scheduler Configuration & Usage

The SALSA scheduler uses a Python 3.12 environment to execute training and inference loops.

## 1. Python Environment

Ensure Python 3.12 and development libraries are installed.

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Configuration (`specs/`)

The scheduler operation is defined by YAML files in the `specs/` directory.

### Application Specs (`specs/apps/`)

Define microservices, resource requests, and Service Level Objectives (SLOs).

**Example `specs/apps/app1.yaml`:**

```yaml
name: app1
entrypoint: s0
migration_interval: 20  # Seconds before migration is allowed
scaling_interval: 20    # Seconds before scaling is allowed

microservices:
  - name: s0
    desiredReplicas: 2
    maxToleratedReplicas: 4
    migrationCost: 1
    resourceRequests:
      cpu: 2000m
      mem: 100Ki

dependencyGraph: !include serviceGraphs/app1.yaml

slo:
  latency: 140
  throughput: 50
  penaltyCoefficient: 1
  ViolationPredictor:
    lookaheadInSeconds: 120
```

### Cluster Specs (`specs/clusters.yaml`)

Define the available computational resources and cost models for the member clusters.

```yaml
clusters:
  - name: kube1
    tierType: Cloud
    cpuCores: 320
    memGb: 502
    cost:
      cpuCoreHour: 0.08
      memGbHour: 0.04
```

## 3. Running the Scheduler

Ensure all application manifests intended for deployment are located in the `work/` directory and referenced in `specs/apps.yaml`.

**Start Training:**

```bash
python src/salsa/main.py
```

**Advanced Configuration:**
Internal scheduler parameters (learning rates, buffer sizes) can be modified in `src/salsa/config/config.toml`.
