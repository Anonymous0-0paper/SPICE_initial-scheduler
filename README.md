# SALSA: SLO Aware MADRL-based Elastic Scheduling in Multi-Cluster Computing Continuum

SALSA (SLO Aware Elastic Scheduling of Applications) is a Multi-Agent Reinforcement Learning (MARL) scheduler designed for multi-cluster Kubernetes environments managed by Karmada. It optimizes microservice placement across Cloud and Edge tiers based on Service Level Objectives (SLOs) and energy/cost constraints.

## Project Overview

This project implements a custom scheduler that interfaces with the Karmada API Server. It utilizes a shared replay buffer and distributed agents to manage application placement and migration dynamically.

## Prerequisites

* **Python:** 3.12+
* **Container Runtime:** Docker
* **Orchestration:** Karmada (running on K3s)
* **Infrastructure:** Minimum 5 nodes (Recommended: Chameleon Cloud machine with `CC-ubuntu22.04` image)

## Documentation & Setup

Please follow the documentation in the numbered order to set up the environment, network, and scheduler.

1.  **[Infrastructure Setup](docs/01_infrastructure_setup.md)**: Provisioning nodes, installing K3s, and setting up the Karmada Control Plane.
2.  **[Networking & Service Discovery](docs/02_networking.md)**: Configuring MultiClusterService and Submariner for cross-cluster communication.
3.  **[Monitoring Stack](docs/03_monitoring.md)**: Installing Thanos, Prometheus, and Kepler for observability and reward calculation.
4.  **[Scheduler Configuration](docs/04_scheduler_configuration.md)**: Configuring the SALSA agents, defining applications, and running the training loop.
5.  **[Benchmarking](docs/05_benchmarking.md)**: Deploying µBench to validate scheduler performance.

## Project Structure

* `src/salsa`: Core source code for the RL agents, environment, and coordinator.
* `specs/`: Configuration files for clusters, applications, and service graphs.
* `scripts/`: Helper scripts for node bootstrapping and monitoring installation.

