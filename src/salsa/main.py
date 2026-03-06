import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import toml
from ruamel.yaml import YAML

from salsa.config_verification import InfrastructureConfig, ApplicationListConfig
from salsa.core.agents.salsaAgent import SalsaAgent
from salsa.core.globalCoordinator.globalCoordinator import GlobalCoordinator
from salsa.core.globalCoordinator.sharedReplayBuffer import SharedReplayBuffer
from salsa.core.rewardSystems.salsaRewardSystem import SalsaRewardSystem
from salsa.core.states.monitor import MetricMonitor
from salsa.core.states.systemState import SystemState
from salsa.entities.cluster import Cluster
from salsa.externals.clock import EventClock
from salsa.externals.karmadaEventProducer import MultiDeploymentMigrationMonitor
from salsa.sloViolationPredictor.base_predictor import BasePredictor
from salsa.sloViolationPredictor.statistical_predictor import StatisticalSLOPredictor
from salsa.utils.typing import agent_id, application_id, microservice_id
from salsa.core.env.salsaEnv import SalsaEnv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # parent of src/
config_path = BASE_DIR / "src" / "salsa" / "config" / "config.toml"
config = toml.load(config_path)

specs_path = BASE_DIR / "specs"
work_path = BASE_DIR / "work"

def include(loader, node):
    try:
        current_filepath = Path(loader.loader.stream.name)
        base_dir = current_filepath.parent
        source_name = current_filepath.name
    except AttributeError:
        base_dir = specs_path
        source_name = "<Stream/String>"

    filename = loader.construct_scalar(node)
    full_path = (base_dir / filename).resolve()

    print(f"   -> Parsing: {source_name}")
    print(f"      -> Including: {filename} (resolved to {full_path})")

    if not full_path.exists():
        raise FileNotFoundError(f"Could not find included file: {full_path} (referenced in {source_name})")

    sub_yaml = YAML()
    sub_yaml.constructor.add_constructor("!include", include)

    with open(full_path) as f:
        return sub_yaml.load(f)

def run_salsa_scheduler(agents: Dict[agent_id, SalsaAgent], global_coordinator: GlobalCoordinator):
    """
    Main loop for SALSA: SLO Aware MADRL-based Elastic Scheduling.
    Based on Algorithm 1.
    """

    while global_coordinator.current_episode < config['scheduler']['max_episodes']:
        global_coordinator.run_episode()

    return agents

def init_system_state() -> SystemState:
    apps_path = specs_path / "apps.yaml"
    clusters_path = specs_path / "clusters.yaml"

    yaml_parser = YAML()
    yaml_parser.constructor.add_constructor("!include", include)
    with open(apps_path) as f:
        apps_data = yaml_parser.load(f)
    apps_conf = ApplicationListConfig.model_validate(apps_data)
    apps = [a.to_domain_entity(a.name) for a in apps_conf.apps]


    with open(clusters_path) as f:
        clusters_data = yaml_parser.load(f)
    clusters_conf = InfrastructureConfig.model_validate(clusters_data)
    clusters = [c.to_domain_entity() for c in clusters_conf.clusters]

    state = SystemState()

    # Create all applications
    for app in apps:
        state.add_application(app)

    # Create all clusters
    for cluster in clusters:
        state.add_cluster(cluster)

    # Create all microservices
    for app in state.get_all_applications():
        for ms in app.microservices:
            state.add_microservice(ms)

    return state

def init_agents(clusters: List[Cluster], ms_list: List[microservice_id]):
    agents = {}

    agent_dims = {
        "cluster_dim": 2,
        "ms_dim": 9,
        "slo_dim": 8,
        "neighbor_dim": 4
    }

    num_clusters = len(clusters)
    num_place_actions = len(ms_list)
    num_scale_actions = len(ms_list) * 2
    num_mig_actions = len(ms_list) * (num_clusters - 1)
    action_dim = 1 + num_place_actions + num_scale_actions + num_mig_actions

    actor_lr = config['scheduler']['actor_lr']
    critic_lr = config['scheduler']['critic_lr']
    device = config['scheduler']['device']
    for c in clusters:
        cid = c.id
        agent = SalsaAgent(aid='agent_' + cid, cluster=c, dims=agent_dims, action_dim=action_dim, device=device, actor_lr=actor_lr, critic_lr=critic_lr, load_path=config['scheduler']['load_path'])
        agents[agent.aid] = agent
    return agents

def main():
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    state = init_system_state()
    reward_system = SalsaRewardSystem(state, config)

    metric_monitor = MetricMonitor(state, config)
    metric_monitor.start()

    karmada_event_monitor = MultiDeploymentMigrationMonitor(
            cnfg=config,
            target_deployments=[ms.id.removesuffix(f"_{ms.app_id}") for ms in state.get_all_microservices()],
            namespaces=[app.id for app in state.get_all_applications()],
            kube_context="karmada-apiserver")
    karmada_event_monitor.start()

    print("Starting Monitors ...")
    time.sleep(5)

    clock = EventClock()

    horizon: int = config["slo_predictor"]["horizon_in_seconds"] // config["scheduler"]["step_interval"]

    latency_predictors: Dict[application_id, BasePredictor] = {app.id: StatisticalSLOPredictor(horizon=horizon, is_min_metric=True, threshold=app.slos["latency"]) for app in state.get_all_applications()}
    throughput_predictors: Dict[application_id, BasePredictor] = {app.id: StatisticalSLOPredictor(horizon=horizon, is_min_metric=False, threshold=app.slos["throughput"]) for app in state.get_all_applications()}

    env = SalsaEnv(state=state,
                   reward_system=reward_system,
                   monitor=metric_monitor,
                   karmada_event_monitor=karmada_event_monitor,
                   latency_predictors=latency_predictors,
                   throughput_predictors=throughput_predictors,
                   clock=clock,
                   config=config)

    ms_list = [ms.id for app in state.get_all_applications() for ms in app.microservices]
    agents: Dict[agent_id, SalsaAgent] = init_agents(state.get_all_clusters(), ms_list)

    global_coordinator = GlobalCoordinator(env=env, agents=agents, buffer=SharedReplayBuffer(config), config=config)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    try:
        agents = run_salsa_scheduler(agents, global_coordinator)
        for agent in agents.values():
            filename = f"final_{agent.aid}.pt"
            full_path = os.path.join(save_dir, filename)
            agent.save_checkpoint(full_path)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving metrics...")

    global_coordinator.close_loggers()
    print("Metrics saved to CSVs.")

    metric_monitor.stop()
    karmada_event_monitor.stop()
    print("Stopped monitors. Exiting...")
    print(f"Make sure to clean up PropagationPolicies, MultiClusterServices and deployed Resources in namespaces {[app.id for app in state.get_all_applications()]}.")
if __name__ == "__main__":
    main()
