from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

from salsa.entities.application import Application
from salsa.entities.cluster import Cluster, TierType
from salsa.entities.dependencyGraph import DependencyGraph
from salsa.entities.microservice import Microservice
from salsa.utils.sizeParser import parse_cpu, parse_memory_bytes


class ServiceLink(BaseModel):
    services: List[str]

class ServiceNode(BaseModel):
    external_services: List[ServiceLink] = []

class ResourceRequests(BaseModel):
    cpu: str
    mem: str

    @field_validator('mem')
    @classmethod
    def validate_memory_format(cls, v: str) -> str:
        valid_units = ['Ki', 'Mi', 'Gi', 'Ti']

        for unit in valid_units:
            if v.endswith(unit):
                number_part = v[:-len(unit)]
                if not number_part.isdigit():
                    raise ValueError(
                        f"Memory value '{v}' has invalid number format. Unit '{unit}' requires an integer prefix.")
                return v

        if v.endswith('m'):
            number_part = v[:-1]
            if not number_part.isdigit():
                raise ValueError(f"Memory value '{v}' has invalid number format. Unit 'm' requires an integer prefix.")
            return v

        if not v.isdigit():
            raise ValueError(f"Memory value '{v}' must be an integer or end with Ki, Mi, Gi, Ti, m.")

        return v

    @field_validator('cpu')
    @classmethod
    def validate_cpu_format(cls, v: str) -> str:
        valid_suffixes = ['m', 'n', 'u']

        for suffix in valid_suffixes:
            if v.endswith(suffix):
                number_part = v[:-len(suffix)]
                try:
                    float(number_part)
                    return v
                except ValueError:
                    raise ValueError(f"CPU value '{v}' has invalid number format. Prefix must be a number.")

        try:
            float(v)
        except ValueError:
            raise ValueError(f"CPU value '{v}' must be a number or end with m, n, u.")

        return v


class MicroserviceConfig(BaseModel):
    name: str
    desiredReplicas: int
    maxToleratedReplicas: int
    migrationCost: float
    resourceRequests: ResourceRequests

class ViolationPredictorConfig(BaseModel):
    lookaheadInSeconds: int


class SloConfig(BaseModel):
    latency: float
    throughput: float
    penaltyCoefficient: float
    ViolationPredictor: ViolationPredictorConfig


class DependencyItem(BaseModel):
    target: str = Field(alias="dependsOn")

class ServiceGraphConfig(BaseModel):
    dependencies: Dict[str, Optional[List[DependencyItem]]]
    
class ApplicationConfig(BaseModel):
    name: str
    microservices: List[MicroserviceConfig]

    entrypoint: str

    service_graph: ServiceGraphConfig = Field(alias="dependencyGraph")

    migration_interval: int
    scaling_interval: int

    slo: SloConfig

    def to_domain_entity(self, app_id: str) -> 'Application':
        dg = DependencyGraph(
            id=f"graph_{self.name}",
            app_id=app_id,
            edges={},
            critical_path=None
        )

        def ms_id(name):
            return f"{name}_{app_id}"

        raw_graph = self.service_graph.dependencies

        for svc_name, dep_list in raw_graph.items():
            src = ms_id(svc_name)
            if src not in dg.edges:
                dg.edges[src] = {}
            if dep_list:
                for item in dep_list:
                    dst = ms_id(item.target)
                    dg.edges[src][dst] = 1.0

        return Application(
            id=app_id,
            microservices=[
                Microservice(id=m.name, app_id=app_id, mem_demands_bytes=parse_memory_bytes(m.resourceRequests.mem), cpu_core_demands=parse_cpu(m.resourceRequests.cpu), max_tolerated_replicas=m.maxToleratedReplicas, migration_cost=m.migrationCost)
                for m in self.microservices
            ],
            dependency_graph=dg,
            entrypoint=ms_id(self.entrypoint) if self.entrypoint else None,
            migration_interval=self.migration_interval,
            scaling_interval=self.scaling_interval,
            slos={
                "latency": self.slo.latency,
                "throughput": self.slo.throughput
            },
            penalty_coefficient=self.slo.penaltyCoefficient,
            horizon=self.slo.ViolationPredictor.lookaheadInSeconds
        )

class ApplicationListConfig(BaseModel):
    apps: List[ApplicationConfig]

class ClusterCostConfig(BaseModel):
    cpuCoreHour: float
    memGbHour: float

class ClusterConfig(BaseModel):
    name: str
    tierType: str
    cost: ClusterCostConfig

    cpu_cores: int = Field(alias="cpuCores")
    mem_gb: float = Field(alias="memGb")

    @field_validator('tierType')
    @classmethod
    def validate_tier(cls, v: str) -> str:
        try:
            TierType[v.upper()]
        except KeyError:
            valid_tiers = [t.name for t in TierType]
            raise ValueError(f"Invalid tierType '{v}'. Must be one of: {valid_tiers}")
        return v

    def to_domain_entity(self) -> 'Cluster':
        tier_enum = TierType[self.tierType.upper()]

        return Cluster(
            id=self.name,
            tier_type=tier_enum,

            cost_per_core_hour=self.cost.cpuCoreHour,
            cost_per_mem_hour=self.cost.memGbHour,

            mem_gb=self.mem_gb,
            cpu_cores=self.cpu_cores,

            cpu_core_utilization=0,
            mem_utilization=0,
            microservices=[],
            applications=[]
        )

class InfrastructureConfig(BaseModel):
    clusters: List[ClusterConfig]
