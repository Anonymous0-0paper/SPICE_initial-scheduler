import torch
import torch.nn as nn
from salsa.core.agents.nets.deepSetEncoder import DeepSetEncoder


class CriticDeepSet(nn.Module):
    def __init__(
            self,
            cluster_dim: int,
            ms_dim: int,
            slo_dim: int,
            neighbor_dim: int,
            ego_action_dim: int,
            neighbor_action_dim: int,
            hidden_dim: int = 128
    ):
        super(CriticDeepSet, self).__init__()

        # 1. State Encoders
        self.cluster_encoder = nn.Sequential(
            nn.Linear(cluster_dim, hidden_dim),
            nn.ReLU()
        )
        self.ms_encoder = DeepSetEncoder(ms_dim, hidden_dim)
        self.slo_encoder = DeepSetEncoder(slo_dim, hidden_dim)

        # 2. Neighbor + Action Encoder
        self.neighbor_encoder = DeepSetEncoder(neighbor_dim + neighbor_action_dim, hidden_dim)

        # 3. The Value Head
        # Cluster + MS + SLO + Neighbors + EgoAction
        combined_dim = (hidden_dim * 4) + ego_action_dim

        self.ln_pooled = nn.LayerNorm(combined_dim)

        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, cluster, microservices, slos, neighbors, neighbor_actions, ego_action):
        # 1. Process Static
        cluster_emb = self.cluster_encoder(cluster)

        # 2. Process Local Sets
        ms_pooled = self._aggregate(self.ms_encoder(microservices))
        slo_pooled = self._aggregate(self.slo_encoder(slos))

        # 3. Process Neighbors + Actions
        neighbor_combined = torch.cat([neighbors, neighbor_actions], dim=2)
        neighbor_pooled = self._aggregate(self.neighbor_encoder(neighbor_combined))

        # 4. Fusion
        fusion = torch.cat([
            cluster_emb,
            ms_pooled,
            slo_pooled,
            neighbor_pooled,
            ego_action
        ], dim=1)

        fusion = self.ln_pooled(fusion)

        return self.head(fusion)

    def _aggregate(self, logits: torch.Tensor):
        if logits.size(1) == 0:
            return torch.zeros(logits.size(0), logits.size(2), device=logits.device)
        return torch.mean(logits, dim=1)
