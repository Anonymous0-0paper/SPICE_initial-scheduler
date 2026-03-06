import torch
import torch.nn as nn

from salsa.core.agents.nets.deepSetEncoder import DeepSetEncoder


class ActorDeepSet(nn.Module):
    def __init__(
            self,
            cluster_dim: int,
            ms_dim: int,
            slo_dim: int,
            neighbor_dim: int,
            action_dim: int,
            hidden_dim: int = 128
    ):
        self.action_dim = action_dim
        self.cluster_dim = cluster_dim
        self.ms_dim = ms_dim
        self.slo_dim = slo_dim
        self.neighbor_dim = neighbor_dim
        self.hidden_dim = hidden_dim

        super(ActorDeepSet, self).__init__()

        # --- 1. The Encoders ---
        # Cluster State encoder
        self.cluster_encoder = nn.Sequential(
            nn.Linear(cluster_dim, hidden_dim),
            nn.ReLU()
        )

        # Variable Sets (DeepSet Encoders)
        self.ms_encoder = DeepSetEncoder(ms_dim, hidden_dim)
        self.slo_encoder = DeepSetEncoder(slo_dim, hidden_dim)
        self.neighbor_encoder = DeepSetEncoder(neighbor_dim, hidden_dim)

        # --- 2. The Policy Head ---
        combined_dim = hidden_dim * 4

        self.ln_pooled = nn.LayerNorm(combined_dim)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, cluster, microservices, slos, neighbors):
        """
        cluster:       (Batch, Cluster_Dim)
        microservices: (Batch, Num_Microservices, MS_Dim)
        slos:          (Batch, Num_SLOs, SLO_Dim)
        neighbors:     (Batch, Num_Neighbors, Neighbor_Dim)
        """

        # 1. Process Static Cluster State (Renamed variable used here)
        cluster_emb = self.cluster_encoder(cluster)

        # 2. Process Sets (Encode per-item -> Sum over the set)
        ms_pooled = self._aggregate_fn(self.ms_encoder(microservices))
        slo_pooled = self._aggregate_fn(self.slo_encoder(slos))
        neighbor_pooled = self._aggregate_fn(self.neighbor_encoder(neighbors))

        # 3. Concatenate everything into one context vector
        combined = torch.cat([
            cluster_emb,
            ms_pooled,
            slo_pooled,
            neighbor_pooled,
        ], dim=1)

        combined = self.ln_pooled(combined)

        return self.head(combined)

    def _aggregate_fn(self, logits: torch.Tensor):
        if logits.size(1) == 0:
            return torch.zeros(logits.size(0), logits.size(2), device=logits.device)
        return torch.max(logits, dim=1)[0]
