import torch
import torch.nn as nn

from salsa.core.agents.nets.deepSetEncoder import DeepSetEncoder


class MessageDeepSet(nn.Module):
    def __init__(
            self,
            cluster_dim: int,
            ms_dim: int,
            slo_dim: int,
            message_dim: int = 64,
            hidden_dim: int = 128
    ):
        super(MessageDeepSet, self).__init__()

        self.cluster_encoder = nn.Sequential(
            nn.Linear(cluster_dim, hidden_dim),
            nn.ReLU()
        )
        self.ms_encoder = DeepSetEncoder(ms_dim, hidden_dim)
        self.slo_encoder = DeepSetEncoder(slo_dim, hidden_dim)

        # Cluster + MS + SLO
        combined_dim = hidden_dim * 3

        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()
        )

    def forward(self, cluster_state, microservices, slos):
        # 1. Static
        cluster_emb = self.cluster_encoder(cluster_state)

        # 2. Sets
        ms_pooled = torch.sum(self.ms_encoder(microservices), dim=1)
        slo_pooled = torch.sum(self.slo_encoder(slos), dim=1)

        # 3. Concatenate
        combined = torch.cat([cluster_emb, ms_pooled, slo_pooled], dim=1)

        return self.head(combined)