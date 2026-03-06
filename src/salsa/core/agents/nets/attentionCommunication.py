import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCommunication(nn.Module):
    def __init__(self, query_dim, msg_dim, hidden_dim=64):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim)
        self.key_layer = nn.Linear(msg_dim, hidden_dim)
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, my_cluster_state, neighbor_msgs):
        if neighbor_msgs is None or neighbor_msgs.size(1) == 0:
            return torch.zeros(my_cluster_state.size(0), self.key_layer.in_features, device=my_cluster_state.device)

        query = self.query_layer(my_cluster_state).unsqueeze(1)

        keys = self.key_layer(neighbor_msgs)

        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)

        context = torch.bmm(weights, neighbor_msgs)
        return context.squeeze(1)
