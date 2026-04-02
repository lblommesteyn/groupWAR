from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class GraphLineupModel(nn.Module):
    """Graph + DeepSets model generalized from the NHL notebook implementation."""

    def __init__(
        self,
        *,
        node_in_dim: int,
        gcn_hidden: int,
        gcn_layers: int,
        deepset_phi_dim: int,
        deepset_rho_dim: int,
        n_nodes: int,
        vector_size: int,
        home_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.home_size = home_size if home_size is not None else n_nodes // 2

        self.embed = nn.Linear(node_in_dim, vector_size)
        self.project = nn.Linear(vector_size, vector_size)

        layers: list[nn.Module] = []
        for index in range(gcn_layers):
            in_dim = vector_size if index == 0 else gcn_hidden
            layers.extend(
                [
                    RGCNConv(in_dim, gcn_hidden, num_relations=2),
                    nn.Linear(gcn_hidden, gcn_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gcn_hidden, gcn_hidden),
                ]
            )
        self.gcn = nn.ModuleList(layers)
        self.phi = nn.Sequential(
            nn.Linear(gcn_hidden, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_phi_dim, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(deepset_phi_dim, deepset_rho_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_rho_dim, 1),
        )
        self.norm = nn.LayerNorm(gcn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(
        self,
        adjacency: torch.Tensor,
        features: torch.Tensor,
        home_mask: torch.Tensor,
        away_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_nodes, _ = features.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} nodes, received {n_nodes}")

        if away_mask is None:
            away_mask = 1.0 - home_mask

        hidden = self.relu(self.embed(features))
        hidden = self.project(hidden).reshape(batch_size * n_nodes, -1)

        mask = adjacency != 0
        batch_index, row_index, col_index = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([batch_index * n_nodes + row_index, batch_index * n_nodes + col_index], dim=0)
        edge_weight = adjacency[mask]
        edge_type = (adjacency[mask] > 0).long()

        aggregated = torch.zeros_like(hidden)
        for layer in self.gcn:
            if isinstance(layer, (nn.Linear, nn.Dropout, nn.ReLU)):
                aggregated = layer(aggregated)
                continue

            hidden = hidden + aggregated
            aggregated = torch.zeros_like(hidden)
            messages = layer(hidden, edge_index, edge_type)
            rows, cols = edge_index
            weighted_messages = messages[cols] * edge_weight.abs().unsqueeze(-1)
            aggregated.index_add_(0, rows, weighted_messages)
            aggregated = self.dropout(self.relu(self.norm(aggregated)))

        hidden = (hidden + aggregated).reshape(batch_size, n_nodes, -1)
        phi = self.phi(hidden.reshape(batch_size * n_nodes, -1)).reshape(batch_size, n_nodes, -1)

        home_sum = (phi * home_mask.unsqueeze(-1).float()).sum(dim=1)
        away_sum = (phi * away_mask.unsqueeze(-1).float()).sum(dim=1)
        score = self.rho(home_sum).squeeze(-1) - self.rho(away_sum).squeeze(-1)
        return 0.5 + 0.5 * self.output_activation(score)
