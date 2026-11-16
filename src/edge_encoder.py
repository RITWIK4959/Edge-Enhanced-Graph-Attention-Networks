import torch

from torch import nn


class LearnedEdgeEncoder(nn.Module):

    """
    Lightweight trainable edge encoder.

    Input:
        source node representation
        target node representation

    Uses:
        source
        target
        absolute difference

    This is intentionally small because the datasets
    contain very few labelled training nodes.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=16,
        dropout=0.10,
    ):

        super().__init__()

        input_dim = (
            3 * node_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                edge_dim,
            ),
        )

        self.norm = nn.LayerNorm(
            edge_dim
        )

    def forward(
        self,
        source,
        target,
    ):

        difference = torch.abs(
            source - target
        )

        pair = torch.cat(
            [
                source,
                target,
                difference,
            ],
            dim=-1,
        )

        edge = self.network(
            pair
        )

        edge = self.norm(
            edge
        )

        return edge


def create_static_edge_attributes(
    node_features,
    edge_index,
    edge_dim,
):

    """
    Static edge representation.

    No trainable parameters.

    e_ij = 0.5 * (x_i + x_j)

    The representation is then truncated/padded
    to EDGE_DIM.
    """

    source = node_features[
        edge_index[0]
    ]

    target = node_features[
        edge_index[1]
    ]

    edge = (
        source + target
    ) / 2.0

    current_dim = (
        edge.size(-1)
    )

    if current_dim > edge_dim:

        edge = edge[
            :, :edge_dim
        ]

    elif current_dim < edge_dim:

        padding = torch.zeros(
            edge.size(0),
            edge_dim -
            current_dim,
            device=edge.device,
            dtype=edge.dtype,
        )

        edge = torch.cat(
            [
                edge,
                padding,
            ],
            dim=-1,
        )

    return edge