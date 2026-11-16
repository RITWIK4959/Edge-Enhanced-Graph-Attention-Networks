import torch.nn.functional as F

from torch import nn

from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
)

from .edge_encoder import (
    LearnedEdgeEncoder,
)


# ============================================================
# INPUT PROJECTION
# ============================================================

class InputProjection(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout,
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_channels,
            hidden_channels,
        )

        self.norm = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.linear(x)

        x = self.norm(x)

        x = F.elu(x)

        x = self.dropout(x)

        return x


# ============================================================
# SMALL CLASSIFIER
# ============================================================

class Classifier(nn.Module):

    """
    Smaller classifier to reduce overfitting.

    Instead of:
        hidden -> hidden -> classes

    use:
        hidden -> classes
    """

    def __init__(
        self,
        hidden_channels,
        num_classes,
        dropout,
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, x):

        x = self.dropout(x)

        return self.linear(x)


# ============================================================
# GAT
# ============================================================

class GATModel(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=64,
        heads=4,
        feature_dropout=0.45,
        attention_dropout=0.35,
        classifier_dropout=0.45,
    ):

        super().__init__()

        self.input = InputProjection(
            in_channels,
            hidden_channels,
            feature_dropout,
        )

        # Layer 1
        self.conv1 = GATConv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
        )

        # Layer 2
        self.conv2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=attention_dropout,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            feature_dropout
        )

        self.classifier = Classifier(
            hidden_channels,
            num_classes,
            classifier_dropout,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
    ):

        x = self.input(x)

        h = self.conv1(
            x,
            edge_index,
        )

        h = self.norm1(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        # Residual connection
        x = x + h

        h = self.conv2(
            x,
            edge_index,
        )

        h = self.norm2(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        return self.classifier(
            x
        )


# ============================================================
# GATv2
# ============================================================

class GATv2Model(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=64,
        heads=4,
        feature_dropout=0.45,
        attention_dropout=0.35,
        classifier_dropout=0.45,
    ):

        super().__init__()

        self.input = InputProjection(
            in_channels,
            hidden_channels,
            feature_dropout,
        )

        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
        )

        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=attention_dropout,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            feature_dropout
        )

        self.classifier = Classifier(
            hidden_channels,
            num_classes,
            classifier_dropout,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
    ):

        x = self.input(x)

        h = self.conv1(
            x,
            edge_index,
        )

        h = self.norm1(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        h = self.conv2(
            x,
            edge_index,
        )

        h = self.norm2(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        return self.classifier(
            x
        )


# ============================================================
# GATv2 + STATIC EDGE
# ============================================================

class GATv2StaticEdgeModel(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=64,
        heads=4,
        edge_dim=16,
        feature_dropout=0.45,
        attention_dropout=0.35,
        classifier_dropout=0.45,
        edge_dropout=0.10,
    ):

        super().__init__()

        self.input = InputProjection(
            in_channels,
            hidden_channels,
            feature_dropout,
        )

        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=edge_dim,
        )

        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=edge_dim,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            feature_dropout
        )

        self.edge_dropout = nn.Dropout(
            edge_dropout
        )

        self.classifier = Classifier(
            hidden_channels,
            num_classes,
            classifier_dropout,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):

        x = self.input(x)

        edge_attr = self.edge_dropout(
            edge_attr
        )

        h = self.conv1(
            x,
            edge_index,
            edge_attr,
        )

        h = self.norm1(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        h = self.conv2(
            x,
            edge_index,
            edge_attr,
        )

        h = self.norm2(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        return self.classifier(
            x
        )


# ============================================================
# GATv2 + LEARNED EDGE
# ============================================================

class GATv2LearnedEdgeModel(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=64,
        heads=4,
        edge_dim=16,
        edge_encoder_hidden=16,
        feature_dropout=0.45,
        attention_dropout=0.35,
        classifier_dropout=0.45,
    ):

        super().__init__()

        self.input = InputProjection(
            in_channels,
            hidden_channels,
            feature_dropout,
        )

        self.edge_encoder1 = (
            LearnedEdgeEncoder(
                node_dim=hidden_channels,
                edge_dim=edge_dim,
                hidden_dim=edge_encoder_hidden,
            )
        )

        self.edge_encoder2 = (
            LearnedEdgeEncoder(
                node_dim=hidden_channels,
                edge_dim=edge_dim,
                hidden_dim=edge_encoder_hidden,
            )
        )

        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=edge_dim,
        )

        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=edge_dim,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            feature_dropout
        )

        self.classifier = Classifier(
            hidden_channels,
            num_classes,
            classifier_dropout,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
    ):

        x = self.input(x)

        # ----------------------------------------------------
        # First learned edge representation
        # ----------------------------------------------------

        source = x[
            edge_index[0]
        ]

        target = x[
            edge_index[1]
        ]

        learned_edge1 = (
            self.edge_encoder1(
                source,
                target,
            )
        )

        h = self.conv1(
            x,
            edge_index,
            learned_edge1,
        )

        h = self.norm1(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        # ----------------------------------------------------
        # Second learned edge representation
        # ----------------------------------------------------

        source = x[
            edge_index[0]
        ]

        target = x[
            edge_index[1]
        ]

        learned_edge2 = (
            self.edge_encoder2(
                source,
                target,
            )
        )

        h = self.conv2(
            x,
            edge_index,
            learned_edge2,
        )

        h = self.norm2(
            h
        )

        h = F.elu(h)

        h = self.dropout(
            h
        )

        x = x + h

        return self.classifier(
            x
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(
    model_name,
    in_channels,
    num_classes,
):

    common = dict(

        in_channels=in_channels,

        num_classes=num_classes,

        hidden_channels=64,

        heads=4,

        feature_dropout=0.45,

        attention_dropout=0.35,

        classifier_dropout=0.45,
    )

    if model_name == "GAT":

        return GATModel(
            **common
        )

    if model_name == "GATv2":

        return GATv2Model(
            **common
        )

    if model_name == (
        "GATv2 + Static Edge"
    ):

        return GATv2StaticEdgeModel(
            **common,
            edge_dim=16,
            edge_dropout=0.10,
        )

    if model_name == (
        "GATv2 + Learned Edge"
    ):

        return GATv2LearnedEdgeModel(
            **common,
            edge_dim=16,
            edge_encoder_hidden=16,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )