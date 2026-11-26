"""
models.py - the predictors that read one view and output class scores for every node
(the loss / metrics only use the stub nodes).

    mlp          no graph at all: each room is classified from its own features (control)
    sage         GraphSAGE, every connection treated alike            (closest to Paudel et al.)
    tagcn        TAGCN, every connection treated alike                (closest to Paudel et al.)
    sage_typed   GraphSAGE with a SEPARATE weight set per edge type   (doorway / passage / wall)
    tagcn_typed  TAGCN with a separate weight set per edge type
    gine         GINEConv, which takes the one-hot edge type as an edge feature

Why 'typed': PyG's SAGEConv takes no edge features and TAGConv takes only one number per edge, so a
one-hot edge type cannot be fed to them. A separate conv per edge type, summed, is exactly what
HeteroConv(aggr='sum') does for one node type; it is written out by hand here because it batches
with plain Data objects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, SAGEConv, TAGConv

from simulator.views import N_EDGE_TYPES, N_FEATURES, N_TYPES

MODEL_NAMES = ["mlp", "sage", "tagcn", "sage_typed", "tagcn_typed", "gine"]


def _make_conv(kind: str, din: int, dout: int):
    if kind == "sage":
        return SAGEConv(din, dout)
    if kind == "tagcn":
        return TAGConv(din, dout, K=2)
    raise ValueError(kind)


class TypedConv(nn.Module):
    """Sum over edge types of a separate conv per type (each sees only the edges of its own type)."""

    def __init__(self, kind: str, din: int, dout: int):
        super().__init__()
        self.convs = nn.ModuleList([_make_conv(kind, din, dout) for _ in range(N_EDGE_TYPES)])

    def forward(self, x, edge_index, edge_type):
        out = 0
        for t, conv in enumerate(self.convs):
            m = edge_type == t
            out = out + conv(x, edge_index[:, m])      # an empty edge set just gives the root term
        return out


class Net(nn.Module):
    def __init__(self, name: str, hidden: int = 64, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        assert name in MODEL_NAMES, name
        self.name = name
        self.dropout = dropout
        self.inp = nn.Linear(N_FEATURES, hidden)
        self.convs = nn.ModuleList()
        if name != "mlp":
            for _ in range(layers):
                if name in ("sage", "tagcn"):
                    self.convs.append(_make_conv(name, hidden, hidden))
                elif name in ("sage_typed", "tagcn_typed"):
                    self.convs.append(TypedConv(name.split("_")[0], hidden, hidden))
                elif name == "gine":
                    mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    self.convs.append(GINEConv(mlp, edge_dim=N_EDGE_TYPES))
        else:
            for _ in range(layers):
                self.convs.append(nn.Linear(hidden, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.convs])
        self.out = nn.Linear(hidden, N_TYPES)

    def forward(self, data):
        x = F.relu(self.inp(data.x))
        edge_attr = None
        if self.name == "gine":
            edge_attr = F.one_hot(data.edge_type, N_EDGE_TYPES).float()
        for conv, norm in zip(self.convs, self.norms):
            if self.name == "mlp":
                h = conv(x)
            elif self.name in ("sage_typed", "tagcn_typed"):
                h = conv(x, data.edge_index, data.edge_type)
            elif self.name == "gine":
                h = conv(x, data.edge_index, edge_attr)
            else:
                h = conv(x, data.edge_index)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h                                   # residual
        return self.out(x)
