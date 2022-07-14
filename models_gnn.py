import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, EdgeConv
from torch_geometric.utils.dropout import dropout_adj


class SPELL(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False, proj_dim=64):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()

        self.layerspf = nn.Linear(4, proj_dim) # projection layer for spatial features (4 -> 64)
        self.layer011 = nn.Linear(self.feature_dim//2+proj_dim, self.channels[0])
        self.layer012 = nn.Linear(self.feature_dim//2, self.channels[0])

        self.batch01 = BatchNorm(self.channels[0])

        self.layer11 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch11 = BatchNorm(self.channels[0])
        self.layer12 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch12 = BatchNorm(self.channels[0])
        self.layer13 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch13 = BatchNorm(self.channels[0])

        self.layer21 = SAGEConv(self.channels[0], self.channels[1])
        self.batch21 = BatchNorm(self.channels[1])

        self.layer31 = SAGEConv(self.channels[1], 1)
        self.layer32 = SAGEConv(self.channels[1], 1)
        self.layer33 = SAGEConv(self.channels[1], 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        spf = x[:, self.feature_dim:self.feature_dim+4] # coordinates for the spatial features (dim: 4)
        edge_index1 = edge_index[:, edge_attr>=0]
        edge_index2 = edge_index[:, edge_attr<=0]

        x_visual = self.layer011(torch.cat((x[:,self.feature_dim//2:self.feature_dim], self.layerspf(spf)), dim=1))
        x_audio = self.layer012(x[:,:self.feature_dim//2])
        x = x_audio + x_visual

        x = self.batch01(x)
        x = F.relu(x)

        edge_index1m, _ = dropout_adj(edge_index=edge_index1, p=self.dropout_a, training=self.training if not self.da_true else True)
        x1 = self.layer11(x, edge_index1m)
        x1 = self.batch11(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.layer21(x1, edge_index1)
        x1 = self.batch21(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        edge_index2m, _ = dropout_adj(edge_index=edge_index2, p=self.dropout_a, training=self.training if not self.da_true else True)
        x2 = self.layer12(x, edge_index2m)
        x2 = self.batch12(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.layer21(x2, edge_index2)
        x2 = self.batch21(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Undirected graph
        edge_index3m, _ = dropout_adj(edge_index=edge_index, p=self.dropout_a, training=self.training if not self.da_true else True)
        x3 = self.layer13(x, edge_index3m)
        x3 = self.batch13(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        x1 = self.layer31(x1, edge_index1)
        x2 = self.layer32(x2, edge_index2)
        x3 = self.layer33(x3, edge_index)

        x = x1 + x2 + x3
        x = torch.sigmoid(x)

        return x
