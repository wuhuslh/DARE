import torch
import torch.nn as nn
# from torch.nn import Transformer
import torch.nn.functional as F
from layers.gcn_layer import GCNLayer, ConvReadoutLayer
# from torch_geometric.nn import GATConv
# import torch_geometric.utils as utils
from layers.conv_layer import ConvLayer, MAXPoolLayer
import math
import random
import os
import numpy as np
from layers.mlp_readout_layer import MLPReadout
from nets.RNA_graph_classification.transformer import *

class AvgPoolLayer(nn.Module):
    def __init__(self):
        super(AvgPoolLayer, self).__init__()

    def forward(self, input):
        return torch.mean(input, dim=0)


class TransGCNNet(nn.Module):
    def __init__(self, net_params, n_layers, n_head, d_model, d_ff, dropout):
        super().__init__()
        self.device = net_params['device']
        self.sequence = None
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers_gnn = nn.ModuleList()

        self.layers_gnn.append(GCNLayer(
            hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        for _ in range(self.n_layers * 2 - 2):
            self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # GNN end
        self.conv_readout_layer = ConvReadoutLayer(self.readout)

        # self.transformer_encoder = TransformerEncoder(
        #     d_model, n_head, n_layers, d_ff, dropout)
        self.embedding = nn.Linear(in_features=36, out_features=d_model)
        self.pooling = AvgPoolLayer()  # 平均池化层

        self.encoder_layers = TransformerEncoderLayer(512, 2, 1024, 0.1)

        self.encoder_feat = TransformerEncoder(self.encoder_layers, 2)
        self.encoder_str = TransformerEncoder(self.encoder_layers, 2)

        window_size = 501
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0] // 2, 0], 1
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0] // 2, 0], 2
        width_o1 = math.ceil(
            (window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o1 = math.ceil(
            (width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)
        width_o2 = math.ceil(
            (width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o2 = math.ceil(
            (width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)

        self.layers_cnn = nn.ModuleList()
        self.layers_cnn.append(ConvLayer(
            1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        for _ in range(self.n_layers - 1):
            self.layers_cnn.append(
                ConvLayer(32, 32, conv_kernel2, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))

        self.layers_pool = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers_pool.append(MAXPoolLayer(
                pooling_kernel, stride=pooling_stride, padding=pooling_padding))

        # self.classifier = MLPReadout(512 + 4032, 2)
        self.classifier = MLPReadout(512, 2)

    def forward(self, g, tqt=None):
        ##GCN
        h1 = self.embedding_h(g.ndata['feat'])
        h1 = self.in_feat_dropout(h1)

        for i in range(self.n_layers):
            # GNN
            h1 = self.layers_gnn[2 * i](g, h1)
            h1 = self.layers_gnn[2 * i + 1](g, h1)

        g.ndata['h'] = h1

        hg = self.conv_readout_layer(g, h1).squeeze(3)
        hg = hg.permute(0, 2, 1)

        ##concat seq featrue
        hs = self._graph2feature(g).squeeze(1)
        hs = hs.to(self.device)

        hg = torch.cat([hg, hs], dim=2)


        ##transformer
        h2 = self.embedding(hg)
        h2 = h2.permute(1, 0, 2)
        # output = self.transformer_encoder(h2)
        # output = self.pooling(output)
        output = self.encoder_feat(h2)
        output = self.pooling(output)

        # h2 = self._graph2feature(g)
        # self.sequence = h2
        # h2 = h2.to(self.device)
        # h2 = self.embedding(h2)
        # h2 = h2.permute(1, 0, 2)
        # output = self.transformer_encoder(h2)
        # output = self.pooling(output)  # 对节点表示进行汇总

        ##CNN
        hc = self._graph2feature(g)
        self.sequence = hc
        hc = hc.to(self.device)

        for i in range(self.n_layers):
            hc = self.layers_cnn[i](hc)
            if i == 0:
                self.filter_out = hc
            hc = self.layers_pool[i](hc)

        ##MLP
        hc = torch.flatten(hc, start_dim=1)
        h_final = torch.cat([output, hc], dim=1)
        pred = self.classifier(output)

        return pred

    def _graph2feature(self, g):
        feat = g.ndata['feat']
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes():
            if first_flag == 0:
                output = torch.transpose(
                    feat[start:start + batch_num], 1, 0).unsqueeze(0)
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(
                    feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        return output
