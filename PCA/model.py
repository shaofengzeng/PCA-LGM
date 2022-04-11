import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.feature_align import feature_align, feature_align_superpoint
from PCA.gconv import Siamese_Gconv
from PCA.affinity_layer import Affinity
from utils.config import cfg
import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.PCA.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        ##
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))


    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
        if type == 'img' or type == 'image':
            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)
            s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)

            if i == self.gnn_layer - 2:
                emb1_new = emb1 + torch.bmm(s, emb2)
                emb2_new = emb2 + torch.bmm(s.transpose(1, 2), emb1)
                emb1 = emb1_new
                emb2 = emb2_new
                # cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                # emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                # emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                # emb1 = emb1_new
                # emb2 = emb2_new


        d, _ = self.displacement_layer(s, P_src, P_tgt)#for offect loss
        return s, d

    # def _forward_super_point(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H,
    #                          type='img'):
    #     if type != "img":
    #         raise TypeError("Arg 'type' must be 'img'")
    #     # src
    #     src_feat = self.feat_layer(src)
    #     src_semi = self.det_layer(src_feat)
    #     src_desc = self.des_layer(src_feat)
    #     src_dn = torch.norm(src_desc, p=2, dim=1)  # Compute the norm.
    #     src_desc = src_desc.div(torch.unsqueeze(src_dn, 1))  # Divide by norm to normalize.
    #     # tgt
    #     tgt_feat = self.feat_layer(tgt)
    #     tgt_semi = self.det_layer(tgt_feat)
    #     tgt_desc = self.des_layer(tgt_feat)
    #     tgt_dn = torch.norm(tgt_desc, p=2, dim=1)  # Compute the norm.
    #     tgt_desc = tgt_desc.div(torch.unsqueeze(tgt_dn, 1))  # Divide by norm to normalize.
    #
    #     #TODO
    #     # --- select points.
    #     dense = F.softmax(src_semi, dim=1)
    #     # Remove dustbin.
    #     nodust = dense[:,:-1, :, :]
    #     # Reshape to get full resolution heatmap.
    #     H,W = cfg.PAIR.RESCALE
    #     Hc = int(H / 8)
    #     Wc = int(W / 8)
    #     nodust = dense[:, :-1, :, :]
    #     nodust = nodust.permute(0, 2, 3, 1)
    #     heatmap = nodust.view(nodust.shape[0],
    #                           nodust.shape[1],
    #                           nodust.shape[2],
    #                           self.cell,
    #                           self.cell)
    #
    #     heatmap = heatmap.permute(0, 1, 3, 2, 4)
    #     heatmap = heatmap.reshape(heatmap.shape[0],
    #                               heatmap.shape[1] * self.cell,
    #                               heatmap.shape[3] * self.cell)
    #     max_value, max_ind = torch.nn.functional.max_pool2d_with_indices(heatmap,
    #                                                 kernel_size=int(heatmap.shape[1]/8),
    #                                                 stride=int(heatmap.shape[1]/8))
    #     #index to coordinate
    #     rs, cs = max_ind//heatmap.shape[2], max_ind%heatmap.shape[2]
    #     #TODO
    #
    #     # get point feature
    #     emb1 = feature_align_superpoint(src_desc, P_src, ns_src, cfg.PAIR.RESCALE)
    #     emb2 = feature_align_superpoint(tgt_desc, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
    #     # adjacency matrices
    #
    #     A_src = torch.bmm(G_src, H_src.transpose(1, 2))
    #     A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))
    #
    #     emb1, emb2 = emb1.transpose(1, 2), emb2.transpose(1, 2)
    #
    #     for i in range(self.gnn_layer):
    #         gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
    #         emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
    #         affinity = getattr(self, 'affinity_{}'.format(i))
    #         s = affinity(emb1, emb2)
    #         s = self.voting_layer(s, ns_src, ns_tgt)
    #         s = self.bi_stochastic(s, ns_src, ns_tgt)
    #
    #         if i == self.gnn_layer - 2:
    #             cross_graph = getattr(self, 'cross_graph_{}'.format(i))
    #             emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
    #             emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
    #             emb1 = emb1_new
    #             emb2 = emb2_new
    #
    #     d, _ = self.displacement_layer(s, P_src, P_tgt)
    #     return s, d
    #
    # def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
    #     if cfg.BACKBONE == "VGG16_bn":
    #         return self._forward_vgg(src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H,
    #                                  type='img')
    #     elif cfg.BACKBONE == "SuperPointNet":
    #         return self._forward_super_point(src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G,
    #                                          K_H, type='img')
