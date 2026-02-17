#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class DualLightGCN(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        
        # Check if MLP score is enabled
        self.score_mlp = self.conf.get("score_mlp", 0)

        self.init_emb()
        
        if self.score_mlp:
            # Input: concat([u, b, u*b]) -> 3 * (2 * emb_size) because of fusion (cat 2 views)
            # Fusion size for u and b is 2 * embedding_size
            self.fused_dim = 2 * self.embedding_size
            self.mlp = nn.Sequential(
                nn.Linear(self.fused_dim * 3, self.fused_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.fused_dim * 2, 1)
            )

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # Generates graph for testing (no dropout)
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # Generates graph for training (with dropout if aug_type is ED)
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature):
        # LightGCN propagation
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            all_features.append(features)
        
        all_features = torch.stack(all_features, dim=1)
        all_features = torch.mean(all_features, dim=1)
        
        A_rep, B_rep = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_rep, B_rep


    def get_multi_modal_representations(self, test=False):
        # Select graphs based on mode
        if test:
            UB_graph = self.UB_propagation_graph_ori
            UI_graph = self.UI_propagation_graph_ori
            BI_graph = self.BI_aggregation_graph_ori
        else:
            UB_graph = self.UB_propagation_graph
            UI_graph = self.UI_propagation_graph
            BI_graph = self.BI_aggregation_graph

        # 1. UB LightGCN
        UB_users_rep, UB_bundles_rep = self.propagate(UB_graph, self.users_feature, self.bundles_feature)

        # 2. UI LightGCN
        UI_users_rep, UI_items_rep = self.propagate(UI_graph, self.users_feature, self.items_feature)

        # 3. BI Mean Aggregation
        BI_bundles_rep = torch.matmul(BI_graph, UI_items_rep)

        # 4. Fusion (Concatenation)
        # users: [e_u^UB | e_u^UI]
        users_rep = torch.cat([UB_users_rep, UI_users_rep], dim=1)
        
        # bundles: [e_b^UB | e_b^BI]
        bundles_rep = torch.cat([UB_bundles_rep, BI_bundles_rep], dim=1)

        return users_rep, bundles_rep


    def compute_score(self, users_emb, bundles_emb):
        # users_emb: [batch_size, dim]
        # bundles_emb: [batch_size, dim]
        
        if self.score_mlp:
            # MLP Score
            element_wise = users_emb * bundles_emb
            mlp_input = torch.cat([users_emb, bundles_emb, element_wise], dim=1)
            scores = self.mlp(mlp_input).squeeze(-1)
        else:
            # Dot Product Score
            scores = torch.sum(users_emb * bundles_emb, dim=1)
            
        return scores


    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        users, bundles = batch
        # bundles: [bs, 1+neg_num]
        
        users_rep, bundles_rep = self.get_multi_modal_representations(test=False)

        # Prepare embeddings for scoring
        # users_embedding: [bs, 1+neg_num, dim]
        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        
        # bundles_embedding: [bs, 1+neg_num, dim]
        bundles_embedding = bundles_rep[bundles]

        # Flatten for simpler scoring calculation
        batch_size, neg_plus_one, dim = users_embedding.shape
        users_flat = users_embedding.view(-1, dim)
        bundles_flat = bundles_embedding.view(-1, dim)

        scores_flat = self.compute_score(users_flat, bundles_flat)
        pred = scores_flat.view(batch_size, neg_plus_one)

        bpr_loss = cal_bpr_loss(pred)
        c_loss = torch.zeros(1, device=self.device)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        # users: [batch_size] 
        # propagate_result: (users_rep, bundles_rep) (Entire set)
        
        users_rep, bundles_rep = propagate_result
        batch_users_emb = users_rep[users] # [batch_size, dim]
        
        if self.score_mlp:
            # MLP scoring must be chunked to avoid OOM
            # We need to compute scores for [batch_size] users vs [all_bundles]
            
            num_bundles = bundles_rep.shape[0]
            batch_size = batch_users_emb.shape[0]
            scores = torch.zeros(batch_size, num_bundles, device=self.device)
            
            chunk_size = 1000 # Adjust based on memory
            
            for i in range(0, num_bundles, chunk_size):
                end = min(i + chunk_size, num_bundles)
                bundle_chunk = bundles_rep[i:end] # [chunk_size, dim]
                chunk_len = bundle_chunk.shape[0]
                
                # Broad cast: [batch_size, chunk_size, dim]
                u_exp = batch_users_emb.unsqueeze(1).expand(-1, chunk_len, -1)
                b_exp = bundle_chunk.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Flatten
                u_flat = u_exp.reshape(-1, self.fused_dim)
                b_flat = b_exp.reshape(-1, self.fused_dim)
                
                chunk_scores = self.compute_score(u_flat, b_flat)
                scores[:, i:end] = chunk_scores.view(batch_size, chunk_len)
                
            return scores
            
        else:
            # Dot product scoring (Standard Matrix Multiplication)
            scores = torch.mm(batch_users_emb, bundles_rep.t())
            return scores
