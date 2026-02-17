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


class CoreFringeSynergy(nn.Module):
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
        
        # Hyperparameters
        self.core_k = self.conf.get("core_k", 3)
        self.rerank_topM = self.conf.get("rerank_topM", 300)

        # MLPs
        # Core Logit: [r_UI, r_BI] -> scalar
        self.mlp_core = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        # Synergy: [h_core, h_fringe] -> d
        self.mlp_syn = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        # Pre-process BI graph for efficient item lookup
        self.prepare_bi_lookup()

        # Generates graph for testing (no dropout)
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)

        # Generates graph for training (with dropout if aug_type is ED)
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        
    def prepare_bi_lookup(self):
        # Convert BI sparse graph to padded tensor for fast lookup
        # self.bi_graph is a scipy sparse matrix [num_bundles, num_items]
        # We want a tensor [num_bundles, max_items_per_bundle]
        
        # Calculate max degree
        degrees = np.diff(self.bi_graph.indptr)
        max_degree = degrees.max()
        
        # Create padded tensor
        # Fill with num_items (which is an out-of-bound index, serving as padding)
        # We will need to handle this padding carefully
        self.padding_idx = self.num_items
        self.bundle_items = torch.full((self.num_bundles, max_degree), self.padding_idx, dtype=torch.long)
        
        for i in range(self.num_bundles):
            items = self.bi_graph.indices[self.bi_graph.indptr[i]:self.bi_graph.indptr[i+1]]
            if len(items) > 0:
                self.bundle_items[i, :len(items)] = torch.from_numpy(items)
                
        self.bundle_items = self.bundle_items.to(self.device)


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def propagate(self, graph, A_feature, B_feature):
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
        if test:
            UB_graph = self.UB_propagation_graph_ori
            UI_graph = self.UI_propagation_graph_ori
            BI_graph = self.BI_propagation_graph_ori
        else:
            UB_graph = self.UB_propagation_graph
            UI_graph = self.UI_propagation_graph
            BI_graph = self.BI_propagation_graph

        UB_u_rep, UB_b_rep = self.propagate(UB_graph, self.users_feature, self.bundles_feature)
        UI_u_rep, UI_i_rep = self.propagate(UI_graph, self.users_feature, self.items_feature)
        BI_b_rep, BI_i_rep = self.propagate(BI_graph, self.bundles_feature, self.items_feature)

        return (UB_u_rep, UB_b_rep, UI_u_rep, UI_i_rep, BI_b_rep, BI_i_rep)


    def calculate_core_fringe_synergy(self, users, bundles, reps):
        """
        users: [batch_size]
        bundles: [batch_size, neg_plus_one] (in training) or [batch_size, num_candidates] (in evaluation)
        reps: tuple of embeddings from get_multi_modal_representations
        """
        UB_u, UB_b, UI_u, UI_i_rep, BI_b, BI_i_rep = reps
        
        batch_size = bundles.shape[0]
        num_candidates = bundles.shape[1]
        
        # Flatten users and bundles for processing
        if users.dim() > 1:
            users = users.view(-1)

        # users: [batch_size, num_candidates] (broadcasting user to match bundles)
        users_expanded = users.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        bundles_flat = bundles.reshape(-1)
        
        # 1. Get Items for bundles
        # bundle_items: [total_pairs, max_items]
        batch_bundle_items = self.bundle_items[bundles_flat] # Padding index is self.num_items
        
        # Mask for valid items: [total_pairs, max_items]
        mask = (batch_bundle_items != self.padding_idx)
        
        # Gather item embeddings
        # We need UI_i_rep for Core/Fringe construction and UI score
        # We need BI_i_rep for BI score
        # To avoid index out of bounds with padding_idx, we can clamp or use a dummy embedding
        # Let's append a zero embedding at the end of item reps
        
        # [num_items + 1, dim]
        UI_i_rep_pad = torch.cat([UI_i_rep, torch.zeros(1, self.embedding_size, device=self.device)], dim=0)
        BI_i_rep_pad = torch.cat([BI_i_rep, torch.zeros(1, self.embedding_size, device=self.device)], dim=0)
        
        # items_ui_emb: [total_pairs, max_items, dim]
        items_ui_emb = UI_i_rep_pad[batch_bundle_items]
        items_bi_emb = BI_i_rep_pad[batch_bundle_items]
        
        # User and Bundle embeddings for scoring
        # u_ui_emb: [total_pairs, dim]
        u_ui_emb = UI_u[users_expanded]
        # b_bi_emb: [total_pairs, dim]
        b_bi_emb = BI_b[bundles_flat]
        
        # 2. Calculate Representativeness
        # r_UI = <u_ui, i_ui>: [total_pairs, max_items]
        r_UI = torch.sum(u_ui_emb.unsqueeze(1) * items_ui_emb, dim=2)
        
        # r_BI = <b_bi, i_bi>: [total_pairs, max_items]
        r_BI = torch.sum(b_bi_emb.unsqueeze(1) * items_bi_emb, dim=2)
        
        # MLP Core Input: [total_pairs, max_items, 2]
        mlp_input = torch.stack([r_UI, r_BI], dim=2)
        
        # Core Logits: [total_pairs, max_items]
        core_logits = self.mlp_core(mlp_input).squeeze(-1)
        
        # Mask logits (set score for padded items to -inf)
        core_logits = core_logits.masked_fill(~mask, float('-inf'))
        
        # Softmax -> Attention weights
        pi = F.softmax(core_logits, dim=1) # [total_pairs, max_items]
        
        # 3. Top-K Selection
        # Handle cases where bundle has fewer items than K
        valid_counts = mask.sum(dim=1) # [total_pairs]
        # We need to ensure we don't select padding indices if k > valid_counts
        # Softmax handles padded items by giving them 0 prob (due to -inf)
        
        k = min(self.core_k, batch_bundle_items.shape[1])
        topk_vals, topk_indices = torch.topk(pi, k=k, dim=1)
        
        # Re-normalize Top-K probabilities
        topk_pi = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-10)
        
        # 4. Core Representation
        # Gather core items from items_ui_emb
        # topk_indices: [total_pairs, k]
        # We need to gather from items_ui_emb: [total_pairs, max_items, dim]
        # Use gather
        
        # Expand indices for dim: [total_pairs, k, dim]
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.embedding_size)
        core_items_emb = torch.gather(items_ui_emb, 1, gather_indices)
        
        # Weighted Sum: [total_pairs, dim]
        h_core = torch.sum(core_items_emb * topk_pi.unsqueeze(-1), dim=1)
        
        # 5. Fringe Representation
        # We need to average items NOT in topk.
        # It's easier to compute (Total - Core) / Count approach?
        # Or construct a mask.
        
        # Create a mask for Top-K items
        # zeros: [total_pairs, max_items]
        is_core = torch.zeros_like(pi, dtype=torch.bool)
        is_core.scatter_(1, topk_indices, True)
        
        # Fringe mask: Valid items AND NOT Core
        is_fringe = mask & (~is_core)
        
        # Expand mask for multiplication: [total_pairs, max_items, 1]
        fringe_mask_expanded = is_fringe.unsqueeze(-1).float()
        
        # Sum of fringe items
        fringe_sum = torch.sum(items_ui_emb * fringe_mask_expanded, dim=1) # [total_pairs, dim]
        fringe_count = is_fringe.sum(dim=1, keepdim=True).clamp(min=1.0) # Avoid div by zero
        
        h_fringe = fringe_sum / fringe_count
        
        # If no fringe items (bundle size <= k), h_fringe is 0 (handled by masked sum)
        # But if fringe_count was 0, we divided by 1.0, so result is 0. Correct.
        
        # 6. Synergy
        # [total_pairs, 2*dim]
        syn_input = torch.cat([h_core, h_fringe], dim=1)
        phi = self.mlp_syn(syn_input)
        
        # 7. Final BI Part
        hat_e_BI = h_core + phi
        
        # 8. User Final and Bundle Final
        # User: [e_u_UI | e_u_UB]
        # u_ui_emb is already fetched for batch. e_u_UB needs fetching.
        u_ub_emb = UB_u[users_expanded]
        
        user_final_flat = torch.cat([u_ui_emb, u_ub_emb], dim=1)
        
        # Bundle: [e_b_UB | hat_e_BI]
        b_ub_emb = UB_b[bundles_flat]
        
        bundle_final_flat = torch.cat([b_ub_emb, hat_e_BI], dim=1)
        
        # Calculate Score
        scores = torch.sum(user_final_flat * bundle_final_flat, dim=1)
        
        return scores.reshape(batch_size, num_candidates)


    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])

        users, bundles = batch
        # bundles: [bs, 1+neg_num]
        
        reps = self.get_multi_modal_representations(test=False)
        
        scores = self.calculate_core_fringe_synergy(users, bundles, reps)

        bpr_loss = cal_bpr_loss(scores)
        c_loss = torch.zeros(1, device=self.device)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        # Full-Ranking Evaluation with Reranking Step
        # users: [batch_size] (validation/test users)
        
        UB_u, UB_b, UI_u, UI_i, BI_b, BI_i = propagate_result
        
        # 1. Baseline Scoring (for Candidate Generation)
        # User Base: [UI | UB]
        # Bundle Base: [UB | BI]
        
        batch_size = users.shape[0]
        num_bundles = self.num_bundles
        
        # User Embeddings (Base)
        u_ui = UI_u[users]
        u_ub = UB_u[users]
        u_base = torch.cat([u_ui, u_ub], dim=1) # [bs, 2d]
        
        # Bundle Embeddings (Base) - All bundles
        # b_ub: [num_bundles, d]
        # b_bi: [num_bundles, d]
        b_ub = UB_b
        b_bi = BI_b
        b_base = torch.cat([b_ub, b_bi], dim=1) # [num_bundles, 2d]
        
        # Dot Product for Base Scores
        base_scores = torch.mm(u_base, b_base.t()) # [bs, num_bundles]
        
        # If rerank_topM is -1, rerank EVERYTHING (slow)
        topM = self.rerank_topM
        if topM == -1:
            topM = num_bundles
            
        # 2. Select Candidates using Base Scores
        # If topM is smaller than num_bundles, select topM
        if topM < num_bundles:
            top_vals, top_indices = torch.topk(base_scores, k=topM, dim=1)
            # top_indices: [bs, topM] -- these are the bundles to rerank
            
            # 3. Rerank Candidates
            # We call calculate_core_fringe_synergy for these (user, candidate_bundle) pairs
            
            rerank_scores = self.calculate_core_fringe_synergy(users, top_indices, propagate_result)
            # rerank_scores: [bs, topM]
            
            # Now we need to construct the full score matrix efficiently
            # We want to return a [bs, num_bundles] score matrix where:
            # - Indices in top_indices have rerank_scores
            # - Indices NOT in top_indices have -infinity (effectively ignoring them for topK metric calc if K << M)
            # But wait, original train.py generic evaluate expects full scores to sort top K.
            # If we set non-candidates to -inf, they won't be in top K [20, 40].
            # This is fine as long as rerank_topM >> topK. 300 >> 40.
            
            final_scores = torch.full_like(base_scores, float('-inf'))
            final_scores.scatter_(1, top_indices, rerank_scores)
            
            return final_scores
            
        else:
            # Rerank ALL bundles
            # This requires passing [bs, num_bundles] tensor which might OOM if we simply call calculate...
            # We need to chunk it.
            
            # Creating a full bundle index tensor [bs, num_bundles] is too big (e.g. 2048 * 30000 ints)
            # We must iterate over chunks of bundles
            
            final_scores = torch.zeros_like(base_scores)
            
            chunk_size = 200 # Process 200 bundles at a time for all users
            all_bundles = torch.arange(num_bundles, device=self.device)
            
            for i in range(0, num_bundles, chunk_size):
                end = min(i + chunk_size, num_bundles)
                bundle_chunk = all_bundles[i:end] # [chunk_len]
                
                # Expand to batch
                # bundles_in: [bs, chunk_len]
                bundles_in = bundle_chunk.unsqueeze(0).expand(batch_size, -1)
                
                chunk_scores = self.calculate_core_fringe_synergy(users, bundles_in, propagate_result)
                final_scores[:, i:end] = chunk_scores
                
            return final_scores
