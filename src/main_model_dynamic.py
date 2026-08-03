"""
模块职责：
1. 承载 Carmen 中面向个体化推荐的动态药物表示与集合建模实验。
2. 实现 DFHD 风格的历史信息融合、SSPNet 风格的集合编码与个性化药物缩放。
3. 在 `main_model` 基础上扩展更强的动态预测头，供 `main_train.py` 选择调用。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_models import main_model


def _masked_last(sequence, mask):
    batch_size = sequence.size(0)
    lengths = mask.long().sum(dim=1)
    last_index = torch.clamp(lengths - 1, min=0)
    batch_index = torch.arange(batch_size, device=sequence.device)
    last_hidden = sequence[batch_index, last_index]
    last_hidden = last_hidden * (lengths > 0).unsqueeze(-1)
    return last_hidden


def _masked_mean(sequence, mask):
    weight = mask.unsqueeze(-1).float()
    denom = weight.sum(dim=2).clamp(min=1.0)
    mean_hidden = (sequence * weight).sum(dim=2) / denom
    return mean_hidden


class DynamicEmbeddingComposer(nn.Module):
    """
    Patient-specific dynamic medication embeddings:
        lambda_t = softmax(W_lambda c_t)
        Delta e_(t,m) = sum_k lambda_(t,k) a_(m,k) b_k
        e_dyn_(t,m) = e_base_m + g_(t,m) * Delta e_(t,m)
    """

    def __init__(self, emb_dim, num_medications, num_bases):
        super().__init__()
        self.dynamic_bases = nn.Parameter(torch.empty(num_bases, emb_dim))   # 可学习参数
        self.medication_coeff = nn.Parameter(torch.empty(num_medications, num_bases))   # 每个药物都有自己的动态组合系数
        self.lambda_proj = nn.Linear(emb_dim, num_bases)   # 根据患者状态生成动态基权重
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )   # gate生成器
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.dynamic_bases, gain=1.414)
        nn.init.xavier_normal_(self.medication_coeff, gain=1.414)
        nn.init.xavier_normal_(self.lambda_proj.weight, gain=1.414)
        nn.init.zeros_(self.lambda_proj.bias)
        for layer in self.gate_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.414)
                nn.init.zeros_(layer.bias)

    def forward(self, patient_state, base_med_emb):  # patient_state患者表示，base_med_embedding是药物embedding表
        lambda_t = torch.softmax(self.lambda_proj(patient_state), dim=-1)  # 根据患者状态生成动态基权重
        mixed_coeff = lambda_t.unsqueeze(1) * self.medication_coeff.unsqueeze(0)   # 融合患者动态权重和药物系数
        delta_emb = torch.einsum("bmk,kd->bmd", mixed_coeff, self.dynamic_bases)   # 组合动态基，得到药物偏移量

        expanded_patient = patient_state.unsqueeze(1).expand(-1, base_med_emb.size(0), -1)    # 扩展患者表示到每个药物
        expanded_base = base_med_emb.unsqueeze(0).expand(patient_state.size(0), -1, -1)   # 扩展基础药物 embedding 到每个患者
        gate_input = torch.cat([expanded_patient, expanded_base], dim=-1)   # 拼接患者表示  患者表示；药物表示
        gate = torch.sigmoid(self.gate_proj(gate_input)).squeeze(-1)

        dynamic_med_emb = expanded_base + gate.unsqueeze(-1) * delta_emb   # 得到个性化药物embedding
        return dynamic_med_emb, lambda_t, gate


class DFHDDualGranularityFusion(nn.Module):
    """
    DFHD-inspired fusion:
    1. coarse-grained: historical medication sequence -> patient query fusion
    2. fine-grained: historical drug memory -> medication embedding fusion
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.coarse_fusion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )  # 粗粒度融合模块
        self.history_memory_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
        )   # 历史记忆投影层
        self.fine_gate = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )   # 细粒度的gate

    def coarse_grained_fuse(self, patient_query, history_repr):  # 粗粒度
        return self.coarse_fusion(torch.cat([patient_query, history_repr], dim=-1))  # 患者当前状态 + 历史药物状态 → 更完整的患者表示

    def fine_grained_fuse(self, dynamic_med_emb, base_med_emb, history_multihot):   # 细粒度
        history_denominator = history_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)    # 每个患者历史中出现过多少种药物
        history_memory = torch.mm(history_multihot, base_med_emb) / history_denominator   # 把患者历史用过的药物 embedding 求平均
        history_memory = self.history_memory_proj(history_memory)   # 不是直接用原始历史药物平均 embedding，而是先映射一下

        history_expand = history_memory.unsqueeze(1).expand(-1, dynamic_med_emb.size(1), -1)   # 把每个患者的历史药物记忆复制到所有药物位置上
        base_expand = base_med_emb.unsqueeze(0).expand(dynamic_med_emb.size(0), -1, -1)  # 把基础药物 embedding 表复制到每个患者上
        fusion_input = torch.cat([dynamic_med_emb, base_expand, history_expand], dim=-1)   # 动态emb+药物emb+患者用药记忆
        fine_gate = torch.sigmoid(self.fine_gate(fusion_input))   # gate
        fused_med_emb = dynamic_med_emb + fine_gate * history_expand   # 最终药物 embedding = 动态药物 embedding + gate 控制后的历史药物记忆
        return fused_med_emb, history_memory, fine_gate


class SetAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(  # 多头注意力
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        batch_size = x.size(0)  # 获取batchsize
        seed = self.seed.expand(batch_size, -1, -1)   # batch 里的每个样本都有一个 seed token
        pooled, _ = self.attn(
            seed, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )   # 用 seed 对 x 做 attention pooling
        pooled = self.norm(seed + self.dropout(pooled))   # 残差连接 + LayerNorm
        return pooled.squeeze(1)


class SSPNetPersonalizedDrugScaler(nn.Module):
    """
    SSPNet-inspired PDRM:
    1. PMA pools diagnoses/procedures inside each visit
    2. Dual-RNN models historical visits
    3. Current-vs-history attention aggregates previous medications
    4. Historical medication relevance scales medication embeddings
    """

    def __init__(self, emb_dim, med_vocab_size, num_heads=4, dropout=0.1):
        super().__init__()
        # 诊断和手术的 SetAttentionBlock
        self.diagnosis_set_encoder = SetAttentionBlock(emb_dim, num_heads=num_heads, dropout=dropout)
        self.procedure_set_encoder = SetAttentionBlock(emb_dim, num_heads=num_heads, dropout=dropout)
        # 池化
        self.diagnosis_pool = PoolingByMultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout)
        self.procedure_pool = PoolingByMultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout)
        # 历史诊断和历史手术的 GRU
        self.history_diag_rnn = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.history_pro_rnn = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.current_visit_proj = nn.Linear(2 * emb_dim, emb_dim)
        self.history_visit_proj = nn.Linear(2 * emb_dim, emb_dim)
        # 药物 gate 生成模块
        self.medication_scale = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.medication_logit = nn.Linear(emb_dim, med_vocab_size) # 把患者个性化信号映射到每一个药物上
        self.medication_ffn = nn.Linear(emb_dim, emb_dim)
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

    def encode_visit_sets(self, diag_emb, pro_emb, diag_mask, pro_mask):  # 单个 visit 内部的诊断和手术 token
        diag_encoded = self.diagnosis_set_encoder(diag_emb, key_padding_mask=~diag_mask)
        pro_encoded = self.procedure_set_encoder(pro_emb, key_padding_mask=~pro_mask)
        diag_visit_repr = self.diagnosis_pool(diag_encoded, key_padding_mask=~diag_mask)
        pro_visit_repr = self.procedure_pool(pro_encoded, key_padding_mask=~pro_mask)
        return diag_encoded, pro_encoded, diag_visit_repr, pro_visit_repr

    def forward(self, diag_emb_seq, pro_emb_seq, diag_mask, pro_mask, visit_mask, med_history, base_med_emb):
        batch_size, num_visits = diag_emb_seq.size(0), diag_emb_seq.size(1)
        diag_token_dim = diag_emb_seq.size(-1)
        pro_token_dim = pro_emb_seq.size(-1)

        # 展平
        flat_diag = diag_emb_seq.reshape(-1, diag_emb_seq.size(2), diag_token_dim)
        flat_pro = pro_emb_seq.reshape(-1, pro_emb_seq.size(2), pro_token_dim)
        flat_diag_mask = diag_mask.reshape(-1, diag_mask.size(2))
        flat_pro_mask = pro_mask.reshape(-1, pro_mask.size(2))
        # 编码每个 visit 内部的诊断和手术集合
        flat_diag_encoded, flat_pro_encoded, flat_diag_visit, flat_pro_visit = self.encode_visit_sets(
            flat_diag, flat_pro, flat_diag_mask, flat_pro_mask
        )

        diag_encoded = flat_diag_encoded.reshape(batch_size, num_visits, -1, diag_token_dim)
        pro_encoded = flat_pro_encoded.reshape(batch_size, num_visits, -1, pro_token_dim)
        diag_visit_repr = flat_diag_visit.reshape(batch_size, num_visits, diag_token_dim)
        pro_visit_repr = flat_pro_visit.reshape(batch_size, num_visits, pro_token_dim)
        # 用 GRU 建模历史 visit
        diag_history_out, _ = self.history_diag_rnn(diag_visit_repr)
        pro_history_out, _ = self.history_pro_rnn(pro_visit_repr)
        # 取当前 visit 表示
        current_diag = _masked_last(diag_history_out, visit_mask)
        current_pro = _masked_last(pro_history_out, visit_mask)
        current_visit_repr = self.current_visit_proj(torch.cat([current_diag, current_pro], dim=-1))
        # 计算历史 visit 和当前 visit 的相关性
        history_visit_repr = self.history_visit_proj(torch.cat([diag_history_out, pro_history_out], dim=-1))
        history_scores = torch.einsum("btd,bd->bt", history_visit_repr, current_visit_repr)
        history_scores = history_scores.masked_fill(~visit_mask, float("-inf"))
        # 去掉当前 visit，只保留历史 visit
        history_only_mask = visit_mask.clone()
        last_index = torch.clamp(visit_mask.long().sum(dim=1) - 1, min=0)
        history_only_mask[torch.arange(batch_size, device=visit_mask.device), last_index] = False
        history_scores = history_scores.masked_fill(~history_only_mask, float("-inf"))
        # 处理没有历史 visit 的情况
        has_history = history_only_mask.any(dim=1, keepdim=True)
        safe_scores = torch.where(
            has_history,
            history_scores,
            torch.zeros_like(history_scores)
        )
        # 计算历史 visit attention
        history_attention = torch.softmax(safe_scores, dim=-1)
        history_attention = history_attention * history_only_mask.float()
        history_attention = history_attention / history_attention.sum(dim=1, keepdim=True).clamp(min=1.0)
        # 聚合历史用药相关性
        history_med_relevance = torch.einsum("bt,btm->bm", history_attention, med_history)
        personalized_signal = self.medication_scale(current_visit_repr) # 根据当前状态生成个性化信号
        medication_gate = torch.sigmoid(
            self.medication_logit(personalized_signal) + self.scale_factor * history_med_relevance
        )  # 生成 medication gate

        # 调整基础药物 embedding
        personalized_med_emb = base_med_emb.unsqueeze(0) * (1.0 + medication_gate.unsqueeze(-1))
        personalized_med_emb = self.medication_ffn(personalized_med_emb)

        # 取当前 visit 的诊断 token 和手术 token
        current_diag_tokens = diag_encoded[
            torch.arange(batch_size, device=diag_encoded.device), last_index
        ]
        current_pro_tokens = pro_encoded[
            torch.arange(batch_size, device=pro_encoded.device), last_index
        ]
        current_diag_mask = diag_mask[
            torch.arange(batch_size, device=diag_mask.device), last_index
        ]
        current_pro_mask = pro_mask[
            torch.arange(batch_size, device=pro_mask.device), last_index
        ]

        return (
            personalized_med_emb,
            current_visit_repr,
            current_diag_tokens,
            current_pro_tokens,
            current_diag_mask,
            current_pro_mask,
            medication_gate,
        )


class SSPNetSetDecoder(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.diag_cross_attn = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pro_cross_attn = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(emb_dim, 1)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, med_emb, diag_tokens, pro_tokens, diag_mask, pro_mask):
        z, _ = self.self_attn(med_emb, med_emb, med_emb, need_weights=False)
        med_emb = self.norm1(med_emb + self.dropout(z))

        diag_out, _ = self.diag_cross_attn(
            med_emb, diag_tokens, diag_tokens, key_padding_mask=~diag_mask, need_weights=False
        )
        med_emb = self.norm2(med_emb + self.dropout(diag_out))

        pro_out, _ = self.pro_cross_attn(
            med_emb, pro_tokens, pro_tokens, key_padding_mask=~pro_mask, need_weights=False
        )
        med_emb = self.norm3(med_emb + self.dropout(pro_out))

        med_context = self.ffn(med_emb)
        logits = self.output(med_context).squeeze(-1)
        return logits, med_context


class SSPNetPairwiseMatchingHead(nn.Module):
    """
    Explicit patient-drug pairing head.
    Final medication scores are produced from the pair
    (current patient state, personalized drug embedding, decoder context),
    instead of relying only on the decoder output.
    """

    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.med_proj = nn.Linear(emb_dim, emb_dim)
        self.context_proj = nn.Linear(emb_dim, emb_dim)
        self.match_proj = nn.Sequential(
            nn.Linear(4 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, query, med_emb, med_context):
        query_hidden = self.query_proj(query).unsqueeze(1).expand(-1, med_emb.size(1), -1)
        med_hidden = self.med_proj(med_emb)
        context_hidden = self.context_proj(med_context)
        pair_feature = torch.cat(
            [
                query_hidden,
                med_hidden,
                context_hidden,
                query_hidden * context_hidden,
            ],
            dim=-1,
        )
        return self.match_proj(pair_feature).squeeze(-1)


class main_model_dynamic_embedding(main_model):
    """
    Dynamic embedding model enhanced with DFHD-style historical drug fusion.
    """

    def __init__(
        self,
        vocab_size,
        ddi_adj,
        encoder,
        ddi_encoder,
        emb_dim=256,
        device=torch.device("cpu:0"),
        use_aug=True,
        ehr_adj=None,
        med2diag=None,
        med2pro=None,
        num_dynamic_basis=8,
        args=None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            ddi_adj=ddi_adj,
            encoder=encoder,
            ddi_encoder=ddi_encoder,
            emb_dim=emb_dim,
            device=device,
            use_aug=use_aug,
            ehr_adj=ehr_adj,
            med2diag=med2diag,
            med2pro=med2pro,
            args=args,
        )
        self.num_dynamic_basis = num_dynamic_basis
        self.med_pad_idx = vocab_size[2]

        self.history_embedding = nn.Embedding(
            vocab_size[2] + 1, emb_dim, padding_idx=self.med_pad_idx
        )
        self.history_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.dynamic_embedding = DynamicEmbeddingComposer(
            emb_dim=emb_dim,
            num_medications=vocab_size[2],
            num_bases=num_dynamic_basis,
        )
        self.dual_fusion = DFHDDualGranularityFusion(emb_dim)
        self.dynamic_layernorm = nn.LayerNorm(vocab_size[2])

    def get_inputs(self, dataset, MaxVisit=2):
        diag_list, pro_list, history_med_list = [], [], []
        med_list, med_ml_list, len_list = [], [], []

        max_visit = min(max([len(cur) for cur in dataset]), MaxVisit)
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        ml_med = max([len(dataset[i][j][2]) for i in range(len(dataset)) for j in range(len(dataset[i]))])

        for patient in dataset:
            for ad_idx in range(len(patient)):
                cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0], dtype=torch.long)
                cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1], dtype=torch.long)
                cur_hist_med = torch.full((max_visit, ml_med), self.med_pad_idx, dtype=torch.long)

                start_idx = max(0, ad_idx - max_visit + 1)
                active_window = patient[start_idx : ad_idx + 1]
                history_window = patient[start_idx:ad_idx]

                for row_idx, visit in enumerate(active_window):
                    d_list, p_list, _ = visit
                    cur_diag[row_idx, : len(d_list)] = torch.LongTensor(d_list)
                    cur_pro[row_idx, : len(p_list)] = torch.LongTensor(p_list)

                for row_idx, visit in enumerate(history_window):
                    _, _, hist_med = visit
                    cur_hist_med[row_idx, : len(hist_med)] = torch.LongTensor(hist_med)

                d_list, p_list, m_list = patient[ad_idx]
                len_list.append(len(active_window))

                diag_list.append(cur_diag.clone())
                pro_list.append(cur_pro.clone())
                history_med_list.append(cur_hist_med.clone())

                cur_med = torch.zeros(self.vocab_size[2])
                cur_med[m_list] = 1
                med_list.append(cur_med)

                cur_med_ml = torch.full((self.vocab_size[2],), -1, dtype=torch.long)
                cur_med_ml[: len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_med_ml)

        diag_tensor = torch.stack(diag_list).to(self.device)
        pro_tensor = torch.stack(pro_list).to(self.device)
        history_med_tensor = torch.stack(history_med_list).to(self.device)
        med_tensor_bce_target = torch.stack(med_list).to(self.device)
        med_tensor_ml_target = torch.stack(med_ml_list).to(self.device)
        len_tensor = torch.LongTensor(len_list).to(self.device)

        return (
            diag_tensor,
            pro_tensor,
            history_med_tensor,
            med_tensor_bce_target,
            med_tensor_ml_target,
            len_tensor,
        )

    def _encode_history_medications(self, med_history):
        hist_mask = med_history.ne(self.med_pad_idx).any(dim=-1)
        hist_visit_repr = self.dropout(self.history_embedding(med_history).sum(dim=-2))
        hist_output, _ = self.history_encoder(hist_visit_repr)
        history_repr = _masked_last(hist_output, hist_mask)

        history_multihot = torch.zeros(
            med_history.size(0), self.vocab_size[2], device=med_history.device
        )
        valid_mask = med_history.ne(self.med_pad_idx)
        if valid_mask.any():
            batch_idx = torch.arange(med_history.size(0), device=med_history.device).view(-1, 1, 1)
            batch_idx = batch_idx.expand_as(med_history)
            valid_batch = batch_idx[valid_mask]
            valid_med = med_history[valid_mask]
            valid_value = torch.ones(valid_med.size(0), device=med_history.device)
            history_multihot.index_put_((valid_batch, valid_med), valid_value, accumulate=True)
            history_multihot = history_multihot.clamp(max=1.0)

        return history_repr, history_multihot

    def forward(self, input, visit_len):
        if len(input) == 4:
            diag, pro, med_history, labels = input
        else:
            diag, pro, labels = input
            med_history = None

        base_query, _ = self._get_query(diag, pro, visit_len)

        if med_history is None:
            history_repr = torch.zeros_like(base_query)
            history_multihot = torch.zeros(
                base_query.size(0), self.vocab_size[2], device=base_query.device
            )
        else:
            history_repr, history_multihot = self._encode_history_medications(med_history)

        ablation = getattr(self.args, 'ablation', 'none')
        if ablation == 'no_history_context':
            history_repr = torch.zeros_like(history_repr)
            history_multihot = torch.zeros_like(history_multihot)

        if ablation in ('no_query_history_fusion', 'no_query_history_fusion_history_score'):
            fused_query = base_query
        else:
            fused_query = self.dual_fusion.coarse_grained_fuse(base_query, history_repr)

        base_med_emb, rec_loss = self.molecular_network(
            self.embeddings,
            self.med2diag,
            self.med2pro,
            self.ehr_adj,
        )

        if self.ddi_encoder:
            ddi_embedding = self.ddi_encoder(self.x, self.edge_index)
            self.ddi_embedding = ddi_embedding
            base_med_emb = base_med_emb + ddi_embedding

        if ablation in (
            'no_personalized_drug',
            'no_personalized_drug_no_history_score',
            'no_gate_cooccurrence_personalized_drug',
        ):
            fused_med_emb = base_med_emb.unsqueeze(0).expand(base_query.size(0), -1, -1)
        else:
            dynamic_med_emb, _, _ = self.dynamic_embedding(fused_query, base_med_emb)
            fused_med_emb, _, _ = self.dual_fusion.fine_grained_fuse(
                dynamic_med_emb, base_med_emb, history_multihot
            )

        normed_query = F.normalize(fused_query, p=2, dim=1)
        normed_dynamic_med_emb = F.normalize(fused_med_emb, p=2, dim=2)

        result = torch.einsum("bd,bmd->bm", normed_query, normed_dynamic_med_emb)
        result = self.dynamic_layernorm(result)

        if self.args.ddi:
            neg_pred_prob = torch.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)
            tmp_right = neg_pred_prob.unsqueeze(1)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        else:
            batch_neg = 0

        return result, batch_neg, rec_loss

    def save_embedding(self):
        super().save_embedding()
        dynamic_basis_file = os.path.join("saved", self.args.model_name, "dynamic_basis.tsv")
        medication_coeff_file = os.path.join("saved", self.args.model_name, "medication_coeff.tsv")
        history_embedding_file = os.path.join("saved", self.args.model_name, "history_med_embedding.tsv")

        dynamic_basis = self.dynamic_embedding.dynamic_bases.detach().cpu().numpy()
        medication_coeff = self.dynamic_embedding.medication_coeff.detach().cpu().numpy()
        history_embedding = self.history_embedding.weight[:-1].detach().cpu().numpy()

        np.savetxt(dynamic_basis_file, dynamic_basis, fmt="%.4f", delimiter="\t")
        np.savetxt(medication_coeff_file, medication_coeff, fmt="%.4f", delimiter="\t")
        np.savetxt(history_embedding_file, history_embedding, fmt="%.4f", delimiter="\t")
        print("saved dynamic embedding files")


class main_model_sspnet_fusion(main_model_dynamic_embedding):
    """
    Ultra-light SSPNet-inspired model:
    1. use the base patient query from the original model
    2. summarize historical medications with a simple visit average
    3. use a history-aware medication gate on top of molecular embeddings
    """

    def __init__(
        self,
        vocab_size,
        ddi_adj,
        encoder,
        ddi_encoder,
        emb_dim=256,
        device=torch.device("cpu:0"),
        use_aug=True,
        ehr_adj=None,
        med2diag=None,
        med2pro=None,
        num_dynamic_basis=8,
        args=None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            ddi_adj=ddi_adj,
            encoder=encoder,
            ddi_encoder=ddi_encoder,
            emb_dim=emb_dim,
            device=device,
            use_aug=use_aug,
            ehr_adj=ehr_adj,
            med2diag=med2diag,
            med2pro=med2pro,
            num_dynamic_basis=num_dynamic_basis,
            args=args,
        )
        self.patient_query_fusion = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.history_to_med = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, vocab_size[2]),
        )
        self.medication_residual = nn.Linear(emb_dim, emb_dim)
        self.ssp_layernorm = nn.LayerNorm(vocab_size[2])
        self.model_variant = getattr(args, 'model_variant', 'sspnet_fusion')
        self.use_patient_conditioned_matching = self.model_variant == 'sspnet_pcm'
        self.use_param_matched_matching = self.model_variant == 'sspnet_param_match'
        if self.use_patient_conditioned_matching or self.use_param_matched_matching:
            self.pairwise_matching_head = SSPNetPairwiseMatchingHead(emb_dim, dropout=0.1)
            self.pairwise_residual_scale = nn.Parameter(torch.tensor(0.1))

    def _build_visit_mask(self, visit_len, max_visit):
        arange = torch.arange(max_visit, device=visit_len.device).unsqueeze(0)
        return arange < visit_len.unsqueeze(1)

    def _build_history_multihot_per_visit(self, med_history):
        batch_size, num_visits, _ = med_history.shape
        history_multihot = torch.zeros(
            batch_size, num_visits, self.vocab_size[2], device=med_history.device
        )
        valid_mask = med_history.ne(self.med_pad_idx)
        if valid_mask.any():
            batch_idx = torch.arange(batch_size, device=med_history.device).view(-1, 1, 1)
            visit_idx = torch.arange(num_visits, device=med_history.device).view(1, -1, 1)
            batch_idx = batch_idx.expand_as(med_history)
            visit_idx = visit_idx.expand_as(med_history)
            valid_batch = batch_idx[valid_mask]
            valid_visit = visit_idx[valid_mask]
            valid_med = med_history[valid_mask]
            valid_value = torch.ones(valid_med.size(0), device=med_history.device)
            history_multihot.index_put_((valid_batch, valid_visit, valid_med), valid_value, accumulate=True)
            history_multihot = history_multihot.clamp(max=1.0)
        return history_multihot

    def forward(self, input, visit_len):
        if len(input) == 4:
            diag, pro, med_history, labels = input
        else:
            diag, pro, labels = input
            med_history = None

        base_query, _ = self._get_query(diag, pro, visit_len)
        visit_mask = self._build_visit_mask(visit_len, diag.size(1))

        if med_history is None:
            med_history = torch.full(
                (diag.size(0), diag.size(1), 1),
                self.med_pad_idx,
                dtype=torch.long,
                device=diag.device,
            )
        history_multihot = self._build_history_multihot_per_visit(med_history)  # 历史药物multi-hot

        base_med_emb, rec_loss = self.molecular_network(
            self.embeddings,
            self.med2diag,
            self.med2pro,
            self.ehr_adj,
        )  # 基础药物 embedding

        if self.ddi_encoder:
            ddi_embedding = self.ddi_encoder(self.x, self.edge_index)
            self.ddi_embedding = ddi_embedding
            base_med_emb = base_med_emb + ddi_embedding  # 药物 embedding = EHR 药物表示 + DDI 药物表示

        # 只包含历史 visit 的 mask
        history_only_mask = visit_mask.clone()
        last_index = torch.clamp(visit_len - 1, min=0)
        history_only_mask[torch.arange(visit_mask.size(0), device=visit_mask.device), last_index] = False

        history_weight = history_only_mask.float()
        history_weight = history_weight / history_weight.sum(dim=1, keepdim=True).clamp(min=1.0)  # 计算历史 visit 的权重
        history_med_relevance = torch.einsum("bt,btm->bm", history_weight, history_multihot) # 得到历史药物相关性 history_med_relevance
        history_denominator = history_med_relevance.sum(dim=1, keepdim=True).clamp(min=1.0)
        history_context = torch.mm(history_med_relevance, base_med_emb) / history_denominator  # 计算历史药物上下文 history_context

        ablation = getattr(self.args, 'ablation', 'none')
        if ablation == 'no_history_context':
            history_med_relevance = torch.zeros_like(history_med_relevance)
            history_context = torch.zeros_like(history_context)

        if ablation in ('no_query_history_fusion', 'no_query_history_fusion_history_score'):
            fused_query = base_query
        else:
            fused_query = self.patient_query_fusion(torch.cat([base_query, history_context], dim=-1))  # 融合患者当前表示和历史用药上下文

        no_personalized_ablation = ablation in (
            'no_personalized_drug',
            'no_personalized_drug_no_history_score',
            'no_gate_cooccurrence_personalized_drug',
        )
        no_history_score_ablation = ablation in (
            'no_history_score',
            'no_personalized_drug_no_history_score',
            'no_query_history_fusion_history_score',
        )

        if no_personalized_ablation:
            personalized_med_emb = base_med_emb.unsqueeze(0).expand(base_query.size(0), -1, -1)
        else:
            gate_input = torch.cat([fused_query, history_context], dim=-1)  # 生成药物门控 medication_gate
            medication_gate = torch.sigmoid(self.history_to_med(gate_input) + history_med_relevance)   # 生成药物残差 med_residual
            if ablation in ('no_medication_gate', 'no_gate_cooccurrence_personalized_drug'):
                medication_gate = torch.zeros_like(medication_gate)

            # 得到个性化药物 embedding
            med_residual = self.medication_residual(base_med_emb)
            personalized_med_emb = base_med_emb.unsqueeze(0) + medication_gate.unsqueeze(-1) * med_residual.unsqueeze(0)

        normed_query = F.normalize(fused_query, p=2, dim=1)
        normed_med = F.normalize(personalized_med_emb, p=2, dim=2)
        match_logits = torch.einsum("bd,bmd->bm", normed_query, normed_med)
        if self.use_patient_conditioned_matching:
            pair_logits = self.pairwise_matching_head(
                fused_query, personalized_med_emb, personalized_med_emb
            )
            match_logits = match_logits + self.pairwise_residual_scale * pair_logits
        elif self.use_param_matched_matching:
            zero_query = torch.zeros_like(fused_query)
            pair_logits = self.pairwise_matching_head(
                zero_query, personalized_med_emb, personalized_med_emb
            )
            match_logits = match_logits + self.pairwise_residual_scale * pair_logits

        if no_history_score_ablation:
            result = self.ssp_layernorm(match_logits)
        else:
            result = self.ssp_layernorm(
                match_logits + 0.1 * history_med_relevance
            )

        if self.args.ddi:
            neg_pred_prob = torch.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)
            tmp_right = neg_pred_prob.unsqueeze(1)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        else:
            batch_neg = 0

        return result, batch_neg, rec_loss


main_model_dfhd_fusion = main_model_dynamic_embedding
main_model_sspnet = main_model_sspnet_fusion
