"""
模块职责：
1. 定义 Carmen 主干模型使用的核心分子编码器、图卷积组件和药物匹配头。
2. 提供基础 `main_model`，以及在其上扩展的动态 basis 个性化打分版本。
3. 负责把患者就诊表示与药物分子表示对齐，输出最终的多标签用药预测分数。
"""

from collections import defaultdict
from copy import deepcopy
import os
from tkinter.messagebox import NO
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from dnc import DNC
except ImportError:
    DNC = None
from layers import FALayer, GCNLayer
try:
    import dgl
except Exception:
    from dgl_fallback import dgl
import math
import pdb
from torch.nn.parameter import Parameter


class Fagcn_main(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=1):
        super().__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):   # 多层卷积，传入图，隐藏层，dropout
            self.layers.append(FALayer(self.g, hidden_dim, dropout))
            # self.layers.append(GCNLayer(self.g, hidden_dim, dropout))


        self.t0 = nn.Linear(in_dim, hidden_dim)   # 输入->隐藏层
        self.t1 = nn.Linear(hidden_dim, out_dim)   # 隐藏层->输出
        self.context_attn = nn.Linear(hidden_dim, hidden_dim)  # 上下文学习
        self.reset_parameters()

    def reset_parameters(self):   # 防止梯度爆炸
        nn.init.xavier_normal_(self.t0.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.context_attn.weight, gain=1.414)

    def forward(self, h, context=None):  # h是节点特征
        raw = h   # 这里后面没有用到
        for i in range(self.layer_num):
            m = self.layers[i](h)  # 根据layer_num来聚合邻居信息

            # update h with context，上下文学习
            attn = torch.tanh(self.context_attn(context))  # eq 7
            m = attn * m
            
            h = self.eps * h + m  # 残差链接
            h = torch.relu(h)
        return h


class MolecularGraphNeuralNetwork_record(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device, fingers, avg_projection, g=None, args=None):
        super().__init__()
        self.device = device
        self.args = args
        self.avg_projection = avg_projection.to(device)
        self.embed_fingerprint = nn.Embedding(N_fingerprint+1, dim, padding_idx=N_fingerprint).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = fingers
        self.fingerprints = torch.cat(fingerprints)
        self.molecular_sizes = [int(size) for size in molecular_sizes]
        if g is None:
            g = self.build_graph(adjacencies)
            g = dgl.to_simple(g)
            g = dgl.remove_self_loop(g)
            g = dgl.to_bidirected(g)
            dill.dump(g, open("g.pkl", 'wb'))

        g = g.to(self.device)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.encoder: Fagcn_main = Fagcn_main(
            g, dim, dim, dim, dropout=0.5, eps=0.3, layer_num=8)

        self.beta = 1
        Nmed = avg_projection.shape[0]
        self.viewcat = nn.Linear(2*dim, dim)
        self.fc_selector = nn.Linear(dim, dim)

    def build_graph(self, adjacencies):
        edge_u, edge_v = [], []
        offset = 0
        for adjacency in adjacencies:
            adjacency = adjacency.cpu()
            edges = adjacency.nonzero(as_tuple=False)
            if edges.numel() > 0:
                edge_u.append(edges[:, 0] + offset)
                edge_v.append(edges[:, 1] + offset)
            offset += adjacency.shape[0]

        if edge_u:
            U = torch.cat(edge_u)
            V = torch.cat(edge_v)
        else:
            U = torch.empty(0, dtype=torch.int64)
            V = torch.empty(0, dtype=torch.int64)

        num_nodes = self.fingerprints.shape[0]
        return dgl.graph((U, V), num_nodes=num_nodes).to('cpu')

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def max(self, vectors, axis):
        max_vectors = [torch.max(v, 0).values for v in torch.split(vectors, axis)]
        return torch.stack(max_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, *rec_args):
        """
        visit_emb(:Tensor) with shape (Nbatch, dim)
        labels(:Tensor) with shape (Nbatch, Nmed) each row is a mult-hot vector
        """

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(self.fingerprints)
        context = self.update_recemb(*rec_args)
        repeat_sizes = torch.as_tensor(self.molecular_sizes, device=context.device, dtype=torch.long)
        context = torch.repeat_interleave(context, repeat_sizes, dim=0)
        fingerprint_vectors = self.encoder(fingerprint_vectors, context)  # eq 7, 8 9

        # Molecular vector by sum or mean of the fingerprint vectors
        molecular_vectors = self.sum(fingerprint_vectors, self.molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
        mpnn_emb = torch.mm(self.avg_projection, molecular_vectors)

        return mpnn_emb, 0
    
    def update_recemb(self, embeddings, med2diag, med2pro, ehradj_idx):
        diag_emb, pro_emb = embeddings[0], embeddings[1]
        Ndiag, Npro = med2diag.shape[1], med2pro.shape[1]
        diag_emb = diag_emb(torch.arange(Ndiag).to(self.device))
        pro_emb = pro_emb(torch.arange(Npro).to(self.device))
        # pdb.set_trace()
        med_diagview = torch.mm(med2diag, diag_emb)
        med_proview = torch.mm(med2pro, pro_emb)
        med_rec = torch.cat((med_diagview, med_proview), -1)
        med_rec = self.viewcat(med_rec)
        med_rec = med_rec + self.cooccu_aug(med_rec, ehradj_idx)
        return med_rec
    
    def cooccu_aug(self, context, ehr_adj):
        aug_emb = torch.mm(ehr_adj, context)
        sel_attn = self.fc_selector(context.clone()).tanh()
        aug_emb = sel_attn * aug_emb
        return aug_emb


class LearnableMedicationIDEncoder(nn.Module):
    """
    Ablation encoder for Ours w/o Mol.
    It removes molecular-structure inputs and uses one learnable embedding
    vector per medication as the base medication representation.
    """

    def __init__(self, num_medications, dim, layer_hidden=None, device=torch.device('cpu:0'),
                 fingers=None, avg_projection=None, g=None, args=None):
        super().__init__()
        self.device = device
        self.medication_embedding = nn.Embedding(num_medications, dim).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.medication_embedding.weight, gain=1.414)

    def forward(self, *rec_args):
        medication_ids = torch.arange(
            self.medication_embedding.num_embeddings,
            device=self.device,
            dtype=torch.long,
        )
        return self.medication_embedding(medication_ids), 0


class DynamicBasisDrugScorer(nn.Module):
    """
    Build patient-specific medication embeddings with:
    1. shared dynamic bases B
    2. medication-specific coefficients a_m
    3. patient-dependent weights lambda_t = softmax(W_lambda c_t)
    4. a learnable gate g_{t,m} that controls the correction strength
    """
    def __init__(self, emb_dim, num_medications, num_bases):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_medications = num_medications
        self.num_bases = num_bases

        self.dynamic_bases = nn.Parameter(torch.empty(num_bases, emb_dim))
        self.drug_basis_coeff = nn.Parameter(torch.empty(num_medications, num_bases))
        self.lambda_proj = nn.Linear(emb_dim, num_bases)
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.dynamic_bases, gain=1.414)
        nn.init.xavier_normal_(self.drug_basis_coeff, gain=1.414)
        nn.init.xavier_normal_(self.lambda_proj.weight, gain=1.414)
        nn.init.zeros_(self.lambda_proj.bias)
        for layer in self.gate_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.414)
                nn.init.zeros_(layer.bias)

    def forward(self, patient_state, base_med_emb):
        """
        patient_state: (B, D)
        base_med_emb:  (M, D)

        Returns:
            dynamic_med_emb: (B, M, D)
            lambda_t:        (B, K)
            gate:            (B, M)
        """
        lambda_t = torch.softmax(self.lambda_proj(patient_state), dim=-1)
        basis_mix = lambda_t.unsqueeze(1) * self.drug_basis_coeff.unsqueeze(0)
        delta_emb = torch.einsum('bmk,kd->bmd', basis_mix, self.dynamic_bases)

        patient_expand = patient_state.unsqueeze(1).expand(-1, base_med_emb.size(0), -1)
        base_expand = base_med_emb.unsqueeze(0).expand(patient_state.size(0), -1, -1)
        gate_input = torch.cat([patient_expand, base_expand], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input)).squeeze(-1)

        dynamic_med_emb = base_expand + gate.unsqueeze(-1) * delta_emb
        return dynamic_med_emb, lambda_t, gate


class main_model(nn.Module):
    def __init__(self, vocab_size, ddi_adj, encoder, ddi_encoder,
                 emb_dim=256,
                 device=torch.device('cpu:0'),
                 use_aug=True,
                 ehr_adj=None,
                 med2diag=None,
                 med2pro=None,
                 args=None):
        super().__init__()
        self.use_aug = use_aug
        self.args = args
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.med2diag = med2diag
        self.med2pro = med2pro
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device) if ehr_adj is not None else None
        self.device = device
        self.vocab_size = vocab_size

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i]+1, emb_dim, padding_idx=vocab_size[i]) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        self.molecular_network = encoder

        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.aug_MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.aug_MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.fc_selector = nn.Linear(emb_dim, emb_dim)

        self.ddi_encoder = ddi_encoder

        adj_tensor = torch.tensor(ddi_adj)
        self.edge_index = adj_tensor.nonzero().t().contiguous()
        # x = np.ones((np.size(ddi_adj, 0),1))
        x = np.random.rand(np.size(ddi_adj, 0),1)
        self.x = torch.Tensor(x)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)

        self.ddi_embedding = None

    def get_inputs(self, dataset, MaxVisit=2):
        # 将list的数据形式转换为tensor形式的
        # use the pad index to make th same length tensor
        diag_list, pro_list, med_list, med_ml_list, len_list = [], [], [], [], []
        # 将药物操作visit变成一样大小
        max_visit = min(max([len(cur) for cur in dataset]), MaxVisit)
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        # [v1, v2, v3] -> [v1], [v1, v2], [v1, v2, v3]
        for p in dataset:
            # 填充
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            for ad_idx in range(len(p)):   # 遍历每一次就诊
                d_list, p_list, m_list = p[ad_idx]  # 拆分本次就诊的诊断、检查、用药列表
                if ad_idx >= max_visit:  # 用滑动窗口来填充特征
                    cur_diag[:-1] = cur_diag[1:]
                    cur_pro[:-1] = cur_pro[1:]  # 整体后移一个位置
                    cur_diag[-1] = self.vocab_size[0]
                    cur_pro[-1] = self.vocab_size[1]   # 最后一个位置填上初始值
                    cur_diag[-1, :len(d_list)] = torch.LongTensor(d_list)
                    cur_pro[-1, :len(p_list)] = torch.LongTensor(p_list)  # 填充并移到张量上
                    # visit len mask
                    len_list.append(max_visit)
                else:  # 如果就诊数很少，直接填充
                    cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
                    cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
                    # visit len mask
                    len_list.append(ad_idx + 1)

                # 克隆一遍
                diag_list.append(cur_diag.long().clone())
                pro_list.append(cur_pro.long().clone())
                # bce target，生成两种用药标签（适配不同损失函数）
                cur_med = torch.zeros(self.vocab_size[2])
                cur_med[m_list] = 1
                med_list.append(cur_med)
                # multi-label margin target
                cur_med_ml = torch.full((self.vocab_size[2],), -1)
                cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_med_ml)


        # 移动设备
        diag_tensor = torch.stack(diag_list).to(self.device)
        pro_tensor = torch.stack(pro_list).to(self.device)
        med_tensor_bce_target = torch.stack(med_list).to(self.device)
        med_tensor_ml_target = torch.stack(med_ml_list).to(self.device)
        len_tensor = torch.LongTensor(len_list).to(self.device)

        return diag_tensor, pro_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor
    
    def get_batch(self, data, batchsize=None):  # 支持 “全量返回” 和 “随机打乱后按批次返回” 两种模式
        # diag_tensor, pro_tensor, med_tensor, len_tensor
        # data = self.get_inputs(dataset)
        if batchsize is None:  # 验证集 / 测试集
            yield data
        else:
            # 获取总样本数
            N = data[0].shape[0]
            # 生成索引
            idx = np.arange(N)
            # 打乱，增加泛化能力
            np.random.shuffle(idx)
            i = 0
            # 分批次处理
            while i < N:
                cur_idx = idx[i:i+batchsize]
                res = [cur_data[cur_idx] for cur_data in data]
                yield res
                i += batchsize

    # 患者表示学习模块
    def _get_query(self, diag, pro, visit_len):  # 处理成张量
        diag_emb_seq = self.dropout(self.embeddings[0](diag).sum(-2))
        pro_emb_seq = self.dropout(self.embeddings[1](pro).sum(-2))
        o1, h1 = self.encoders[0](diag_emb_seq)
        o2, h2 = self.encoders[1](pro_emb_seq)  # o2 with shape (B, M, D)
        # NOTE: select by len
        # o1, o2 with shape (B, D)
        o1 = torch.stack([o1[i, visit_len[i]-1, :] for i in range(visit_len.shape[0])])
        o2 = torch.stack([o2[i, visit_len[i]-1, :] for i in range(visit_len.shape[0])])   # 剔除填充的无效就诊步骤，只保留每个样本的最后一次有效就诊特征
        # 因为padding了一些无用的值，所以不一定最后一个就是有效值


        # 多模态融合和生成Q
        patient_representations = torch.cat([o1, o2], dim=-1)  # (B, dim*2)
        query = self.query(patient_representations)  # (B, dim)


        # 归一化
        norm_of_query = torch.norm(query, 2, 1, keepdim=True)
        normed_query = (norm_of_query / (1 + norm_of_query)) * (query / norm_of_query)  # 张量归一化
        return query, normed_query


    # 把 “患者编码→药物编码→匹配→DDI 惩罚” 的所有步骤封装在forward里
    def forward(self, input, visit_len):
        """
        Args:
            input(:list) with shape [(B, M, N_x)]. x can be diag, pro, med 
            len(:list/LongTensor) with shape (B, 1)
        """
        diag, pro, labels = input
        query, normed_query = self._get_query(diag, pro, visit_len)  # (Batch, dim)，嵌入

        # 药物分子结构编码
        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro, self.ehr_adj)  # (N_medication, dim)

        # DDI编码
        if self.ddi_encoder:
            ddi_embedding = self.ddi_encoder(self.x, self.edge_index)
            self.ddi_embedding = ddi_embedding
            # print("self.ddi_embedding", self.ddi_embedding)
            MPNN_emb += ddi_embedding

        #  cosine samilarity，计算余弦相似度
        # MPNN_emb: (M, dim), normed_query (dim,)
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)  # 归一化
        # normed_MPNN_emb = self.ddi_embedding
        # print("normed_MPNN_emb", normed_MPNN_emb)
        MPNN_match = (torch.mm(normed_query, normed_MPNN_emb.t()))  # (B, N_med)，计算余弦相似度
        MPNN_att = self.MPNN_layernorm(MPNN_match)  # 层归一化
        result = MPNN_att  # result: (M,)


        if self.args.ddi:   # 计算ddi惩罚项
            neg_pred_prob = F.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)  # (B, Nmed, 1)
            tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, Nmed)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)  # (N, Nmed, Nmed)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        else:
            batch_neg = 0

        return result, batch_neg, 0

    def save_embedding(self):
        #  生成并处理药物（MPNN）嵌入
        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro, self.ehr_adj)  # 生成嵌入
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)   # 归一化
        med_emb = MPNN_emb.detach().cpu().numpy()  # 张量转化为numpy
        normed_med_emb = normed_MPNN_emb.detach().cpu().numpy()
        # 模型的嵌入张量是计算图的一部分，带有梯度信息。保存时不需要梯度，detach()可以把张量从计算图中剥离，节省内存，也避免后续操作意外修改计算图

        # 处理诊断（diag）嵌入
        diag_emb = self.embeddings[0].weight[:-1]  # .detach().cpu().numpy()，提取嵌入层权重，去掉填充位
        print("save no pad diag_emb: {} -> {}".format(self.embeddings[0].weight.shape, diag_emb.shape))
        normed_diag_emb = diag_emb / torch.norm(diag_emb, 2, 1, keepdim=True)  # 归一化
        diag_emb = diag_emb.detach().cpu().numpy()  # 张量转化为numpy
        normed_diag_emb = normed_diag_emb.detach().cpu().numpy()

        pro_emb = self.embeddings[1].weight[:-1].detach().cpu().numpy()  # 提取操作嵌入

        #  定义文件保存路径
        diag_file = os.path.join('saved', self.args.model_name, 'diag.tsv')
        normed_diag_file = os.path.join('saved', self.args.model_name, 'diag_normed.tsv')
        pro_file = os.path.join('saved', self.args.model_name, 'pro.tsv')
        med_file = os.path.join('saved', self.args.model_name, 'med.tsv')
        normed_med_file = os.path.join('saved', self.args.model_name, 'med_normed.tsv')

        # 处理并保存 DDI 嵌入
        if self.ddi_embedding != None:
            normed_ddi_embedding = self.ddi_embedding / torch.norm(self.ddi_embedding, 2, 1, keepdim=True)
            normed_ddi_emb = normed_ddi_embedding.detach().cpu().numpy()
            ddi_emb = self.ddi_embedding.detach().cpu().numpy()
            normed_ddi_file = os.path.join('saved', self.args.model_name, 'ddi_normed.tsv')
            ddi_file = os.path.join('saved', self.args.model_name, 'ddi_emb.tsv')
            np.savetxt(normed_ddi_file, normed_ddi_emb, fmt="%.4f", delimiter='\t')
            np.savetxt(ddi_file, ddi_emb, fmt="%.4f", delimiter='\t')

        # 保存所有嵌入文件并提示
        np.savetxt(diag_file, diag_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_diag_file, normed_diag_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(pro_file, pro_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(med_file, med_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_med_file, normed_med_emb, fmt="%.4f", delimiter='\t')

        print("saved embedding files")
        return

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
            item.weight.data[:, -1] = 0.


class main_model_dynamic_basis(main_model):
    """
    Replace the static medication embedding in the matching head with
    a patient-specific dynamic embedding:
        e_dyn(t,m) = e_base(m) + g_(t,m) * sum_k lambda_(t,k) a_(m,k) b_k
    """
    def __init__(self, vocab_size, ddi_adj, encoder, ddi_encoder,
                 emb_dim=256,
                 device=torch.device('cpu:0'),
                 use_aug=True,
                 ehr_adj=None,
                 med2diag=None,
                 med2pro=None,
                 num_dynamic_basis=8,
                 args=None):
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
            args=args
        )
        self.num_dynamic_basis = num_dynamic_basis
        self.dynamic_scorer = DynamicBasisDrugScorer(
            emb_dim=emb_dim,
            num_medications=vocab_size[2],
            num_bases=num_dynamic_basis
        )
        self.dynamic_layernorm = nn.LayerNorm(vocab_size[2])

    def forward(self, input, visit_len):
        diag, pro, labels = input
        query, _ = self._get_query(diag, pro, visit_len)
        normed_query = F.normalize(query, p=2, dim=1)

        base_med_emb, rec_loss = self.molecular_network(
            self.embeddings,
            self.med2diag,
            self.med2pro,
            self.ehr_adj
        )

        if self.ddi_encoder:
            ddi_embedding = self.ddi_encoder(self.x, self.edge_index)
            self.ddi_embedding = ddi_embedding
            base_med_emb = base_med_emb + ddi_embedding

        dynamic_med_emb, _, _ = self.dynamic_scorer(query, base_med_emb)
        normed_dynamic_med_emb = F.normalize(dynamic_med_emb, p=2, dim=2)

        result = torch.einsum('bd,bmd->bm', normed_query, normed_dynamic_med_emb)
        result = self.dynamic_layernorm(result)

        if self.args.ddi:
            neg_pred_prob = F.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)
            tmp_right = neg_pred_prob.unsqueeze(1)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        else:
            batch_neg = 0

        return result, batch_neg, rec_loss

    def save_embedding(self):
        super().save_embedding()
        dynamic_basis_file = os.path.join('saved', self.args.model_name, 'dynamic_basis.tsv')
        drug_coeff_file = os.path.join('saved', self.args.model_name, 'drug_basis_coeff.tsv')

        dynamic_basis = self.dynamic_scorer.dynamic_bases.detach().cpu().numpy()
        drug_coeff = self.dynamic_scorer.drug_basis_coeff.detach().cpu().numpy()

        np.savetxt(dynamic_basis_file, dynamic_basis, fmt="%.4f", delimiter='\t')
        np.savetxt(drug_coeff_file, drug_coeff, fmt="%.4f", delimiter='\t')
        print("saved dynamic basis files")
