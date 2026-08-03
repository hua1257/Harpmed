"""
模块职责：
1. 作为 Carmen 项目的统一训练、验证和测试入口，负责解析命令行参数。
2. 加载数据集、词表、EHR/DDI 图以及分子图特征，并按配置组装不同模型变体。
3. 执行训练循环、评估指标统计、模型保存与断点恢复。
"""

from ast import arg
from lib2to3.pytree import Node
import dill
import numpy as np
import argparse
import csv
import json
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import jaccard_score
from torch.optim import Adam, SGD
import os
import pdb
import torch
import time
from main_models import (
    main_model,
    MolecularGraphNeuralNetwork_record,
    main_model_dynamic_basis,
)
from main_model_dynamic import main_model_dfhd_fusion, main_model_sspnet
from main_baseline import (
    MolecularGraphNeuralNetwork_fagcn,
    MolecularGraphNeuralNetwork_ContextIndependent)
from util import buildMPNN_main, buildMPNN_multihot, llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN, buildMPNN_ecfp, dangerous_pair_num
from util import Metrics, get_ehr_adj
import torch.nn.functional as F
import scipy.sparse as sp
# from torch_sparse import SparseTensor
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data
try:
    from models_gnn import GNN
except ImportError:
    GNN = None
# from main_models import GCNConv

# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# setting
default_model_variant = 'sspnet_fusion'
default_model_name = 'sspnet_fusion_run'
default_resume_path = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--label_soft', action='store_true', default=False, help="soft label")
parser.add_argument('--focal_loss', action='store_true', default=False, help="focal loss")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=default_model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=default_resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
# parser.add_argument('--target_ddi', type=float, default=0.01346, help='target ddi')
parser.add_argument('--target_ddi', type=float, default=0.005, help='target ddi')
parser.add_argument('--threshold', type=float, default=0.5, help='prediction threshold')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--kp', type=float, default=0.07, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension') # 改回64
parser.add_argument('--datadir', type=str, default=r"D:\代码们\Carmen-main\data", help='datadir')
parser.add_argument('--encoder', type=str, default="main", help='molecular encoder type')
parser.add_argument('--model_variant', type=str, default=default_model_variant,
                    choices=[
                        'late_fusion',
                        'dynamic_basis',
                        'dynamic_embedding',
                        'dfhd_fusion',
                        'sspnet_fusion',
                        'sspnet_pcm',
                        'sspnet_param_match',
                    ],
                    help='prediction head variant')
parser.add_argument('--num_dynamic_basis', type=int, default=2,
                    help='number of shared dynamic basis vectors')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--seed', type=int, default=1203, help='random seed')
parser.add_argument('--epoch', type=int, default=400, help='# of epoches')
parser.add_argument('--early_stop', type=int, default=5, help='early stop number')
parser.add_argument('--load', action='store_true', default=False, help='load resume file')
parser.add_argument('--noaug', action='store_true', default=False, help='do not use aug part')
parser.add_argument('--ddi', action='store_true', default=False, help='use ddi')

parser.add_argument('--ddi_encoding', action='store_true', default=False, help='use ddi encoding')
parser.add_argument('--num_layer', type=int, default=1,
                        help='number of GNN message passing layers (default: 5).')
parser.add_argument('--gnn_type', type=str, default="gat")
parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--dropout_ratio', type=float, default=0.4,
                        help='dropout ratio (default: 0)')
parser.add_argument('--p_or_m', type=str, default="minus")
parser.add_argument('--MIMIC', type=int, default=3, help="mimic3 or mimic4")
parser.add_argument('--AnalyzeSeenNew', action='store_true', default=False,
                    help='analyze seen vs new medication predictions from a loaded checkpoint')
parser.add_argument('--analysis_split', type=str, default='test',
                    choices=['train', 'test', 'eval'],
                    help='split used by --AnalyzeSeenNew')
parser.add_argument('--analysis_output', type=str, default='',
                    help='output prefix for --AnalyzeSeenNew; default is saved/<model_name>/seen_new_<split>')


args = parser.parse_args()
print(args)
if not args.resume_path and (args.Test or args.load or args.AnalyzeSeenNew):
    args.resume_path = os.path.join("saved", args.model_name, "best.model")

print("[startup] running variant '{}'.".format(args.model_variant))
print("[startup] model_name='{}'.".format(args.model_name))
print("[startup] MIMIC={}".format(args.MIMIC))

if args.model_variant == 'late_fusion':
    print("[warning] current run is still using the old baseline head 'late_fusion'.")

if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved", args.model_name))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [move_to_device(cur, device) for cur in data]


def split_batch_tensors(data_tensors):
    if len(data_tensors) == 5:
        cur_diag, cur_pro, cur_med_target, cur_med_ml_target, cur_len = data_tensors
        cur_hist_med = None
    elif len(data_tensors) == 6:
        cur_diag, cur_pro, cur_hist_med, cur_med_target, cur_med_ml_target, cur_len = data_tensors
    else:
        raise ValueError("Unexpected batch tensor size: {}".format(len(data_tensors)))
    return cur_diag, cur_pro, cur_hist_med, cur_med_target, cur_med_ml_target, cur_len


def build_model_input(cur_diag, cur_pro, cur_hist_med, labels=None):
    if cur_hist_med is None:
        return (cur_diag, cur_pro, labels)
    return (cur_diag, cur_pro, cur_hist_med, labels)


def eval(model: main_model, data_eval, voc_size, epoch, metric_obj: Metrics):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for data_tensors in model.get_batch(data_eval, 128):
        data_tensors = move_to_device(data_tensors, model.device)
        cur_diag, cur_pro, cur_hist_med, cur_med_target, _, cur_len = split_batch_tensors(data_tensors)
        result, _, _ = model(build_model_input(cur_diag, cur_pro, cur_hist_med, None), cur_len)
        result = F.sigmoid(result).detach().cpu().numpy()
        preds = np.zeros_like(result)
        preds[result>=args.threshold] = 1
        preds[result<args.threshold] = 0
        visit_cnt += cur_med_target.shape[0]
        med_cnt += preds.sum()
        cur_med_target = cur_med_target.detach().cpu().numpy()
        metric_obj.feed_data(cur_med_target, preds, result)

    if args.Test:
        pass
        # model.save_embedding()

    metric_obj.set_data(save=args.Test)
    pred_list = [np.nonzero(row)[0].tolist() for row in metric_obj.pred]
    ddi_rate = ddi_rate_score([pred_list], metric_obj.ddi_adj)

    ops = 'd'
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run(ops=ops)
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def iter_sequential_batches(data, batchsize):
    n_rows = data[0].shape[0]
    for start in range(0, n_rows, batchsize):
        end = min(start + batchsize, n_rows)
        yield [cur_data[start:end] for cur_data in data]


def build_seen_medication_mask(dataset, num_medications):
    seen_rows = []
    for patient in dataset:
        history = set()
        for visit in patient:
            seen = np.zeros(num_medications, dtype=bool)
            if history:
                seen[list(history)] = True
            seen_rows.append(seen)
            history.update(visit[2])
    if not seen_rows:
        return np.zeros((0, num_medications), dtype=bool)
    return np.stack(seen_rows, axis=0)


def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def safe_average_precision(y_true, y_prob):
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    return float(metrics.average_precision_score(y_true, y_prob))


def subset_seen_new_metrics(name, y_gt, y_pred, y_prob, candidate_mask):
    y_gt_bool = y_gt.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    masked_gt = y_gt_bool & candidate_mask
    masked_pred = y_pred_bool & candidate_mask
    true_positive = int((masked_gt & masked_pred).sum())
    target_count = int(masked_gt.sum())
    pred_count = int(masked_pred.sum())
    all_target_count = int(y_gt_bool.sum())
    all_pred_count = int(y_pred_bool.sum())

    jaccard_scores = []
    for row_idx in range(masked_gt.shape[0]):
        gt_idx = set(np.nonzero(masked_gt[row_idx])[0].tolist())
        pred_idx = set(np.nonzero(masked_pred[row_idx])[0].tolist())
        union = gt_idx | pred_idx
        inter = gt_idx & pred_idx
        jaccard_scores.append(0.0 if len(union) == 0 else len(inter) / len(union))

    precision = safe_divide(true_positive, pred_count)
    recall = safe_divide(true_positive, target_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    flat_mask = candidate_mask.reshape(-1)
    flat_gt = y_gt_bool.reshape(-1)[flat_mask].astype(np.int32)
    flat_prob = y_prob.reshape(-1)[flat_mask]

    return {
        'subset': name,
        'visits': int(y_gt.shape[0]),
        'candidate_positions': int(candidate_mask.sum()),
        'target_labels': target_count,
        'predicted_labels': pred_count,
        'true_positive': true_positive,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        'prauc': safe_average_precision(flat_gt, flat_prob),
        'avg_target_per_visit': safe_divide(target_count, y_gt.shape[0]),
        'avg_pred_per_visit': safe_divide(pred_count, y_gt.shape[0]),
        'target_share': safe_divide(target_count, all_target_count),
        'prediction_share': safe_divide(pred_count, all_pred_count),
    }


def analyze_seen_new_medications(model, data_split, split_name, voc_size, args):
    model.eval()
    data_tensors = model.get_inputs(data_split)
    y_gt_full, y_pred_full, y_prob_full = [], [], []

    with torch.no_grad():
        for data_tensors_batch in iter_sequential_batches(data_tensors, 128):
            data_tensors_batch = move_to_device(data_tensors_batch, model.device)
            cur_diag, cur_pro, cur_hist_med, cur_med_target, _, cur_len = split_batch_tensors(data_tensors_batch)
            logits, _, _ = model(build_model_input(cur_diag, cur_pro, cur_hist_med, None), cur_len)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob >= args.threshold).astype(np.int32)
            y_gt_full.append(cur_med_target.detach().cpu().numpy().astype(np.int32))
            y_pred_full.append(pred)
            y_prob_full.append(prob.astype(np.float64))

    y_gt = np.concatenate(y_gt_full, axis=0)
    y_pred = np.concatenate(y_pred_full, axis=0)
    y_prob = np.concatenate(y_prob_full, axis=0)
    seen_mask = build_seen_medication_mask(data_split, voc_size[2])
    if seen_mask.shape != y_gt.shape:
        raise ValueError(
            "seen/new mask shape {} does not match prediction shape {}.".format(
                seen_mask.shape, y_gt.shape
            )
        )

    all_mask = np.ones_like(seen_mask, dtype=bool)
    rows = [
        subset_seen_new_metrics('all', y_gt, y_pred, y_prob, all_mask),
        subset_seen_new_metrics('seen', y_gt, y_pred, y_prob, seen_mask),
        subset_seen_new_metrics('new', y_gt, y_pred, y_prob, ~seen_mask),
    ]

    for row in rows:
        row.update({
            'model_name': args.model_name,
            'model_variant': args.model_variant,
            'split': split_name,
            'threshold': args.threshold,
        })

    output_prefix = args.analysis_output
    if not output_prefix:
        output_prefix = os.path.join('saved', args.model_name, 'seen_new_{}'.format(split_name))
    output_root, output_ext = os.path.splitext(output_prefix)
    if output_ext.lower() in ('.csv', '.json'):
        output_prefix = output_root
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_path = output_prefix + '.csv'
    json_path = output_prefix + '.json'
    fieldnames = [
        'model_name', 'model_variant', 'split', 'threshold',
        'subset', 'visits', 'candidate_positions', 'target_labels',
        'predicted_labels', 'true_positive', 'precision', 'recall', 'f1',
        'jaccard', 'prauc', 'avg_target_per_visit', 'avg_pred_per_visit',
        'target_share', 'prediction_share',
    ]
    with open(csv_path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w') as fout:
        json.dump({'rows': rows}, fout, indent=2)

    print("[seen-new] wrote {}".format(csv_path))
    print("[seen-new] wrote {}".format(json_path))
    for row in rows:
        print(
            "[seen-new] {subset}: F1={f1:.4f} PRAUC={prauc:.4f} "
            "target_share={target_share:.4f} prediction_share={prediction_share:.4f}".format(**row)
        )
    return rows


def main():
    # load data
    if args.MIMIC == 4:
        data_path = os.path.join(args.datadir, 'records_final_4.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
        ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl')
        molecule_path = os.path.join(args.datadir, 'ndc2SMILES_4.pkl')
    else:
        data_path = os.path.join(args.datadir, 'records_final.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final.pkl')
        ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')
        molecule_path = os.path.join(args.datadir, 'ndc2SMILES.pkl')

    if not os.path.exists(molecule_path):
        fallback_molecule_path = os.path.join(args.datadir, 'ndc2SMILES.pkl')
        if os.path.exists(fallback_molecule_path):
            molecule_path = fallback_molecule_path

    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    metric_obj = Metrics(data, med_voc, args)
    use_aug = not args.noaug

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    ehr_adj, med2diag, med2pro, g = None, None, None, None

    # print("ddi_embedding", ddi_embedding)

    def create_matrices():
        Ndiag, Npro, Nmed = voc_size
        med_count_in_train = np.zeros(Nmed)
        med2diag = np.zeros((Nmed, Ndiag))
        med2pro = np.zeros((Nmed, Npro))
        for p in data_train:
            for m in p:
                cur_diag, cur_pro, cur_med = m
                for cm in cur_med:
                    med2diag[cm][cur_diag] += 1
                    med2pro[cm][cur_pro] += 1
                    med_count_in_train[cm] += 1
        med_count_in_train[med_count_in_train==0] = 1

        DTH, PTH = 0, 0
        med2diag = torch.FloatTensor(med2diag)
        med2pro = torch.FloatTensor(med2pro)

        med2diag = med2diag / med_count_in_train.reshape(-1,1)
        med2diag = F.normalize(med2diag, p=1, dim=1)
        med2diag = med2diag.to(torch.float32).to(device)

        med2pro = med2pro / med_count_in_train.reshape(-1,1)
        med2pro = F.normalize(med2pro, p=1, dim=1)
        med2pro = med2pro.to(torch.float32).to(device)


        ehr_adj = get_ehr_adj(data_train, Nmed, no_weight=False)
        ehr_sim = torch.from_numpy(ehr_adj) / (torch.from_numpy(med_count_in_train).reshape(1,-1))
        ehr_sim = ehr_sim.to(torch.float32).to(device)
        ehr_sim = F.normalize(ehr_sim, p=1, dim=1)

        norm = torch.relu(med2pro.sum(1).reshape(-1, 1) - 1) + 1
        med2pro = med2pro / norm
        med2pro = med2pro.to(device)
        ehr_adj = get_ehr_adj(data_train, Nmed, no_weight=False)
        
        ehr_norm = ehr_adj.sum(1).reshape(-1, 1)
        ehr_norm[ehr_norm==0] = 1
        ehr_adj = ehr_adj / ehr_norm
        return ehr_adj, med2diag, med2pro, ehr_sim
 
    if args.encoder == "main":
        # build_fun 主要是为了将药物分子按照一定的格式处理成邻接矩阵的输入
        build_fun = buildMPNN_main
        encoder_cls = MolecularGraphNeuralNetwork_record
        ehr_adj, med2diag, med2pro, ehr_sim = create_matrices()
        # print ("matrices, ", ehr_adj, med2diag, med2pro, ehr_sim)
        g = None

    elif args.encoder == 'fagcn': # vanilla gnn
        # Carmen_{c-}
        build_fun = buildMPNN_ecfp
        encoder_cls = MolecularGraphNeuralNetwork_fagcn

    ddi_encoder = None
    if args.ddi_encoding:
        if GNN is None:
            raise ImportError(
                "--ddi_encoding requires torch-geometric dependencies. "
                "Install torch_geometric, torch_scatter, and torch_sparse first."
            )
        ddi_encoder = GNN(p_or_m=args.p_or_m, device=device, num_layer=args.num_layer, emb_dim=args.dim, gnn_type=args.gnn_type)

    MPNNSet, N_fingerprint, average_projection = build_fun(molecule,
                                                           med_voc.idx2word,
                                                           radius=2,
                                                           device=device)
    print(f"N_fingerprint: {N_fingerprint}")
    MPNN_molecule_Set = list(zip(*MPNNSet))
    encoder = encoder_cls(N_fingerprint, args.dim, 2,
                          device=device,
                          fingers=MPNN_molecule_Set,
                          avg_projection=average_projection,
                          g=g,
                          args=args)
    
    if args.model_variant == 'late_fusion':
        model_cls = main_model
    elif args.model_variant in ('dynamic_basis', 'dynamic_embedding'):
        model_cls = main_model_dynamic_basis
    elif args.model_variant == 'dfhd_fusion':
        model_cls = main_model_dfhd_fusion
    elif args.model_variant in ('sspnet_fusion', 'sspnet_pcm', 'sspnet_param_match'):
        model_cls = main_model_sspnet
    else:
        raise ValueError("Unsupported model_variant: {}".format(args.model_variant))

    if args.model_variant == 'dynamic_embedding':
        print("[model_variant] 'dynamic_embedding' currently uses the dynamic-basis model. "
              "Use 'dfhd_fusion' to run the history-drug fusion model.")

    model_kwargs = dict(
        vocab_size=voc_size,
        ddi_adj=ddi_adj,
        encoder=encoder,
        ddi_encoder=ddi_encoder,
        emb_dim=args.dim,
        device=device,
        use_aug=use_aug,
        ehr_adj=ehr_adj,
        med2diag=med2diag,
        med2pro=med2pro,
        args=args
    )
    if args.model_variant in (
        'dynamic_basis',
        'dynamic_embedding',
        'dfhd_fusion',
        'sspnet_fusion',
        'sspnet_pcm',
        'sspnet_param_match',
    ):
        model_kwargs['num_dynamic_basis'] = args.num_dynamic_basis
    model = model_cls(**model_kwargs)
    print("[model] variant={} class={} params={}".format(
        args.model_variant, model.__class__.__name__, get_n_params(model)
    ))
    model.to(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model.ddi_encoding()
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=1e-5)
    epoch_begin = 0

    if args.Test or args.load or args.AnalyzeSeenNew:
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(args.resume_path))
        checkpoint = torch.load(args.resume_path, map_location=device)
        checkpoint_variant = checkpoint.get('model_variant')
        checkpoint_class = checkpoint.get('model_class')
        if checkpoint_variant is not None and checkpoint_variant != args.model_variant:
            raise ValueError(
                "Checkpoint variant '{}' does not match current variant '{}'.".format(
                    checkpoint_variant, args.model_variant
                )
            )
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch'] + 1
        print(
            "Load {} finish... checkpoint_variant={} checkpoint_class={}".format(
                args.resume_path, checkpoint_variant, checkpoint_class
            )
        )

    if args.AnalyzeSeenNew:
        split_map = {
            'train': data_train,
            'test': data_test,
            'eval': data_eval,
        }
        analyze_seen_new_medications(
            model,
            split_map[args.analysis_split],
            args.analysis_split,
            voc_size,
            args,
        )
        return

    if args.Test:
    # if True:
        model.to(device=device)
        tic = time.time()
        data_test_tensors = model.get_inputs(data_test)

        result = []
        metrics = eval(model, data_test_tensors, voc_size, 0, metric_obj) # ddi_adj删了
        model.save_embedding()
        result.append(list(metrics))
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\\pm$ {:.4f} & ".format(m, s)

        print (outstring)

        print ('test time: {}'.format(time.time() - tic))
        return 

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0


    # ddi_embedding = ddi_encoding(ddi_adj, args.dim)
    # ddi_embedding = ddi_embedding.to(device)

    # 将list数据转化为tensor
    use_ddi_loss = args.ddi

    data_train_tensors = model.get_inputs(data_train)
    data_eval_tensors = model.get_inputs(data_eval)
    EPOCH = args.epoch
    for epoch in range(epoch_begin, EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        
        model.train()
        step = 0
        trian_visit_num = sum([len(p) for p in data_train])
        for cur_batch in model.get_batch(data_train_tensors, 16):
            cur_diag, cur_pro, cur_hist_med, cur_med_bce_target, cur_med_ml_target, cur_len = split_batch_tensors(cur_batch)
            result, loss_ddi, _ = model(build_model_input(cur_diag, cur_pro, cur_hist_med, cur_med_bce_target), cur_len)
            # NOTE: batch of these loss function
            if args.label_soft:
                cur_med_bce_target = torch.matmul(cur_med_bce_target, ehr_sim) + cur_med_bce_target
            loss_bce = F.binary_cross_entropy_with_logits(result, cur_med_bce_target)
            if args.focal_loss:
                alpha = 0.25
                gamma = 1
                BCE_loss = F.binary_cross_entropy_with_logits(result, cur_med_bce_target, reduce=False)
                pt = torch.exp(-BCE_loss)
                F_loss = alpha * (1-pt)**gamma * BCE_loss
                loss_bce = torch.mean(F_loss)*10
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), cur_med_ml_target)

            # NOTE: value range of loss_ddi 
            loss = 0.95 * loss_bce + 0.05 * loss_multi  #  + loss_ddi
            if use_ddi_loss:
                pred_binary = (torch.sigmoid(result).detach() >= args.threshold).long().cpu().numpy()
                pred_label_list = [np.where(row == 1)[0].tolist() for row in pred_binary]
                cur_ddi_rate = ddi_rate_score([pred_label_list], ddi_adj)
                if cur_ddi_rate > args.target_ddi:   # 如果当前ddi率大于目标ddi率，则加入ddi loss
                    beta = max(0.0, min(1.0, 1 + (args.target_ddi - cur_ddi_rate) / 0.05))
                    loss = beta * loss + (1 - beta) * loss_ddi
 
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()

            step += cur_diag.shape[0]
            llprint('\rtraining step: {} / {}'.format(step, trian_visit_num))

        print ()
        tic2 = time.time() 
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval_tensors, voc_size, epoch, metric_obj)
        print ('training time: {}, test time: {}'.format(tic2 - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        savefile = os.path.join('saved', args.model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate))
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "model_variant": args.model_variant,
                    "model_class": model.__class__.__name__,
                    "MIMIC": args.MIMIC}, open(savefile, 'wb'))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            savefile = os.path.join('saved', args.model_name, 'best.model')
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "model_variant": args.model_variant,
                        "model_class": model.__class__.__name__,
                        "MIMIC": args.MIMIC}, open(savefile, 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

        if epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()
