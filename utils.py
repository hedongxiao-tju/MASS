import os
import torch
import dgl
import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import random
import sklearn.metrics as metric


def load_blogcatalog():
    total_nodes = pd.read_csv('./dataset/blogcatalog/data/nodes.csv', header=None).max()[0]
    total_classes = pd.read_csv('./dataset/blogcatalog/data/groups.csv', header=None).max()[0]
    node_belong = pd.read_csv('./dataset/blogcatalog/data/group-edges.csv', header=None, delimiter=',',
                              names=['node', 'group'])
    label_matrix = np.zeros((total_nodes, total_classes), dtype=int)
    for index, row in node_belong.iterrows():
        i = row[0] - 1
        c = row[1] - 1
        label_matrix[i, c] = 1
    blog_edge = pd.read_csv('./dataset/blogcatalog/data/edges.csv', delimiter=',', header=None, names=['src', 'dst'])
    blog_edge = blog_edge.apply(subtract_one)
    src = blog_edge['src'].to_numpy()
    dst = blog_edge['dst'].to_numpy()
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.rand(total_nodes, 100)
    g.ndata['label'] = torch.LongTensor(label_matrix)
    return g


def load_pcg():
    label_data_df = pd.read_csv('./dataset/pcg/labels.csv', header=None, delimiter=',')
    label_data = label_data_df.values
    feat_data_df = pd.read_csv('./dataset/pcg/features.csv', header=None, delimiter=',')
    feat_data = feat_data_df.values
    pcg_edge = pd.read_csv('./dataset/pcg/edges_undir.csv', header=None, delimiter=',', names=['src', 'dst'])
    src = pcg_edge['src'].to_numpy()
    dst = pcg_edge['dst'].to_numpy()
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(feat_data)
    g.ndata['label'] = torch.LongTensor(label_data)
    return g


def load_humloc():
    label_data_df = pd.read_csv('./dataset/humloc/labels.csv', header=None, delimiter=',')
    label_data = label_data_df.values
    feat_data_df = pd.read_csv('./dataset/humloc/features.csv', header=None, delimiter=',')
    feat_data = feat_data_df.values
    humloc_edge = pd.read_csv('./dataset/humloc/edge_list.csv', header=0, delimiter=',')
    src = humloc_edge['prot1'].to_numpy().astype('int64')
    dst = humloc_edge['prot2'].to_numpy().astype('int64')
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(feat_data)
    g.ndata['label'] = torch.LongTensor(label_data)
    return g


def load_ppi():
    feats = np.load('./dataset/ppi/feats.npy')
    class_map = json.load(open('./dataset/ppi/class_map.json', 'r'))
    label_mat = np.zeros((feats.shape[0], len(list(class_map["0"]))), dtype=int)
    for k, v in class_map.items():
        label_mat[int(k)] = list(v)
    adj_full = sp.load_npz('./dataset/ppi/adj_full.npz')
    g = dgl.from_scipy(adj_full)
    g.ndata['feat'] = torch.FloatTensor(feats)
    g.ndata['label'] = torch.LongTensor(label_mat)
    return g


def get_labeled_idx(g):
    labeled_data = (g.ndata['label'].sum(dim=-1) >= 1)
    labeled_idx = torch.nonzero(labeled_data).squeeze(-1)
    return labeled_idx


def subtract_one(x):
    return x - 1


def create_split(g, train_rate, val_rate):
    labeled_nodes = get_labeled_idx(g)
    labels = g.ndata['label'][labeled_nodes].cpu().numpy()
    num_labeled_nodes = labeled_nodes.shape[0]
    n_labels = labels.shape[1]

    num_train = int(num_labeled_nodes * train_rate)
    num_val = int(num_labeled_nodes * val_rate)
    num_test = num_labeled_nodes - num_train - num_val

    best_split = None
    best_score = -float('inf')

    for i in range(1000):
        indices = torch.randperm(num_labeled_nodes)
        shuffled = labeled_nodes[indices]

        train = shuffled[:num_train]
        val = shuffled[num_train:num_train + num_val]
        test = shuffled[num_train + num_val:]

        train_labels = g.ndata['label'][train].sum(0) > 0
        val_labels = g.ndata['label'][val].sum(0) > 0
        test_labels = g.ndata['label'][test].sum(0) > 0

        score = (train_labels.sum() + val_labels.sum() + test_labels.sum()).item()

        if score == n_labels * 3:
            return train, val, test

        if score > best_score:
            best_score = score
            best_split = (train, val, test)

    return best_split


def build_mask(graph, train_idx, val_idx, test_idx, device='cpu'):
    graph = graph.to(device)

    train_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    train_mask[train_idx] = True

    val_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    val_mask[val_idx] = True

    test_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    test_mask[test_idx] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph


def graph_process(graph, to_bidirect=True, self_loop=True):
    if to_bidirect:
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()

    return graph


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_metrics(true_results, pred_label_results, pred_prob_results):
    hamming_loss = metric.hamming_loss(true_results, pred_label_results)
    ranking_loss = metric.label_ranking_loss(true_results, pred_prob_results)
    macro_f1 = metric.f1_score(true_results, pred_label_results, average='macro')
    micro_f1 = metric.f1_score(true_results, pred_label_results, average='micro')
    macro_ap = metric.average_precision_score(true_results, pred_prob_results, average='macro')
    micro_ap = metric.average_precision_score(true_results, pred_prob_results, average='micro')
    macro_auc = metric.roc_auc_score(true_results, pred_prob_results, average='macro')
    micro_auc = metric.roc_auc_score(true_results, pred_prob_results, average='micro')
    lrap = metric.label_ranking_average_precision_score(true_results, pred_prob_results)
    return hamming_loss, ranking_loss, macro_f1, micro_f1, macro_ap, micro_ap, macro_auc, micro_auc, lrap


def train_downstream_classifier(representations, labels, train_mask, val_mask, test_mask,
                                input_dim, num_classes, hidden_dim=128):
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(hidden_dim, num_classes)
    ).to(representations.device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_state = None
    labels = labels.float()

    for epoch in range(200):
        classifier.train()
        optimizer.zero_grad()

        with torch.no_grad():
            train_repr = representations[train_mask].detach()

        train_outputs = classifier(train_repr)
        train_loss = criterion(train_outputs, labels[train_mask])
        train_loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_repr = representations[val_mask].detach()
            val_outputs = classifier(val_repr)
            val_loss = criterion(val_outputs, labels[val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = classifier.state_dict().copy()

        if epoch % 50 == 0:
            print(f'Downstream Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)

    return classifier
