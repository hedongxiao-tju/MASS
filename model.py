import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k - 1):
            x = self.activation(self.conv[i](x, edge_index))
        x = self.conv[self.k - 1](x, edge_index)
        return x


class MASSModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_codebooks, codebook_dim,
                 neg_par=1.0, pos_par=0.001, ali_par=1, consist_par=1,
                 temperature=0.1, threshold=0.5,
                 gnn_layers=2, activation=F.relu,
                 fea_stru_par=0.1,
                 walk_length=10, num_walks=20, window_size=5):
        super().__init__()
        self.num_labels = num_labels
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.ali_par = ali_par
        self.consist_par = consist_par
        self.neg_par = neg_par
        self.pos_par = pos_par
        self.threshold = threshold
        self.fea_stru_par = fea_stru_par
        self.hidden_dim = hidden_dim

        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size

        self.encoder = Encoder(
            in_channels=input_dim,
            out_channels=hidden_dim,
            activation=activation,
            base_model=GCNConv,
            k=gnn_layers
        )

        self.deepwalk_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.deepwalk_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.codebooks = nn.Parameter(
            torch.randn(num_labels, num_codebooks, codebook_dim)
        )

        self.label_projections = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_dim) for _ in range(num_labels)
        ])

        self.codebook_attention = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim // 2),
            nn.ReLU(),
            nn.Linear(codebook_dim // 2, 1)
        )

        self.deepwalk_embeddings = None
        self.deepwalk_embeddings_tensor = None

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)

        if self.deepwalk_embeddings_tensor is not None:
            za = self.deepwalk_mlp(self.deepwalk_embeddings_tensor)
            fused_z = torch.cat([z, za], dim=1)
        else:
            fused_z = z

        return fused_z

    def dgl_to_nx(self, graph):
        g_nx = nx.Graph()
        src, dst = graph.edges()
        edges = list(zip(src.cpu().numpy(), dst.cpu().numpy()))
        g_nx.add_edges_from(edges)
        num_nodes = graph.num_nodes()
        for node in range(num_nodes):
            if node not in g_nx:
                g_nx.add_node(node)
        return g_nx

    def random_walk(self, start_node, nx_graph):
        walk = [start_node]
        current = start_node
        for _ in range(self.walk_length - 1):
            neighbors = list(nx_graph.neighbors(current))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current = next_node
        return [str(node) for node in walk]

    def generate_walks(self, nx_graph):
        walks = []
        nodes = list(nx_graph.nodes())
        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                try:
                    walk = self.random_walk(node, nx_graph)
                    if len(walk) > 1:
                        walks.append(walk)
                except Exception:
                    continue
        return walks

    def generate_deepwalk_embeddings(self, graph):
        print("Generating DeepWalk embeddings...")
        nx_graph = self.dgl_to_nx(graph)
        walks = self.generate_walks(nx_graph)

        if not walks:
            print("Warning: No walks generated! Using random embeddings.")
            num_nodes = graph.num_nodes()
            self.deepwalk_embeddings = np.random.randn(num_nodes, self.hidden_dim)
            return self.deepwalk_embeddings

        print(f"Training Word2Vec on {len(walks)} walks...")
        try:
            model = Word2Vec(
                sentences=walks,
                vector_size=self.hidden_dim,
                window=self.window_size,
                min_count=0,
                sg=1,
                hs=0,
                negative=5,
                workers=4,
                epochs=5,
                seed=2025
            )
        except Exception as e:
            print(f"Error training Word2Vec: {e}")
            num_nodes = graph.num_nodes()
            self.deepwalk_embeddings = np.random.randn(num_nodes, self.hidden_dim)
            return self.deepwalk_embeddings

        num_nodes = graph.num_nodes()
        self.deepwalk_embeddings = np.zeros((num_nodes, self.hidden_dim))

        for node_idx in range(num_nodes):
            try:
                self.deepwalk_embeddings[node_idx] = model.wv[str(node_idx)]
            except KeyError:
                self.deepwalk_embeddings[node_idx] = np.random.randn(self.hidden_dim)

        print(f"DeepWalk embeddings generated! Dimension: {self.hidden_dim}")
        return self.deepwalk_embeddings

    def set_deepwalk_embeddings(self, device):
        if self.deepwalk_embeddings is not None:
            self.deepwalk_embeddings_tensor = torch.tensor(
                self.deepwalk_embeddings, dtype=torch.float32
            ).to(device)
        else:
            raise ValueError("DeepWalk embeddings not generated. Call generate_deepwalk_embeddings first.")

    def get_gnn_representation(self, x, edge_index):
        return self.encoder(x, edge_index)

    def get_projected_deepwalk_representation(self):
        if self.deepwalk_embeddings_tensor is None:
            return None
        za = self.deepwalk_mlp(self.deepwalk_embeddings_tensor)
        return self.deepwalk_projector(za)

    def get_node_label_representations(self, z):
        node_label_reps = []
        for proj in self.label_projections:
            label_rep = proj(z)
            node_label_reps.append(label_rep)
        return torch.stack(node_label_reps, dim=1)

    def get_pseudo_labels(self):
        C, K, n = self.codebooks.shape
        codebook_flat = self.codebooks.view(-1, n)
        attention_weights = self.codebook_attention(codebook_flat).view(C, K)
        attention_weights = F.softmax(attention_weights, dim=-1)
        pseudo_labels = torch.bmm(
            attention_weights.unsqueeze(1),
            self.codebooks
        ).squeeze(1)
        return pseudo_labels

    def get_similarity_matrix(self, node_label_reps, pseudo_labels):
        batch_size = node_label_reps.size(0)
        return torch.bmm(
            node_label_reps,
            pseudo_labels.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2)
        )

    def fea_stru_alignment_loss(self, z, ha):
        z_norm = F.normalize(z, dim=1)
        ha_norm = F.normalize(ha, dim=1)
        similarity = torch.mm(z_norm, ha_norm.t())
        positive_loss = (1 - similarity.diag()).mean()
        mask = torch.eye(z.size(0), device=z.device).bool()
        negative_similarity = similarity.masked_fill(mask, -float('inf'))
        negative_loss = F.relu(negative_similarity).mean()
        return positive_loss + negative_loss

    def ssl_loss_sep(self, z):
        z = F.normalize(z, dim=1)
        variance = z.var(dim=0)
        return 1 - variance.mean()

    def ssl_loss_pos_neigh(self, z, edge_index):
        row, col = edge_index
        non_self_mask = row != col
        row = row[non_self_mask]
        col = col[non_self_mask]
        return (z[row] - z[col]).pow(2).mean()

    def label_alignment_loss(self, similarity_matrix, node_label_reps, pseudo_labels):
        batch_size, C, _ = similarity_matrix.shape
        label_scores = similarity_matrix.diagonal(dim1=1, dim2=2)
        label_probs = torch.sigmoid(label_scores)

        total_high_loss = 0
        total_low_loss = 0
        high_count = 0
        low_count = 0

        for i in range(C):
            node_repr_i = node_label_reps[:, i, :]
            pseudo_label_i = pseudo_labels[i]
            similarity = F.cosine_similarity(node_repr_i, pseudo_label_i.unsqueeze(0))
            prob_i = label_probs[:, i]

            high_mask = prob_i > self.threshold
            if high_mask.any():
                total_high_loss += (1 - similarity[high_mask]).sum()
                high_count += high_mask.sum()

            low_mask = prob_i <= self.threshold
            if low_mask.any():
                total_low_loss += similarity[low_mask].sum()
                low_count += low_mask.sum()

        high_loss = total_high_loss / max(high_count, 1)
        low_loss = total_low_loss / max(low_count, 1)
        return high_loss + low_loss + 1

    def consistency_loss(self, similarity_matrix):
        label_scores = similarity_matrix.diagonal(dim1=1, dim2=2)
        pseudo_probs = F.softmax(label_scores / self.temperature, dim=1)
        sharpened_probs = F.softmax(label_scores / (self.temperature * 0.5), dim=1)
        return 1 + F.kl_div(
            pseudo_probs,
            sharpened_probs.detach(),
            reduction='batchmean'
        )

    def total_loss(self, x, edge_index):
        z = self.get_gnn_representation(x, edge_index)
        ha = self.get_projected_deepwalk_representation()

        if ha is not None:
            fea_stru_align_loss = self.fea_stru_alignment_loss(z, ha)
        else:
            fea_stru_align_loss = torch.tensor(0.0).to(z.device)

        node_label_reps = self.get_node_label_representations(z)
        pseudo_labels = self.get_pseudo_labels()
        similarity_matrix = self.get_similarity_matrix(node_label_reps, pseudo_labels)

        label_alignment_loss = self.label_alignment_loss(similarity_matrix, node_label_reps, pseudo_labels)
        consistency_loss = self.consistency_loss(similarity_matrix)
        ssl_sep_loss = self.ssl_loss_sep(z)
        ssl_pos_loss = self.ssl_loss_pos_neigh(z, edge_index)

        total_loss = (self.ali_par * label_alignment_loss +
                      self.consist_par * consistency_loss +
                      self.neg_par * ssl_sep_loss +
                      self.pos_par * ssl_pos_loss +
                      self.fea_stru_par * fea_stru_align_loss)

        return total_loss.mean(), {
            'total_loss': total_loss.item(),
            'fea_stru_align_loss': fea_stru_align_loss.item() if ha is not None else 0.0,
            'label_alignment_loss': label_alignment_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'ssl_sep_loss': ssl_sep_loss.item(),
            'ssl_pos_loss': ssl_pos_loss.item()
        }

    def get_fused_representations(self, x, edge_index):
        return self.forward(x, edge_index)