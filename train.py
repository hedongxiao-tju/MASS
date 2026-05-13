import os
import numpy as np
import argparse
import torch

from utils import load_humloc, load_pcg, load_blogcatalog, load_ppi
from utils import graph_process, create_split, build_mask
from utils import set_seed, get_metrics, train_downstream_classifier

from model import MASSModel


def main():
    argparser = argparse.ArgumentParser("MASS training",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--use_gpu", action="store_true")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--seed", type=int, default=2025)
    argparser.add_argument("--gnn_layers", type=int, default=2)
    argparser.add_argument("--num_epochs", type=int, default=200)
    argparser.add_argument("--train_rate", type=float, default=0.1)
    argparser.add_argument("--val_rate", type=float, default=0.1)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--data_name", type=str, default="pcg")
    argparser.add_argument("--runs", type=int, default=3)
    argparser.add_argument("--learning_rate", type=float, default=0.0005)
    argparser.add_argument("--hidden_dim", type=int, default=128)
    argparser.add_argument("--codebooks_num", type=int, default=15)
    argparser.add_argument("--codebook_dim", type=int, default=32)
    argparser.add_argument("--temperature", type=float, default=0.3)
    argparser.add_argument("--thr", type=float, default=0.8)
    argparser.add_argument("--sep_par", type=float, default=1)   # 1
    argparser.add_argument("--pos_par", type=float, default=0.3)  # 0.3
    argparser.add_argument("--ali_par", type=float, default=1000)  # 1000
    argparser.add_argument("--consist_par", type=float, default=700)  # 700
    argparser.add_argument("--fea_stru_par", type=float, default=10)  # 10
    argparser.add_argument("--walk_length", type=int, default=10)
    argparser.add_argument("--num_walks", type=int, default=20)
    argparser.add_argument("--window_size", type=int, default=5)

    args = argparser.parse_args()

    if args.use_gpu:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    test_hamming_loss, test_ranking_loss, test_ma_f1, test_mi_f1 = [], [], [], []
    test_macro_ap, test_micro_ap = [], []
    test_macro_auc, test_micro_auc = [], []
    test_lrap = []

    for run in range(1, args.runs + 1):
        if args.runs == 1:
            set_seed(args.seed)
        else:
            set_seed(run)

        if args.data_name == 'humloc':
            graph = load_humloc()
        elif args.data_name == 'pcg':
            graph = load_pcg()
        elif args.data_name == 'blogcatalog':
            graph = load_blogcatalog()
        elif args.data_name == 'ppi':
            graph = load_ppi()
        else:
            raise Exception("None of the available dataset is selected!")

        graph = graph_process(graph, to_bidirect=False).to(device)
        feat_dim = graph.ndata['feat'].shape[1]
        n_classes = graph.ndata['label'].shape[1]
        train_idx, val_idx, test_idx = create_split(graph, args.train_rate, args.val_rate)
        graph = build_mask(graph, train_idx, val_idx, test_idx, device=device)
        print(f"feature dimension: {feat_dim}, total classes: {n_classes}")

        model = MASSModel(
            feat_dim,
            args.hidden_dim,
            n_classes,
            args.codebooks_num,
            args.codebook_dim,
            args.sep_par,
            args.pos_par,
            args.ali_par,
            args.consist_par,
            args.temperature,
            args.thr,
            args.gnn_layers,
            torch.nn.functional.relu,
            fea_stru_par=args.fea_stru_par,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            window_size=args.window_size
        ).to(device)

        print("=== Generating DeepWalk Embeddings ===")
        model.generate_deepwalk_embeddings(graph.cpu())
        model.set_deepwalk_embeddings(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        edge_index = torch.stack(graph.edges()).to(device)
        x = graph.ndata['feat'].to(device)

        print("=== Start Training ===")
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            loss, loss_dict = model.total_loss(x, edge_index)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f'(T) | Epoch={epoch:03d}, Total Loss={loss_dict["total_loss"]:.4f}, '
                      f'Feature Structure Alignment Loss={loss_dict["fea_stru_align_loss"]:.4f}, '
                      f'Alignment Loss={loss_dict["label_alignment_loss"]:.4f}')

        print("\n=== Downstream Classifier Training ===")
        model.eval()
        with torch.no_grad():
            node_representations = model.get_fused_representations(x, edge_index).detach()

        train_mask = graph.ndata['train_mask'].bool()
        val_mask = graph.ndata['val_mask'].bool()
        test_mask = graph.ndata['test_mask'].bool()
        labels = graph.ndata['label'].to(device).float()

        classifier = train_downstream_classifier(
            node_representations,
            labels,
            train_mask,
            val_mask,
            test_mask,
            input_dim=args.hidden_dim * 2,  # z + za concatenated
            num_classes=n_classes,
            hidden_dim=args.hidden_dim
        )

        print("\n=== Testing Phase ===")
        classifier.eval()
        with torch.no_grad():
            test_repr = node_representations[test_mask].detach()
            test_outputs = classifier(test_repr)
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            test_preds = (test_probs > 0.5).astype(float)
            test_true = labels[test_mask].cpu().numpy()

            hamming_loss, ranking_loss, macro_f1, micro_f1, macro_ap, micro_ap, macro_auc, micro_auc, lrap = get_metrics(
                test_true, test_preds, test_probs
            )

            test_hamming_loss.append(hamming_loss)
            test_ranking_loss.append(ranking_loss)
            test_ma_f1.append(macro_f1)
            test_mi_f1.append(micro_f1)
            test_macro_ap.append(macro_ap)
            test_micro_ap.append(micro_ap)
            test_macro_auc.append(macro_auc)
            test_micro_auc.append(micro_auc)
            test_lrap.append(lrap)

            print("\n=== Final Test Results ===")
            print(f"Hamming Loss: {hamming_loss:.4f}")
            print(f"Ranking Loss: {ranking_loss:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Micro F1: {micro_f1:.4f}")
            print(f"Macro AP: {macro_ap:.4f}")
            print(f"Micro AP: {micro_ap:.4f}")
            print(f"Macro AUC: {macro_auc:.4f}")
            print(f"Micro AUC: {micro_auc:.4f}")
            print(f"Label Ranking AP: {lrap:.4f}")

    if args.runs > 1:
        hamming_loss_mean, hamming_loss_std = np.mean(np.array(test_hamming_loss)), np.std(np.array(test_hamming_loss))
        ranking_loss_mean, ranking_loss_std = np.mean(np.array(test_ranking_loss)), np.std(np.array(test_ranking_loss))
        ma_f1_mean, ma_f1_std = np.mean(np.array(test_ma_f1)), np.std(np.array(test_ma_f1))
        mi_f1_mean, mi_f1_std = np.mean(np.array(test_mi_f1)), np.std(np.array(test_mi_f1))
        macro_ap_mean, macro_ap_std = np.mean(np.array(test_macro_ap)), np.std(np.array(test_macro_ap))
        micro_ap_mean, micro_ap_std = np.mean(np.array(test_micro_ap)), np.std(np.array(test_micro_ap))
        macro_auc_mean, macro_auc_std = np.mean(np.array(test_macro_auc)), np.std(np.array(test_macro_auc))
        micro_auc_mean, micro_auc_std = np.mean(np.array(test_micro_auc)), np.std(np.array(test_micro_auc))
        lrap_mean, lrap_std = np.mean(np.array(test_lrap)), np.std(np.array(test_lrap))

        print(f"\n{'=' * 60}")
        print(f"Average Results over {args.runs} runs:")
        print(f"{'=' * 60}")
        print(f"Final test hamming loss performance: {hamming_loss_mean:.4f} ± {hamming_loss_std:.4f}")
        print(f"Final test ranking loss performance: {ranking_loss_mean:.4f} ± {ranking_loss_std:.4f}")
        print(f"Final test macro-F1 performance: {ma_f1_mean:.4f} ± {ma_f1_std:.4f}")
        print(f"Final test micro-F1 performance: {mi_f1_mean:.4f} ± {mi_f1_std:.4f}")
        print(f"Final test macro-AP performance: {macro_ap_mean:.4f} ± {macro_ap_std:.4f}")
        print(f"Final test micro-AP performance: {micro_ap_mean:.4f} ± {micro_ap_std:.4f}")
        print(f"Final test macro-AUC performance: {macro_auc_mean:.4f} ± {macro_auc_std:.4f}")
        print(f"Final test micro-AUC performance: {micro_auc_mean:.4f} ± {micro_auc_std:.4f}")
        print(f"Final test LRAP performance: {lrap_mean:.4f} ± {lrap_std:.4f}")


if __name__ == "__main__":
    main()
