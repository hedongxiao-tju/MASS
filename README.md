
# Multi-Semantic Aware Self-Supervised Learning for Multi-Label Node Classification
This repository is for the paper "Multi-Semantic Aware Self-Supervised Learning for Multi-Label Node Classification".
![](/MASS.png)

# 1.  Environment Configurations
```
  python == 3.9.18
  torch == 2.0.1
  torch_geometric == 2.5.2
  dgl == 1.1.2
  scikit-learn == 1.4.1
  networkx == 3.1
  numpy == 1.26.4
  pandas == 2.2.1
  gensim == 4.4.0
```

# 2. How to use MASS
You can reproduce the experiments easily by running the following command.
```
 # humloc
 python train.py --data_name humloc --hidden_dim 256 --neg_par 1 --pos_par 0.3 --ali_par 1000 --consist_par 700 --fea_stru_par 10 

 # pcg
 python train.py --data_name pcg --hidden_dim 256 --neg_par 1 --pos_par 0.3 --ali_par 1000 --consist_par 700 --fea_stru_par 10 

 # blogcatalog
 python train.py --data_name blogcatalog --hidden_dim 256 --neg_par 1 --pos_par 0.8 --ali_par 4 --consist_par 3 --fea_stru_par 20 

 # ppi
 python train.py --data_name ppi --hidden_dim 128 --neg_par 1 --pos_par 0.3 --ali_par 1 --consist_par 1 --fea_stru_par 50
``` 

The implementation of `utils.py` is adapted from [CorGCN](https://github.com/YuanchenBei/CorGCN) under the MIT License.

# Cite
Please cite our paper if MASS helps your work.
```
@article{MASS,
  title={Multi-Semantic Aware Self-Supervised Learning for Multi-Label Node Classification},
  author={Jiayu Zhang, Jitao Zhao, Dongxiao He, Cuiying Huo, Zhiyong Feng,
  journal={IJCAI},
  year={2026}
}
```

