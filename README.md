# SANSA 推荐模型复现与性能分析

复现 RecSys 2023 论文《Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering》(SANSA)，并在 ReChorus 推荐框架中与 BPRMF、BUIR 进行对比实验。

## 项目背景

SANSA 是对 EASE 的稀疏化、因子化改进，通过稀疏近似 Cholesky 分解与近似逆技术，将稠密权重矩阵分解为两个稀疏矩阵，提升大规模物品集下的可扩展性。

本仓库主要包含 SANSA 的 Embedding 化重构实现。原论文实现基于稀疏矩阵运算，本工作将其重构为基于 Embedding 查表与聚合的形式，使其训练与推理复杂度与 BPRMF 等 Embedding 模型对齐，便于在 ReChorus 框架内进行快速实验对比。

## 我的工作

- ReChorus 框架学习与适配
- SANSA 模型 Embedding 化重构实现与调试
- 实验运行、结果记录与性能分析
- 实验报告“实验及分析”部分撰写

## 实验结果

在 Amazon Grocery_and_Gourmet_Food 数据集上对比 BPRMF、BUIR 与 SANSA（Ours），评估 Top-K 推荐指标。

| 模型 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| BPRMF | 0.2897 | 0.1981 | 0.4108 | 0.2372 | 0.5199 | 0.2648 |
| BUIR | 0.2993 | 0.1963 | 0.4149 | 0.2339 | 0.5295 | 0.2627 |
| SANSA (Ours) | 0.2046 | 0.1303 | 0.3143 | 0.1658 | 0.4388 | 0.1972 |

**结论**：SANSA 在强个性化食品推荐场景下排序性能不及 BPRMF、BUIR，原因包括任务匹配度差异、数据特性适配、信息利用方式不同以及 Embedding 化重构的近似性。该结果说明模型设计需与业务场景、数据特性紧密结合。

## 运行说明

1. 将本仓库中的 `SANSA.py` 放入 ReChorus 框架的 general 模型区。
2. 确保已安装 Python、PyTorch 及 ReChorus 依赖。
3. 在 ReChorus 项目根目录运行：

```bash
python main.py --model_name SANSA
