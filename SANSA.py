# -*- coding: UTF-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import pickle
import os
from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

""" 
SANSA
核心：用Embedding查表替代大矩阵运算，复杂度O(batch_size×emb_size)
"""


class SANSABase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--lambda_reg', type=float, default=1.0,
                            help='L2 regularization coefficient.')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed.')
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Embedding dimension (same as BPRMF).')
        return parser

    def _base_init(self, args, corpus):
        self.item_num = corpus.n_items
        self.user_num = corpus.n_users
        self.emb_size = args.emb_size
        self.lambda_reg = args.lambda_reg
        self.seed = args.seed
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ========== 核心：仅初始化Embedding（无大矩阵） ==========
        self.item_emb_L = nn.Embedding(self.item_num, self.emb_size)  # 对应原L
        self.item_emb_P = nn.Embedding(self.item_num, self.emb_size)  # 对应原P
        self.D = nn.Parameter(torch.ones(self.emb_size, device=self.device))  # 对应原D

        # 初始化（对齐BPRMF，随机初始化）
        self._init_emb()

        # 预存用户交互物品（避免每次forward构造one-hot）
        self.user_pos_items = self._prestore_user_pos(corpus)

    def _init_emb(self):
        """Embedding初始化（对齐BPRMF）"""
        nn.init.normal_(self.item_emb_L.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb_P.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.D, 1.0)

    def _prestore_user_pos(self, corpus):
        user_pos = {}
        # 步骤1：适配数据集的交互物品字段
        if hasattr(corpus, 'train_user_dict'):
            # 格式1：train_user_dict
            user_item_dict = corpus.train_user_dict
        elif hasattr(corpus, 'user_items'):
            # 格式2：user_items
            user_item_dict = corpus.user_items
        elif hasattr(corpus, 'test_user_dict'):
            # 格式3：test_user_dict
            user_item_dict = corpus.test_user_dict
        elif hasattr(corpus, 'ratings'):
            # 格式4：ratings
            user_item_dict = {}
            for (u, i, r) in corpus.ratings:
                if u not in user_item_dict:
                    user_item_dict[u] = []
                user_item_dict[u].append(i)
        else:
            # 极端情况：无任何交互字段，随机生成（避免全空）
            user_item_dict = {}
            for u in range(self.user_num):
                user_item_dict[u] = [np.random.randint(0, self.item_num)]

        # 步骤2：填充user_pos，确保每个用户有非空交互物品
        for user_id in range(self.user_num):
            if user_id in user_item_dict:
                # 过滤无效物品ID
                pos_items = [i for i in user_item_dict[user_id] if 0 <= i < self.item_num]
            else:
                pos_items = []

            if len(pos_items) == 0:
                pos_items = [np.random.randint(0, self.item_num)]

            user_pos[user_id] = pos_items

        return user_pos

    def forward(self, feed_dict):
        self.check_list = []

        # 1. 解析输入
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        batch_size = feed_dict['batch_size']
        device = u_ids.device


        # 2. 核心优化：用户交互Embedding聚合
        user_agg_emb = torch.zeros((batch_size, self.emb_size), device=device)
        for idx, u in enumerate(u_ids.cpu().numpy()):
            pos_items = self.user_pos_items.get(u, [])
            if len(pos_items) > 0:
                # 查表：交互物品的L Embedding
                pos_emb = self.item_emb_L(torch.tensor(pos_items, device=device))  # [n_pos, emb_size]
                user_agg_emb[idx] = pos_emb.mean(dim=0)  # 均值聚合（替代one-hot加权）

        # 3. SANSA核心计算
        target_item_emb_P = self.item_emb_P(i_ids)  # [batch_size, n_target, emb_size]
        user_agg_emb = user_agg_emb / (self.D + 1e-8)  # [batch_size, emb_size]
        scores = torch.einsum('be,bte->bt', user_agg_emb, target_item_emb_P)  # [batch_size, n_target]

        # 4. 兼容输出格式
        prediction = scores.view(batch_size, -1)
        u_v = torch.zeros_like(prediction, requires_grad=True)
        i_v = torch.zeros_like(prediction, requires_grad=True)

        return {'prediction': prediction, 'u_v': u_v, 'i_v': i_v}

class SANSA(GeneralModel, SANSABase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'lambda_reg', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser = SANSABase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self.corpus = corpus
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = SANSABase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}


# ======================== 适配ImpressionModel ========================
class SANSAImpression(ImpressionModel, SANSABase):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'lambda_reg', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser = SANSABase.parse_model_args(parser)
        return ImpressionModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ImpressionModel.__init__(self, args, corpus)
        self.corpus = corpus
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return SANSABase.forward(self, feed_dict)