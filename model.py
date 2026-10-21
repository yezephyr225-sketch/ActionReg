# -*- coding: utf-8 -*-
"""
================================================================================
 model.py  ——  轻量时空图卷积网络 + 创新模块 CA-AGC
================================================================================
模型 = 轻量 ST-GCN 主干 + 创新模块「置信度感知的自适应图卷积 (CA-AGC)」。

【创新点：CA-AGC（Confidence-Aware Adaptive Graph Convolution）】
  传统 ST-GCN 的图邻接矩阵 A 是固定的（或仅数据自适应），所有关节被同等看待，
  完全丢弃了姿态估计器输出的“置信度/可见度”这一免费且有价值的信息。
  在真实手势中，握手(手伸向镜头)、拥抱(手臂交叉自遮挡)、快速挥手(运动模糊)等
  都会导致部分关节定位不可靠。

  CA-AGC 让邻接矩阵随【每帧关节置信度】动态变化：
      G_ij(t) = sqrt( c_i(t) * c_j(t) )          # 置信度门控
      A_eff(t) = RowNorm( A_static ⊙ G(t) ) + B  # B 为可学习自适应项
  低置信度关节相连的边被自动削弱，特征聚合时不可靠关节贡献减小，
  模型自然地把注意力集中在可靠关节上。该模块几乎不增加计算量
  （仅一次逐元素乘法 + 行归一化），置信度直接来自 MediaPipe，无需额外网络。

  可通过 config.USE_CA_AGC=False 关闭以做消融实验对比。
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from graph import build_adjacency, NUM_JOINTS


# ----------------------------------------------------------------------------
# 创新模块：置信度感知自适应图卷积
# ----------------------------------------------------------------------------
class CAAGConv(nn.Module):
    """空间图卷积（含 CA-AGC）。输入 x[N,Cin,T,V] 与 conf[N,1,T,V]。"""

    def __init__(self, in_ch, out_ch, A, use_ca_agc=True):
        super().__init__()
        self.use_ca_agc = use_ca_agc
        self.register_buffer("A", torch.from_numpy(A.astype(np.float32)))  # [V,V] 已归一化
        V = A.shape[0]
        # 可学习自适应邻接（数据驱动），初始化为 0，初始等价于纯静态图
        self.B = nn.Parameter(torch.zeros(V, V))
        # 1x1 卷积做特征变换
        self.theta = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, conf):
        N, C, T, V = x.shape
        h = self.theta(x)                       # [N,Cout,T,V]

        if self.use_ca_agc:
            c = conf[:, 0, :, :].clamp(0.0, 1.0)        # [N,T,V]
            cs = torch.sqrt(c + 1e-6)
            G = cs.unsqueeze(-1) * cs.unsqueeze(-2)      # [N,T,V,V] 置信度门控
            A_gated = self.A.unsqueeze(0).unsqueeze(0) * G   # [N,T,V,V]
            denom = A_gated.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            A_gated = A_gated / denom                    # 行归一化
            A_eff = A_gated + self.B.unsqueeze(0).unsqueeze(0)   # 叠加自适应项
            out = torch.einsum("nctv,ntvw->nctw", h, A_eff)
        else:
            # 消融：不使用置信度门控，仅静态图 + 自适应项
            A_eff = self.A + self.B                       # [V,V]
            out = torch.einsum("nctv,vw->nctw", h, A_eff)
        return out


# ----------------------------------------------------------------------------
# 时间卷积
# ----------------------------------------------------------------------------
class TemporalConv(nn.Module):
    def __init__(self, ch, kernel=9, stride=1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(ch, ch, kernel_size=(kernel, 1),
                              stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(self.conv(x))


# ----------------------------------------------------------------------------
# ST-GCN 基本块：空间(CA-AGC) + 时间 + 残差
# ----------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, kernel=9, dropout=0.0,
                 use_ca_agc=True, residual=True):
        super().__init__()
        self.gcn = CAAGConv(in_ch, out_ch, A, use_ca_agc=use_ca_agc)
        self.gcn_bn = nn.BatchNorm2d(out_ch)
        self.tcn = TemporalConv(out_ch, kernel=kernel, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.stride = stride

        if not residual:
            self.residual = None
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda v: v
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    @staticmethod
    def _down_conf(conf, T_out):
        """把 conf 的时间维下采样/对齐到 T_out。"""
        T = conf.shape[2]
        if T == T_out:
            return conf
        idx = torch.linspace(0, T - 1, T_out, device=conf.device).round().long()
        return conf[:, :, idx, :]

    def forward(self, x, conf):
        res = 0 if self.residual is None else self.residual(x)
        h = self.relu(self.gcn_bn(self.gcn(x, conf)))     # 空间
        h = self.tcn(h)                                    # 时间(可能下采样)
        h = self.drop(h)
        out = self.relu(h + res)
        conf_out = self._down_conf(conf, out.shape[2])
        return out, conf_out



# ----------------------------------------------------------------------------
# 完整模型
# ----------------------------------------------------------------------------
class LightCAAGCN(nn.Module):
    def __init__(self, num_classes, in_channels=None, num_joints=NUM_JOINTS,
                 blocks=None, kernel=None, dropout=None, use_ca_agc=None):
        super().__init__()
        in_channels = in_channels if in_channels is not None else config.IN_CHANNELS
        blocks = blocks if blocks is not None else config.MODEL_BLOCKS
        kernel = kernel if kernel is not None else config.TEMPORAL_KERNEL
        dropout = dropout if dropout is not None else config.DROPOUT
        use_ca_agc = use_ca_agc if use_ca_agc is not None else config.USE_CA_AGC

        A = build_adjacency(num_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        layers = []
        cin = in_channels
        for i, (cout, stride) in enumerate(blocks):
            layers.append(STGCNBlock(
                cin, cout, A, stride=stride, kernel=kernel,
                dropout=dropout, use_ca_agc=use_ca_agc,
                residual=(i != 0),     # 第一层不用残差
            ))
            cin = cout
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x, conf):
        # x:[N,C,T,V]  conf:[N,1,T,V]
        N, C, T, V = x.shape
        # 输入标准化
        h = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        h = self.data_bn(h)
        h = h.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        for layer in self.layers:
            h, conf = layer(h, conf)

        # 全局池化 [N,C,T,V] -> [N,C]
        h = F.adaptive_avg_pool2d(h, 1).view(h.size(0), -1)
        return self.fc(h)


def build_model(num_classes):
    """根据 config 构造模型。"""
    return LightCAAGCN(num_classes=num_classes)
