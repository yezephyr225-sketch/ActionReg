# -*- coding: utf-8 -*-
"""
================================================================================
 graph.py  ——  人体骨架图（MediaPipe Pose 33 关键点）
================================================================================
定义关节之间的连接关系（边），并构造归一化邻接矩阵，供图卷积使用。
同时给出左右对称的关节对，用于“水平镜像”数据增强。

MediaPipe Pose 33 个关键点索引（部分）：
  0 鼻子  1-6 眼/眼角  7-8 耳  9-10 嘴
  11 左肩 12 右肩  13 左肘 14 右肘  15 左腕 16 右腕
  17-22 手部细节点  23 左髋 24 右髋  25 左膝 26 右膝
  27 左踝 28 右踝  29-32 足部点
================================================================================
"""

import numpy as np

NUM_JOINTS = 33

# 标准 MediaPipe Pose 连接（无向边）
POSE_EDGES = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 躯干与手臂
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # 髋与腿
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# 左右对称关节对（用于水平镜像增强时交换）
FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def build_adjacency(num_joints: int = NUM_JOINTS, edges=None, self_loop: bool = True):
    """
    构造对称归一化邻接矩阵 A_norm = D^{-1/2} (A + I) D^{-1/2}。
    返回形状 [V, V] 的 float32 numpy 数组。
    """
    if edges is None:
        edges = POSE_EDGES
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        A += np.eye(num_joints, dtype=np.float32)
    # 对称归一化
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)


def get_flip_index(num_joints: int = NUM_JOINTS):
    """
    返回长度为 num_joints 的索引数组：镜像后第 i 个关节应取原来第 flip_index[i] 个。
    """
    idx = np.arange(num_joints)
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    return idx
