# -*- coding: utf-8 -*-
"""
================================================================================
 dataset.py  ——  PyTorch 数据集与数据增强
================================================================================
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from graph import get_flip_index


def normalize_skeleton(seq):
    """seq [T,V,3] -> xy [T,V,2] (中心化+尺度归一化), vis [T,V,1]"""
    xy = seq[:, :, :2].astype(np.float32).copy()
    vis = seq[:, :, 2:3].astype(np.float32).copy()

    hips = xy[:, [23, 24], :].mean(axis=1)
    shoulders = xy[:, [11, 12], :].mean(axis=1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    valid = torso > 1e-6
    scale = float(np.median(torso[valid])) if np.any(valid) else 1.0
    if scale < 1e-6:
        scale = 1.0

    xy = (xy - hips[:, None, :]) / scale
    return xy, vis


def resample_indices(L, T, jitter=False):
    if L <= 0:
        return np.zeros(T, dtype=np.int64)
    base = np.linspace(0, L - 1, T)
    if jitter and L > 1:
        step = max(1.0, (L - 1) / T)
        base = base + np.random.uniform(-step / 2, step / 2, size=T)
    idx = np.clip(np.round(base), 0, L - 1).astype(np.int64)
    return idx


def augment(xy, vis):
    xy = xy.copy()
    vis = vis.copy()
    V = xy.shape[1]

    if np.random.rand() < config.AUG_HFLIP_PROB:
        xy[:, :, 0] = -xy[:, :, 0]
        fidx = get_flip_index(V)
        xy = xy[:, fidx, :]
        vis = vis[:, fidx, :]

    ang = np.deg2rad(np.random.uniform(-config.AUG_ROTATE_DEG, config.AUG_ROTATE_DEG))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy @ R.T

    sc = 1.0 + np.random.uniform(-config.AUG_SCALE_RANGE, config.AUG_SCALE_RANGE)
    xy = xy * sc

    tr = np.random.uniform(-config.AUG_TRANSLATE, config.AUG_TRANSLATE,
                           size=(1, 1, 2)).astype(np.float32)
    xy = xy + tr

    if config.AUG_JOINT_NOISE > 0:
        xy = xy + np.random.normal(0, config.AUG_JOINT_NOISE,
                                   size=xy.shape).astype(np.float32)

    if config.AUG_JOINT_MASK_PROB > 0:
        mask = np.random.rand(V) < config.AUG_JOINT_MASK_PROB
        vis[:, mask, :] = 0.0

    return xy.astype(np.float32), vis.astype(np.float32)


def build_features(xy, vis, use_motion):
    """xy [T,V,2], vis [T,V,1] -> feat [C,T,V], conf [1,T,V]"""
    feats = [xy]
    if use_motion:
        motion = np.zeros_like(xy)
        motion[1:] = xy[1:] - xy[:-1]
        feats.append(motion)
    feat = np.concatenate(feats, axis=2)
    feat = np.transpose(feat, (2, 0, 1))
    conf = np.transpose(vis, (2, 0, 1))
    return feat.astype(np.float32), conf.astype(np.float32)


class SkeletonDataset(Dataset):
    def __init__(self, records, clip_len, training=False):
        self.records = records
        self.clip_len = clip_len
        self.training = training
        self.use_motion = config.USE_MOTION
        self.cache = {}
        for r in records:
            if r["npy"] not in self.cache:
                self.cache[r["npy"]] = np.load(r["npy"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        seq = self.cache[r["npy"]]
        L = seq.shape[0]
        idx = resample_indices(L, self.clip_len,
                               jitter=(self.training and config.AUG_TEMPORAL_JITTER))
        seq = seq[idx]
        xy, vis = normalize_skeleton(seq)
        if self.training and config.AUG_ENABLE:
            xy, vis = augment(xy, vis)
        feat, conf = build_features(xy, vis, self.use_motion)
        return (torch.from_numpy(feat),
                torch.from_numpy(conf),
                int(r["label"]))


def load_meta():
    with open(config.META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
