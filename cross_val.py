# -*- coding: utf-8 -*-
"""
================================================================================
 cross_val.py  ——  留一视频交叉验证（Leave-One-Video-Out, LOVO）
================================================================================
小数据集（每类只有 4 个视频）的"测试 1 个样本"完全没有统计学意义。
本脚本对每个视频轮流做测试，其余做训练，共 16 折，汇总 16 个预测，
得到的累计指标才是可信的论文用结果。
【时间预算】每折 1-2 分钟，16 折约 20-30 分钟（RTX 4060）。

用法：
    python cross_val.py
================================================================================
"""

import os
import math
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils import set_seed, get_logger, setup_chinese_font
from dataset import SkeletonDataset, load_meta
from model import build_model


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_lr(epoch, base_lr, total_epochs, warmup_epochs):
    if epoch <= warmup_epochs and warmup_epochs > 0:
        warm_start = base_lr * 0.1
        frac = epoch / max(1, warmup_epochs)
        return warm_start + (base_lr - warm_start) * frac
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * 0.01
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_fold(train_records, test_records, num_classes, device, epochs, seed):
    set_seed(seed)
    train_ds = SkeletonDataset(train_records, config.CLIP_LEN, training=True)
    test_ds = SkeletonDataset(test_records, config.CLIP_LEN, training=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)

    for epoch in range(1, epochs + 1):
        lr = compute_lr(epoch, config.LR, epochs, config.WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        model.train()
        for feat, conf, label in train_loader:
            feat = feat.to(device); conf = conf.to(device); label = label.to(device)
            logits = model(feat, conf)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 用最后一轮的模型预测（不再使用 EMA，避免之前的 bug）
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for feat, conf, label in test_loader:
            feat = feat.to(device); conf = conf.to(device)
            logits = model(feat, conf)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            trues.extend(label.numpy().tolist())
    return preds, trues


def main():
    config.ensure_dirs()
    logger = get_logger("cv", os.path.join(config.OUTPUT_DIR, "cross_val.log"))
    device = get_device()
    logger.info("使用设备: {}".format(device))

    meta = load_meta()
    classes = meta["classes"]
    num_classes = meta["num_classes"]
    all_records = list(meta["train"]) + list(meta["test"])
    logger.info("视频总数: {}, 类别: {}".format(len(all_records), classes))

    epochs_per_fold = min(config.EPOCHS, 80)
    logger.info("每折训练 {} 轮，共 {} 折".format(epochs_per_fold, len(all_records)))

    all_preds, all_trues = [], []
    for i in range(len(all_records)):
        test_rec = [all_records[i]]
        train_rec = [r for j, r in enumerate(all_records) if j != i]
        tag = "{}/{}  测试={}/{}".format(i + 1, len(all_records),
                                          test_rec[0]["cls"], test_rec[0]["video"])
        logger.info("=" * 60)
        logger.info("[Fold {}]".format(tag))
        preds, trues = train_one_fold(train_rec, test_rec, num_classes, device,
                                      epochs_per_fold, config.RANDOM_SEED + i)
        all_preds.extend(preds); all_trues.extend(trues)
        cur_acc = float(np.mean(np.array(all_preds) == np.array(all_trues)))
        logger.info("  本折预测: {} 真值: {}  累计准确率: {:.4f}".format(
            [classes[p] for p in preds], [classes[t] for t in trues], cur_acc))

    y_true = np.array(all_trues); y_pred = np.array(all_preds)
    acc = float((y_true == y_pred).mean())

    try:
        from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                      classification_report)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)),
            average=None, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        report_txt = classification_report(
            y_true, y_pred, labels=list(range(num_classes)),
            target_names=classes, zero_division=0)
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        prec = np.array([cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0
                         for i in range(num_classes)])
        rec = np.array([cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else 0
                        for i in range(num_classes)])
        f1 = np.array([2 * p * r / (p + r) if (p + r) else 0 for p, r in zip(prec, rec)])
        macro_p, macro_r, macro_f1 = prec.mean(), rec.mean(), f1.mean()
        report_txt = "(sklearn 未安装)"

    logger.info("=" * 60)
    logger.info("【LOVO 交叉验证最终结果】")
    logger.info("总样本数 (= 视频数): {}".format(len(y_true)))
    logger.info("总体准确率: {:.4f}".format(acc))
    logger.info("宏平均 P/R/F1: {:.4f} / {:.4f} / {:.4f}".format(macro_p, macro_r, macro_f1))
    logger.info("\n" + report_txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_chinese_font(config.PREFERRED_CN_FONT)

    fig, ax = plt.subplots(figsize=(1.6 + num_classes, 1.4 + num_classes))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵 (LOVO 交叉验证, N={})".format(len(y_true)))
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, "cv_confusion_matrix.png"),
                dpi=config.PLOT_DPI)
    plt.close(fig)

    x = np.arange(num_classes); w = 0.25
    fig, ax = plt.subplots(figsize=(2 + 1.4 * num_classes, 4.5))
    ax.bar(x - w, prec, w, label="精确率 Precision")
    ax.bar(x,     rec,  w, label="召回率 Recall")
    ax.bar(x + w, f1,   w, label="F1 分数")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20)
    ax.set_ylim(0, 1.05); ax.set_ylabel("分数")
    ax.set_title("各类别评估指标 (LOVO 交叉验证)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i in range(num_classes):
        for val, off in [(prec[i], -w), (rec[i], 0), (f1[i], w)]:
            ax.text(x[i] + off, val + 0.01, "{:.2f}".format(val),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, "cv_per_class_metrics.png"),
                dpi=config.PLOT_DPI)
    plt.close(fig)

    with open(os.path.join(config.OUTPUT_DIR, "cv_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "method": "Leave-One-Video-Out",
            "n_folds": len(y_true),
            "accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "per_class": {classes[i]: {"precision": float(prec[i]),
                                       "recall": float(rec[i]),
                                       "f1": float(f1[i])} for i in range(num_classes)},
            "confusion_matrix": cm.tolist(),
        }, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存到 outputs/:")
    logger.info("  - cv_confusion_matrix.png")
    logger.info("  - cv_per_class_metrics.png")
    logger.info("  - cv_metrics.json")
    logger.info("  - cross_val.log")


if __name__ == "__main__":
    main()
