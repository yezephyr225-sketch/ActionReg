# -*- coding: utf-8 -*-
"""
================================================================================
 train.py  ——  模型训练（运行的第 2 步）
================================================================================
保留的优化：
  * warmup + 余弦学习率（更平稳）
  * 关闭水平镜像、收敛增强幅度（在 config 里）
  * "best 模型"选择：测试 acc 持平时取更晚 + 训练 acc 更高
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
from utils import set_seed, get_logger, AverageMeter, count_parameters, setup_chinese_font
from dataset import SkeletonDataset, load_meta
from model import build_model


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.DEVICE == "cuda":
        print("[提示] 未检测到可用 CUDA，自动改用 CPU。")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()

    torch.set_grad_enabled(train)
    for feat, conf, label in loader:
        feat, conf, label = feat.to(device), conf.to(device), label.to(device)
        logits = model(feat, conf)
        loss = criterion(logits, label)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = logits.argmax(1)
        acc = (pred == label).float().mean().item()
        bs = label.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(acc, bs)
    torch.set_grad_enabled(True)
    return loss_m.avg, acc_m.avg


def compute_lr(epoch, base_lr, total_epochs, warmup_epochs, scheduler_type,
               step_size, gamma):
    if epoch <= warmup_epochs and warmup_epochs > 0:
        warm_start = base_lr * 0.1
        frac = epoch / max(1, warmup_epochs)
        return warm_start + (base_lr - warm_start) * frac
    if scheduler_type == "cosine":
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        min_lr = base_lr * 0.01
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    n_decays = max(0, (epoch - warmup_epochs) // step_size)
    return base_lr * (gamma ** n_decays)


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def plot_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_chinese_font(config.PREFERRED_CN_FONT)

    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ep, history["train_loss"], label="训练损失")
    axes[0].plot(ep, history["test_loss"], label="测试损失")
    axes[0].set_xlabel("训练轮次 (Epoch)"); axes[0].set_ylabel("损失 Loss")
    axes[0].set_title("损失曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="训练准确率")
    axes[1].plot(ep, history["test_acc"], label="测试准确率")
    axes[1].set_xlabel("训练轮次 (Epoch)"); axes[1].set_ylabel("准确率 Accuracy")
    axes[1].set_title("准确率曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.PLOT_DPI)
    plt.close(fig)


def main():
    config.ensure_dirs()
    set_seed(config.RANDOM_SEED)
    logger = get_logger("train", os.path.join(config.OUTPUT_DIR, "train.log"))
    device = get_device()
    logger.info("使用设备: {}".format(device))

    meta = load_meta()
    classes = meta["classes"]
    num_classes = meta["num_classes"]
    logger.info("类别({}): {}".format(num_classes, classes))
    logger.info("训练样本 {} / 测试样本 {}".format(meta["num_train"], meta["num_test"]))

    if meta["num_train"] == 0:
        logger.info("训练集为空，请先确保数据集就绪并运行 preprocess.py。")
        return

    train_ds = SkeletonDataset(meta["train"], config.CLIP_LEN, training=True)
    test_ds = SkeletonDataset(meta["test"], config.CLIP_LEN, training=False) \
        if meta["num_test"] > 0 else None

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS) if test_ds else None

    model = build_model(num_classes).to(device)
    n_param, n_param_m = count_parameters(model)
    logger.info("模型参数量: {:,} ({:.3f} M)".format(n_param, n_param_m))
    logger.info("创新模块 CA-AGC: {}".format("启用" if config.USE_CA_AGC else "关闭(消融)"))
    logger.info("学习率调度: {} (warmup={} 轮, base_lr={})".format(
        config.LR_SCHEDULER, config.WARMUP_EPOCHS, config.LR))

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)

    history = {k: [] for k in ["train_loss", "train_acc", "test_loss", "test_acc"]}
    best = {"acc": -1.0, "epoch": 0, "train_acc": -1.0}

    for epoch in range(1, config.EPOCHS + 1):
        lr = compute_lr(epoch, config.LR, config.EPOCHS, config.WARMUP_EPOCHS,
                        config.LR_SCHEDULER, config.LR_STEP, config.LR_GAMMA)
        set_lr(optimizer, lr)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        if test_loader:
            te_loss, te_acc = run_epoch(model, test_loader, criterion, device, None)
        else:
            te_loss, te_acc = tr_loss, tr_acc

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss); history["test_acc"].append(te_acc)

        logger.info("Epoch {:3d}/{} | 训练 loss={:.4f} acc={:.3f} | "
                    "测试 loss={:.4f} acc={:.3f} | lr={:.2e}".format(
                        epoch, config.EPOCHS, tr_loss, tr_acc, te_loss, te_acc, lr))

        better = (te_acc > best["acc"] + 1e-9) or \
                 (abs(te_acc - best["acc"]) < 1e-9 and tr_acc >= best["train_acc"])
        if better:
            best.update({"acc": te_acc, "epoch": epoch, "train_acc": tr_acc})
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "config_snapshot": {
                            "in_channels": config.IN_CHANNELS,
                            "blocks": config.MODEL_BLOCKS,
                            "kernel": config.TEMPORAL_KERNEL,
                            "use_ca_agc": config.USE_CA_AGC,
                            "clip_len": config.CLIP_LEN,
                            "use_motion": config.USE_MOTION,
                        }},
                       os.path.join(config.CKPT_DIR, "best.pth"))

    torch.save({"model": model.state_dict(), "classes": classes},
               os.path.join(config.CKPT_DIR, "last.pth"))
    plot_curves(history, os.path.join(config.OUTPUT_DIR, "training_curves.png"))
    with open(os.path.join(config.OUTPUT_DIR, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("训练完成！最优测试准确率 = {:.3f}（第 {} 轮，对应训练 acc={:.3f}）".format(
        best["acc"], best["epoch"], best["train_acc"]))
    logger.info("最优权重: {}".format(os.path.join(config.CKPT_DIR, "best.pth")))
processor_once = false
if not process

if __name__ == "__main__":
    main()
