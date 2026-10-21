# -*- coding: utf-8 -*-
"""
================================================================================
 preprocess.py  ——  数据预处理（运行的第 1 步）
================================================================================
功能：
  1. 自动扫描 datasets/ 下的子文件夹作为“动作类别”（新增动作只需加文件夹）。
  2. 用 MediaPipe 把每个视频提取成骨架序列，保存为 .npy（一次提取，反复使用）。
  3. 按【视频级别】划分训练/测试集（同一视频不会同时进训练和测试，避免数据泄漏）。
  4. 把类别表、划分清单写入 processed/meta.json，供训练/测试/演示读取。
用法（在项目根目录下）：
    python preprocess.py
强制重新提取（忽略已有 .npy 缓存）：
    python preprocess.py --force
================================================================================
"""

import os
import json
import argparse
import random

import numpy as np

import config
from utils import set_seed, get_logger
from pose_extractor import PoseExtractor


def discover_classes():
    """从 datasets/ 子文件夹发现类别。若 config.CLASSES 指定则用指定的。"""
    if config.CLASSES:
        return list(config.CLASSES)
    if not os.path.isdir(config.DATASET_DIR):
        raise FileNotFoundError("数据集目录不存在: {}".format(config.DATASET_DIR))
    classes = sorted(
        d for d in os.listdir(config.DATASET_DIR)
        if os.path.isdir(os.path.join(config.DATASET_DIR, d))
    )
    if not classes:
        raise RuntimeError("datasets/ 下没有任何动作类别文件夹。")
    return classes


def list_videos(class_dir):
    return sorted(
        f for f in os.listdir(class_dir)
        if f.lower().endswith(config.VIDEO_EXTS)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重新提取骨架")
    args = parser.parse_args()

    config.ensure_dirs()
    set_seed(config.RANDOM_SEED)
    logger = get_logger("preprocess", os.path.join(config.OUTPUT_DIR, "preprocess.log"))

    classes = discover_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    logger.info("发现 {} 个动作类别: {}".format(len(classes), classes))

    extractor = PoseExtractor(
        model_complexity=config.POSE_MODEL_COMPLEXITY,
        min_det=config.POSE_MIN_DET_CONFIDENCE,
        min_track=config.POSE_MIN_TRACK_CONFIDENCE,
        static_image_mode=False,
    )
    logger.info("姿态提取后端: {}".format(extractor._backend))

    samples = []        # 每个元素: dict(npy, label, cls, video, n_frames)
    per_class_videos = {c: [] for c in classes}

    for cls in classes:
        class_dir = os.path.join(config.DATASET_DIR, cls)
        videos = list_videos(class_dir)
        if not videos:
            logger.info("[跳过] 类别 {} 下没有视频。".format(cls))
            continue
        for v in videos:
            vpath = os.path.join(class_dir, v)
            npy_name = "{}__{}.npy".format(cls, os.path.splitext(v)[0])
            npy_path = os.path.join(config.SKELETON_DIR, npy_name)

            if os.path.exists(npy_path) and not args.force:
                seq = np.load(npy_path)
                status = "缓存"
            else:
                seq = extractor.extract_video(
                    vpath, frame_stride=config.FRAME_SAMPLE_STRIDE
                )
                np.save(npy_path, seq)
                status = "提取"

            n = seq.shape[0]
            # 检测成功率（visibility 大于 0 的帧占比）
            ok_rate = float(np.mean(seq[:, :, 2].sum(axis=1) > 0)) if n > 0 else 0.0
            logger.info("  [{}] {}/{}  帧数={}  有效检测率={:.0%}".format(
                status, cls, v, n, ok_rate))

            if n < 2:
                logger.info("    [警告] 帧数过少，跳过该视频。")
                continue

            rec = {"npy": npy_path, "label": class_to_idx[cls],
                   "cls": cls, "video": v, "n_frames": int(n)}
            samples.append(rec)
            per_class_videos[cls].append(rec)

    extractor.close()

    # ----------------- 按视频级别划分训练/测试集 -----------------
    rng = random.Random(config.RANDOM_SEED)
    train_set, test_set = [], []
    for cls, recs in per_class_videos.items():
        if not recs:
            continue
        recs = recs[:]
        rng.shuffle(recs)
        n_total = len(recs)
        n_test = max(config.MIN_TEST_PER_CLASS,
                     int(round(n_total * (1 - config.TRAIN_RATIO))))
        n_test = min(n_test, max(0, n_total - 1))  # 至少留 1 个做训练
        test_recs = recs[:n_test]
        train_recs = recs[n_test:]
        train_set.extend(train_recs)
        test_set.extend(test_recs)
        logger.info("类别 {}: 共 {} 视频 -> 训练 {} / 测试 {}".format(
            cls, n_total, len(train_recs), len(test_recs)))

    meta = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "clip_len": config.CLIP_LEN,
        "train": train_set,
        "test": test_set,
        "num_train": len(train_set),
        "num_test": len(test_set),
    }
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("预处理完成！训练样本 {} 个，测试样本 {} 个。".format(
        len(train_set), len(test_set)))
    logger.info("清单已保存到: {}".format(config.META_PATH))
    logger.info("骨架缓存目录: {}".format(config.SKELETON_DIR))
    if len(train_set) == 0 or len(test_set) == 0:
        logger.info("[提示] 训练或测试集为空，请确认每个类别至少有 2 个视频。")


if __name__ == "__main__":
    main()
