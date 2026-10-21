# -*- coding: utf-8 -*-
"""
================================================================================
 utils.py  ——  通用工具函数
================================================================================
包含：
  1. setup_chinese_font()  解决 matplotlib 中文 / 负号显示成“方框 □”的问题
  2. set_seed()            固定随机种子，保证实验可复现
  3. get_logger()          统一的控制台 + 文件日志
  4. AverageMeter          训练时统计平均值
  5. count_parameters()    统计模型参数量
================================================================================
"""

import os
import sys
import random
import logging

import numpy as np


# ----------------------------------------------------------------------------
# 1. 中文字体修复（核心：解决标题/坐标/数字出现方框的问题）
# ----------------------------------------------------------------------------
def setup_chinese_font(preferred: str = ""):
    """
    让 matplotlib 正确显示中文和负号。

    方框 □ 的根本原因：matplotlib 默认字体（DejaVu Sans）不含中文字形，
    且默认用特殊的 Unicode 负号，部分字体不支持。
    解决方法：把字体切换成系统中存在的中文字体，并关闭 Unicode 负号。

    本函数会：
      (1) 如果传入 preferred 且系统里存在，则直接使用；
      (2) 否则在常见中文字体里自动挑选一个可用的；
      (3) 同时设置 axes.unicode_minus=False 修复负号方框。
    返回最终使用的字体名（字符串）。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 候选中文字体（覆盖 Windows / Linux / Mac 常见情况）
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",  # Windows
        "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",     # macOS
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "Source Han Sans SC", "Source Han Sans CN",
        "AR PL UMing CN", "Droid Sans Fallback",                      # Linux
    ]

    # 系统中已安装的字体名集合
    installed = {f.name for f in font_manager.fontManager.ttflist}

    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break

    # 若按名字没找到，尝试按文件路径扫描常见中文字体文件并注册
    if chosen is None:
        possible_files = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        for fp in possible_files:
            if os.path.exists(fp):
                try:
                    font_manager.fontManager.addfont(fp)
                    chosen = font_manager.FontProperties(fname=fp).get_name()
                    break
                except Exception:
                    continue

    if chosen is not None:
        # 把选中的中文字体放到 sans-serif 列表最前面
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["font.family"] = "sans-serif"
    else:
        print("[警告] 未在系统中找到中文字体，图中中文可能仍显示为方框。"
              "请安装 SimHei 或 Microsoft YaHei 后重试，或在 config.PREFERRED_CN_FONT 指定字体。")

    # 关键：修复负号显示为方框
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ----------------------------------------------------------------------------
# 2. 随机种子
# ----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """固定 random / numpy / torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 3. 日志
# ----------------------------------------------------------------------------
def get_logger(name: str = "ActionRec", log_file: str = None):
    """返回一个同时输出到控制台和（可选）文件的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:            # 避免重复添加 handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------------------
# 4. 平均值统计
# ----------------------------------------------------------------------------
class AverageMeter:
    """累计并计算平均值（如 loss、acc）。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


# ----------------------------------------------------------------------------
# 5. 参数量统计
# ----------------------------------------------------------------------------
def count_parameters(model):
    """返回模型可训练参数量（以百万 M 为单位）。"""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n / 1e6
