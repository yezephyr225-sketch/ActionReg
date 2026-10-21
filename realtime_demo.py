# -*- coding: utf-8 -*-
"""
================================================================================
 realtime_demo.py  ——  实时动作识别可视化（运行的第 4 步）
================================================================================
本脚本是项目“实时识别”这一核心需求的最终呈现，一个脚本同时支持：

  【模式一：弹窗实时播放】(默认开启)
      打开摄像头或视频文件，逐帧提取骨架、滑动窗口识别动作，
      在画面上叠加：当前预测类别 + 置信度 + 实时 FPS + 骨架连线。
      右上角的 FPS 数字就是“能否在低端显卡(如 RTX 3050)实时运行”的直接证据。

  【模式二：导出标注视频】(默认开启)
      把识别结果逐帧画到画面上，导出成一个标注好的 MP4 文件，
      方便写论文、做汇报、存档。

  默认两种模式同时开（弹窗预览的同时导出视频）。
  如不想导出，加 --no-save；不想弹窗，加 --no-show。

--------------------------------------------------------------------------------
用法示例（在项目根目录 D:\\G25070602_x3_800 下）：

  # 1) 识别一个视频文件：弹窗 + 自动导出到 outputs\\demo_<源文件名>.mp4
  python realtime_demo.py --source datasets\\Hug\\VID_20260621_130432.mp4

  # 2) 摄像头实时识别（弹窗 + 导出 outputs\\demo_camera.mp4，按 q 退出）
  python realtime_demo.py --source 0

  # 3) 只弹窗、不导出
  python realtime_demo.py --source datasets\\Hug\\VID_20260621_130432.mp4 --no-save

  # 4) 只导出、不弹窗（无显示器/批处理）
  python realtime_demo.py --source datasets\\Hug\\VID_20260621_130432.mp4 --no-show

  # 5) 自定义导出位置：--out 可以是目录(自动拼文件名)，也可以是完整 .mp4 路径
  python realtime_demo.py --source xxx.mp4 --out D:\\G25070602_x3_800\\outputs
  python realtime_demo.py --source xxx.mp4 --out D:\\G25070602_x3_800\\outputs\\result.mp4

按 q 键退出弹窗。
--------------------------------------------------------------------------------
【中文显示】OpenCV 的 putText 不支持中文，本脚本用 Pillow 绘制中文标签，
未安装 Pillow 时自动退回英文，绝不出现乱码方框。
================================================================================
"""

import os
import time
import argparse
from collections import deque, Counter

import numpy as np
import cv2
import torch

import config
from model import build_model
from pose_extractor import PoseExtractor
from dataset import normalize_skeleton, build_features, resample_indices
from graph import POSE_EDGES


# ----------------------------------------------------------------------------
def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
def draw_skeleton(frame, landmarks):
    """在画面上画骨架连线与关节点。landmarks: [V,3] 归一化坐标(x,y,visibility)。"""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, v in landmarks]
    for i, j in POSE_EDGES:
        if landmarks[i, 2] > 0.3 and landmarks[j, 2] > 0.3:
            cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    for (x, y, v) in landmarks:
        if v > 0.3:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 128, 255), -1)


# ----------------------------------------------------------------------------
_FONT_CACHE = {}

def _load_cn_font(size):
    """缓存式加载中文字体。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    try:
        from PIL import ImageFont
        for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\simsun.ttc",
                   "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def put_text_cn(frame, text, org, color=(255, 255, 0), size=28):
    """用 Pillow 绘制中文，失败则退回 OpenCV 英文。color 为 BGR。"""
    try:
        from PIL import Image, ImageDraw
        font = _load_cn_font(size)
        if font is None:
            raise RuntimeError("no font")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame


# ----------------------------------------------------------------------------
def draw_panel(frame, label_text, conf_text, fps, infer_ms, device_name):
    """在左上角画一个半透明信息面板，叠加预测、置信度、FPS。"""
    h, w = frame.shape[:2]
    panel_w = min(int(w * 0.46), 430)
    panel_h = 116
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    frame = put_text_cn(frame, "动作: {}  {}".format(label_text, conf_text),
                        (22, 18), color=(0, 255, 255), size=30)
    fps_color = (0, 255, 0) if fps >= 25 else ((0, 200, 255) if fps >= 15 else (0, 80, 255))
    frame = put_text_cn(frame, "实时帧率 FPS: {:.1f}".format(fps),
                        (22, 56), color=fps_color, size=26)
    frame = put_text_cn(frame, "推理: {:.1f} ms/帧  ({})".format(infer_ms, device_name),
                        (22, 88), color=(220, 220, 220), size=22)
    return frame


# ----------------------------------------------------------------------------
def resolve_out_path(out_arg, source):
    """
    把 --out 解析成一个【具体的 .mp4 文件路径】。
    支持三种情况：
      (1) out_arg 为空 -> 用 outputs/demo_<源名>.mp4
      (2) out_arg 是目录(或以分隔符结尾/已存在的文件夹/无扩展名) -> 在其中拼 demo_<源名>.mp4
      (3) out_arg 是 .xxx 文件名 -> 直接用
    """
    if isinstance(source, str):
        # 同时兼容 Windows(\) 与 Unix(/) 分隔符，避免跨平台时文件名提取错误
        leaf = source.replace("\\", "/").rstrip("/").split("/")[-1]
        base = os.path.splitext(leaf)[0]
    else:
        base = "camera"
    default_name = "demo_{}.mp4".format(base)

    if not out_arg:
        return os.path.join(config.OUTPUT_DIR, default_name)

    # 判断 out_arg 到底是目录还是文件
    _, ext = os.path.splitext(out_arg)
    is_dir = os.path.isdir(out_arg) or out_arg.endswith(("\\", "/")) or ext == ""
    if is_dir:
        return os.path.join(out_arg, default_name)
    return out_arg


def make_writer(out_path, fps, size):
    """创建 VideoWriter，mp4v 失败时自动退回 MJPG(.avi)。返回 (writer, 实际路径) 或 (None, None)。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # 首选 mp4v
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, out_path
    # 退回 MJPG + .avi（兼容性最好）
    alt_path = os.path.splitext(out_path)[0] + ".avi"
    writer = cv2.VideoWriter(alt_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        print("[提示] mp4v 编码器不可用，已改用 MJPG 导出: {}".format(alt_path))
        return writer, alt_path
    return None, None


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="实时动作识别可视化")
    parser.add_argument("--source",
                        default=r"D:\G25070602_x3_800\datasets\Hug\VID_20260621_130432.mp4",
                        help="摄像头编号(如 0)或视频文件路径")
    parser.add_argument("--ckpt", default=os.path.join(config.CKPT_DIR, "best.pth"),
                        help="模型权重路径")
    # 默认就导出；不想导出用 --no-save
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="不导出视频（默认会导出）")
    parser.set_defaults(save=True)
    # --out 留空 -> outputs/demo_<源名>.mp4；可填目录或完整文件名
    parser.add_argument("--out", default=r"D:\G25070602_x3_800\outputs\VID_20260621_130432.mp4",
                        help="导出路径，可为目录或完整 .mp4 文件名；留空=outputs/demo_<源名>.mp4")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="不弹窗")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    config.ensure_dirs()
    device = get_device()

    # ---------------- 加载模型 ----------------
    if not os.path.exists(args.ckpt):
        print("未找到权重 {}，请先运行 train.py 训练。".format(args.ckpt))
        return
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("模型已加载，类别: {}".format(classes))
    print("设备: {}".format(device.type.upper()))

    extractor = PoseExtractor(model_complexity=config.POSE_MODEL_COMPLEXITY,
                              min_det=config.POSE_MIN_DET_CONFIDENCE,
                              min_track=config.POSE_MIN_TRACK_CONFIDENCE)

    # ---------------- 打开视频源 ----------------
    source = args.source if args.source is not None else config.DEMO_SOURCE
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开视频源:", source)
        extractor.close()
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or src_fps > 240:
        src_fps = 30.0

    # ---------------- 视频导出 writer ----------------
    writer = None
    real_out = None
    if args.save:
        out_path = resolve_out_path(args.out, source)
        writer, real_out = make_writer(out_path, src_fps, (in_w, in_h))
        if writer is None:
            print("[警告] 无法创建视频写出器，将仅弹窗不导出。")
        else:
            print("将导出标注视频到: {}".format(real_out))

    # ---------------- 滑动窗口与平滑 ----------------
    buf = deque(maxlen=config.DEMO_BUFFER_LEN)
    vote = deque(maxlen=config.DEMO_SMOOTH)
    frame_id = 0
    label_text, conf_text = "...", ""
    ema_fps = 0.0
    last_infer_ms = 0.0
    n_written = 0
    t_prev = time.time()

    win_name = "Action Recognition (press q to quit)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = extractor.extract_frame(frame)
        buf.append(lm)
        draw_skeleton(frame, lm)

        if len(buf) >= max(8, config.DEMO_BUFFER_LEN // 2) and \
                frame_id % config.DEMO_PRED_EVERY == 0:
            seq = np.stack(buf, axis=0)
            idx = resample_indices(seq.shape[0], config.CLIP_LEN, jitter=False)
            seq = seq[idx]
            xy, vis = normalize_skeleton(seq)
            feat, conf = build_features(xy, vis, config.USE_MOTION)
            feat = torch.from_numpy(feat).unsqueeze(0).to(device)
            conf = torch.from_numpy(conf).unsqueeze(0).to(device)

            t0 = time.time()
            with torch.no_grad():
                prob = torch.softmax(model(feat, conf), dim=1)[0].cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            last_infer_ms = (time.time() - t0) * 1000.0

            pred = int(prob.argmax())
            vote.append(pred)
            smoothed = Counter(vote).most_common(1)[0][0]
            label_text = classes[smoothed]
            conf_text = "{:.0%}".format(float(prob[smoothed]))

        now = time.time()
        inst_fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        ema_fps = inst_fps if ema_fps == 0 else (0.9 * ema_fps + 0.1 * inst_fps)

        frame = draw_panel(frame, label_text, conf_text, ema_fps,
                           last_infer_ms, device.type.upper())

        if writer is not None:
            writer.write(frame)
            n_written += 1
        if args.show:
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
        print("标注视频已保存（共写入 {} 帧）: {}".format(n_written, real_out))
    if args.show:
        cv2.destroyAllWindows()
    extractor.close()
    print("演示结束。端到端平均 FPS ≈ {:.1f}（{}）".format(ema_fps, device.type.upper()))


if __name__ == "__main__":
    main()