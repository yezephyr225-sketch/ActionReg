# -*- coding: utf-8 -*-
"""
================================================================================
 pose_extractor.py  ——  人体骨架提取（封装 MediaPipe Pose）
================================================================================
把一段视频或单帧图像转成骨架关键点序列。
每个关键点输出 (x, y, visibility)：
   x, y       —— 归一化到 [0,1] 的图像坐标
   visibility —— 该关键点的置信度/可见度（创新模块 CA-AGC 会用到）

兼容性（重要）：
   1) 优先使用经典 API  mp.solutions.pose（老版本 mediapipe，如 0.9.x 有）。
   2) 若经典 API 不存在（新版本 mediapipe 0.10+ 已移除它），自动回退到
      新版 Tasks API（PoseLandmarker）。
   3) Tasks API 需要一个 .task 模型文件；若本地找不到，会【自动联网下载】
      官方模型并缓存到项目 models/ 目录（只需下载一次，之后离线可用）。
   两条路径对外接口完全一致，上层代码无需关心用的是哪一个。
================================================================================
"""

import os
import numpy as np
import cv2

NUM_JOINTS = 33

# MediaPipe 官方 Pose Landmarker 模型下载地址（Google 官方存储）
_TASK_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_TASK_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


class PoseExtractor:
    """统一的姿态提取器。支持逐帧调用与整段视频提取。"""

    def __init__(self, model_complexity=1, min_det=0.5, min_track=0.5,
                 static_image_mode=False):
        self.num_joints = NUM_JOINTS
        self._backend = None
        self._impl = None

        import mediapipe as mp

        # ---------- 优先：经典 solutions API ----------
        # 注意：新版 mediapipe(0.10+) 虽然可能保留 mp.solutions 命名空间，
        # 但 pose 子模块可能已被移除，因此这里用 try 实际实例化来判断是否真正可用。
        classic_ok = False
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self._impl = mp.solutions.pose.Pose(
                    static_image_mode=static_image_mode,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_det,
                    min_tracking_confidence=min_track,
                )
                self._backend = "solutions"
                classic_ok = True
            except Exception as e:
                print("[提示] 经典 solutions API 初始化失败，将回退到 Tasks API：", e)

        if not classic_ok:
            # ---------- 回退：新版 Tasks API ----------
            self._backend = "tasks"
            self._init_tasks_api(model_complexity, min_det)

    # ------------------------------------------------------------------
    def _download_task_model(self, model_complexity, dest_path):
        """从 Google 官方地址下载 .task 模型到 dest_path。"""
        url = _TASK_MODEL_URLS.get(model_complexity, _TASK_MODEL_URLS[1])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print("[下载] 正在下载 MediaPipe 姿态模型（仅首次需要）...")
        print("       来源:", url)
        print("       保存到:", dest_path)
        try:
            import urllib.request
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            raise RuntimeError(
                "自动下载模型失败：{}\n"
                "请手动用浏览器打开下面的链接下载，并放到项目 models/ 目录：\n  {}\n"
                "目标文件名应为：{}".format(e, url, os.path.basename(dest_path))
            )
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1024:
            raise RuntimeError("模型下载不完整，请重试或手动下载：{}".format(url))
        print("[下载] 完成。")

    # ------------------------------------------------------------------
    def _init_tasks_api(self, model_complexity, min_det):
        """初始化新版 Tasks API（需要本地 .task 模型文件，缺失则自动下载）。"""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        fname = _TASK_MODEL_NAMES.get(model_complexity, "pose_landmarker_full.task")
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # 在常见位置寻找已下载的模型文件
        search = [
            os.path.join(models_dir, fname),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
            os.path.join(os.getcwd(), fname),
        ]
        model_path = next((p for p in search if os.path.exists(p)), None)

        # 找不到则自动下载到 models/ 目录
        if model_path is None:
            model_path = os.path.join(models_dir, fname)
            self._download_task_model(model_complexity, model_path)

        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_det,
        )
        self._impl = vision.PoseLandmarker.create_from_options(opts)
        self._mp_vision = vision
        self._ts = 0

    # ------------------------------------------------------------------
    def extract_frame(self, frame_bgr):
        """
        输入一帧 BGR 图像，返回 [33, 3] 的 (x, y, visibility) 数组。
        若未检测到人体，返回全 0（visibility=0），表示该帧不可靠。
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = np.zeros((self.num_joints, 3), dtype=np.float32)

        if self._backend == "solutions":
            res = self._impl.process(rgb)
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    out[i] = (lm.x, lm.y, lm.visibility)
        else:
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            res = self._impl.detect_for_video(mp_img, self._ts)
            # 新版 Tasks API 返回的是 pose_landmarks（列表，每个人体一组）
            lm_groups = getattr(res, "pose_landmarks", None)
            if lm_groups:
                lms = lm_groups[0]
                for i, lm in enumerate(lms):
                    if i >= self.num_joints:
                        break
                    vis = getattr(lm, "visibility", None)
                    if vis is None:
                        vis = getattr(lm, "presence", 1.0)
                    out[i] = (lm.x, lm.y, float(vis))
        return out

    # ------------------------------------------------------------------
    def extract_video(self, video_path, frame_stride=1, max_frames=None):
        """
        提取整段视频的骨架序列。
        返回 [T, 33, 3] 的数组（T 为有效帧数）。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("无法打开视频: {}".format(video_path))
        seq = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_stride == 0:
                seq.append(self.extract_frame(frame))
                if max_frames is not None and len(seq) >= max_frames:
                    break
            idx += 1
        cap.release()
        if len(seq) == 0:
            return np.zeros((0, self.num_joints, 3), dtype=np.float32)
        return np.stack(seq, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    def close(self):
        try:
            self._impl.close()
        except Exception:
            pass