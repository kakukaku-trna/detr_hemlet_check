#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from pathlib import Path

def video_to_images(video_path, output_dir, interval=1):
    """
    将视频按帧输出为图片
    :param video_path: 输入视频路径 (.mp4)
    :param output_dir: 输出图片文件夹
    :param interval:   每隔多少帧保存 1 次 (默认 1 表示每帧保存)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ 无法打开视频:", video_path)
        return

    frame_id = 0
    save_id = 0

    print(f"开始提取视频帧：{video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 interval 帧保存
        if frame_id % interval == 0:
            save_path = output_dir / f"frame_{save_id:06d}.jpg"
            cv2.imwrite(str(save_path), frame)
            print("✔ 保存：", save_path)
            save_id += 1

        frame_id += 1

    cap.release()
    print("🎉 完成！总共保存图片：", save_id)


if __name__ == "__main__":
    # 你可以改这里
    video_path = r"L:\detr-main\detr-main\inference_demo\detect_demo\对照.mp4"     # 输入视频
    output_dir = r"L:\detr-main\detr-main\inference_demo\detect_demo\tupian"       # 输出目录
    interval = 1                       # 每 1 帧保存一次 (可以改成 5、10)

    video_to_images(video_path, output_dir, interval)
