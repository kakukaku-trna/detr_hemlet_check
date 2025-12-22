#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized video inference script for DETR-like models.
- Uses proper Resize + Normalize transforms (match resnet50 standard)
- Robust checkpoint loading (handles "module." prefixes)
- Uses postprocessors['bbox'] when available to map boxes to original image size
- Lower confidence threshold (default 0.3)
- Optional mixed precision and optional NMS (disabled by default)
- Simple temporal smoothing option
"""
import argparse
import os
import time
from pathlib import Path

import random
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torchvision.ops.boxes import batched_nms

# Import your model builder (assumes build_model returns model, criterion, postprocessors)
# from models import build_model  # <-- Uncomment / adapt if your repo layout differs
# For this script to run you must have `build_model` available in models.py
from models import build_model

# ---------------------------
# Utility functions
# ---------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('DETR video inference', add_help=False)
    parser.add_argument('--video', type=str, default="inference_demo/detect_demo/2.mp4", help="Path to input video")
    parser.add_argument('--output_dir', default='./inference_output', help='path where to save results')
    parser.add_argument('--resume', default='/root/detr/data/output/checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--device', default='cuda', help='device for inference (cuda or cpu)')
    parser.add_argument('--confidence', default=0.3, type=float, help='confidence threshold (lower for higher recall)')
    parser.add_argument('--apply_nms', action='store_true', help='apply NMS after filtering (not necessary for DETR)')
    parser.add_argument('--nms_iou', default=0.5, type=float, help='NMS IoU threshold if apply_nms is set')
    parser.add_argument('--resize_short', default=800, type=int, help='short side resize used in transform')
    parser.add_argument('--amp', action='store_true', help='use mixed precision (torch.cuda.amp) for inference')
    parser.add_argument('--smooth_alpha', default=0.0, type=float, help='temporal smoothing alpha in [0,1], 0=disabled')
    parser.add_argument('--num_queries', default=100, type=int, help='number of queries (for info)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_file', default='coco', type=str,
                        help="Dataset type: coco, voc, etc.")
    parser.add_argument('--num_classes', default=3, type=int,
                        help="Number of object classes (91 for COCO)")
    # 模型结构参数
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Transformer 隐藏层维度")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="backbone name")
    parser.add_argument('--dilation', action='store_true',
                        help="是否使用空洞卷积")
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int,
                        help="Transformer 注意力头数")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Transformer encoder 层数")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Transformer decoder 层数")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true',
                        help="是否分割模式（推理视频时不需要）")
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned', 'v2'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help="backbone 学习率，>0 表示训练时更新 backbone 权重，推理可随便设")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Transformer FFN 隐藏层维度")
    parser.add_argument('--aux_loss', action='store_true',
                        help="是否使用解码器中间层的辅助损失（推理时可开可关）")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="匹配代价：分类损失权重")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="匹配代价：L1 框回归损失权重")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="匹配代价：GIoU 损失权重")
    parser.add_argument('--bbox_loss_coef', default=5, type=float,
                        help="分类损失权重")
    parser.add_argument('--giou_loss_coef', default=2, type=float,
                        help="GIoU 损失权重")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="空类（背景）损失权重")
    return parser





import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import build_model
import cv2
import torchvision
from torchvision.ops.boxes import batched_nms
import sys
sys.path.append(r"F:\不知名\垃圾文件\sort-master")  # 外层目录
from sort.sort import Sort
# ----------------- 工具函数 -----------------
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
    return scores, boxes

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color if color is not None else [255, 0, 0]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_text = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2_text, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

# ----------------- 类别设置 -----------------
CLASSES = ['N/A', 'helmet', 'no_helmet']
COLORS = {
    'helmet': [0, 255, 0],        # 绿色
    'no_helmet': [0, 0, 255],     # 红色
    'N/A': [0, 0, 255],       # 白色（可随便）
}


# ----------------- 视频推理 -----------------
def main(args):
    device = torch.device(args.device)
    model, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    already_alerted_ids = set()  # 用于记录已经报警的 track_id

    image_Totensor = torchvision.transforms.ToTensor()

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = args.output_dir + "/output_video1111111.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # 帧计数

        # 转成Tensor
        img_tensor = image_Totensor(frame).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0].cpu(), (width, height))
        scores, boxes = filter_boxes(probas, bboxes_scaled)

        scores_np = scores.numpy()
        boxes_np = boxes.numpy()
        # 画框
        for i in range(boxes_np.shape[0]):
            class_id = scores_np[i].argmax()
            raw_label = CLASSES[class_id]
            confidence = scores_np[i].max()

            # 将 "N/A" 统一为 "no_helmet"
            if raw_label == "N/A":
                raw_label = "no_helmet"

            x1, y1, x2, y2 = boxes_np[i]
            box_area = (x2 - x1) * (y2 - y1)

            # ------------------------------
            #   🎯 增强版分类逻辑
            # ------------------------------
            # 1. 小框忽略（避免噪声）
            if box_area < 200:  # 你可以调为 200~600
                continue
            # 2. 默认标签
            label = raw_label

            # 4. 低置信度也判为 no_helmet（更保守）
            if confidence < 0.45 and raw_label != "helmet":
                label = "no_helmet"

            if raw_label == "helmet" and confidence < 0.85:
                continue  # 跳过这个框，不画框也不报警

            # ------------------------------
            #   🎨 画框
            # ------------------------------
            text = f"{label} {confidence:.3f}"
            plot_one_box(boxes_np[i], frame, color=COLORS[label], label=text)
        # 准备 SORT 输入：只追踪 no_helmet 或 N/A
        dets = []
        for i in range(boxes_np.shape[0]):
            class_id = scores_np[i].argmax()
            raw_label = CLASSES[class_id]

            # 将 N/A 当 no_helmet
            if raw_label == "N/A":
                raw_label = "no_helmet"

            if raw_label == "no_helmet":
                x1, y1, x2, y2 = boxes_np[i]
                score = scores_np[i].max()
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)

        # ✅ 修改这里，防止空数组导致报错
        if dets.shape[0] > 0:
            tracks = tracker.update(dets)
        else:
            tracks = np.empty((0, 5))  # 返回空数组，保证 shape=[0,5] 与 SORT 一致# 返回 [x1, y1, x2, y2, track_id]

        # ------------ 每 10 帧检查一次是否存在 no_helmet ------------
        # ------------ 每 10 帧检查一次警报 ------------
        # ------------ 每 10 帧检查一次警报（带追踪去重） ------------
        if frame_count % 10 == 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)

                if track_id not in already_alerted_ids:
                    # 保存报警帧
                    save_path = os.path.join(args.output_dir, f"alert_frame_{frame_count}_id{track_id}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(
                        f"⚠️ 帧 {frame_count} — 未佩戴安全帽！track_id={track_id}, 已保存：{os.path.abspath(save_path)}")

                    # 标记为已报警
                    already_alerted_ids.add(track_id)

        out_video.write(frame)

    cap.release()
    out_video.release()
    print(f"视频处理完成，保存至：{output_path}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
