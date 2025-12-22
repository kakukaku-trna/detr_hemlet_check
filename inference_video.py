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
    parser.add_argument('--num_classes', default=91, type=int,
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

# generate deterministic random colors but in BGR for OpenCV
def make_colors(classes):
    random.seed(0)
    colors = {}
    for cls in classes:
        # random color in BGR order
        colors[cls] = [random.randint(0, 255) for _ in range(3)]
    return colors

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # x: iterable (x0,y0,x1,y1) in pixel coords
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    if color is None:
        color = [0, 0, 255]  # default red (BGR)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # try placing above box, if overflow place below
        text_x1 = c1[0]
        text_y1 = c1[1] - t_size[1] - 3
        if text_y1 < 0:
            text_y1 = c1[1] + t_size[1] + 3
        text_x2 = text_x1 + t_size[0]
        text_y2 = text_y1 + t_size[1] + 3
        # clip to image bounds
        text_x2 = min(text_x2, img.shape[1] - 1)
        text_y2 = min(text_y2, img.shape[0] - 1)
        cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (text_x1, text_y2 - 2), 0, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

def filter_scores_nms(scores, labels, boxes, confidence=0.3, apply_nms=False, iou=0.5):
    """
    scores: Tensor [N]
    labels: Tensor [N]
    boxes:  Tensor [N,4] in xyxy
    returns filtered (scores, labels, boxes) as CPU tensors
    """
    if scores.numel() == 0:
        return torch.tensor([]), torch.tensor([], dtype=torch.long), torch.tensor([])

    keep_mask = scores > confidence
    if keep_mask.sum() == 0:
        return torch.tensor([]), torch.tensor([], dtype=torch.long), torch.tensor([])

    scores = scores[keep_mask]
    labels = labels[keep_mask]
    boxes = boxes[keep_mask]

    if apply_nms:
        keep_idx = batched_nms(boxes, scores, labels, iou)
        if keep_idx.numel() == 0:
            return torch.tensor([]), torch.tensor([], dtype=torch.long), torch.tensor([])
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    return scores.cpu(), labels.cpu(), boxes.cpu()

# ---------------------------
# Main inference
# ---------------------------

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print("Using device:", device)

    # -- model
    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    # -- load checkpoint robustly
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        # strip "module." prefix if present
        new_state = {}
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                new_key = k
                if k.startswith("module."):
                    new_key = k[len("module."):]
                new_state[new_key] = v
        else:
            new_state = state_dict
        try:
            model.load_state_dict(new_state, strict=False)
            print("Loaded checkpoint into model (strict=False).")
        except Exception as e:
            print("Warning: load_state_dict failed with strict=False:", e)
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Fallback: loaded raw checkpoint dict (strict=False).")
            except Exception as e2:
                print("Error loading checkpoint:", e2)
                print("Continuing without checkpoint.")
    else:
        print("No checkpoint found at", args.resume)

    # print parameter count
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_parameters)

    # -- prepare video IO
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video: {video_path}, fps: {fps}, size: {width}x{height}")

    out_path = Path(args.output_dir) / ("result_" + Path(video_path).name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # -- transforms (match common DETR/resnet50 training)
    transform = transforms.Compose([
        transforms.Resize(args.resize_short),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -- classes: adapt if your model uses a different class list
    # If your checkpoint/model was trained on Crowdhuman(single-class), keep this:
    CLASSES = ['N/A', 'pedestrian']
    #COLORS = make_colors(CLASSES)
    COLORS = {'pedestrian': (255, 0, 0)}

    frame_id = 0
    last_scores_dict = {}  # for optional smoothing keyed by track idx (simple)
    smoothing_alpha = float(args.smooth_alpha)  # 0.0 disabled

    warmup = 5
    timings = []

    # Inference loop
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_w, img_h = pil_img.size

            image_tensor = transform(pil_img).unsqueeze(0).to(device)

            t0 = time.time()
            if args.amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(image_tensor)
            else:
                outputs = model(image_tensor)
            t1 = time.time()
            if frame_id > warmup:
                timings.append(t1 - t0)
            if frame_id % int(max(1, fps)) == 0:
                avg = np.mean(timings) if len(timings) else (t1 - t0)
                print(f"Frame {frame_id}: time per frame ~ {avg:.3f}s")

            # Prefer postprocessor if available (recommended)
            boxes_cpu = torch.tensor([])
            scores_cpu = torch.tensor([])
            labels_cpu = torch.tensor([], dtype=torch.long)

            if isinstance(postprocessors, dict) and 'bbox' in postprocessors:
                try:
                    target_sizes = torch.tensor([[img_h, img_w]], device=device)
                    results = postprocessors['bbox'](outputs, target_sizes)[0]
                    # results: dict with keys 'scores','labels','boxes'
                    scores = results.get('scores', torch.tensor([]))
                    labels = results.get('labels', torch.tensor([], dtype=torch.long))
                    boxes = results.get('boxes', torch.tensor([]))
                    scores_cpu, labels_cpu, boxes_cpu = filter_scores_nms(scores.cpu(), labels.cpu(), boxes.cpu(),
                                                                         confidence=args.confidence,
                                                                         apply_nms=args.apply_nms,
                                                                         iou=args.nms_iou)
                except Exception as e:
                    # fallback if postprocessor fails
                    print("postprocessors['bbox'] failed:", e)
                    # continue to fallback path below
                    pass

            # fallback: directly decode from model outputs (assume outputs contain pred_logits & pred_boxes)
            if boxes_cpu.numel() == 0:
                # try to extract directly (old-style)
                if "pred_logits" in outputs and "pred_boxes" in outputs:
                    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1].cpu()  # [num_queries, num_classes]
                    # if single-class training, ensure probas shape aligns (we assume class idx 0 is background)
                    # get highest class prob and label per query:
                    top_scores, labels = probas.max(-1)  # per-query top score and label
                    bboxes_scaled = outputs["pred_boxes"][0].cpu()
                    # rescale normalized cxcywh -> xyxy in pixel coords
                    # convert cxcywh [0,1] to pixels
                    cx, cy, w, h = bboxes_scaled.unbind(1)
                    x0 = (cx - 0.5 * w) * img_w
                    y0 = (cy - 0.5 * h) * img_h
                    x1 = (cx + 0.5 * w) * img_w
                    y1 = (cy + 0.5 * h) * img_h
                    boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=1)
                    # filter & optional nms
                    scores_cpu, labels_cpu, boxes_cpu = filter_scores_nms(top_scores, labels, boxes_xyxy,
                                                                         confidence=args.confidence,
                                                                         apply_nms=args.apply_nms,
                                                                         iou=args.nms_iou)
                else:
                    # nothing we can do
                    scores_cpu = torch.tensor([])
                    labels_cpu = torch.tensor([], dtype=torch.long)
                    boxes_cpu = torch.tensor([])

            # Optional temporal smoothing: simple exponential smoothing on scores only
            if smoothing_alpha > 0 and boxes_cpu.numel() > 0:
                # We'll smooth by box coordinates presence; this is a naive approach (no tracking)
                # Smooth by comparing exact box coords; if same box appears across frames, smooth the score.
                smoothed_scores = []
                for s, l, b in zip(scores_cpu, labels_cpu, boxes_cpu):
                    key = (int(l.item()), int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item()))
                    prev = last_scores_dict.get(key, None)
                    if prev is not None:
                        s = smoothing_alpha * s + (1 - smoothing_alpha) * prev
                    smoothed_scores.append(s)
                    last_scores_dict[key] = s
                if len(smoothed_scores) > 0:
                    scores_cpu = torch.stack(smoothed_scores)

            # Draw boxes
            if boxes_cpu.numel() > 0:
                boxes_np = boxes_cpu.numpy()
                scores_np = scores_cpu.numpy()
                labels_np = labels_cpu.numpy()
                for i in range(boxes_np.shape[0]):
                    label_id = int(labels_np[i])
                    label_name = CLASSES[label_id] if label_id < len(CLASSES) else str(label_id)
                    conf = float(scores_np[i])
                    text = f"{label_name} {conf:.2f}"
                    color = COLORS.get(label_name, [255, 0, 0])
                    plot_one_box(boxes_np[i], frame, color=color, label=text, line_thickness=2)

            writer.write(frame)

    cap.release()
    writer.release()
    print("Saved result video to:", out_path)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR video inference', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
