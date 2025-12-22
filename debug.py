#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict Helmet / No-Helmet Inference (DETR)
- Outputs only "Helmet" or "No Helmet" per person (no "unknown"/"head"/"person" boxes)
- Matching: helmet/head -> person via IoU
- Color correction: convert head->helmet if helmet-color ratio high and bright
- Attempts to auto-detect class index mapping (person/helmet/head) from model outputs,
  but you can force mapping by setting CLASS_NAME_BY_IDX manually in the code below.
"""
import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from models import build_model

CLASS_NAME_BY_IDX = {1:'helmet',2:'head',3:'person'}
# -------------------------
# 参数解析（保持独立函数）
# -------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('DETR Helmet Strict Inference', add_help=False)
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--resume', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', default='./inference_output', help='Output dir')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--confidence', default=0.3, type=float, help='base confidence to consider a pred')
    parser.add_argument('--conf_person', default=0.2, type=float, help='min conf for person')
    parser.add_argument('--conf_helmet', default=0.90, type=float, help='min conf for helmet')
    parser.add_argument('--conf_head', default=0.3, type=float, help='min conf for head')
    parser.add_argument('--iou_match', default=0.5, type=float, help='IoU threshold to match helmet/head -> person')
    parser.add_argument('--helmet_color_ratio', default=0.18, type=float, help='color ratio threshold for helmet correction')
    parser.add_argument('--min_brightness_for_color', default=100, type=float, help='min V (HSV) to trust color mask')
    parser.add_argument('--apply_nms', action='store_true', help='optional: apply NMS per class to reduce duplicates')
    parser.add_argument('--nms_iou', default=0.5, type=float, help='NMS IoU threshold')
    parser.add_argument('--debug', action='store_true', help='print extra debug info')

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


# -------------------------
# 类别映射（根据训练）
# -------------------------
CLASS_NAME_BY_IDX = {1:'helmet', 2:'head', 3:'person'}

# -------------------------
# 工具函数
# -------------------------
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5*w), (y_c - 0.5*h), (x_c + 0.5*w), (y_c + 0.5*h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    inter = interW*interH
    areaA = max(0, boxA[2]-boxA[0])*max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0])*max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

def plot_one_box(x,img,color=(0,255,0),label=None,thickness=2):
    x1,y1,x2,y2 = [int(v) for v in x]
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
    if label:
        t_size = cv2.getTextSize(label,0,fontScale=thickness/3,thickness=thickness)[0]
        y_text = max(0,y1-6)
        cv2.rectangle(img,(x1,y_text-t_size[1]),(x1+t_size[0],y_text),color,-1)
        cv2.putText(img,label,(x1,y_text-2),0,thickness/3,(255,255,255),thickness=thickness)

def compute_helmet_color_ratio(region_bgr,min_brightness=80):
    h,w = region_bgr.shape[:2]
    if h==0 or w==0: return 0.0,0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[...,2].mean())
    if v_mean<min_brightness: return 0.0, v_mean
    mask_white = cv2.inRange(hsv,(0,0,180),(180,40,255))
    mask_yellow = cv2.inRange(hsv,(20,80,80),(35,255,255))
    mask_red1 = cv2.inRange(hsv,(0,100,100),(10,255,255))
    mask_red2 = cv2.inRange(hsv,(160,100,100),(180,255,255))
    mask_red = cv2.bitwise_or(mask_red1,mask_red2)
    mask = cv2.bitwise_or(mask_white,cv2.bitwise_or(mask_yellow,mask_red))
    ratio = cv2.countNonZero(mask)/float(h*w)
    return ratio, v_mean

# -------------------------
# 推理逻辑
# -------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, _, _ = build_model(args)
    # debug.py 第 153 行
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(args.video).stem
    out_path = os.path.join(args.output_dir, f"{base}_helmet.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # ---------- 输入预处理 ----------
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        img_t = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)

        # -------------------
        # 处理输出
        # -------------------
        pred_logits = outputs['pred_logits'][0].cpu()
        pred_boxes = outputs['pred_boxes'][0].cpu()
        prob = pred_logits.softmax(-1)
        cls_probs = prob[:,:-1]
        scores, labels = cls_probs.max(-1)
        boxes_xyxy = rescale_bboxes(pred_boxes,(width,height)).numpy()

        helmet_boxes, head_boxes, person_boxes = [],[],[]
        for i in range(len(labels)):
            cls_idx = labels[i].item()
            sc = scores[i].item()
            box = boxes_xyxy[i].tolist()
            cls_name = CLASS_NAME_BY_IDX.get(cls_idx)
            if cls_name=='helmet' and sc>=args.conf_helmet:
                helmet_boxes.append(box)
            elif cls_name=='head' and sc>=args.conf_head:
                head_boxes.append(box)
            elif cls_name=='person' and sc>=args.conf_person:
                person_boxes.append(box)

        # ------------ 严格 Helmet / No Helmet ------------
        # 严格 Helmet / No Helmet 判断
        for idx, p_box in enumerate(person_boxes):
            decision = "No Output"
            best_he_iou, best_he_score = 0.0, 0.0
            best_hd_iou, best_hd_score = 0.0, 0.0
            color_ratio, brightness = 0.0, 0.0

            # --- Helmet 判定 ---
            for hb, hs in zip(helmet_boxes, helmet_scores):
                iou_val = iou(p_box, hb)
                if hs >= conf_helmet and iou_val > best_he_iou:
                    best_he_iou = iou_val
                    best_he_score = hs
            if best_he_iou >= iou_match:
                decision = "Helmet"
                plot_one_box(p_box, frame, color=(0, 255, 0), label=decision)
                print(
                    f"Frame {frame_idx}, Person {idx + 1}: decision={decision}, best_helmet_score={best_he_score:.3f}")
                continue

            # --- head (No Helmet) 判定 ---
            for hd, hs in zip(head_boxes, head_scores):
                iou_val = iou(p_box, hd)
                if hs >= conf_head and iou_val > best_hd_iou:
                    best_hd_iou = iou_val
                    best_hd_score = hs
                    # 颜色校正仅作为辅助分数加成
                    x1, y1, x2, y2 = map(int, hd)
                    region = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    cr, br = compute_helmet_color_ratio(region, min_brightness_for_color)
                    color_ratio = cr
                    brightness = br

            # --- 结合分数比值 + 颜色辅助判断 No Helmet ---
            if best_hd_iou >= iou_match:
                # 如果 helmet 分数低或 head 分数明显高，则判 No Helmet
                if best_he_score < conf_helmet or best_hd_score > best_he_score * 1.2:
                    decision = "No Helmet"
                    plot_one_box(p_box, frame, color=(0, 0, 255), label=decision)
                # 否则严格模式下不输出
            # 打印调试信息
            print(f"Frame {frame_idx}, Person {idx + 1}: decision={decision}, "
                  f"best_helmet_score={best_he_score:.3f}, best_head_score={best_hd_score:.3f}, "
                  f"best_helmet_iou={best_he_iou:.2f}, best_head_iou={best_hd_iou:.2f}, "
                  f"color_ratio={color_ratio:.2f}, brightness={brightness:.1f}")
            # 严格模式：没有输出

        writer.write(frame)
        print(f"Frame {frame_idx}: persons={len(person_boxes)}, helmets={len(helmet_boxes)}, heads={len(head_boxes)}")

    cap.release()
    writer.release()
    print("Output saved to:", out_path)

# -------------------------
# -------------------------
# 主逻辑（调试版）
# -------------------------
def main_debug(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    # ---- 强制类别映射 ----
    CLASS_NAME_BY_IDX = {1:'helmet', 2:'head', 3:'person'}
    AUTO_DETECT_CLASS_MAPPING = False

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(args.video).stem
    out_path = os.path.join(args.output_dir, f"{base}_debug.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    transform = torchvision.transforms.ToTensor()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        img_t = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)

        pred_logits = outputs['pred_logits'][0].cpu()
        pred_boxes = outputs['pred_boxes'][0].cpu()
        prob = pred_logits.softmax(-1)
        cls_probs = prob[:, :-1]
        scores, labels = cls_probs.max(-1)
        boxes_xyxy = rescale_bboxes(pred_boxes, (width, height)).numpy()

        # ---- 打印分数和类别 ----
        print(f"\nFrame {frame_idx}:")
        for i in range(len(labels)):
            cls_idx = labels[i].item()
            sc = scores[i].item()
            cls_name = CLASS_NAME_BY_IDX.get(cls_idx, "unknown")
            print(f"  cls_idx={cls_idx}, cls_name={cls_name}, score={sc:.3f}")

        # ---- 临时阈值 ----
        conf_person, conf_helmet, conf_head = 0.2, 0.2, 0.2

        # ---- 收集各类框 ----
        helmet_boxes, head_boxes, person_boxes = [], [], []
        for i in range(len(labels)):
            cls_idx = labels[i].item()
            sc = scores[i].item()
            box = boxes_xyxy[i].tolist()
            cls_name = CLASS_NAME_BY_IDX.get(cls_idx)
            if cls_name=='helmet' and sc>=conf_helmet:
                helmet_boxes.append(box)
            elif cls_name=='head' and sc>=conf_head:
                head_boxes.append(box)
            elif cls_name=='person' and sc>=conf_person:
                person_boxes.append(box)

        # ---- 绘制所有框 ----
        for hb in helmet_boxes:
            plot_one_box(hb, frame, color=(0,255,0), label="Helmet")
        for hd in head_boxes:
            plot_one_box(hd, frame, color=(0,0,255), label="Head")
        for pb in person_boxes:
            plot_one_box(pb, frame, color=(0,255,255), label="Person")

        writer.write(frame)

    cap.release()
    writer.release()
    print("Debug video saved to:", out_path)




# -------------------------
if __name__=='__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
