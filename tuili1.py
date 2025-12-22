#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.ops.boxes import batched_nms
from models import build_model



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
    parser.add_argument('--image', type=str, default="inference_demo/images",
                        help="Path to image or directory")

    return parser


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

def plot_one_box(x, img, color, label=None, line_thickness=2):
    tl = line_thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
        c2_t = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2_t, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - 2),
                    0, tl/3, (255,255,255), thickness=tf, lineType=cv2.LINE_AA)

# ----------------- 类别设置 -----------------
CLASSES = ['N/A', 'helmet', 'no_helmet']
COLORS = {
    'helmet': [0, 255, 0],     # 绿
    'no_helmet': [0, 0, 255],  # 红
    'N/A': [255, 255, 255]
}

# ----------------- 图片推理 -----------------
def main(args):
    device = torch.device(args.device)

    # 构建模型
    model, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    to_tensor = torchvision.transforms.ToTensor()

    # 判断输入是单张图片还是文件夹
    input_path = Path(args.image)
    img_list = []

    if input_path.is_file():
        img_list = [input_path]
    elif input_path.is_dir():
        img_list = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        print("❌ 输入路径错误")
        return

    print(f"共检测 {len(img_list)} 张图像")

    for img_path in img_list:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0].cpu(), (w, h))
        scores, boxes = filter_boxes(probas, bboxes_scaled,
                                     confidence=args.confidence,
                                     apply_nms=args.apply_nms,
                                     iou=args.nms_iou)

        scores_np = scores.numpy()
        boxes_np = boxes.numpy()

        # ---- 绘制识别框 ----
        for i in range(boxes_np.shape[0]):
            class_id = scores_np[i].argmax()
            label = CLASSES[class_id]
            conf = scores_np[i].max()
            text = f"{label} {conf:.2f}"

            plot_one_box(
                boxes_np[i], img,
                color=COLORS[label],
                label=text
            )

        # ---- 保存结果 ----
        save_name = f"{img_path.stem}_result.jpg"
        save_path = Path(args.output_dir) / save_name
        cv2.imwrite(str(save_path), img)
        print(f"✔ 已保存：{save_path}")

    print("🎉 图片检测完成")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)