import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import build_model
from PIL import Image
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2

# 设置参数
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', default='False', help="Disables auxiliary decoding losses (loss at each layer)")

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default="/root/detr/data/Crowdhuman/coco")
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/root/detr/inference_demo/inference_output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/root/detr/data/output/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default="True")
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def box_cxcywh_to_xyxy(x):
    # 将DETR的检测框坐标(x_center,y_center,w,h)转化成coco数据集的检测框坐标(x0,y0,x1,y1)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    # 把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    # 1. 置信度过滤
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    # 2. NMS
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
        labels = labels[keep]          # 别忘了 labels 也要过滤
    else:
        labels = scores.argmax(-1)

    # 3. 返回顺序：scores, labels, boxes
    return scores.max(-1).values, labels, boxes

# COCO classes
CLASSES = ['N/A', 'pedestrian']

# 生成随机颜色的函数
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

# 创建类别颜色字典
COLORS = {cls: random_color() for cls in CLASSES}

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # 把检测框画到图片上
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = [255, 0, 0]  # 固定为红色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    torch.manual_seed(args.seed)

    # 导入网络
    model, criterion, postprocessors = build_model(args)

    # 加载权重
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint found at", args.resume)

    model.to(device)
    model.eval()

    # 打印参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    # 输入视频
    video_path = "inference_demo/detect_demo/2.mp4"   # TODO: 修改为你的视频路径
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.output_dir) / ("result_" + Path(video_path).name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    transform = torchvision.transforms.ToTensor()
    frame_id = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR -> RGB
            pil_img = Image.fromarray(rgb_frame)
            img_w, img_h = pil_img.size

            # tensor [1,C,H,W]
            image_tensor = transform(pil_img).unsqueeze(0).to(device)

            t0 = time.time()
            outputs = model(image_tensor)
            t1 = time.time()
            if frame_id % int(fps) == 0:
                print(f"Frame {frame_id}, time per frame: {t1-t0:.3f}s")

            probas = outputs["pred_logits"].softmax(-1)[0, :, :-1].cpu()
            bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0].cpu(), (img_w, img_h))
            scores, labels, boxes = filter_boxes(probas, bboxes_scaled, confidence=0.5, apply_nms=True, iou=0.5)

            # 绘制检测结果
            if boxes.numel() > 0:
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                labels_np = labels.cpu().numpy()
                for i in range(boxes_np.shape[0]):
                    label_id = int(labels_np[i])
                    label_name = CLASSES[label_id] if label_id < len(CLASSES) else str(label_id)
                    conf = float(scores_np[i])
                    text = f"{label_name} {conf:.2f}"
                    color = COLORS.get(label_name, [255, 0, 0])
                    plot_one_box(boxes_np[i], frame, color=color, label=text, line_thickness=2)

            writer.write(frame)  # BGR

    cap.release()
    writer.release()
    print("Saved result video to", out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
