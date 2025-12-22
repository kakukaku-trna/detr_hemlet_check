#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETR Fine-tune Script (self-contained)
- Automatically freezes backbone
- Low lr for classification head
- Validation each epoch
- Save checkpoint
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models import build_model  # 确保 models/build_model.py 可用
from datasets.coco import build as build_coco  # 如果你的 COCO Dataset 在 datasets/coco.py

# === collate_fn 替代 utils.collate_fn ===
def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------
# Argument parser
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser('DETR Fine-tune helmet/head', add_help=True)
    parser.add_argument('--resume', required=True, help='Path to pretrained checkpoint (.pth)')
    parser.add_argument('--output_dir', default='output_refine', help='Save directory')
    parser.add_argument('--coco_path', required=True, help='Path to dataset root')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset_file', default='coco', type=str, help='dataset type (coco, helmet, etc.)')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcherg6tfdcx  rewewqwq
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters


    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # === 输出文件夹 ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 加载模型 ===
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    # ✅ 新增：从checkpoint读取epoch
    start_epoch = checkpoint.get('epoch', 0)
    print(f"从 epoch {start_epoch} 继续训练...")

    model, criterion, postprocessors = build_model(args)
    pretrained_dict = checkpoint['model']
    if 'class_embed.weight' in pretrained_dict:
        pretrained_dict['class_embed.weight'][1] = 0
        pretrained_dict['class_embed.bias'][1] = 0
    model.load_state_dict(pretrained_dict, strict=False)

    model.to(device)

    # === 冻结 backbone ===
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    # === 数据集 ===
    dataset_train = build_coco('train', args)
    dataset_val = build_coco('val', args)
    #加
    # 新增：加载验证集标注用于评估
    #coco_gt = COCO(Path(args.coco_path) / "annotations" / "instances_val2017.json")
    #加完

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # === 优化器 ===
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone}
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    # ✅ 新增：加载优化器状态
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("已加载优化器状态")

    scheduler_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,total_iters=len(train_loader) * 3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    # ✅ 新增：加载调度器状态
    # ✅ 加载调度器状态（带兼容性检查）
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("已加载学习率调度器状态")

    if 'scheduler_warm' in checkpoint:  # ✅ 修复KeyError
        scheduler_warm.load_state_dict(checkpoint['scheduler_warm'])
        print("已加载Warmup调度器状态")
    else:
        print("⚠️  Checkpoint中未找到Warmup调度器状态，将重新初始化")

    # 在 optimizer 定义之后，加一行
    weight_dict = criterion.weight_dict
    weight_dict['loss_ce'] = 3.0  # head 分类 loss 3 倍权重

    # === 训练循环 ===
    # for epoch in range(args.epochs):
    #     print(f"\n=== Epoch [{epoch + 1}/{args.epochs}] ===")
    # ✅ 修改：从 start_epoch 开始
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch [{epoch + 1}/{args.epochs}] ===")

        # --- 训练 ---
        model.train()
        # for i, (samples, targets) in enumerate(train_loader):
        #     samples = list(s.to(device) for s in samples)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #
        #     loss_dict = criterion(model(samples), targets)
        #     losses = sum(loss_dict[k] for k in loss_dict)
        #     optimizer.zero_grad()
        #     losses.backward()
        #     optimizer.step()
        for i, (samples, targets) in enumerate(train_loader):
            samples = list(s.to(device) for s in samples)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = criterion(model(samples), targets)
            losses = sum(loss_dict.values())

            # ① 保险丝：loss > 50 就跳过
            #if losses.item() > 160:
            if epoch >= 10 and losses.item() > 50:
                print(f'⚠️  跳过爆炸 batch idx={i}, loss={losses.item()}')
                continue

            optimizer.zero_grad()
            #改
            # losses = sum(loss_dict.values())
            # losses = torch.clamp(losses, max=30.0)

            losses = sum(loss_dict.values())
            #改完
            losses.backward()
            # ② 梯度裁剪，防止偶发大梯度
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
            optimizer.step()
            if epoch < 3:  # 前 3 epoch 用 warm-up
                scheduler_warm.step()

            if i % 50 == 0:
                print(f"Step {i}, Loss: {losses.item():.4f}")

        lr_scheduler.step()
#原本
        # --- 验证 ---
        # model.eval()
        # print("Evaluating...")
        # for i, (samples, targets) in enumerate(val_loader):
        #     samples = list(s.to(device) for s in samples)
        #     with torch.no_grad():
        #         outputs = model(samples)
        #     # postprocessors 可用于 bbox/segmentation 处理，按需打印结果
        #
 #改
        # --- 验证 --- (替换原有验证代码块)
        model.eval()
        print("Evaluating...")
        max_val_steps = 3000  # 限制验证步数，调试用
        with torch.no_grad():
            for i, (samples, targets) in enumerate(val_loader):
                if i >= max_val_steps:  # 提前退出，避免卡太久
                    print(f"验证提前结束于 {max_val_steps} 步")
                    break

                samples = list(s.to(device) for s in samples)
                outputs = model(samples)

                # 每10步打印一次进度
                if i % 10 == 0:
                    print(f"Val step {i}/{len(val_loader)}")

                # 释放缓存
                torch.cuda.empty_cache()
#改完
        # --- 保存 checkpoint ---
        ckpt_path = output_dir / f'checkpoint_epoch{epoch +  1}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': vars(args)
        }, ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":

    main()
