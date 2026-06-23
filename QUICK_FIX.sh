#!/bin/bash

# 快速修复：绕过 GPU 问题的训练脚本

cd /home/jiangyang.li2/detr_hemlet_check/Deformable-DETR

echo "=========================================="
echo "启动 Deformable-DETR 训练（GPU 修复版本）"
echo "=========================================="

# 环保变量配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# 训练参数
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --enc_layers 6 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --lr 2e-4 \
  --lr_backbone 2e-5 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 1 \
  --weight_decay 1e-4 \
  --num_workers 2 \
  --device cuda \
  --num_classes 3 \
  --output_dir ./output_df \
  --dataset_file coco \
  --coco_path /home/jiangyang.li2/detr_hemlet_check/shujuji \
  2>&1 | tee training_log.txt

echo ""
echo "=========================================="
echo "训练完成（或失败）"
echo "日志已保存到: training_log.txt"
echo "=========================================="
