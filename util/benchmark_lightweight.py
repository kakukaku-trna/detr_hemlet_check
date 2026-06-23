import torch
import time
from thop import profile
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr_drop', default=120, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

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
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/output_dd',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #我自己改过的
    parser.add_argument('--num_classes', default=3, type=int)

    return parser

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_tensor):
    model.eval()
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params


def compute_fps(model, input_tensor, device="cuda", runs=100):
    model.eval()

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    t1 = time.time()

    return runs / (t1 - t0)


# if __name__ == "__main__":
#     from models import build_model
#     from util.misc import NestedTensor
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="cuda")
#     args = parser.parse_args()

#     model, _, _ = build_model(args)
#     model = model.to(args.device)

#     dummy = torch.randn(1, 3, 640, 640).to(args.device)

#     print("========== Params ==========")
#     print(count_params(model))

#     print("========== FLOPs ==========")
#     flops, params = compute_flops(model, dummy)
#     print("FLOPs:", flops)
#     print("Params (thop):", params)

#     print("========== FPS ==========")
#     fps = compute_fps(model, dummy, args.device)
#     print("FPS:", fps)
if __name__ == "__main__":
    from models import build_model
    import argparse

    parser = get_args_parser()

    args = parser.parse_args()

    model, _, _ = build_model(args)
    model = model.to(args.device)

    dummy = torch.randn(1, 3, 640, 640).to(args.device)

    print("========== Params ==========")
    print(count_params(model))

    print("========== FLOPs ==========")
    flops, params = compute_flops(model, dummy)
    print("FLOPs:", flops)
    print("Params (thop):", params)

    print("========== FPS ==========")
    fps = compute_fps(model, dummy, args.device)
    print("FPS:", fps)