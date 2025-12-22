# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import torchvision.models as models
# -------------------------
# # 1. 定义 SE 注意力模块
# # -------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.num_channels = num_channels
        # self.se = SEBlock(self.num_channels)#这句是加的可能有问题！
#开始替换
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
#上面是源文件，下面是改动的！
    # def forward(self, tensor_list):
    #     """
    #     Args:
    #         tensor_list (NestedTensor): 包含 tensor 和 mask
    #             tensor_list.tensors: shape [B, C, H, W]
    #             tensor_list.mask: shape [B, H, W]
    #     Returns:
    #         out: dict[str, Tensor], key为输出层名，value为特征图
    #         pos: dict[str, Tensor], key为输出层名，value为positional encoding
    #     """
    #     x = tensor_list.tensors  # 拆出 tensor
    #     mask = tensor_list.mask  # 拆出 mask，如果有的话
    #
    #     # backbone 前向
    #     xs = self.body(x)  # ResNet + SE 模块
    #
    #     # 如果 self.body 返回单 Tensor，则封装成 dict
    #     if isinstance(xs, torch.Tensor):
    #         xs = {"0": xs}
    #
    #     out = {}
    #     pos = {}
    #
    #     # 遍历每个输出层
    #     for name, x_layer in xs.items():
    #         out[name] = x_layer
    #
    #         # 下采样 mask 到特征图大小
    #         if mask is not None:
    #             mask_downsampled = F.interpolate(mask[None].float(), size=x_layer.shape[-2:]).to(torch.bool)[0]
    #         else:
    #             mask_downsampled = None
    #
    #         # 生成 positional encoding
    #         pos[name] = self.position_embedding(NestedTensor(x_layer, mask_downsampled))
    #
    #     return out, pos
#替换结束

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

# # 新backbone
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# Backbone modules with SE support (方案B) and NestedTensor.
# """
# from collections import OrderedDict
# from typing import Dict, List
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision.models._utils import IntermediateLayerGetter
#
# from util.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding
#
# # -------------------- Frozen BatchNorm -------------------- #
# class FrozenBatchNorm2d(nn.Module):
#     """BatchNorm2d with fixed statistics and affine parameters."""
#     def __init__(self, n):
#         super().__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
#
#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]
#         super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
#                                       missing_keys, unexpected_keys, error_msgs)
#
#     def forward(self, x):
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias
#
# # -------------------- SE Module -------------------- #
# class SEModule(nn.Module):
#     """Squeeze-and-Excitation module."""
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
# # -------------------- Backbone Base -------------------- #
# class BackboneBase(nn.Module):
#     """Base backbone that outputs NestedTensors, with SE applied to layer outputs."""
#     def __init__(self, backbone: nn.Module, train_backbone: bool,
#                  return_interm_layers: bool):
#         super().__init__()
#
#         # -------------------- 冻结梯度逻辑 -------------------- #
#         for name, param in backbone.named_parameters():
#             # 冻结 conv1 和 layer1，如果 train_backbone=False
#             if not train_backbone and ('conv1' in name or 'bn1' in name or 'layer1' in name):
#                 param.requires_grad_(False)
#
#         # -------------------- IntermediateLayerGetter -------------------- #
#         if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#         else:
#             return_layers = {"layer4": "0"}
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#
#         # -------------------- 动态获取通道数 -------------------- #
#         self.num_channels_dict = {}
#         for name, module in backbone.named_modules():
#             if name in return_layers:
#                 # 输出通道数取最后一个卷积层的 out_channels
#                 last_conv = None
#                 for m in module.modules():
#                     if isinstance(m, nn.Conv2d):
#                         last_conv = m
#                 if last_conv is not None:
#                     self.num_channels_dict[name] = last_conv.out_channels
#
#         # -------------------- SE 模块 -------------------- #
#         self.se_modules = nn.ModuleDict({
#             layer_name: SEModule(self.num_channels_dict[layer_name])
#             for layer_name in return_layers.keys()
#         })
#
#         # 映射 IntermediateLayerGetter 返回 key -> SE 模块 key
#         self.layer_name_map = {v: k for k, v in return_layers.items()}
#
#         # 保存总输出通道数
#         if return_interm_layers:
#             self.num_channels = [self.num_channels_dict[k] for k in return_layers.keys()]
#         else:
#             self.num_channels = self.num_channels_dict[list(return_layers.keys())[0]]
#
#     def forward(self, tensor_list: NestedTensor):
#         xs = self.body(tensor_list.tensors)
#         mask = tensor_list.mask
#         assert mask is not None
#         out: Dict[str, NestedTensor] = {}
#
#         for name, x in xs.items():
#             se_name = self.layer_name_map[name]  # 映射 "0"/"1"/"2"/"3" -> "layer1"/...
#             x = self.se_modules[se_name](x)      # 对每层输出加 SE
#             mask_down = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out[name] = NestedTensor(x, mask_down)
#
#         return out
#
# # -------------------- Backbone -------------------- #
# class Backbone(nn.Module):
#     """ResNet backbone with frozen BN, SE support (方案B), NestedTensor output."""
#     def __init__(self, name: str, train_backbone: bool,
#                  return_interm_layers: bool, dilation: bool):
#         super().__init__()
#
#         # 构建 torchvision ResNet
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(),
#             norm_layer=FrozenBatchNorm2d
#         )
#
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
#         self.body = BackboneBase(backbone, train_backbone, return_interm_layers)
#         self.num_channels = num_channels
#
#     def forward(self, tensor_list: NestedTensor):
#         return self.body(tensor_list)
#
# # -------------------- Joiner -------------------- #
# class Joiner(nn.Sequential):
#     """Backbone + Position Embedding joiner for DETR."""
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)
#
#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)
#         out: List[NestedTensor] = []
#         pos: List[torch.Tensor] = []
#         for name, x in xs.items():
#             out.append(x)
#             pos.append(self[1](x).to(x.tensors.dtype))
#         return out, pos
#
# # -------------------- Build Backbone -------------------- #
# def build_backbone(args):
#     """Build backbone + position embedding model."""
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model
#
