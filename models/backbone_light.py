# # ------------------------------------------------------------------------
# # Deformable DETR Backbone (ORIGINAL + MODIFIED)
# # ------------------------------------------------------------------------

# from collections import OrderedDict
# import torch
# import torch.nn.functional as F
# import torchvision
# from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
# from typing import Dict, List

# from util.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding


# # =========================================================
# # FrozenBatchNorm2d（原版，未改）
# # =========================================================
# class FrozenBatchNorm2d(torch.nn.Module):
#     def __init__(self, n, eps=1e-5):
#         super().__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
#         self.eps = eps

#     def forward(self, x):
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         scale = w * (rv + self.eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias


# # =========================================================
# # BackboneBase（原版逻辑被替换）
# # =========================================================
# class BackboneBase(nn.Module):

#     def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
#         super().__init__()

#         # =========================================================
#         # ❌ ORIGINAL (ResNet版本)
#         # =========================================================
#         """
#         for name, parameter in backbone.named_parameters():
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#                 parameter.requires_grad_(False)

#         if return_interm_layers:
#             return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
#             self.strides = [8, 16, 32]
#             self.num_channels = [512, 1024, 2048]
#         else:
#             return_layers = {'layer4': "0"}
#             self.strides = [32]
#             self.num_channels = [2048]

#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         """

#         # =========================================================
#         # ✔ MODIFIED (MobileNetV3 + FPN 不再使用 IntermediateLayerGetter)
#         # =========================================================
#         self.body = backbone

#         # 统一输出：P3/P4/P5/P6
#         self.strides = [8, 16, 32, 64]
#         self.num_channels = [256, 256, 256, 256]

#     def forward(self, tensor_list: NestedTensor):

#         # =========================================================
#         # ❌ ORIGINAL
#         # xs = self.body(tensor_list.tensors)
#         # =========================================================

#         # =========================================================
#         # ✔ MODIFIED (MobileNetV3 returns dict directly)
#         # =========================================================
#         xs = self.body(tensor_list.tensors)

#         out: Dict[str, NestedTensor] = {}

#         for name, x in xs.items():
#             mask = tensor_list.mask
#             if mask is not None:
#                 mask = F.interpolate(mask[None].float(), size=x.tensors.shape[-2:]).to(torch.bool)[0]

#             out[name] = NestedTensor(x.tensors, mask)

#         return out


# # # =========================================================
# # # ✔ NEW: MobileNetV3 + FPN（新增模块）
# # # =========================================================
# # class MobileNetV3FPN(nn.Module):

# #     def __init__(self, train_backbone=True):
# #         super().__init__()
# #         # =========================================================
# #         # Deformable DETR required attributes
# #         # =========================================================
# #         self.num_channels = [256, 256, 256, 256]
# #         self.strides = [8, 16, 32, 64]

# #         # =========================================================
# #         # ✔ MODIFIED: backbone replace ResNet
# #         # =========================================================
# #         backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features

# #         # self.stage1 = nn.Sequential(*backbone[0:4])   # C3
# #         # self.stage2 = nn.Sequential(*backbone[4:7])   # C4
# #         # self.stage3 = nn.Sequential(*backbone[7:])    # C5
# #         self.stage1 = nn.Sequential(*backbone[:3])   # 3x3 conv + BN + ReLU, 输出16
# #         self.stage2 = nn.Sequential(*backbone[3:6])  # 输出24
# #         self.stage3 = nn.Sequential(*backbone[6:12]) # 输出40
# #         self.stage4 = nn.Sequential(*backbone[12:16])# 输出112
# #         self.stage5 = nn.Sequential(*backbone[16:])  # 输出160-960
# #         # freeze optional
# #         for p in self.parameters():
# #             p.requires_grad = train_backbone

# #         # =========================================================
# #         # ✔ MODIFIED: channel alignment to 256
# #         # =========================================================
# #         self.lateral_c3 = nn.Conv2d(40, 256, 1)
# #         self.lateral_c4 = nn.Conv2d(112, 256, 1)
# #         # self.lateral_c5 = nn.Conv2d(160, 256, 1)
# #         self.lateral_c5 = nn.Conv2d(960, 256, 1)


# #     def forward(self, x):

# #         # =========================================================
# #         # feature extraction
# #         # =========================================================
# #         x_tensor = x.tensors
# #         mask = x.mask  
# #         # c3 = self.stage1(x)
# #         # c4 = self.stage2(c3)
# #         # c5 = self.stage3(c4)

# #         #x = self.stage1(x)
# #         c1 = self.stage1(x_tensor)  # 16
# #         c2 = self.stage2(c1)        # 24
# #         c3 = self.stage3(c2)        # 40
# #         c4 = self.stage4(c3)        # 112
# #         c5 = self.stage5(c4)        # 960

# #         # ✔ 关键：mask 需要同步下采样
# #         mask_c5 = F.interpolate(mask[None].float(), size=c5.shape[-2:]).to(torch.bool)[0]
# #         # =========================================================
# #         # ✔ MODIFIED: FPN top-down
# #         # =========================================================
# #         p5 = self.lateral_c5(c5)
# #         p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2)
# #         p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2)
# #         p6 = F.max_pool2d(p5, 2)


# #         # =========================================================
# #         # output format (Deformable DETR required)
# #         # =========================================================
# #         return {
# #             "0": NestedTensor(p3, mask_c5),
# #             "1": NestedTensor(p4, mask_c5),
# #             "2": NestedTensor(p5, mask_c5),
# #             "3": NestedTensor(p6, mask_c5),
# #         }


# # =========================================================
# # ✔ MobileNetV3 + FPN (Deformable DETR Compatible)
# # =========================================================

# class MobileNetV3FPN(nn.Module):

#     def __init__(self, train_backbone=True):
#         super().__init__()

#         backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features

#         # =====================================================
#         # ✔ 正确 stage 划分（已对齐 torchvision 结构）
#         # =====================================================
#         self.stage1 = nn.Sequential(*backbone[:3])    # 16
#         self.stage2 = nn.Sequential(*backbone[3:5])   # 24
#         self.stage3 = nn.Sequential(*backbone[5:7])   # 40
#         self.stage4 = nn.Sequential(*backbone[7:13])  # 80
#         self.stage5 = nn.Sequential(*backbone[13:16]) # 112
#         self.stage6 = nn.Sequential(*backbone[16:18]) # 160
#         self.stage7 = nn.Sequential(*backbone[18:])   # 960

#         # freeze
#         if not train_backbone:
#             for p in self.parameters():
#                 p.requires_grad = False

#         # =====================================================
#         # ✔ FPN lateral convs (channel 对齐关键)
#         # =====================================================
#         self.lateral_c3 = nn.Conv2d(40, 256, 1)
#         self.lateral_c4 = nn.Conv2d(80, 256, 1)
#         self.lateral_c5 = nn.Conv2d(112, 256, 1)
#         self.lateral_c6 = nn.Conv2d(160, 256, 1)
#         self.lateral_c7 = nn.Conv2d(960, 256, 1)

#         self.strides = [8, 16, 32, 64]
#         self.num_channels = [256, 256, 256, 256]

#     # =========================================================
#     # ✔ forward（核心修复版）
#     # =========================================================
#     def forward(self, x):

#         x_tensor = x.tensors
#         mask = x.mask

#         # =====================================================
#         # ✔ backbone forward (strict order)
#         # =====================================================
#         x = self.stage1(x_tensor)
#         x = self.stage2(x)
#         c3 = self.stage3(x)   # 40
#         c4 = self.stage4(c3)  # 80
#         c5 = self.stage5(c4)  # 112
#         c6 = self.stage6(c5)  # 160
#         c7 = self.stage7(c6)  # 960

#         # =====================================================
#         # ✔ mask resize (only once per level)
#         # =====================================================
#         def resize_mask(m, ref):
#             return F.interpolate(
#                 m[None].float(),
#                 size=ref.shape[-2:]
#             ).to(torch.bool)[0]

#         m7 = resize_mask(mask, c7)

#         # =====================================================
#         # ✔ FPN top-down
#         # =====================================================
#         p7 = self.lateral_c7(c7)
#         p6 = self.lateral_c6(c6) + F.interpolate(p7, scale_factor=2)
#         p5 = self.lateral_c5(c5) + F.interpolate(p6, scale_factor=2)
#         p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2)
#         p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2)

#         p6 = F.max_pool2d(p7, 2)

#         # =====================================================
#         # ✔ output format (Deformable DETR required)
#         # =====================================================
#         return {
#             "0": NestedTensor(p4, m7),
#             "1": NestedTensor(p5, m7),
#             "2": NestedTensor(p6, m7),
#             "3": NestedTensor(p7, m7),
#         }

# # =========================================================
# # Joiner（原版不改）
# # =========================================================
# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)
#         self.num_channels = backbone.num_channels
#         self.strides = backbone.strides

#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)

#         out: List[NestedTensor] = []
#         pos = []

#         for k in sorted(xs.keys()):
#             out.append(xs[k])

#         for x in out:
#             pos.append(self[1](x).to(x.tensors.dtype))

#         return out, pos


# # =========================================================
# # build_backbone（入口修改）
# # =========================================================
# def build_backbone(args):

#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0

#     # =========================================================
#     # ❌ ORIGINAL (ResNet)
#     # =========================================================
#     """
#     backbone = Backbone(
#         args.backbone,
#         train_backbone,
#         return_interm_layers,
#         args.dilation
#     )
#     """

#     # =========================================================
#     # ✔ MODIFIED (MobileNetV3 + FPN)
#     # =========================================================
#     backbone = MobileNetV3FPN(train_backbone=train_backbone)

#     model = Joiner(backbone, position_embedding)

#     return model


import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from util.misc import NestedTensor

# =========================================================
# MobileNetV3 + FPN for Deformable DETR
# =========================================================
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor
import torchvision

class MobileNetV3FPN(nn.Module):
    def __init__(self, train_backbone=True):
        super().__init__()

        backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features

        # MobileNetV3 stages
        self.stage1 = nn.Sequential(*backbone[:4])    # c3 -> 40
        self.stage2 = nn.Sequential(*backbone[4:7])   # c4 -> 40 (actually 40+?) 验证一下
        self.stage3 = nn.Sequential(*backbone[7:13])  # c5 -> 112
        self.stage4 = nn.Sequential(*backbone[13:])   # c6 -> 960

        # lateral conv: 输入通道必须 = 上面 stage 输出
        self.lateral_c3 = nn.Conv2d(24, 256, 1)
        self.lateral_c4 = nn.Conv2d(40, 256, 1)
        self.lateral_c5 = nn.Conv2d(112, 256, 1)
        self.lateral_c6 = nn.Conv2d(960, 256, 1)

        for p in self.parameters():
            p.requires_grad = train_backbone

    def forward(self, x: NestedTensor):
        tensor = x.tensors
        mask = x.mask

        c3 = self.stage1(tensor)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        c6 = self.stage4(c5)
        # #临时打印，用于参数对齐
        # print("c3", c3.shape)
        # print("c4", c4.shape)
        # print("c5", c5.shape)
        # print("c6", c6.shape)
        # #临时打印，用于参数对齐
        
        # 同步 mask 下采样
        mask_c3 = F.interpolate(mask[None].float(), size=c3.shape[-2:]).to(torch.bool)[0]
        mask_c4 = F.interpolate(mask[None].float(), size=c4.shape[-2:]).to(torch.bool)[0]
        mask_c5 = F.interpolate(mask[None].float(), size=c5.shape[-2:]).to(torch.bool)[0]
        mask_c6 = F.interpolate(mask[None].float(), size=c6.shape[-2:]).to(torch.bool)[0]

        # FPN top-down
        p6 = self.lateral_c6(c6)
        p5 = self.lateral_c5(c5) + F.interpolate(p6, size=c5.shape[-2:])
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:])
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:])

        # optional P7
        p7 = F.max_pool2d(p6, 2)

        return {
        "0": NestedTensor(p3, mask_c3),
        "1": NestedTensor(p4, mask_c4),
        "2": NestedTensor(p5, mask_c5),
        "3": NestedTensor(p6, mask_c6),
        }
# =========================================================
# Joiner (Backbone + Position Embedding)
# =========================================================
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = [256, 256, 256, 256]
        self.strides = [8, 16, 32, 64]

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list = []
        pos: list = []

        for k in sorted(xs.keys()):
            out.append(xs[k])

        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

# =========================================================
# 构建 backbone
# =========================================================
def build_backbone(args):
    from .position_encoding import build_position_encoding
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = MobileNetV3FPN(train_backbone=train_backbone)
    model = Joiner(backbone, position_embedding)
    return model