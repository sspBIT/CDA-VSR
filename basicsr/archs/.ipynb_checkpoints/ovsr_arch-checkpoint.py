'''
 Copyright 2023 xtudbxk
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmengine.model import constant_init
# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from torchvision.ops import deform_conv2d

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def overlapping_window_partition(x, window_size, stride):
    """
    手动实现带有重叠的窗口划分。
    
    Args:
        x: 输入特征图，形状为 (B, C, H, W)
        window_size: 窗口的大小
        stride: 步幅，表示滑动窗口的移动步长
    
    Returns:
        windows: 划分后的窗口，形状为 (num_windows, window_size, window_size, C)
    """
    B, C, H, W = x.shape

    # 计算每个维度上的窗口数量
    out_h = (H - window_size) // stride + 1
    out_w = (W - window_size) // stride + 1

    # 使用 unfold 操作来实现窗口划分，stride 影响滑动步幅，window_size 定义窗口大小
    unfolded = F.unfold(x, kernel_size=window_size, stride=stride)
    
    # unfolded 形状为 (B, C*window_size*window_size, out_h*out_w)
    unfolded = unfolded.view(B, C, window_size * window_size, out_h, out_w)

    # 重新排列为 (num_windows, window_size, window_size, C)
    unfolded = unfolded.permute(0, 3, 4, 1, 2).contiguous()

    return unfolded.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def overlapping_window_reverse(windows, window_size, overlap, H, W):
    """
    将窗口重组为原始特征图。
    Args:
        windows: 划分后的窗口 (num_windows, window_size, window_size, C)。
        window_size: 窗口大小。
        overlap: 重叠大小。
        H, W: 原始特征图的高度和宽度。
    Returns:
        重组后的特征图 (B, H, W, C)。
    """
    stride = window_size - overlap
    num_windows, window_size, _, C = windows.shape

    # 计算每个样本的窗口数量
    num_vertical_windows = (H - overlap) // stride  # 每列窗口数
    num_horizontal_windows = (W - overlap) // stride  # 每行窗口数

    # 推断批量大小 B
    B = num_windows // (num_vertical_windows * num_horizontal_windows)

    # 确定恢复后的特征图的大小
    H_pad = H + (stride - H % stride) % stride
    W_pad = W + (stride - W % stride) % stride

    # 初始化空的特征图和权重掩码
    full_feat = torch.zeros(B, H_pad, W_pad, C, device=windows.device)
    weight_mask = torch.zeros_like(full_feat)

    # 计算窗口的大小和步长
    count = 0
    for i in range(0, H_pad - window_size + 1, stride):
        for j in range(0, W_pad - window_size + 1, stride):
            window = windows[count]  # 取出一个窗口
            full_feat[:, i:i + window_size, j:j + window_size, :] += window  # 把窗口加到原图相应位置
            weight_mask[:, i:i + window_size, j:j + window_size, :] += 1  # 更新权重掩码
            count += 1

    # 最后将结果归一化，去除重复覆盖的区域
    full_feat /= weight_mask
    return full_feat[:, :H, :W, :]  # 去掉 padding 部分，返回 (B, H, W, C)


@ARCH_REGISTRY.register()
class OVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5, 
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False):

        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.hr_in = hr_in

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # conv align
        self.convs = nn.Sequential(
            nn.Conv2d(num_feat * 3, num_feat * 3, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat * 3, num_feat, 3, padding=1))
        
        self.deform_align1 = DeformableAlignment_npu(in_channels=num_feat, out_channels=num_feat)
        self.deform_align2 = DeformableAlignment_npu(in_channels=num_feat, out_channels=num_feat)
        
        # CrossAttention align
        window_size = 8
        num_heads = 8
        self.ln_l = nn.LayerNorm([window_size * window_size, num_feat])
        self.ln_h = nn.LayerNorm([window_size * window_size, num_feat])
        self.ln_current = nn.LayerNorm([window_size * window_size, num_feat])
        
        self.attn_l = WindowAttention(dim=num_feat, window_size=to_2tuple(window_size), num_heads=num_heads)
        self.attn_h = WindowAttention(dim=num_feat, window_size=to_2tuple(window_size), num_heads=num_heads)

        self.reconstruction_I = make_layer(ResidualBlockNoBN, 32, num_feat=num_feat)
        self.reconstruction_P = make_layer(ResidualBlockNoBN, 16, num_feat=num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    # def check_image_size(self, x):
    #     _, _, _, h, w = x.size()
    #     window_size = 16
    #     mod_pad_h = (window_size - h % window_size) % window_size
    #     mod_pad_w = (window_size - w % window_size) % window_size
    #     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, 0), 'reflect')
    #     return x

    def forward(self, x, mv, hidden_key=None, return_hs=False):
        H, W = x.shape[3:]
        # x = self.check_image_size(x)
        # mv = self.check_image_size(mv)
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # extract features for each frame
        feat_origin = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat = self.feature_extraction(feat_origin)
        feat = feat.view(b, t, -1, h, w)

        # align
        reconstruction_feats = []
        for fi in range(t):
            current_feat = feat[:, fi]
            if fi == 0 and hidden_key is None:
                # feat_warp2 = torch.zeros(b,64,h,w,device=x.device)
                rec_feat = self.reconstruction_I(current_feat)
                
            else:
                current_mv = mv[:, fi]  # [8, 2, 64, 64]
                current_mv = -current_mv
                feat_l, feat_h = hidden_key[0],hidden_key[1]
                feat_warp_l = mv_warp_avg_patch(feat_l,current_mv,interpolation='nearest', padding_mode='zeros', align_corners=True)
                feat_warp_h = mv_warp_avg_patch(feat_h,current_mv,interpolation='nearest', padding_mode='zeros', align_corners=True)

#                 # 交叉注意力细对齐-基于窗口
#                 window_size = 8
#                 feat_warp_l_windows = window_partition(feat_warp_l, window_size=window_size)  # nW*B, window_size, window_size, C
#                 feat_warp_h_windows = window_partition(feat_warp_h, window_size=window_size)  # nW*B, window_size, window_size, C
#                 current_feat_windows = window_partition(current_feat, window_size=window_size)   # nW*B, window_size, window_size, C
                
#                 feat_warp_l_windows = self.ln_l(feat_warp_l_windows.view(-1, window_size * window_size, 32))  # nW*B, window_size*window_size, C
#                 feat_warp_h_windows = self.ln_h(feat_warp_h_windows.view(-1, window_size * window_size, 32))  # nW*B, window_size*window_size, C
#                 current_feat_windows = self.ln_current(current_feat_windows.view(-1, window_size * window_size, 32))  # nW*B, window_size*window_size, C
                             
#                 aligned_feat_l_windows = self.attn_l(feat_warp_l_windows, current_feat_windows,current_feat_windows).view(-1, window_size, window_size, 32)  # nW*B, window_size*window_size, C
#                 aligned_feat_h_windows = self.attn_h(feat_warp_h_windows, current_feat_windows,current_feat_windows).view(-1, window_size, window_size, 32)  # nW*B, window_size*window_size, C
                
#                 aligned_feat_l = window_reverse(aligned_feat_l_windows, window_size, h, w).permute(0, 3, 1, 2) + feat_warp_l # b c h w
#                 aligned_feat_h = window_reverse(aligned_feat_h_windows, window_size, h, w).permute(0, 3, 1, 2) + feat_warp_h # b c h w


                # mv-guided deformable convolution
                cond_l = torch.cat([feat_warp_l, current_feat], dim=1)
                aligned_feat_l = self.deform_align1(feat_l, cond_l, current_mv)
                cond_h = torch.cat([feat_warp_h, current_feat], dim=1)
                aligned_feat_h = self.deform_align2(feat_h, cond_h, current_mv)            
            
                aligned_feat = self.convs(torch.cat([aligned_feat_l,aligned_feat_h,current_feat], dim=1))
                
                rec_feat = self.reconstruction_P(aligned_feat)
            reconstruction_feats.append(rec_feat)
            hidden_key = (current_feat,rec_feat)

        reconstruction_feats = torch.stack(reconstruction_feats, dim=1)

        # upsample
        out = reconstruction_feats.view(b*t, -1, h, w)
        out = self.pixel_shuffle(self.upconv1(out)).view(b, t, c, 4*h, 4*w)
        if self.hr_in:
            base = x
        else:
            base = F.interpolate(x.view(-1, c, h, w), scale_factor=4, mode='bilinear', align_corners=False).view(b, t, c, 4*h, 4*w)
        out += base
        out1 = out[:, :, :, :H*4, :W*4]

        if return_hs is True:
            return out1, hidden_key
        else:
            return out1

def flow_warp_avg_patch(x, flow, interpolation='nearest', padding_mode='zeros', align_corners=True):
    """Patch Alignment

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, 2,h, w). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    # if x.size()[-2:] != flow.size()[1:3]:
    #     raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
    #                      f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # patch size is set to 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    flow = F.pad(flow, (0, pad_w, 0, pad_h), mode='reflect')
    hp = h + pad_h
    wp = w + pad_w
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, hp), torch.arange(0, wp))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = F.avg_pool2d(flow, 8)
    flow = F.interpolate(flow, scale_factor=8, mode='nearest')
    flow = flow.permute(0, 2, 3, 1)
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output

def mv_warp_avg_patch(x, mv, interpolation='nearest', padding_mode='zeros', align_corners=True):
    """Patch Alignment

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, 2,h, w). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    # if x.size()[-2:] != flow.size()[1:3]:
    #     raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
    #                      f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # patch size is set to 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    flow = F.pad(mv, (0, pad_w, 0, pad_h), mode='reflect')
    hp = h + pad_h
    wp = w + pad_w
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, hp), torch.arange(0, wp))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = flow.permute(0, 2, 3, 1)
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output

def mv_warp(x, mv):
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    mv = mv.permute(0, 2, 3, 1)
    grid_mv = grid - mv
    y_indices = grid_mv[..., 0].long()
    x_indices = grid_mv[..., 1].long()
    y_indices = torch.clamp(y_indices, min=0, max=h-1)
    x_indices = torch.clamp(x_indices, min=0, max=w-1)

    batch_idx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)  # (batch_size, 1, 1)
    channel_idx = torch.arange(x.size(1), device=x.device).view(1, -1, 1)  # (1, channels, 1)

    # 扩展批次和通道索引以匹配 y_indices 和 x_indices 的形状
    batch_idx = batch_idx.expand(-1, y_indices.size(1), y_indices.size(2))  # (batch_size, h, w)
    channel_idx = channel_idx.expand(y_indices.size(0), -1, y_indices.size(1))  # (batch_size, channels, h)

    # 使用高级索引提取对应位置的值
    output = x[batch_idx, channel_idx, y_indices, x_indices]

    return output

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q,K,V, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = Q.shape
        q = self.q_proj(Q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(K).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(V).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class DeformableAlignment(nn.Module):
    """Deformable alignment module.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deform_groups=16,
                 bias=True, max_residue_magnitude=10):
        super(DeformableAlignment, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        
        offset_channels = (2 * kernel_size * kernel_size + kernel_size * kernel_size) * deform_groups
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, offset_channels, 3, 1, 1)
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=0.1)

        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow):
        out = self.conv_offset(extra_feat)
        offset1, offset2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset1, offset2), dim=1))
        offset = offset + flow.flip(1).repeat(1,offset.size(1) // 2, 1,1)

        # mask
        mask = torch.sigmoid(mask)
        offset_all = torch.cat((offset, mask), dim=1)

        # return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, self.groups,
        #                                self.deform_groups)
        return deform_conv2d(x, offset, self.weight, bias=None, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)
    
    
import torch_npu   
class DeformableAlignment_npu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deform_groups=1,
                 bias=True, max_residue_magnitude=8):
        super(DeformableAlignment_npu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude

        weight = torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size).to('npu')
        nn.init.kaiming_uniform_(weight, a=0.1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias_tensor = torch.zeros(out_channels).to('npu')
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.bias = None
            
        nn.init.kaiming_uniform_(self.weight, a=0.1)

        offset_channels = (2 * kernel_size * kernel_size + kernel_size * kernel_size) * deform_groups

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * out_channels+2, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, offset_channels, 3, 1, 1)
        )

        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat,flow], dim=1).contiguous()
        out = self.conv_offset(extra_feat)
        offset1, offset2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat([offset1, offset2], dim=1).contiguous())
        repeat_times = offset.size(1) // 2
        flow = flow.flip(1).repeat_interleave(repeat_times, dim=1)  # 高效控制复制
        offset = offset + flow  # 完成offset融合

        mask = torch.sigmoid(mask)

        offset_mask = torch.cat([offset, mask], dim=1)

        x_npu = x.contiguous()
        offset_mask_npu = offset_mask.contiguous()

        stride_npu = [1, self.stride, self.stride, 1]
        padding_npu = [self.padding, self.padding, self.padding, self.padding]
        dilation_npu = [1, self.dilation, self.dilation, 1]

        out, _ = torch_npu.npu_deformable_conv2d(
            x_npu,
            self.weight,
            offset_mask_npu,
            bias=self.bias,
            kernel_size=[self.kernel_size, self.kernel_size],
            stride=stride_npu,
            padding=padding_npu,
            dilation=dilation_npu,
            groups=self.groups,
            deformable_groups=self.deform_groups,
            modulated=True
        )

        return out