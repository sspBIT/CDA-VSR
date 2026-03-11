import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, make_layer,flow_warp
from mmcv.ops import ModulatedDeformConv2d

@ARCH_REGISTRY.register()
class CDAVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=3,
                 num_reconstruct_block_I=24,
                 num_reconstruct_block_P=12,
                 center_frame_idx=None,
                 hr_in=False
                 ):

        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.hr_in = hr_in

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # deform align
        self.deform_align = DeformableAlignment(in_channels=num_feat, out_channels=num_feat)

        # gate fusion
        self.conv_weight_l = nn.Sequential(
            nn.Conv2d(1, num_feat, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, 1, 1, 1, 0))
        self.convs = nn.Sequential(
            nn.Conv2d(num_feat * 3, num_feat, 3, padding=1))

        # reconstruction
        self.reconstruction_I = make_layer(ResidualBlockNoBN, num_reconstruct_block_I, num_feat=num_feat)
        self.reconstruction_P = make_layer(ResidualBlockNoBN, num_reconstruct_block_P, num_feat=num_feat)
        self.conv_h = nn.Conv2d(num_feat, num_feat, 3, padding=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, mv, res, hidden_key=None, return_hs=False):
        b, t, c, h, w = x.size()
        # extract features for each frame
        feat = self.conv_first(x.view(-1, c, h, w))
        feat = self.feature_extraction(feat)
        feat = feat.view(b, t, -1, h, w)

        # align
        reconstruction_feats = []
        for fi in range(t):
            current_feat = feat[:, fi]
            x_i = x[:, fi]
            if fi == 0 and hidden_key is None:
                rec_feat = self.reconstruction_I(current_feat)
            else:
                current_mv = mv[:, fi]
                current_res = res[:, fi]
                feat_l, feat_h = hidden_key[0],hidden_key[1]
                feat_lh = torch.cat([feat_l, feat_h], dim=1)
                feat_warp_lh = mv_warp_avg_patch(feat_lh,current_mv,interpolation='nearest', padding_mode='border', align_corners=True)

                cond = torch.cat([feat_warp_lh, current_feat], dim=1)
                feat_warp_lh2 = self.deform_align(feat_lh,cond,current_mv)
                aligned_feat_l_weight = self.conv_weight_l(current_res)
                feat_warp_lh = feat_warp_lh2 * aligned_feat_l_weight + feat_warp_lh2
                aligned_feat = self.convs(torch.cat([feat_warp_lh,current_feat], dim=1))
                rec_feat = self.reconstruction_P(aligned_feat)


            reconstruction_feats.append(rec_feat)
            rec_feat1 = self.conv_h(rec_feat)
            hidden_key = (current_feat,rec_feat1,x_i)

        reconstruction_feats = torch.stack(reconstruction_feats, dim=1)

        # upsample
        out = reconstruction_feats.view(b*t, -1, h, w)
        out = self.pixel_shuffle(self.upconv1(out)).view(b, t, c, 4*h, 4*w)
        if self.hr_in:
            base = x
        else:
            base = F.interpolate(x.view(-1, c, h, w), scale_factor=4, mode='bilinear', align_corners=False).view(b, t, c, 4*h, 4*w)
        out += base
        if return_hs is True:
            return out, hidden_key
        else:
            return out

def mv_warp_avg_patch(x, mv, interpolation='nearest', padding_mode='zeros', align_corners=True):
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = mv.permute(0, 2, 3, 1)
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output

class DeformableAlignment(nn.Module):
    """Deformable alignment module.

    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=4,
                 bias=True, max_residue_magnitude=8):
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
            nn.Conv2d(out_channels * 3 +2, out_channels // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels // 4, offset_channels, 1, 1, 0)
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)
        
        self.dcn = ModulatedDeformConv2d(
            in_channels = 2 * in_channels,
            out_channels = 2 * out_channels,
            kernel_size = kernel_size,
            padding = padding,
            deform_groups=deform_groups,
            bias = False
        )

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat,flow],dim=1)
        out = self.conv_offset(extra_feat)
        offset1, offset2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset1, offset2), dim=1))
        offset = offset + flow.flip(1).repeat(1,offset.size(1) // 2, 1,1)

        # mask
        mask = torch.sigmoid(mask)
        out = self.dcn(x, offset, mask)
        return out


