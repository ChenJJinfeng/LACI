import torch
from torch import nn
import pdb


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3, padding=1, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear",align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        # print(f'x5_up: {x5_up.shape}, x4: {x4.shape}')
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)

    def forward(self, input):
        features = self.encoder(input)
        out_seg, x8_up = self.decoder(features)
        return out_seg # 4, 16, 112, 112, 80

import torch.nn.functional as F
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入输出通道不同，则使用1x1卷积进行匹配
        self.match_channels = nn.Conv1d(in_channels, out_channels, 1, stride=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果通道不匹配，调整residual
        if self.match_channels:
            residual = self.match_channels(residual)

        out += residual  # 跳跃连接
        out = self.relu(out)
        return out

class FeatureProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super(FeatureProcessor, self).__init__()
        self.resnet_block1 = ResNetBlock(in_channels, out_channels)
        self.resnet_block2 = ResNetBlock(out_channels, out_channels)
        self.fc = nn.Linear(4096, 7 * 7 * 5)
        self.relu = nn.ReLU()


    def forward(self, diffs):
        x = self.resnet_block1(diffs)
        x = self.resnet_block2(x)
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), 7, 7, 5)
        return x


#############代码1
class VNet_CF(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_CF, self).__init__()

        # self.labelbs = labeled
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)

        output_shape = (7, 7, 5)
        self.processor1 = FeatureProcessor(in_channels=256, out_channels=256, output_shape=output_shape)
        self.processor2 = FeatureProcessor(in_channels=256, out_channels=256, output_shape=output_shape)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))  # 控制特征交互的权重

    def forward(self, image_input, text_features=None):
        batch_size = image_input.size(0)
        half_batch_size = batch_size // 2

        # Step 1: 正常输出
        encoder_features = self.encoder(image_input)
        output, _ = self.decoder(encoder_features)

        last_features = encoder_features[-1]
        # print(encoder_features[-1].shape)        ## torch.Size([4, 256, 7, 7, 5])       #### torch.Size([4, 256, 17, 17, 12])

        # Step 2: 反事实输出
        if text_features is not None:
            ## Step 2.1: 计算差异性
            diffs = []
            for i in range(half_batch_size):
                diff = torch.abs(text_features[i] - text_features[half_batch_size + i])  ## torch.sie([256, 4096])
                diffs.append(diff)
            diffs = torch.stack(diffs, dim=0)  ## ## torch.sie([2, 256, 4096])

            ## Step 2.2: 计算差异方向
            T_l2u = self.processor1(diffs)   # labeled → unlabeled
            T_u2l = self.processor2(diffs)   # unlabeled → labeled
            T = torch.cat((T_u2l, T_l2u))

            ## Step 2.3: 反事实特征，标记数据的未标记数据特征互相生成
            last_features_1 = torch.cat([last_features[half_batch_size:], last_features[:half_batch_size]])* T #do-operation，实现未标记数据向标记数据的转换，标记数据向未标记数据的转换，如果实现，说明已经学习到了因果差异，因此原操作，可以充当虚假差异，不改变原有的输出结果
            encoder_features[-1] = last_features_1
            ## Step 2.4: 将互相生成的特征交换为标记数据和无标记数据
            counterfactual_output, _ = self.decoder1(encoder_features)

            return output, counterfactual_output
        else:
            return output

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, L]
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class FeatureProcessorC(nn.Module):
    """
    Input:
        diffs: [B_half, 256, 4096]
    Output:
        directional tensor: [B_half, 256, 7, 7, 5]
    """
    def __init__(self, in_channels=256, token_dim=4096, out_shape=(7, 7, 5)):
        super().__init__()
        self.out_shape = out_shape
        self.resblock1 = ResBlock1D(in_channels)
        self.resblock2 = ResBlock1D(in_channels)
        self.fc = nn.Linear(token_dim, out_shape[0] * out_shape[1] * out_shape[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, diffs):
        # diffs: [B_half, 256, 4096]
        x = self.resblock1(diffs)
        x = self.resblock2(x)
        x = self.fc(x)                         # [B_half, 256, 245]
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), *self.out_shape)  # [B_half, 256, 7, 7, 5]
        return x


# === Begin new helper classes ===
class TextSpatialCrossAttention(nn.Module):
    """
    Fuse precomputed text embeddings [B, 256, 4096] with bottleneck image features [B, 256, 7, 7, 5]
    using cross-attention. Text tokens attend to spatial image tokens after projection.
    """
    def __init__(self, img_channels=256, text_dim=4096, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, img_channels)
        self.img_norm = nn.LayerNorm(img_channels)
        self.txt_norm = nn.LayerNorm(img_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=img_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(img_channels, img_channels),
            nn.ReLU(inplace=True),
            nn.Linear(img_channels, img_channels),
        )

    def forward(self, img_feat, text_feat):
        # img_feat: [B, 256, 7, 7, 5]
        # text_feat: [B, 256, 4096]
        B, C, H, W, D = img_feat.shape
        img_tokens = img_feat.view(B, C, -1).permute(0, 2, 1)   # [B, 245, 256]
        txt_tokens = self.text_proj(text_feat)                   # [B, 256, 256]

        q = self.img_norm(img_tokens)
        k = self.txt_norm(txt_tokens)
        v = self.txt_norm(txt_tokens)
        fused_tokens, _ = self.attn(q, k, v)
        fused_tokens = img_tokens + self.out_proj(fused_tokens)
        fused_feat = fused_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W, D)
        return fused_feat


class DecoderV2(nn.Module):
    """
    VNet-style decoder with four upsampling stages and one refinement stage.
    Channel schedule is 128 / 128 / 64 / 32 / n_classes for n_filters=16.
    """
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super().__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        upsampling = UpsamplingDeconvBlock

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)   # 256 -> 128
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)       # 128
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)       # 128 -> 64

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)     # 64
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)     # 64 -> 32

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)     # 32
        self.block_eight_up = upsampling(n_filters * 2, n_filters * 2, normalization=normalization)     # keep 32 for refinement

        self.block_nine = convBlock(1, n_filters * 2, n_filters * 2, normalization=normalization)      # refinement at 32
        self.out_conv = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        # x1 has 16 channels, project it to 32 channels before fusion to match the required schedule
        if x1.shape[1] != x8_up.shape[1]:
            x1_proj = F.pad(x1, (0, 0, 0, 0, 0, 0, 0, x8_up.shape[1] - x1.shape[1]))
        else:
            x1_proj = x1
        x8_up = x8_up + x1_proj

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up
# === End new helper classes ===


class VNet_CFC(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        n_filters=16,
        normalization='batchnorm',
        has_dropout=True,
        has_residual=False,
    ):
        super().__init__()

        # shared encoder
        self.encoder = Encoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )

        # two independent decoders
        self.decoder_std = Decoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )
        self.decoder_cf = Decoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )

        # directional processors
        self.processor_l2u = FeatureProcessorC(in_channels=256, token_dim=4096, out_shape=(7, 7, 5))
        self.processor_u2l = FeatureProcessorC(in_channels=256, token_dim=4096, out_shape=(7, 7, 5))

        # learnable scale for counterfactual perturbation
        self.beta = nn.Parameter(torch.tensor(0.1))

    def _build_counterfactual_bottom(self, last_features, text_features):
        """
        last_features: [B, 256, 7, 7, 5]
        text_features: [B, 256, 4096]
        Assumption: batch is [labeled..., unlabeled...], and B is even.
        """
        B = last_features.size(0)
        assert B % 2 == 0, "Batch size must be even for counterfactual pairing."
        half = B // 2

        feat_l = last_features[:half]     # labeled half
        feat_u = last_features[half:]     # unlabeled half

        txt_l = text_features[:half]
        txt_u = text_features[half:]

        # element-wise absolute differences
        diffs = torch.abs(txt_l - txt_u)  # [half, 256, 4096]

        # directional tensors
        T_l2u = self.processor_l2u(diffs)   # [half, 256, 7, 7, 5]
        T_u2l = self.processor_u2l(diffs)   # [half, 256, 7, 7, 5]

        # residual-style counterfactual perturbation (much stabler than direct multiplication)
        cf_l = feat_l + self.beta * (feat_u * T_u2l)   # unlabeled -> labeled direction
        cf_u = feat_u + self.beta * (feat_l * T_l2u)   # labeled -> unlabeled direction

        cf_bottom = torch.cat([cf_l, cf_u], dim=0)
        return cf_bottom

    def forward(self, image_input, text_features=None):
        # shared encoder
        encoder_features = self.encoder(image_input)

        # standard prediction
        std_features = list(encoder_features)  # shallow copy is enough since we replace only last stage
        std_output, _ = self.decoder_std(std_features)

        if text_features is None:
            return std_output

        # allow text_features to be either a list of [256, 4096] tensors
        # or a single tensor of shape [B, 256, 4096]
        if isinstance(text_features, list):
            text_features = torch.stack(text_features, dim=0)
        elif not torch.is_tensor(text_features):
            raise TypeError(f"text_features should be a list or tensor, got {type(text_features)}")

        assert text_features.dim() == 3, \
            f"text_features should be [B, 256, 4096], got {text_features.shape}"
        assert text_features.size(0) == image_input.size(0), \
            f"Batch mismatch: image batch {image_input.size(0)}, text batch {text_features.size(0)}"

        text_features = text_features.to(image_input.device).float()

        cf_features = list(encoder_features)
        last_features = cf_features[-1]   # [B, 256, 7, 7, 5]

        cf_bottom = self._build_counterfactual_bottom(last_features, text_features)
        cf_features[-1] = cf_bottom

        cf_output, _ = self.decoder_cf(cf_features)

        return std_output, cf_output


# === Begin new network class ===
class VNet_CFC_CA(nn.Module):
    """
    VNet encoder + cross-attention text-image fusion at the bottleneck +
    standard / counterfactual dual decoders.

    Encoder:
      - image encoder follows VNet with channels 16/32/64/128/256
      - text embeddings are precomputed with shape [B, 256, 4096]
      - cross-attention fuses text and spatial image features at the bottleneck

    Decoder:
      - text difference features are computed by element-wise absolute differences
      - two lightweight processors map them to directional tensors [B_half, 256, 7, 7, 5]
      - two decoders share the encoder but use independent parameters
      - each decoder is VNet-style with channels 128/128/64/32/n_classes
    """
    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        n_filters=16,
        normalization='batchnorm',
        has_dropout=True,
        has_residual=False,
        num_heads=8,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )

        self.cross_attn_fusion = TextSpatialCrossAttention(
            img_channels=n_filters * 16,
            text_dim=4096,
            num_heads=num_heads,
            dropout=0.1,
        )

        self.decoder_std = DecoderV2(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )
        self.decoder_cf = DecoderV2(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )

        self.processor_l2u = FeatureProcessorC(in_channels=256, token_dim=4096, out_shape=(7, 7, 5))
        self.processor_u2l = FeatureProcessorC(in_channels=256, token_dim=4096, out_shape=(7, 7, 5))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def _prepare_text_features(self, image_input, text_features):
        if isinstance(text_features, list):
            text_features = torch.stack(text_features, dim=0)
        elif not torch.is_tensor(text_features):
            raise TypeError(f"text_features should be a list or tensor, got {type(text_features)}")

        assert text_features.dim() == 3, \
            f"text_features should be [B, 256, 4096], got {text_features.shape}"
        assert text_features.size(0) == image_input.size(0), \
            f"Batch mismatch: image batch {image_input.size(0)}, text batch {text_features.size(0)}"
        return text_features.to(image_input.device).float()

    def _build_counterfactual_bottom(self, fused_last_features, text_features):
        B = fused_last_features.size(0)
        assert B % 2 == 0, "Batch size must be even for counterfactual pairing."
        assert text_features.size(0) == B, f"text_features batch {text_features.size(0)} does not match feature batch {B}"
        half = B // 2

        feat_l = fused_last_features[:half]
        feat_u = fused_last_features[half:]
        txt_l = text_features[:half]
        txt_u = text_features[half:]

        diffs = torch.abs(txt_l - txt_u)  # [half, 256, 4096]
        T_l2u = self.processor_l2u(diffs)
        T_u2l = self.processor_u2l(diffs)

        cf_l = feat_l + self.beta * (feat_u * T_u2l)
        cf_u = feat_u + self.beta * (feat_l * T_l2u)
        cf_bottom = torch.cat([cf_l, cf_u], dim=0)
        return cf_bottom

    def forward(self, image_input, text_features=None):
        encoder_features = self.encoder(image_input)

        if text_features is None:
            std_features = list(encoder_features)
            std_output, _ = self.decoder_std(std_features)
            return std_output

        text_features = self._prepare_text_features(image_input, text_features)

        fused_features = list(encoder_features)
        fused_last = self.cross_attn_fusion(fused_features[-1], text_features)
        fused_features[-1] = fused_last

        std_features = list(fused_features)
        std_output, _ = self.decoder_std(std_features)

        cf_features = list(fused_features)
        cf_features[-1] = self._build_counterfactual_bottom(fused_last, text_features)
        cf_output, _ = self.decoder_cf(cf_features)

        return std_output, cf_output
# === End new network class ===



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #import pdb; pdb.set_trace()
