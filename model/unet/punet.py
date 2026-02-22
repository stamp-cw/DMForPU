import torch
import torch.nn as nn
import torch.nn.functional as F

from selector.model_selector import register_model


def crop_and_concat(net1, net2):
    """
    net1: skip connection feature map (smaller or equal)
    net2: upsampled feature map (larger)
    shape: [N, C, H, W]
    """
    _, _, h1, w1 = net1.shape
    _, _, h2, w2 = net2.shape

    dh = (h2 - h1) // 2
    dw = (w2 - w1) // 2

    net2_crop = net2[:, :, dh:dh + h1, dw:dw + w1]
    return torch.cat([net1, net2_crop], dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, drop_rate):
        super().__init__()
        padding = (
            (kernel_size[0] // 2) * dilation[0],
            (kernel_size[1] // 2) * dilation[1]
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        return self.block(x)

class DownsampleBlock(nn.Module):
    def __init__(self, ch, kernel_size, pool_size, dilation, drop_rate):
        super().__init__()
        padding = (
            (kernel_size[0] // 2) * dilation[0],
            (kernel_size[1] // 2) * dilation[1]
        )
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size,
                      stride=pool_size,
                      padding=padding,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        return self.block(x)

# class UpsampleBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, pool_size, drop_rate):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_ch, out_ch,
#                 kernel_size=kernel_size,
#                 stride=pool_size,
#                 padding=(
#                     kernel_size[0] // 2,
#                     kernel_size[1] // 2
#                 ),
#                 output_padding=(
#                     pool_size[0] - 1,
#                     pool_size[1] - 1
#                 ),
#                 bias=False
#             ),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=drop_rate)
#         )
#
#     def forward(self, x):
#         return self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size, drop_rate):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=pool_size,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        return self.block(x)

@register_model(name=['PUNet'])
class PUNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # ===== Fixed config (as you set) =====
        self.depths        = 5
        # self.depths        = 4
        self.filters_root  = 8
        # self.kernel_size   = (7, 1)
        # self.kernel_size   = (3, 1)
        self.kernel_size   = (3, 3)
        # self.pool_size     = (4, 1)
        self.pool_size     = (2, 2)
        self.dilation_rate = (1, 1)
        self.drop_rate     = 0.0
        # self.n_class       = 3
        self.n_class       = 1

        # ===== Input =====
        self.input_conv = ConvBlock(
            1,
            self.filters_root,
            self.kernel_size,
            self.dilation_rate,
            self.drop_rate
        )

        # ===== Encoder =====
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch = self.filters_root
        for d in range(self.depths):
            out_ch = self.filters_root * (2 ** d)

            self.down_convs.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    self.kernel_size,
                    self.dilation_rate,
                    self.drop_rate
                )
            )

            if d < self.depths - 1:
                self.down_samples.append(
                    DownsampleBlock(
                        out_ch,
                        self.kernel_size,
                        self.pool_size,
                        self.dilation_rate,
                        self.drop_rate
                    )
                )

            in_ch = out_ch

        # ===== Decoder =====
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for d in reversed(range(self.depths - 1)):
            filters = self.filters_root * (2 ** d)

            # self.up_samples.append(
            #     UpsampleBlock(
            #         filters * 2,
            #         filters,
            #         self.kernel_size,
            #         self.pool_size,
            #         self.drop_rate
            #     )
            # )

            self.up_samples.append(
                UpsampleBlock(
                    filters * 2,
                    filters,
                    self.pool_size,
                    self.drop_rate
                )
            )

            self.up_convs.append(
                ConvBlock(
                    filters * 2,
                    filters,
                    self.kernel_size,
                    self.dilation_rate,
                    self.drop_rate
                )
            )

        # ===== Output =====
        self.output_conv = nn.Conv2d(
            self.filters_root,
            self.n_class,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        skips = []

        x = self.input_conv(x)

        # ===== Encoder =====
        for d in range(self.depths):
            x = self.down_convs[d](x)
            skips.append(x)
            if d < self.depths - 1:
                x = self.down_samples[d](x)

        # bottleneck representation
        self.representation = skips[-1]

        # ===== Decoder =====
        for i, d in enumerate(reversed(range(self.depths - 1))):
            x = self.up_samples[i](x)
            x = crop_and_concat(skips[d], x)
            x = self.up_convs[i](x)

        logits = self.output_conv(x)
        return logits
