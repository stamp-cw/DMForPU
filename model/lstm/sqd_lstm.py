import torch
import torch.nn as nn
import torch.nn.functional as F

from selector.model_selector import register_model


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

@register_model(name=['SqdLstmNet', 'JointConvSQDLSTMNet'])
class JointConvSQDLSTMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_channels=1
        # Encoder
        self.c1 = ConvBlock(input_channels, 16)
        self.p1 = nn.AvgPool2d(2)

        self.c2 = ConvBlock(16, 32)
        self.p2 = nn.AvgPool2d(2)

        self.c3 = ConvBlock(32, 64)
        self.p3 = nn.AvgPool2d(2)

        self.c4 = ConvBlock(64, 128)
        self.p4 = nn.AvgPool2d(2)

        # SQD-LSTM block
        self.lstm_h = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm_v = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=True)
        self.conv_h = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_v = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Decoder
        self.u5 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c5 = ConvBlock(256, 128)

        self.u6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c6 = ConvBlock(128, 64)

        self.u7 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c7 = ConvBlock(64, 32)

        self.u8 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c8 = ConvBlock(32, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # SQD-LSTM
        B, C, H, W = p4.size()

        # Horizontal LSTM
        x_h = p4.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        h_h, _ = self.lstm_h(x_h)                  # [B, H*W, 64]
        H_h = h_h.permute(0, 2, 1).view(B, 64, H, W)
        c_h = self.conv_h(H_h)

        # Vertical LSTM
        x_v = p4.permute(0, 2, 1, 3).contiguous().view(B, C, H*W).permute(0, 2, 1)
        h_v, _ = self.lstm_v(x_v)
        H_v = h_v.permute(0, 2, 1).view(B, 64, W, H).permute(0, 1, 3, 2)  # restore [B, 64, H, W]
        c_v = self.conv_v(H_v)

        H = torch.cat([c_h, c_v], dim=1)  # [B, 128, H, W]

        # Decoder
        u5 = self.u5(H)
        u5 = torch.cat([u5, c4], dim=1)
        c5 = self.c5(u5)

        u6 = self.u6(c5)
        u6 = torch.cat([u6, c3], dim=1)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c2], dim=1)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c1], dim=1)
        c8 = self.c8(u8)

        out = self.out_conv(c8)
        return out
