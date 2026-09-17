import torch
import torch.nn as nn
import torch.nn.functional as F

from selector.model_selector import register_model


def keras_he_normal_(weight):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    std = (2.0 / fan_in) ** 0.5 / 0.8796256610342398
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class KerasSameConvTranspose2d(nn.ConvTranspose2d):
    """Keras SAME for the SQD decoder's 3x3 kernel and stride 2."""
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        return super().forward(x)[..., :-1, :-1]


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        # Match Keras BatchNormalization defaults used by DeepPhaseUnwrap.
        # PyTorch's momentum has the inverse meaning of Keras' momentum:
        # Keras 0.99 -> PyTorch 1 - 0.99 = 0.01.
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        keras_he_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

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
        keras_he_normal_(self.conv_h.weight)
        keras_he_normal_(self.conv_v.weight)
        nn.init.zeros_(self.conv_h.bias)
        nn.init.zeros_(self.conv_v.bias)

        # Decoder
        self.u5 = KerasSameConvTranspose2d(128, 128)
        self.c5 = ConvBlock(256, 128)

        self.u6 = KerasSameConvTranspose2d(128, 64)
        self.c6 = ConvBlock(128, 64)

        self.u7 = KerasSameConvTranspose2d(64, 32)
        self.c7 = ConvBlock(64, 32)

        self.u8 = KerasSameConvTranspose2d(32, 16)
        self.c8 = ConvBlock(32, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        for layer in (self.u5, self.u6, self.u7, self.u8, self.out_conv):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for lstm in (self.lstm_h, self.lstm_v):
            for suffix in ('', '_reverse'):
                nn.init.xavier_uniform_(getattr(lstm, 'weight_ih_l0' + suffix))
                nn.init.orthogonal_(getattr(lstm, 'weight_hh_l0' + suffix))
                bias_ih = getattr(lstm, 'bias_ih_l0' + suffix)
                bias_hh = getattr(lstm, 'bias_hh_l0' + suffix)
                nn.init.zeros_(bias_ih)
                nn.init.zeros_(bias_hh)
                with torch.no_grad():
                    bias_ih[lstm.hidden_size:2 * lstm.hidden_size].fill_(1)
                # Keras has one trainable bias; keep the redundant PyTorch bias zero.
                bias_hh.requires_grad_(False)

    @staticmethod
    def _to_vertical_sequence(x):
        """Match Keras Permute((2, 1, 3)) followed by spatial flattening."""
        B, C, H, W = x.shape
        return x.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)

    @staticmethod
    def _from_vertical_sequence(x, height, width):
        """Restore a width-major vertical sequence to NCHW layout."""
        B, _, C = x.shape
        return x.view(B, width, height, C).permute(0, 3, 2, 1).contiguous()

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
        x_v = self._to_vertical_sequence(p4)
        h_v, _ = self.lstm_v(x_v)
        H_v = self._from_vertical_sequence(h_v, H, W)
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
