import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1, base_features=16, drop_out_prob=0.5):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.base_features = base_features
        self.Conv1 = ConvBlock(img_ch, self.base_features)
        self.Conv2 = ConvBlock(self.base_features, self.base_features*2)
        self.Conv3 = ConvBlock(self.base_features*2, self.base_features*4)
        self.Conv4 = ConvBlock(self.base_features*4, self.base_features*8)
        self.Conv5 = ConvBlock(self.base_features*8, self.base_features*16)

        self.Up5 = UpConv(self.base_features*16, self.base_features*8)
        self.Att5 = AttentionBlock(F_g=self.base_features*8, F_l=self.base_features*8, n_coefficients=self.base_features*4)
        self.UpConv5 = ConvBlock(self.base_features*16, self.base_features*8)

        self.Up4 = UpConv(self.base_features*8, self.base_features*4)
        self.Att4 = AttentionBlock(F_g=self.base_features*4, F_l=self.base_features*4, n_coefficients=self.base_features*2)
        self.UpConv4 = ConvBlock(self.base_features*8, self.base_features*4)

        self.Up3 = UpConv(self.base_features*4, self.base_features*2)
        self.Att3 = AttentionBlock(F_g=self.base_features*2, F_l=self.base_features*2, n_coefficients=self.base_features)
        self.UpConv3 = ConvBlock(self.base_features*4, self.base_features*2)

        self.Up2 = UpConv(self.base_features*2, self.base_features)
        self.Att2 = AttentionBlock(F_g=self.base_features, F_l=self.base_features, n_coefficients=self.base_features//2)
        self.UpConv2 = ConvBlock(self.base_features*2, self.base_features)
        self.dropout = torch.nn.Dropout(p=drop_out_prob)

        self.Conv = nn.Conv2d(self.base_features, output_ch, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)
        d3 = self.dropout(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        d2 = self.dropout(d2)

        out = self.Conv(d2)

        return out 