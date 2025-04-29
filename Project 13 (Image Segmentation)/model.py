import torch
import numpy as np
from torch import nn, math, os
from torch.nn import functional as F

class double_conv_drop(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))

    def forward(self, x):
        return self.conv(x)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(SpatialAttentionBlock, self).__init__()

        self.conv_e_0 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.conv_e_1 = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=padding)

        self.conv_d_0 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.conv_d_1 = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, e, d):
        # Calculate spatial attention map
        avg_pool_e, avg_pool_d = torch.mean(e, dim=1, keepdim=True), torch.mean(d, dim=1, keepdim=True)
        max_pool_e, max_pool_d = torch.max(e, dim=1, keepdim=True)[0], torch.max(d, dim=1, keepdim=True)[0]
        conv_0_e, conv_0_d = self.conv_e_0(e), self.conv_d_0(d)
        conv_1_e, conv_1_d = self.conv_e_1(torch.cat((avg_pool_e, max_pool_e, conv_0_e), dim=1)), self.conv_d_1(torch.cat((avg_pool_d, max_pool_d, conv_0_d), dim=1))
        return F.sigmoid(torch.add(conv_1_e, conv_1_d))

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.ratio = in_channels//16
        self.ca_encoder_shared = nn.Conv2d(in_channels, self.ratio, kernel_size=1)
        self.ca_decoder_shared = nn.Conv2d(in_channels, self.ratio, kernel_size=1)
        self.c_all = nn.Conv2d(self.ratio, in_channels, kernel_size=1)

    def forward(self, e, d):
        e_avg, e_max = self.ca_encoder_shared(F.adaptive_avg_pool2d(e, (1,1))), self.ca_encoder_shared(F.adaptive_max_pool2d(e, (1,1)))
        d_avg, d_max = self.ca_decoder_shared(F.adaptive_avg_pool2d(d, (1,1))), self.ca_decoder_shared(F.adaptive_max_pool2d(d, (1,1)))

        e_concat = torch.add(e_avg, e_max)
        d_concat = torch.add(d_avg, d_max)

        all_concat = self.c_all(torch.add(e_concat, d_concat))
        return F.sigmoid(all_concat)

class SC_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.spatial = SpatialAttentionBlock(in_channels, kernel_size=kernel_size, padding=padding)
        self.channel = ChannelAttentionBlock(in_channels)

    def forward(self, e, d):
        spatial_out = self.spatial(e, d)
        channel_out = self.channel(e, d)
        refined_e = e * channel_out * spatial_out
        return refined_e

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.up(x)
        return x
    

class Unet(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)

        # Contracting path.
        self.down_convolution_1 = double_conv_drop(3, 32, dropout=dropout)
        self.down_convolution_2 = double_conv_drop(32, 64, dropout=dropout)
        self.down_convolution_3 = double_conv_drop(64, 128, dropout=dropout)
        self.down_convolution_4 = double_conv_drop(128, 256, dropout=dropout)


        # Bottle neck.
        self.fc = double_conv_drop(256, 512, dropout=dropout)

        # Expanding path.
        self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_double_conv_1 = double_conv_drop(512, 256, dropout=dropout)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_double_conv_2 = double_conv_drop(256,128, dropout=dropout)

        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_double_conv_3 = double_conv_drop(128, 64, dropout=dropout)

        self.up_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_double_conv_4 = double_conv_drop(64, 32, dropout=dropout)

        # Output
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        # Contracting layer
        x_1 = self.down_convolution_1(x)
        x = self.max_pool(x_1)

        x_2 = self.down_convolution_2(x)
        x = self.max_pool(x_2)

        x_3 = self.down_convolution_3(x)
        x = self.max_pool(x_3)

        x_4 = self.down_convolution_4(x)
        x = self.max_pool(x_4)

        # Bottle neck layer.

        x = self.fc(x)

        # Expanding layer.
        x = self.up_conv1(x)
        x = self.up_double_conv_1(torch.cat((x, x_4), dim=1))

        x = self.up_conv2(x)
        x = self.up_double_conv_2(torch.cat((x, x_3), dim=1))

        x = self.up_conv3(x)
        x = self.up_double_conv_3(torch.cat((x, x_2), dim=1))

        x = self.up_conv4(x)
        x = self.up_double_conv_4(torch.cat((x, x_1), dim=1))

        # Output
        o = self.out(x)

        return o


class AttentionSpatialChannelUnet(nn.Module):
    def __init__(self, num_classes, kernel_size=3, padding=1, dropout=0.2, log_2=5):
        super().__init__()

        self.n_filters = [2**i for i in range(log_2, log_2+5)]

        self.pool = nn.MaxPool2d(2)

        # Contracting block
        self.down_block_1 = double_conv_drop(3, self.n_filters[0], dropout=dropout)
        self.down_block_2 = double_conv_drop(self.n_filters[0], self.n_filters[1], dropout=dropout)
        self.down_block_3 = double_conv_drop(self.n_filters[1], self.n_filters[2], dropout=dropout)
        self.down_block_4 = double_conv_drop(self.n_filters[2], self.n_filters[3], dropout=dropout)

        # Bride block
        self.bridge = double_conv_drop(self.n_filters[3], self.n_filters[4], dropout=0.)
        # Expanding block

        self.up_block_1 = UpConv(self.n_filters[4], self.n_filters[3])
        self.attention_1 = SC_Attention(self.n_filters[3], kernel_size=kernel_size, padding=padding)
        self.up_double_conv_1 = double_conv_drop(self.n_filters[4], self.n_filters[3],dropout=dropout)

        self.up_block_2 = UpConv(self.n_filters[3], self.n_filters[2])
        self.attention_2 = SC_Attention(self.n_filters[2], kernel_size=kernel_size, padding=padding)
        self.up_double_conv_2 = double_conv_drop(self.n_filters[3], self.n_filters[2], dropout=dropout)

        self.up_block_3 = UpConv(self.n_filters[2], self.n_filters[1])
        self.attention_3 = SC_Attention(self.n_filters[1], kernel_size=kernel_size, padding=padding)
        self.up_double_conv_3 = double_conv_drop(self.n_filters[2], self.n_filters[1], dropout=dropout)

        self.up_block_4 = UpConv(self.n_filters[1], self.n_filters[0])
        self.attention_4 = SC_Attention(self.n_filters[0], kernel_size=kernel_size, padding=padding)
        self.up_double_conv_4 = double_conv_drop(self.n_filters[1], self.n_filters[0], dropout=dropout)


        # Output
        self.out = nn.Conv2d(self.n_filters[0], num_classes, kernel_size=1)

    def forward(self, x):

        # Contracting layer
        x_1 = self.down_block_1(x)
        x = self.pool(x_1)

        x_2 = self.down_block_2(x)
        x = self.pool(x_2)

        x_3 = self.down_block_3(x)
        x = self.pool(x_3)

        x_4 = self.down_block_4(x)
        x = self.pool(x_4)

        # Bride layer

        bridge = self.bridge(x)

        # Expanding layer

        u_1 = self.up_block_1(bridge)
        a_1 = self.attention_1(x_4, u_1)
        u_1 = self.up_double_conv_1(torch.cat((u_1, a_1), dim=1))

        u_2 = self.up_block_2(u_1)
        a_2 = self.attention_2(x_3, u_2)
        u_2 = self.up_double_conv_2(torch.cat((u_2, a_2), dim=1))

        u_3 = self.up_block_3(u_2)
        a_3 = self.attention_3(x_2, u_3)
        u_3 = self.up_double_conv_3(torch.cat((u_3, a_3), dim=1))

        u_4 = self.up_block_4(u_3)
        a_4 = self.attention_4(x_1, u_4)
        u_4 = self.up_double_conv_4(torch.cat((u_4, a_4), dim=1))

        return self.out(u_4)

class AttentionSpatialChannelUnetWithDS(nn.Module):
    def __init__(self, num_classes, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Contracting block
        self.down_block_1 = double_conv_drop(3, 32, dropout=dropout)
        self.down_block_2 = double_conv_drop(32, 64, dropout=dropout)
        self.down_block_3 = double_conv_drop(64, 128, dropout=dropout)
        self.down_block_4 = double_conv_drop(128, 256, dropout=dropout)

        # Bride block
        self.bridge = double_conv_drop(256, 512, dropout=dropout)

        # Expanding block

        self.up_block_1 = UpConv(512, 256)
        self.attention_1 = SC_Attention(256, kernel_size=kernel_size, padding=padding)
        self.up_double_conv_1 = double_conv_drop(512, 256,dropout=dropout)

        self.up_block_2 = UpConv(256, 128)
        self.attention_2 = SC_Attention(128, kernel_size=kernel_size, padding=padding)
        self.up_double_conv_2 = double_conv_drop(256, 128, dropout=dropout)

        self.up_block_3 = UpConv(128, 64)
        self.attention_3 = SC_Attention(64, kernel_size=kernel_size, padding=padding)
        self.up_double_conv_3 = double_conv_drop(128, 64, dropout=dropout)

        self.up_block_4 = UpConv(64, 32)
        self.attention_4 = SC_Attention(32, kernel_size=kernel_size, padding=padding)
        self.up_double_conv_4 = double_conv_drop(64, 32, dropout=dropout)

        # Fusion Block
        self.ds_block_1 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8, padding=0)
        self.ds_block_2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)
        self.ds_block_3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.ds_conv = double_conv_drop(32+64+128+256, 32, dropout=dropout)



        # Output
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        # Contracting layer
        x_1 = self.down_block_1(x)
        x = self.pool(x_1)

        x_2 = self.down_block_2(x)
        x = self.pool(x_2)

        x_3 = self.down_block_3(x)
        x = self.pool(x_3)

        x_4 = self.down_block_4(x)
        x = self.pool(x_4)

        # Bride layer

        bridge = self.bridge(x)

        # Expanding layer

        u_1 = self.up_block_1(bridge)
        a_1 = self.attention_1(x_4, u_1)
        u_1 = self.up_double_conv_1(torch.cat((u_1, a_1), dim=1))

        u_2 = self.up_block_2(u_1)
        a_2 = self.attention_2(x_3, u_2)
        u_2 = self.up_double_conv_2(torch.cat((u_2, a_2), dim=1))

        u_3 = self.up_block_3(u_2)
        a_3 = self.attention_3(x_2, u_3)
        u_3 = self.up_double_conv_3(torch.cat((u_3, a_3), dim=1))

        u_4 = self.up_block_4(u_3)
        a_4 = self.attention_4(x_1, u_4)
        u_4 = self.up_double_conv_4(torch.cat((u_4, a_4), dim=1))

        # Fusion layer

        u_1_up = self.ds_block_1(u_1)
        u_2_up = self.ds_block_2(u_2)
        u_3_up = self.ds_block_3(u_3)

        fusion_block = torch.cat((u_4, u_3_up, u_2_up, u_1_up), dim=1)
        fusion_block = self.ds_conv(fusion_block)


        return self.out(fusion_block)