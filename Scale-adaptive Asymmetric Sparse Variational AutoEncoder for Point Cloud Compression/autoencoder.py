# Function: full codec network
# Code for Scale-adaptive Asymmetric Sparse Variational AutoEncoder for Point Cloud Compression
# Author: J. Chen, Y. Zhu, W. Huang, C. Lan and T. Zhao
# Institution: Fuzhou University
# Trimmer: T. Du
# Year: 2024
# Published in:J. Chen, Y. Zhu, W. Huang, C. Lan and T. Zhao, "Scale-Adaptive Asymmetric Sparse Variational AutoEncoder for Point Cloud Compression," in IEEE Transactions on Broadcasting, doi: 10.1109/TBC.2024.3437161.
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import MinkowskiEngine as ME

from data_utils import isin, istopk
from math import sqrt

import torch,os
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class XceptionResNet(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()


        self.relu = ME.MinkowskiReLU(inplace=True)


        self.bottleneck = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)

        self.conv = ME.MinkowskiChannelwiseConvolution(
            in_channels=channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )


    def forward(self, x):

        bx = self.bottleneck(x)

        out = self.conv(bx)
        out = x + out

        return out

def make_layer(block, block_layers, channels):

    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)

class Encoder(torch.nn.Module):
    def __init__(self, channels=[0,1,2,3,4,5]):
        super().__init__()

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.block01 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[2])


        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.block11 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[3]*2)
        self.block12 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[3]*2)


        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3]*2,
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.block21 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[4]*2)
        self.block22 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[4]*2)
        self.block23 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[4]*2)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4]*2,
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)


        self.relu = ME.MinkowskiReLU(inplace=True)

        self.map0 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.map1 = ME.MinkowskiConvolution(
            in_channels=channels[3]*2,
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.dp=ME.MinkowskiDropout()

    def forward(self, x):

        #x = self.dp(x)

        out0 = self.relu(self.down0(self.relu(self.conv0(x))))

        out0 = self.block01(out0)

        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = ME.cat(out1, self.map0(out0))
        out1 = self.block11(out1)
        out1 = self.block12(out1)




        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = ME.cat(out2, self.map1(out1))
        out2 = self.block21(out2)
        out21 = out2
        out2 = self.block22(out2)
        out2 = self.block23(out2)
        out2 = out2 + out21


        out2 = self.conv3(out2)

        return [out2, out1, out0]


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """
    def __init__(self, channels=[0,1,2,3]):
        super().__init__()

        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size= 3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.block01 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[1])
        self.block02 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[1])
        self.block03 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[1])


        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.block11 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[2])
        self.block12 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[2])



        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)


        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size= 3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)
        self.block21 = make_layer(
            block=XceptionResNet,
            block_layers=1,
            channels=channels[3])


        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size= 3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()



    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training: 
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else: 
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, nums_list, ground_truth_list, training=True):


        out0 = self.relu(self.conv0(self.relu(self.up0(x))))

        out0 = self.block01(out0)
        out01 = out0
        out0 = self.block02(out0)
        out0 = self.block03(out0)
        out0 = out0 + out01

        out_cls_0 = self.conv0_cls(out0)

        out = self.prune_voxel(out0, out_cls_0,
            nums_list[0], ground_truth_list[0], training)

        out1 = self.relu(self.conv1(self.relu(self.up1(out))))

        out1 = self.block11(out1)
        out1 = self.block12(out1)
        out_cls_1 = self.conv1_cls(out1)

        out = self.prune_voxel(out1, out_cls_1,
            nums_list[1], ground_truth_list[1], training)

        out2 = self.relu(self.conv2(self.relu(self.up2(out))))


        out2 = self.block21(out2)

        out_cls_2 = self.conv2_cls(out2)

        out = self.prune_voxel(out2, out_cls_2,
            nums_list[2], ground_truth_list[2], training)

        out_cls_list0 = [out_cls_0]
        out_cls_list1 = [out_cls_0, out_cls_1]
        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]

        return out_cls_list,out#out_cls_list1,out_cls_list2, out

