import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class ConvBnActivation2d(nn.Module):
    def __init__(self, in_channels, out_channels, activation='tanh', kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnActivation2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Activation function must be either 'sigmoid' or 'tanh'.")

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))




class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)




class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x




class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)




class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x




class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels, out_channels)
        
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

# Residual block
class BasicBlock(nn.Module):
    def __init__(self, channel_num):
        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        ) 
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )

        self.relu = nn.ReLU()
        self.sobelconv = Sobelxy(channels=channel_num)
    
    def forward(self, x):
        residual = x
        x = self.sobelconv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + self.sobelconv(residual)
        out = self.relu(x)
        return out

