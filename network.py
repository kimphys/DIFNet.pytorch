import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Convolution operation
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, bn=True, activation_type='leaky'):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                padding=pad)
        if bn:
            self.batchNorm = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1e-4)
        else:
            self.batchNorm = None
        
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'leaky':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'mish':
            self.activation = Mish()
        elif activation_type == 'swish':
            self.activation = Swish()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv2d(x)

        if self.batchNorm is not None:
            x = self.batchNorm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

class DIFNet(nn.Module):
    def __init__(self, in_channels=2,out_channels=1):
        super(DIFNet, self).__init__()

        self.C = in_channels

        ## For feature extraction
        ex_channels = [1,16]

        self.ex_conv2d = ConvLayer(in_channels=ex_channels[0],out_channels=ex_channels[1],kernel_size=3,bn=False)
        
        self.ex_resblock_conv2d_1 = ConvLayer(in_channels=ex_channels[1],out_channels=ex_channels[1],kernel_size=3,bn=True)
        self.ex_resblock_conv2d_2 = ConvLayer(in_channels=ex_channels[1],out_channels=ex_channels[1],kernel_size=3,bn=False,activation_type='')

        ## For featurn fusion and reconstruction
        f_channels = [16,out_channels]
        self.fusion = ConvLayer(in_channels=16 * self.C,out_channels=16,kernel_size=5,pad=2,bn=False,activation_type=False)
        
        self.f_resblock_conv2d_1 = ConvLayer(in_channels=f_channels[0],out_channels=f_channels[0],kernel_size=3,bn=True)
        self.f_resblock_conv2d_2 = ConvLayer(in_channels=f_channels[0],out_channels=f_channels[0],kernel_size=3,bn=False,activation_type='')

        self.f_conv2d = ConvLayer(in_channels=f_channels[0],out_channels=f_channels[1],kernel_size=3,bn=False,activation_type='')

    def forward(self, inputs):

        B, H, W = inputs.shape[0], inputs.shape[2], inputs.shape[3]

        out_list = []

        ## Feature extraction ##
        for i in range(self.C):
            out_list.append(self.forward_once(inputs[:,i,:,:].view(B,1,H,W)))

        ## Feature fusion ##
        x = torch.cat(out_list,dim=1)
        x1 = self.fusion(x)

        # 1st ResBlock
        x2 = self.f_resblock_conv2d_1(x1)
        x2 = self.f_resblock_conv2d_2(x2)

        x1 = x1 + x2

        # 2nd ResBlock
        x2 = self.f_resblock_conv2d_1(x1)
        x2 = self.f_resblock_conv2d_2(x2)

        x1 = x1 + x2

        # 3rd ResBlock
        x2 = self.f_resblock_conv2d_1(x1)
        x2 = self.f_resblock_conv2d_2(x2)

        x1 = x1 + x2

        x = self.f_conv2d(x1)

        return x
    
    def forward_once(self, x):

        x1 = self.ex_conv2d(x)

        # 1st ResBlock
        x2 = self.ex_resblock_conv2d_1(x1)
        x2 = self.ex_resblock_conv2d_2(x2)

        x1 = x1 + x2

        # 2nd ResBlock
        x2 = self.ex_resblock_conv2d_1(x1)
        x2 = self.ex_resblock_conv2d_2(x2)

        x = x1 + x2    

        return x

if __name__ == "__main__":
    a = torch.rand(2,3,20,20)
    b = torch.rand(2,3,20,20)

    inputs = torch.cat([a,b],dim=1)

    model = DIFNet()

    output = model(inputs)

    print(output.shape)