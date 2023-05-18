import torch.nn as nn

class MaskedCNN(nn.Conv2d):
    """
    Masked convolution as explained in the PixelCNN variant of
    Van den Oord et al, “Pixel Recurrent Neural Networks”, NeurIPS 2016
    It inherits from Conv2D (uses the same parameters, plus the option to select a mask including
    the center pixel or not, as described in class and in the Fig. 2 of the above paper)
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A': # zero out the weights in the bottom right part of the filters, including the center pixel.
            self.mask[:, :, height//2, width//2:] = 0
            self.mask[:, :, height//2+1:, :] = 0
        else: # zero out the weights in the bottom right part of the filters, excluding the center pixel.
            self.mask[:, :, height//2, width//2+1:] = 0
            self.mask[:, :, height//2+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    A PixelCNN variant you have to implement according to the instructions
    """

    def __init__(self):
        super(PixelCNN, self).__init__()
        
        # Block 1
        self.conv1 = MaskedCNN(in_channels=1, out_channels= 16, kernel_size=3, stride=1, dilation=3, padding_mode='reflect', mask_type='A')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU(negative_slope=0.001)
        
        # Block 2
        self.conv2 = MaskedCNN(in_channels=16, out_channels= 16, kernel_size=3, stride=1, dilation=3, padding_mode='reflect', mask_type='B')
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.LeakyReLU(negative_slope=0.001)
        
        # Block 3
        self.conv3 = MaskedCNN(in_channels=16, out_channels= 16, kernel_size=3, stride=1, dilation=3, padding_mode='reflect', mask_type='B')
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU(negative_slope=0.001)

        # 1x1 Convolution Layer
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=9, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # Block 1
        # print(x.shape)
        x = self.relu1(self.bn1(self.conv1(x)))
        # print(x.shape)
        # Block 2
        x = self.relu2(self.bn2(self.conv2(x)))
        # print(x.shape)
        # Block 3
        x = self.relu3(self.bn3(self.conv3(x)))
        # print(x.shape)
        # 1x1 Convolution Layer
        x = self.conv4(x)
        # print(x.shape)
        # Sigmoid Activation
        x = self.sigmoid(x)
        # print(x.shape)
        return x
