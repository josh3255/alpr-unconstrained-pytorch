import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_size = (3,3)
        self.dimensionality_factor = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(32, 32)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.residual_block2 = ResidualBlock(64, 64)
        self.residual_block3 = ResidualBlock(64, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.residual_block4 = ResidualBlock(64, 64)
        self.residual_block5 = ResidualBlock(64, 64)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.detection_block = DetectionBlock()

    def forward(self, input):
        # input : B X 3 X 208 X 208
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.max_pool1(output)
        # output : B X 16 X 102 X 102

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.residual_block1(output)
        output = self.max_pool2(output)
        # output : B X 32 X 52 X 52

        output = self.conv4(output)
        output = self.relu4(output)
        output = self.residual_block2(output)
        output = self.residual_block3(output)
        output = self.max_pool3(output)
        # output : B X 64 X 26 X 26

        output = self.residual_block4(output)
        output = self.residual_block5(output)
        output = self.max_pool4(output)
        # output : B X 64 X 13 X 13
        output = self.detection_block(output)

class DetectionBlock(nn.Module):
    def __init__(self):
        super(DetectionBlock, self).__init__()
        self.kernel_size = (3, 3)
        self.source_size = (208, 208)

        self.object_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=self.kernel_size, padding=1)
        self.affine_layer = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=self.kernel_size, padding=1)

        self.onebyone_conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(1,1))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        object_probability_map = self.object_layer(input)
        affine_parameter_map = self.affine_layer(input)

        _b, _, _, _ = object_probability_map.shape

        # object_probability_map = object_map[:, 0, :, :]
        # non_object_probability_map = object_map[:, 1, :, :]

        object_probability_map = nnf.interpolate(object_probability_map, size=self.source_size, mode='bicubic', align_corners=False)
        affine_parameter_map = nnf.interpolate(affine_parameter_map, size=self.source_size, mode='bicubic', align_corners=False)

        object_probability_map = object_probability_map.view([object_probability_map.shape[0], object_probability_map.shape[1], object_probability_map.shape[2] * object_probability_map.shape[3]])
        object_probability_map = self.softmax(object_probability_map)
        affine_parameter_map = self.onebyone_conv(affine_parameter_map)
        object_probability_map = object_probability_map.view([object_probability_map.shape[0], object_probability_map.shape[1], self.source_size[0], self.source_size[1]])

        output_map = torch.cat((object_probability_map, affine_parameter_map), dim=1)
        # output_map : B X 8 X 208 X 208
        '''
            B X 0~1 X 208 X 208 : object probability(and non obj)
            B X 2~7 X 208 X 208 : affine transpose parameter
        '''
        return output_map

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + residual
        x = self.relu(x)
        return x
