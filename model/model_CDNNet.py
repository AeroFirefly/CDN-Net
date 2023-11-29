import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class CDNNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, 
                 netdepth=4, scale_method='deconv', deep_supervision=False):
        """
        Initialize CDNNet.

        Parameters:
            num_classes : predicted classes account, default 1
            input_channels : input image channels account, 1 or 3
            block : CDNNet node type
            num_blocks : CDNNet node account
            nb_filter : CDNNet node output channels account
            netdepth : CDNNet depth
            deep_supervision : CDNNet supervise mid-node of the shallowest layer

        Returns:
            None
        """
        super(CDNNet, self).__init__()
        self.netdepth = netdepth
        self.scale_method = scale_method.lower()
        self.deep_supervision = True if deep_supervision.lower() == 'true' else False
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)

        self._gen_scale_conv(netdepth, scale_method, nb_filter)
        
        self._gen_nested_conv(netdepth, block, input_channels, nb_filter, num_blocks)

        self._gen_parallel_1x1conv(netdepth, nb_filter)

        self.__setattr__("conv0_{}_final".format(netdepth-1), 
                         self._make_layer(block, nb_filter[0]*netdepth, nb_filter[0]))

        if self.deep_supervision:
            self._gen_multi_finalconv(netdepth, nb_filter, num_classes)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    
    def _gen_scale_conv(self, netdepth, scale_method, nb_filter):
        if scale_method.lower()=='biinterp':
            # up: bilinear-interp
            self.__setattr__("up", nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.__setattr__("up_4", nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
            self.__setattr__("up_8", nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
            self.__setattr__("up_16", nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))
            # down: bilinear-interp
            self.__setattr__("down", nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True))
        elif scale_method.lower()=='deconv':
            # up: deconv
            for i in range(1, netdepth):
                conv_module_up = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=int(pow(2,i+1)), stride=int(pow(2,i)), padding=int(pow(2,i-1)))
                self.__setattr__("up" if i==1 else "up_{}".format(int(pow(2,i))), conv_module_up)
            for i in range(1, netdepth):
                for j in range(netdepth-i):
                    conv_module_up = nn.ConvTranspose2d(nb_filter[i], nb_filter[i], kernel_size=4, stride=2, padding=1)
                    self.__setattr__("up_{}_{}".format(i,j), conv_module_up)
            # down: conv
            for i in range(netdepth-2):
                for j in range(1, netdepth-1-i):
                    conv_module_down = nn.Conv2d(nb_filter[i], nb_filter[i], kernel_size=4, stride=2, padding=1)
                    self.__setattr__("down_{}_{}".format(i,j), conv_module_down)
        else:
            raise Exception('wrong scale method')

    def _gen_nested_conv(self, netdepth, block, input_channels, nb_filter, num_blocks):
        for i in range(netdepth):
            for j in range(netdepth-i):
                if i==0 and j==0:
                    conv_module = self._make_layer(block, input_channels, nb_filter[0])
                elif j==0:
                    conv_module = self._make_layer(block, nb_filter[i-1], nb_filter[i], num_blocks[i-1])
                elif i==0:
                    conv_module = self._make_layer(block, nb_filter[i]*j + nb_filter[i+1], nb_filter[i])
                else:
                    conv_module = self._make_layer(block, nb_filter[i]*j + nb_filter[i+1]+ nb_filter[i-1], nb_filter[i], num_blocks[i-1])
                self.__setattr__("conv{}_{}".format(i,j), conv_module)
    
    def _gen_parallel_1x1conv(self, netdepth, nb_filter):
        for i in range(1, netdepth):
            conv_module_1x1 = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
            self.__setattr__("conv0_{}_1x1".format(i), conv_module_1x1)

    def _gen_multi_finalconv(self, netdepth, nb_filter, num_classes):
        for i in range(1, netdepth):
            conv_module_final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.__setattr__("final{}".format(i), conv_module_final)

    def forward(self, input):
        netdepth = self.netdepth
        scale_method = self.scale_method
        assert netdepth>=3 and netdepth<=5, "network depth must >=3 and <=5"

        x0_0 = self.conv0_0(input)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_1_0(x1_0)], 1)) if scale_method=='deconv'\
                else self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_2_0(x2_0),self.down_0_1(x0_1)], 1)) if scale_method=='deconv'\
                else self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_1_1(x1_1)], 1)) if scale_method=='deconv'\
                else self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        if netdepth>=4:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up_3_0(x3_0),self.down_1_1(x1_1)], 1)) if scale_method=='deconv'\
                else self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_2_1(x2_1),self.down_0_2(x0_2)], 1)) if scale_method=='deconv'\
                else self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_1_2(x1_2)], 1)) if scale_method=='deconv'\
                else self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        if netdepth>=5:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up_4_0(x4_0),self.down_2_1(x2_1)], 1)) if scale_method=='deconv'\
                else self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up_3_1(x3_1),self.down_1_2(x1_2)], 1)) if scale_method=='deconv'\
                else self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up_2_2(x2_2),self.down_0_3(x0_3)], 1)) if scale_method=='deconv'\
                else self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_1_3(x1_3)], 1)) if scale_method=='deconv'\
                else self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if netdepth==3:
            Final_x0_2 = self.conv0_2_final(torch.cat([self.up_4(self.conv0_2_1x1(x2_0)), 
                                                       self.up(self.conv0_1_1x1(x1_1)), x0_2], 1))

            if self.deep_supervision:
                output1 = self.final1(x0_1)
                output2 = self.final2(Final_x0_2)
                return [output1, output2]
            else:
                output = self.final(Final_x0_2)
                return output
        elif netdepth==4:
            Final_x0_3 = self.conv0_3_final(torch.cat([self.up_8(self.conv0_3_1x1(x3_0)), 
                                                       self.up_4(self.conv0_2_1x1(x2_1)), self.up(self.conv0_1_1x1(x1_2)), x0_3], 1))
            
            if self.deep_supervision:
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(Final_x0_3)
                return [output1, output2, output3]
            else:
                output = self.final(Final_x0_3)
                return output
        elif netdepth==5:
            Final_x0_4 = self.conv0_4_final(torch.cat([self.up_16(self.conv0_4_1x1(x4_0)), self.up_8(self.conv0_3_1x1(x3_1)), 
                                                       self.up_4 (self.conv0_2_1x1(x2_2)), self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

            if self.deep_supervision:
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(x0_3)
                output4 = self.final4(Final_x0_4)
                return [output1, output2, output3, output4]
            else:
                output = self.final(Final_x0_4)
                return output

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()

        # check loaded parameters and created model parameters
        msg = 'If you see this, your model does not fully load the ' + \
              'pre-trained weight. Please make sure ' + \
              'you have correctly specified --arch xxx ' + \
              'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
