import torch
import torch.nn as nn
from torchvision import models


class VGG16_base(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers = self.get_backbone(batch_norm)

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
                edge_list = conv_list
                break

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)

        return node_layers, edge_layers
    

class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)


class VGG16(VGG16_base):
    def __init__(self):
        super(VGG16, self).__init__(False)


class NoBackbone(nn.Module):
    def __init__(self, batch_norm=True):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError



class SuperPointNet(nn.Module):
    def __init__(self, weights_path="./pretrained_weight/superpoint_v1.pth", cuda=True):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        #
        if weights_path != None:
            self.load_weight(weights_path, cuda)
        #
        self.get_backbone()
        #
    def get_backbone(self, ):
        self.feat_layer = nn.Sequential(self.conv1a,
                                        self.relu,
                                        self.conv1b,
                                        self.relu,
                                        self.pool,
                                        self.conv2a,
                                        self.relu,
                                        self.conv2b,
                                        self.relu,
                                        self.pool,
                                        self.conv3a,
                                        self.relu,
                                        self.conv3b,
                                        self.relu,
                                        self.pool,
                                        self.conv4a,
                                        self.relu,
                                        self.conv4b,
                                        self.relu)
        #Detector Head
        self.det_layer = nn.Sequential(self.convPa,
                                       self.relu,
                                       self.convPb)
        # Descriptor Head.
        self.des_layer = nn.Sequential(self.convDa,
                                       self.relu,
                                       self.convDb)

    def load_weight(self, weights_path, cuda=True):
        if cuda:
            # Train on GPU, deploy on GPU.
            self.load_state_dict(torch.load(weights_path))
            self.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))