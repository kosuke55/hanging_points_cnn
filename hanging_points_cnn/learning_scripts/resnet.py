from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision.models.resnet import BasicBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        # torchvision.models.resnet.ResNet.__init__(self, block, layers, num_classes)
        super(ResNet, self).__init__(block, layers, num_classes)
        self.layer4[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        del self.fc


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    for p in model.conv1.parameters():
        p.requires_grad_(False)
    for p in model.bn1.parameters():
        p.requires_grad_(False)
    return model
