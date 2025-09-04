import torch
import torch.nn as nn
from model.backbone import ResNet, ConvNeXt, EfficientNetV2, SwinTransformer, VGG

class Model4Classifier(nn.Module):
    def __init__(self, **kwargs):
        super(Model4Classifier, self).__init__()
        self.class_num = kwargs['class_num']
        self.hidden_dim = kwargs['hidden_dim']
        self.backbone_name = kwargs['backbone']
        self.model_name = kwargs['model_name']
        print(f"使用骨干网络: {self.backbone_name}")
        if kwargs['model_name'] == 'ResNet':
            self.backbone = getattr(ResNet, self.backbone_name)(pretrained=True)
        if kwargs['model_name'] == 'EfficientNetV2':
            self.backbone = getattr(EfficientNetV2, self.backbone_name)(num_classes=self.class_num)
        if kwargs['model_name'] == 'SwinTransformer':
            self.backbone = getattr(SwinTransformer, self.backbone_name)(num_classes=self.class_num)
        if kwargs['model_name'] == 'ConvNeXt':
            self.backbone = getattr(ConvNeXt, self.backbone_name)(num_classes=self.class_num)
        if kwargs['model_name'] == 'VGG':
            self.backbone = getattr(VGG, self.backbone_name)(num_classes=self.class_num)
        # classifier
        self.classifier = nn.Sequential()
        # 全连接层1
        self.classifier.add_module('fc1',nn.Linear(2048, self.hidden_dim))
        self.classifier.add_module('batchnorm1d', nn.BatchNorm1d(self.hidden_dim))
        self.classifier.add_module('relu',nn.ReLU(inplace=True))
        self.classifier.add_module('dropout',nn.Dropout(p=0.5))
        self.classifier.add_module('fc2',nn.Linear(self.hidden_dim,self.class_num))

    def forward(self, x):
        output = self.backbone(x)
        if self.model_name == 'ResNet':
            output = self.classifier(output)
        return output