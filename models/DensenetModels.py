import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
#from resnet import InsResNet50
#from resnet import InsResNet18

import torchvision
import ipdb

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet18, self).__init__()
        if isTrained:
            self.resnet18 = torchvision.models.resnet18(pretrained=False)
            self.resnet18.load_state_dict(torch.load('models/resnet18-5c106cde.pth'))
            kernelCount = self.resnet18.fc.in_features
            self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        else:
            self.resnet18 = InsResNet18()        
            pretrained_weights = torch.load('models/ckpt_epoch_240.pth')['model']
            #del pretrained_weights['encoder.module.fc.weight']
            #del pretrained_weights['encoder.module.fc.bias']
            self.resnet18.load_state_dict(torch.load('models/ckpt_epoch_240.pth')['model'])
            #self.resnet18.load_state_dict(pretrained_weights, strict=False)
            kernelCount = self.resnet18.encoder.module.fc.in_features
            self.resnet18.encoder.module.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
            del self.resnet18.encoder.module.l2norm

    def forward(self, x):
        x = self.resnet18(x)
        return x


class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet50, self).__init__()
        if isTrained:
            self.resnet50 = torchvision.models.resnet50(pretrained=False)
            self.resnet50.load_state_dict(torch.load('models/resnet50-19c8e357.pth'))
            kernelCount = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        else:
            self.resnet50 = InsResNet50()        
            self.resnet50.load_state_dict(torch.load('models/ckpt_epoch_240.pth')['model'])
            kernelCount = self.resnet50.encoder.module.fc.in_features
            self.resnet50.encoder.module.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
            del self.resnet50.encoder.module.l2norm

    def forward(self, x):
        x = self.resnet50(x)
        return x

class DenseNet121(nn.Module):

    def __init__(self, classCount=128, isTrained=False):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Linear(kernelCount, classCount)
        self.densenet121 = nn.DataParallel(self.densenet121)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x


        
