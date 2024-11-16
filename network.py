from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
from networks.googlenet import googlenet
from networks.shufflenetv2 import shufflenetv2
from networks.mobilenet import MobileNet
from networks.wideresnet import WideResNet
from collections import OrderedDict
from util.data import *

def get_network(model_arch, num_classes=1000):

    if model_arch == "googlenet":
        net = googlenet(pretrained=True)
    elif model_arch == "vgg16":
        net = models.vgg16(pretrained=True)
    elif model_arch == "vgg19":
        net = models.vgg19(pretrained=True)
    elif model_arch == "resnet50":
        net = models.resnet50(pretrained=True)
    elif model_arch == 'shufflenetv2':
        net = shufflenetv2(num_classes=num_classes,pretrained=False)
    elif model_arch == 'mobilenet':
        net = MobileNet(num_classes=num_classes, pretrained=False)
    elif model_arch == "wideresnet":
        net = WideResNet()
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = False

def set_parameter_requires_grad_selected(model, set_layer, requires_grad=False):
    for name, param in model.named_parameters():
        print(name)
        if set_layer in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_num_non_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==False, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

#args.model_name, args.model_name, args.dataset
def get_my_model(weight_path, model_name, arch, dataset):
    weight_path = weight_path + '/' + str(model_name)
    num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)

    target_network = get_network(arch, num_classes)

    # Set the target model into evaluation mode
    target_network.eval()

    if dataset == "caltech" or dataset == 'asl':
        if 'repaired' in model_name:
            target_network = torch.load(weight_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)
    elif dataset == 'eurosat':
        target_network = torch.load(weight_path, map_location=torch.device('cpu'))
        if 'repaired' in model_name:
            adaptive = '_adaptive'
    elif dataset == "imagenet" and 'repaired' in model_name:
        target_network = torch.load(weight_path, map_location=torch.device('cpu'))
    elif dataset == "cifar10":
        if 'repaired' in model_name:
            target_network = torch.load(weight_path, map_location=torch.device('cpu'))
        else:
            target_network.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    return target_network

