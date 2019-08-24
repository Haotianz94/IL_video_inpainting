import torch
import torch.nn as nn


class LossNetwork(nn.Module):
    """
    Extract certain feature maps from pretrained VGG model, used for computing perceptual loss
    """
    def __init__(self, vgg_model=None, output_layer=['3', '8', '15']):
        super(LossNetwork, self).__init__()
        if vgg_model is None:
            # prepare fixed VGG16
            conv = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
            conv.weight.data.fill_(1)
            pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            relu = torch.nn.ReLU()
            model = []
            model += ([conv, relu] * 2 + [pool]) * 2
            model += ([conv, relu] * 4  + [pool]) * 2
            model += [conv, relu] * 4 
            self.vgg_layers = model
        else:
            self.vgg_layers = vgg_model.features
        self.output_layer = output_layer
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        feature_list = []
        if type(self.vgg_layers) == list:
            for layer, module in enumerate(self.vgg_layers):
                x = module(x)
                if str(layer) in self.output_layer:
                    feature_list.append(x)
        else:
            for name, module in self.vgg_layers._modules.items():
                x = module(x)
                if name in self.output_layer:
                    feature_list.append(x)
        return feature_list
