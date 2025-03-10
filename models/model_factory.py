from models.resnext import *
from models.resnet import *
from models.se_resnet import *

def model_factory(name):
    if name == 'resnet18':
        return ResNet_18()
    elif name == 'resnext18':
        return ResNeXt_18()
    elif name == 'se-resnet18':
        return SE_ResNet_18()
    else:
        raise ValueError(f"Unknown model name: {name}")