from models.resnext import *
from models.resnet import *
from models.se_resnet import *
from models.wide_resnet import *
from models.shake_wide_resnet import *

def model_factory(name):
    if name == 'resnet18':
        return ResNet_18()
    elif name == 'resnext18':
        return ResNeXt_18()
    elif name == 'se-resnet18':
        return SE_ResNet_18()
    elif name == 'wide-resnet18':
        return Wide_ResNet_18()
    elif name == 'shake-wide-resnet18':
        return Shake_Wide_ResNet_18()
    else:
        raise ValueError(f"Unknown model name: {name}")