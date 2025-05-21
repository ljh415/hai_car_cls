import os
import random
import numpy as np

import timm
import torch


# CFG = {
#     'IMG_SIZE': 224,
#     'BATCH_SIZE': 64,
#     'EPOCHS': 10,
#     'LEARNING_RATE': 1e-4,
#     'SEED' : 42
# }

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def build_model(backbone, num_classes, device=None):
    """
    resnetv2_50x1_bit
    efficientnetv2_l
    efficientvit_l2
    vit_huge_patch14_224
    vit_large_patch16_224
    """
    if backbone == "bit":
        model = timm.create_model(
            "resnetv2_50x1_bit",
            pretrained=True,
            num_classes=num_classes
        )
    elif backbone == "eff_v2":
        model = timm.create_model(
            "efficientnetv2_l",
            pretrained=False,
            num_classes=num_classes
        )
    elif backbone == "eff_vit":
        model = timm.create_model(
            "efficientvit_l2",
            pretrained=True,
            num_classes=num_classes
        )
    elif backbone == "vit_h":
        model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=True,
            num_classes=num_classes
        )
    elif backbone == "vit_l":
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
    else :
        raise Exception("다른거 쓰고 싶으면 너가 추가해")
    
    if device is not None:
        model = model.to(device)
        
    return model