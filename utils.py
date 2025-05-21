import os
import random
import numpy as np
from datetime import datetime

import timm
import torch
from torch.optim import Adam, AdamW, SGD

MODEL_BACKBONE_MAP = {
    "bit": ("resnetv2_50x1_bit", 0.0),
    "eff_v2": ("tf_efficientnetv2_l.in21k", 0.0),
    "eff_vit": ("efficientvit_l2", 0.0),
    "vit_h": ("vit_huge_patch14_224", 0.4),
    "vit_l": ("vit_large_patch16_224", 0.3),
    "convnextv2": ("convnextv2_large.fcmae_ft_in22k_in1k", 0.3),
    "swinv2": ("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k", 0.2),
}

MODEL_IMG_SIZE_MAP = {
    "vit_h": 224,
    "vit_l": 224,
    "convnextv2": 224,
    "swinv2": 384,
    "bit": 224,
    "eff_v2": 224,
    "eff_vit": 224,
}

MODEL_BATCHSIZE_MAP = {
    "vit_h": 16,
    "vit_l": 48,
    "convnextv2": 32,
    "swinv2": 8,
    "bit": 128,
    "eff_v2": 40,
    "eff_vit": 120,
}

def get_timestamp():
    now = datetime.now()  
    timestamp = now.strftime("%y%m%d_%H%M%S")
    return timestamp

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_optimizer(model, name, lr, weight_decay):
    if name == "Adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def build_model(backbone, num_classes, device=None):
    if backbone not in MODEL_BACKBONE_MAP:
        raise Exception(f"[!] Unknown backbone: {backbone}")
    
    model_name, drop_path_rate = MODEL_BACKBONE_MAP[backbone]

    kwargs = {
        "pretrained": True,
        "num_classes": num_classes
    }
    
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    
    model = timm.create_model(
        model_name,
        **kwargs
    )
    
    if device is not None:
        model = model.to(device)
        
    return model