import timm
import torch
import yaml

import torchvision.transforms as transforms
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from main import test
from utils import seed_everything
from dataset import Transform

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("config.yaml", "r") as f:
            CFG = yaml.safe_load(f)
    CFG["lr"] = float(CFG["lr"]) if isinstance(CFG["lr"], str) else CFG["lr"]
    
    batch_size = CFG["BATCH_SIZE"]
    train_data_path = CFG["train_root"]
    test_data_path = CFG["test_root"]
    img_size = CFG["IMG_SIZE"]
    seed = CFG["SEED"]
    submission_sample_path = CFG["submission_path"]
    submission_save_dir = CFG["submission_save_dir"]
    
    seed_everything(seed)
    
    transform = Transform(img_size)

    full_dataset = CustomImageDataset(train_data_path, transform=None)
    class_names = full_dataset.classes
    
    test(
            model_name="eff_v2",
            test_data_path=test_data_path,
            transform=transform["test"],
            batch_size=batch_size,
            device=device,
            ckpt_path=args.ckpt,
            class_names=class_names,
            submission_sample_path=submission_sample_path,
            submission_save_dir=submission_save_dir
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args =parser.parse_args()
    
    main(args)