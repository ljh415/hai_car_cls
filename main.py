import os
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from utils import seed_everything, build_model
from dataset import CustomImageDataset, Transform


def main(args):
    
    with open(args.config, "r") as f:
        CFG = yaml.safe_load(f)
    CFG["lr"] = float(CFG["lr"]) if isinstance(CFG["lr"], str) else CFG["lr"]
    
    print(CFG)
    
    ## init hyperparameter
    epochs = CFG["EPOCHS"]
    lr = CFG["lr"]
    batch_size = CFG["BATCH_SIZE"]
    train_data_path = CFG["train_root"]
    test_data_path = CFG["test_root"]
    img_size = CFG["IMG_SIZE"]
    seed = CFG["SEED"]
    submission_sample_path = CFG["submission_path"]
    model_save_dir = CFG["model_save_dir"]
    submission_save_dir = CFG["submission_save_dir"]
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    seed_everything(seed)
    
    transform = Transform(img_size)
    
    full_dataset = CustomImageDataset(train_data_path, transform=None)
    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes
    
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=94
    )
    
    train_dataset = Subset(CustomImageDataset(train_data_path, transform=transform["train"]), train_idx)
    val_dataset = Subset(CustomImageDataset(train_data_path, transform=transform["train"]), val_idx)
    print(f'train 이미지 수: {len(train_dataset)}, valid 이미지 수: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    ## model init
    model = build_model(args.model, len(class_names), device)
    
    ## train setting
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_logloss = float('inf')
    
    for epoch in range(epochs):
        avg_train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, epochs)
        avg_val_loss, valid_acc, val_logloss = valid(model, val_loader, criterion, device, epoch, epochs, class_names)
        
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {valid_acc:.4f}% | Valid Log Loss : {val_logloss:.5f}")
        
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            model_save_path = os.path.join(model_save_dir, f"{args.model}-{valid_acc}-{val_logloss}-best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")
    
    if args.test:
        del model
        test(
            model_name=args.model,
            test_data_path=test_data_path,
            transform=transform["test"],
            batch_size=batch_size,
            device=device,
            ckpt_path=model_save_path,
            class_names=class_names,
            submission_sample_path=submission_sample_path,
            submission_save_dir=submission_save_dir
        )

def train(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc=f"[Epoch {epoch+1}/{epochs}] Training "):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy

def valid(model, loader, criterion, device, epoch, epochs, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation "):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # acc
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # logloss
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = running_loss / len(loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
    
    return avg_val_loss, val_accuracy, val_logloss

def test(model_name, test_data_path, transform, batch_size, device, ckpt_path, class_names, submission_sample_path, submission_save_dir):
    
    test_dataset = CustomImageDataset(test_data_path, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = build_model(model_name, len(class_names), device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            for prob in probs.cpu():
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)
    
    pred = pd.DataFrame(results)
    
    submission = pd.read_csv(submission_sample_path, encoding='utf-8-sig')
    
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    
    submission_save_path = os.path.join(submission_save_dir, f"{os.path.basename(ckpt_path).split('.')[0]}.csv")
    submission.to_csv(submission_save_path, index=False, encoding='utf-8-sig')
    
    print(f"Submission csv saved: {submission_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--test", "-t", action="store_false")
    parser.add_argument("--model", "-m", default="eff_v2")
    args = parser.parse_args()
    
    main(args)