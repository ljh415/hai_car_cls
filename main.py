import os
import yaml
import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from dataset import CustomImageDataset, Transform
from utils import seed_everything, build_model, get_timestamp, get_optimizer, get_current_lr
from utils import MODEL_BATCHSIZE_MAP, MODEL_IMG_SIZE_MAP

def main(args):
    
    with open(args.config, "r") as f:
        CFG = yaml.safe_load(f)
    CFG["lr"] = float(CFG["lr"]) if isinstance(CFG["lr"], str) else CFG["lr"]
    
    if args.batch is not None:
        CFG["BATCH_SIZE"] = args.batch
    if args.epochs:
        CFG["EPOCHS"] = args.epochs
    if args.lr:
        CFG["lr"] = args.lr
    if args.weight_decay:
        CFG["weight_decay"] = args.weight_decay
    
    print(CFG)
    
    ## init hyperparameter
    epochs = CFG["EPOCHS"]
    lr = CFG["lr"]
    batch_size = CFG["BATCH_SIZE"]
    train_data_path = CFG["train_root"]
    test_data_path = CFG["test_root"]
    seed = CFG["SEED"]
    submission_sample_path = CFG["submission_path"]
    model_save_dir = CFG["model_save_dir"]
    submission_save_dir = CFG["submission_save_dir"]
    
    ##
    wandb.init(project="hai_car_cls")
    
    model_name = wandb.config.model
    optimizer_name = wandb.config.optimizer
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    epochs = wandb.config.epochs
    
    batch_size = MODEL_BATCHSIZE_MAP.get(model_name, CFG["BATCH_SIZE"])
    wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
    
    img_size = MODEL_IMG_SIZE_MAP.get(model_name, CFG["IMG_SIZE"])
    wandb.config.update({"img_size": img_size}, allow_val_change=True)
    
    wandb.run.name = f"{model_name}-{batch_size}-{epochs}-{optimizer_name}-{lr}-{weight_decay}-{get_timestamp()}"
    wandb.run.save()
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    ## model init
    model = build_model(model_name, len(class_names), device)
    
    ## train setting
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)
    
    best_logloss = float('inf')
    
    saved_ckpt_path_set = set()
    
    for epoch in range(epochs):
        avg_train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, epochs)
        avg_val_loss, valid_acc, val_logloss = valid(model, val_loader, criterion, device, epoch, epochs, class_names)
        
        scheduler.step(val_logloss)

        wandb.log({
            "epoch": epoch,
            "lr": get_current_lr(optimizer),
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": valid_acc,
            "val_logloss": val_logloss
        })
        
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {valid_acc:.4f}% | Valid Log Loss : {val_logloss:.5f}")
        
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            timestamp = get_timestamp()
            model_save_path = os.path.join(model_save_dir, timestamp.split("_")[0], f"{model_name}-{epoch}-{val_logloss:.4f}-{timestamp}.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            saved_ckpt_path_set.add(model_save_path)
            torch.save(model.state_dict(), model_save_path)
            print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")
    
    # 최종 best모델 제외하고 체크포인트 삭제
    del_model_set = saved_ckpt_path_set - set([model_save_path])
    for del_model_path in del_model_set:
        os.remove(del_model_path)
    
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
    
    avg_loss = running_loss / len(loader)
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
    
    print("Start Test")
    
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
    
    submission[class_columns] = pred.values
    submission_save_path = os.path.join(submission_save_dir, get_timestamp().split("_")[0], f"{'.'.join(os.path.basename(ckpt_path).split('.')[:-1])}.csv")
    os.makedirs(os.path.dirname(submission_save_path), exist_ok=True)
    submission.to_csv(submission_save_path, index=False, encoding='utf-8-sig')
    
    print(f"Submission csv saved: {submission_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--test", "-t", action="store_false")
    parser.add_argument("--model", "-m", default="eff_v2", help="bit, eff_v2, eff_vit, vit_h, vit_l")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    args = parser.parse_args()
    
    main(args)
    wandb.finish()