"""
Training Script for [Insert Project Name Here]

This script provides a training pipeline with configurable arguments.
Modify the model, dataset, and training logic according to your research needs.

Author: [Insert Author Name Here]
Date: [Insert Date Here]
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training script for [Insert Project Name Here]"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to the dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101", "vit_base", "custom"],
        help="Model architecture to use (default: resnet50)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the model",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of output classes (default: 1000)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) (default: 1e-4)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer type (default: adamw)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine", "plateau"],
        help="Learning rate scheduler (default: cosine)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (default: None)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Frequency (in epochs) to save checkpoints (default: 10)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training (default: cuda)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs to use (default: 0)",
    )

    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for tensorboard logs (default: ./logs)",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Frequency (in iterations) to print training status (default: 10)",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(args):
    """
    Build and return the model based on arguments.
    
    Args:
        args: Command-line arguments.
    
    Returns:
        nn.Module: The constructed model.
    """
    model = None
    
    if args.model == "resnet50":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            pretrained=args.pretrained,
        )
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "resnet101":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet101",
            pretrained=args.pretrained,
        )
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "vit_base":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "vit_b_16",
            pretrained=args.pretrained,
        )
        model.heads.head = nn.Linear(model.heads.head.in_features, args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def build_optimizer(args, model):
    """
    Build and return the optimizer based on arguments.
    
    Args:
        args: Command-line arguments.
        model: The model to optimize.
    
    Returns:
        optim.Optimizer: The constructed optimizer.
    """
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def build_scheduler(args, optimizer):
    """
    Build and return the learning rate scheduler based on arguments.
    
    Args:
        args: Command-line arguments.
        optimizer: The optimizer to schedule.
    
    Returns:
        optim.lr_scheduler: The constructed scheduler.
    """
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    return scheduler


def save_checkpoint(args, model, optimizer, scheduler, epoch, best_acc, filename):
    """
    Save a training checkpoint.
    
    Args:
        args: Command-line arguments.
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The scheduler to save.
        epoch: Current epoch number.
        best_acc: Best accuracy achieved so far.
        filename: Name of the checkpoint file.
    """
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "args": args,
    }
    torch.save(checkpoint, os.path.join(args.save_dir, filename))
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(args, model, optimizer, scheduler):
    """
    Load a training checkpoint.
    
    Args:
        args: Command-line arguments.
        model: The model to load weights into.
        optimizer: The optimizer to load state into.
        scheduler: The scheduler to load state into.
    
    Returns:
        tuple: (start_epoch, best_acc)
    """
    if args.checkpoint is None:
        return 0, 0.0
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return 0, 0.0
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    
    print(f"Checkpoint loaded: {args.checkpoint}")
    print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
    
    return start_epoch, best_acc


def train_one_epoch(args, model, train_loader, criterion, optimizer, epoch):
    """
    Train the model for one epoch.
    
    Args:
        args: Command-line arguments.
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        epoch: Current epoch number.
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % args.print_freq == 0:
            pbar.set_postfix(
                loss=total_loss / (batch_idx + 1),
                acc=100.0 * correct / total,
            )
    
    return total_loss / len(train_loader)


def validate(args, model, val_loader, criterion):
    """
    Validate the model on the validation set.
    
    Args:
        args: Command-line arguments.
        model: The model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(args.device)
            targets = targets.to(args.device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    """
    Main training function.
    """
    args = parse_args()
    
    print("=" * 60)
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    
    model = build_model(args)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    
    train_loader = None
    val_loader = None
    
    start_epoch, best_acc = load_checkpoint(args, model, optimizer, scheduler)
    
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            args, model, train_loader, criterion, optimizer, epoch
        )
        
        val_loss, val_acc = validate(args, model, val_loader, criterion)
        
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                args, model, optimizer, scheduler, epoch, best_acc, "best_model.pth"
            )
        
        if epoch % args.save_freq == 0:
            save_checkpoint(
                args,
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                f"checkpoint_epoch_{epoch}.pth",
            )
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
