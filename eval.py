"""
Evaluation Script for [Insert Project Name Here]

This script provides an evaluation pipeline with configurable arguments.
Modify the model, dataset, and evaluation metrics according to your research needs.

Author: [Insert Author Name Here]
Date: [Insert Date Here]
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for evaluation configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluation script for [Insert Project Name Here]"
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
        help="Batch size for evaluation (default: 32)",
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
        "--num_classes",
        type=int,
        default=1000,
        help="Number of output classes (default: 1000)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint for evaluation (required)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs to use (default: 0)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model predictions to a file",
    )

    # Evaluation arguments
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation results",
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "resnet101":
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "vit_base":
        model = torch.hub.load("pytorch/vision:v0.10.0", "vit_b_16", pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def load_checkpoint(args, model):
    """
    Load model weights from checkpoint.
    
    Args:
        args: Command-line arguments.
        model: The model to load weights into.
    
    Returns:
        nn.Module: The model with loaded weights.
    """
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for k, v in state_dict.items()]
        )
    
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded successfully: {args.checkpoint}")
    
    return model


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(predictions, targets, num_classes):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions (numpy array).
        targets: Ground truth labels (numpy array).
        num_classes: Number of classes.
    
    Returns:
        dict: Dictionary containing computed metrics.
    """
    metrics = {}
    
    accuracy = 100.0 * np.mean(predictions == targets)
    metrics["accuracy"] = accuracy
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        confusion_matrix[t, p] += 1
    metrics["confusion_matrix"] = confusion_matrix.tolist()
    
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    metrics["precision_macro"] = 100.0 * np.mean(precision_per_class)
    metrics["recall_macro"] = 100.0 * np.mean(recall_per_class)
    metrics["f1_macro"] = 100.0 * np.mean(f1_per_class)
    
    metrics["precision_per_class"] = precision_per_class
    metrics["recall_per_class"] = recall_per_class
    metrics["f1_per_class"] = f1_per_class
    
    return metrics


@torch.no_grad()
def evaluate(args, model, data_loader, num_classes):
    """
    Evaluate the model on the dataset.
    
    Args:
        args: Command-line arguments.
        model: The model to evaluate.
        data_loader: Data loader for evaluation.
        num_classes: Number of classes.
    
    Returns:
        tuple: (metrics dict, all predictions, all targets)
    """
    model.eval()
    
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Accuracy")
    
    all_predictions = []
    all_targets = []
    
    num_samples = 0
    max_samples = args.num_samples if args.num_samples else float("inf")
    
    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        if num_samples >= max_samples:
            break
        
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        outputs = model(images)
        
        _, predictions = outputs.max(1)
        
        correct = predictions.eq(targets).sum().item()
        accuracy = 100.0 * correct / targets.size(0)
        
        acc_meter.update(accuracy, targets.size(0))
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        num_samples += targets.size(0)
        
        pbar.set_postfix(acc=acc_meter.avg)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = compute_metrics(all_predictions, all_targets, num_classes)
    metrics["num_samples"] = num_samples
    
    return metrics, all_predictions, all_targets


def save_results(args, metrics, predictions, targets):
    """
    Save evaluation results to files.
    
    Args:
        args: Command-line arguments.
        metrics: Dictionary of computed metrics.
        predictions: Model predictions.
        targets: Ground truth labels.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_path = os.path.join(args.output_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {results_path}")
    
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, "predictions.npz")
        np.savez(predictions_path, predictions=predictions, targets=targets)
        print(f"Predictions saved to: {predictions_path}")


def print_results(metrics, verbose=False):
    """
    Print evaluation results.
    
    Args:
        metrics: Dictionary of computed metrics.
        verbose: Whether to print detailed results.
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision (Macro): {metrics['precision_macro']:.2f}%")
    print(f"Recall (Macro): {metrics['recall_macro']:.2f}%")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.2f}%")
    
    if verbose:
        print("\nPer-class metrics:")
        for i, (p, r, f1) in enumerate(
            zip(
                metrics["precision_per_class"],
                metrics["recall_per_class"],
                metrics["f1_per_class"],
            )
        ):
            print(f"  Class {i}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    print("=" * 60)


def main():
    """
    Main evaluation function.
    """
    args = parse_args()
    
    print("=" * 60)
    print(f"Evaluation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.data_path}")
    print(f"  Split: {args.split}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    
    model = build_model(args)
    model = load_checkpoint(args, model)
    model = model.to(device)
    model.eval()
    
    data_loader = None
    
    metrics, predictions, targets = evaluate(args, model, data_loader, args.num_classes)
    
    print_results(metrics, verbose=args.verbose)
    
    save_results(args, metrics, predictions, targets)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
