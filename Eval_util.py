import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

from Cora_info import class_names

def analyze_class_distribution(data):
    """Analyze the distribution of classes in train/val/test sets."""
    print("=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Count classes in each split
    train_labels = data.y[data.train_mask].cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()
    test_labels = data.y[data.test_mask].cpu().numpy()

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    # Create distribution DataFrame
    df_dist = pd.DataFrame({
        'Class': class_names,
        'Train': [train_counts.get(i, 0) for i in range(len(class_names))],
        'Val': [val_counts.get(i, 0) for i in range(len(class_names))],
        'Test': [test_counts.get(i, 0) for i in range(len(class_names))]
    })

    # Add totals and percentages
    df_dist['Total'] = df_dist['Train'] + df_dist['Val'] + df_dist['Test']
    df_dist['Train %'] = (df_dist['Train'] / df_dist['Train'].sum() * 100).round(2)
    df_dist['Val %'] = (df_dist['Val'] / df_dist['Val'].sum() * 100).round(2)
    df_dist['Test %'] = (df_dist['Test'] / df_dist['Test'].sum() * 100).round(2)
    df_dist['Total %'] = (df_dist['Total'] / df_dist['Total'].sum() * 100).round(2)

    print(df_dist.to_string(index=False))

    return df_dist


def calculate_random_baseline(test_labels, train_labels):
    """Calculate expected accuracy if guessing randomly based on class distribution."""
    # Random uniform guessing
    uniform_random_acc = 1.0 / len(class_names)

    # Random guessing based on training set distribution
    train_dist = np.bincount(train_labels) / len(train_labels)
    test_dist = np.bincount(test_labels) / len(test_labels)

    # Expected accuracy if guessing according to train distribution
    stratified_random_acc = np.sum(train_dist * test_dist)

    return uniform_random_acc, stratified_random_acc


def detailed_performance_analysis(data, pred, true_labels, mask, split_name):
    """Perform detailed per-class analysis."""
    pred_masked = pred[mask].cpu().numpy()
    true_masked = true_labels[mask].cpu().numpy()

    print(f"\n" + "=" * 60)
    print(f"{split_name.upper()} SET DETAILED ANALYSIS")
    print("=" * 60)

    # Overall accuracy
    overall_acc = (pred_masked == true_masked).mean()
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    # Random baselines
    uniform_baseline, stratified_baseline = calculate_random_baseline(true_masked,
                                                                      data.y[data.train_mask].cpu().numpy())
    print(f"Random Uniform Baseline: {uniform_baseline:.4f} ({uniform_baseline * 100:.2f}%)")
    print(f"Random Stratified Baseline: {stratified_baseline:.4f} ({stratified_baseline * 100:.2f}%)")

    # Improvement over random
    uniform_improvement = (overall_acc - uniform_baseline) / uniform_baseline * 100
    stratified_improvement = (overall_acc - stratified_baseline) / stratified_baseline * 100
    print(f"Improvement over Uniform Random: {uniform_improvement:.1f}%")
    print(f"Improvement over Stratified Random: {stratified_improvement:.1f}%")

    # Per-class metrics
    print(f"\nPER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(f"{'Class':<20} {'Count':<8} {'Correct':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
    print("-" * 80)

    # Calculate per-class metrics
    class_metrics = []
    for class_idx in range(len(class_names)):
        # Find indices for this class
        class_mask = true_masked == class_idx
        class_count = class_mask.sum()

        if class_count > 0:
            # True positives, false positives, false negatives
            tp = ((pred_masked == class_idx) & (true_masked == class_idx)).sum()
            fp = ((pred_masked == class_idx) & (true_masked != class_idx)).sum()
            fn = ((pred_masked != class_idx) & (true_masked == class_idx)).sum()

            # Metrics
            accuracy = tp / class_count if class_count > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(
                f"{class_names[class_idx]:<20} {class_count:<8} {tp:<8} {accuracy:<10.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f}")

            class_metrics.append({
                'class': class_names[class_idx],
                'count': class_count,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        else:
            print(f"{class_names[class_idx]:<20} {0:<8} {0:<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}")

    # Summary statistics
    if class_metrics:
        accuracies = [m['accuracy'] for m in class_metrics]
        precisions = [m['precision'] for m in class_metrics if m['precision'] > 0]
        recalls = [m['recall'] for m in class_metrics if m['recall'] > 0]
        f1s = [m['f1'] for m in class_metrics if m['f1'] > 0]

        print("-" * 80)
        print(
            f"{'AVERAGE':<20} {'-':<8} {'-':<8} {np.mean(accuracies):<10.4f} {np.mean(precisions):<10.4f} {np.mean(recalls):<8.4f} {np.mean(f1s):<8.4f}")
        print(
            f"{'STD DEV':<20} {'-':<8} {'-':<8} {np.std(accuracies):<10.4f} {np.std(precisions):<10.4f} {np.std(recalls):<8.4f} {np.std(f1s):<8.4f}")

    return class_metrics


def plot_confusion_matrix(pred, true_labels, mask, split_name):
    """Plot confusion matrix."""
    pred_masked = pred[mask].cpu().numpy()
    true_masked = true_labels[mask].cpu().numpy()

    cm = confusion_matrix(true_masked, pred_masked)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {split_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_class_performance(class_metrics, split_name):
    """Plot per-class performance metrics."""
    if not class_metrics:
        return

    df_metrics = pd.DataFrame(class_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Per-Class Performance Metrics - {split_name} Set', fontsize=16)

    # Accuracy
    axes[0, 0].bar(df_metrics['class'], df_metrics['accuracy'])
    axes[0, 0].set_title('Accuracy per Class')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Precision
    axes[0, 1].bar(df_metrics['class'], df_metrics['precision'])
    axes[0, 1].set_title('Precision per Class')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Recall
    axes[1, 0].bar(df_metrics['class'], df_metrics['recall'])
    axes[1, 0].set_title('Recall per Class')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # F1-Score
    axes[1, 1].bar(df_metrics['class'], df_metrics['f1'])
    axes[1, 1].set_title('F1-Score per Class')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_predictions_confidence(out, mask, split_name):
    """Analyze prediction confidence."""
    probs = F.softmax(out[mask], dim=1)
    max_probs, pred_classes = torch.max(probs, dim=1)

    print(f"\n" + "=" * 60)
    print(f"PREDICTION CONFIDENCE ANALYSIS - {split_name.upper()}")
    print("=" * 60)

    confidence_stats = {
        'mean': max_probs.mean().item(),
        'std': max_probs.std().item(),
        'min': max_probs.min().item(),
        'max': max_probs.max().item(),
        'median': max_probs.median().item()
    }

    print(f"Confidence Statistics:")
    print(f"  Mean: {confidence_stats['mean']:.4f}")
    print(f"  Std:  {confidence_stats['std']:.4f}")
    print(f"  Min:  {confidence_stats['min']:.4f}")
    print(f"  Max:  {confidence_stats['max']:.4f}")
    print(f"  Median: {confidence_stats['median']:.4f}")

    # Confidence distribution
    low_conf = (max_probs < 0.5).sum().item()
    med_conf = ((max_probs >= 0.5) & (max_probs < 0.8)).sum().item()
    high_conf = (max_probs >= 0.8).sum().item()
    total = len(max_probs)

    print(f"\nConfidence Distribution:")
    print(f"  Low confidence (<0.5):  {low_conf:4d} ({low_conf / total * 100:.1f}%)")
    print(f"  Med confidence (0.5-0.8): {med_conf:4d} ({med_conf / total * 100:.1f}%)")
    print(f"  High confidence (>0.8): {high_conf:4d} ({high_conf / total * 100:.1f}%)")
