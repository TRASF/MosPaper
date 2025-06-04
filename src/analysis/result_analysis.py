"""
Result analysis and visualization utilities for mosquito wingbeat classification.

This module provides functions to visualize and analyze experiment results,
including:
- Training history plots (accuracy, loss)
- Confusion matrix visualization with counts and percentages
- Classification report visualization
- Model performance comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Configure logger
logger = logging.getLogger(__name__)

def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]] = None, 
    output_path: Optional[Path] = None, 
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
    hide_zeros: bool = True,
    show_counts: bool = True,
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot confusion matrix with enhanced visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        output_path: Path to save the figure (optional, if None figure is not saved)
        title: Title of the plot
        figsize: Figure size (width, height)
        normalize: Whether to normalize confusion matrix
        hide_zeros: Whether to hide zero values
        show_counts: Whether to show counts along with percentages
        cmap: Colormap name
    
    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # For our mosquito classification, we pre-define class names in config
    # so just fall back to index numbers if needed
    class_names = class_names or [str(i) for i in range(cm.shape[0])]
        
    # For our mosquito analysis, we always use normalized confusion matrices
    # so simplify this logic - just adjust title
    title = f"Normalized {title}"
    
    # Create normalized matrix for percentage display - add small epsilon to avoid div by zero
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Create a single standard heatmap - we always normalize for better visualization
    sns.heatmap(
        cm_norm, 
        annot=False,  # We add custom annotations below
        cmap=cmap, 
        fmt='.2f', 
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax
    )
        
    # Add custom cell annotations (counts + percentages)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Skip if value is zero and hide_zeros is True
            if hide_zeros and cm[i, j] == 0:
                continue
                
            # Create annotation text
            if show_counts:
                if normalize:
                    # Show both percentage and count
                    cell_text = f"{cm_norm[i, j]:.1%}\n({cm[i, j]})"
                else:
                    cell_text = f"{cm[i, j]}"
            else:
                if normalize:
                    cell_text = f"{cm_norm[i, j]:.1%}"
                else:
                    cell_text = f"{cm[i, j]}"
                    
            # Add text with color based on cell value
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j + 0.5, i + 0.5, cell_text,
                   ha="center", va="center", 
                   color=color,
                   fontweight="bold" if i == j else "normal")  # Bold for diagonal
    
    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()
    
    # Calculate and show overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.5, 0.01, f"Overall Accuracy: {accuracy:.2%}", ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save figure if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    
    return fig

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 10),  # Wider figure
    metrics: Optional[List[str]] = None,
    title: str = 'Training History'
) -> plt.Figure:
    """Create a training history plot with multiple rows of metrics."""
    if not metrics:
        metrics = [k for k in history.keys() if not k.startswith('val_')]

    # Calculate optimal layout
    n_plots = len(metrics)
    n_cols = min(4, n_plots)  # Maximum 4 plots per row
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot grid
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    axes = []

    # Create subplots
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        axes.append(fig.add_subplot(gs[row, col]))

    # Plot each metric
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        # Plot training curve
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation curve if available
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            
            # Add best value annotation
            if 'accuracy' in metric:
                best_idx = np.argmax(history[f'val_{metric}'])
            else:  # loss metrics
                best_idx = np.argmin(history[f'val_{metric}'])
            best_value = history[f'val_{metric}'][best_idx]
            ax.axvline(x=best_idx, color='r', linestyle='--', alpha=0.3)
            ax.text(best_idx, best_value, f' Best: {best_value:.4f}',
                   verticalalignment='center')

        # Customize each subplot
        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add suptitle and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Training history plot saved to {output_path}")

    return fig

def generate_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate and save classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save text report
        
    Returns:
        Classification report as dict
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(len(set(y_true)))]
        
    # Generate report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    
    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Classification report saved to {output_path}")
        
    return report_dict

def visualize_predictions(
    model,
    test_data,
    true_labels,
    class_names: Optional[List[str]] = None,
    num_samples: int = 10,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    is_waveform: bool = True
) -> plt.Figure:
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained model
        test_data: Test data samples
        true_labels: Ground truth labels
        class_names: List of class names
        num_samples: Number of samples to visualize
        output_path: Path to save the figure
        figsize: Figure size
        is_waveform: Whether the data is waveform (True) or spectrogram (False)
        
    Returns:
        Matplotlib figure object
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(len(set(true_labels)))]
    
    # Get random samples
    indices = np.random.choice(len(test_data), size=min(num_samples, len(test_data)), replace=False)
    samples = test_data[indices]
    true_labels = true_labels[indices]
    
    # Get model predictions
    predictions = model.predict(samples)
    if predictions.ndim > 1 and predictions.shape[1] > 1:  # If logits/probabilities
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions.ravel().astype(int)
    
    # Create figure
    n_cols = 2
    n_rows = (num_samples + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot each sample
    for i, (sample, true, pred) in enumerate(zip(samples, true_labels, pred_labels)):
        if i >= len(axes):
            break
            
        # Different visualization based on data type
        if is_waveform:
            # Plot waveform
            axes[i].plot(sample.ravel())
            axes[i].set_xlim(0, len(sample.ravel()))
        else:
            # Plot spectrogram
            axes[i].imshow(sample.squeeze(), aspect='auto', origin='lower')
        
        # Color based on prediction correctness
        color = "green" if true == pred else "red"
        title = f"True: {class_names[true]}\nPred: {class_names[pred]}"
        axes[i].set_title(title, color=color)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Hide unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Model Predictions", fontsize=16)
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Prediction visualization saved to {output_path}")
    
    return fig

def compare_models(
    model_results: Dict[str, Dict],
    output_path: Optional[Path] = None,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (12, 6),
) -> pd.DataFrame:
    """
    Compare multiple model configurations.
    
    Args:
        model_results: Dictionary of model results {model_name: result_dict}
        output_path: Path to save comparison plot
        metric: Metric to compare
        figsize: Figure size
        
    Returns:
        DataFrame with results comparison
    """
    # Create DataFrame from results
    df = pd.DataFrame([
        {
            'Model': model_name,
            f'Training {metric}': results.get(f'train_{metric}', results.get('accuracy', 0)),
            f'Validation {metric}': results.get(f'val_{metric}', results.get('val_accuracy', 0)),
            f'Test {metric}': results.get(f'test_{metric}', results.get('test_accuracy', 0))
        }
        for model_name, results in model_results.items()
    ])
    
    # Sort by test metric
    df = df.sort_values(f'Test {metric}', ascending=False)
    
    # Plot comparison
    plt.figure(figsize=figsize)
    df.set_index('Model').plot(kind='bar', ax=plt.gca())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Model comparison plot saved to {output_path}")
    
    return df

def plot_ssl_metrics(
    history: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 20),
    title: str = 'SSL Training Metrics',
    model_type: str = 'fixmatch'
) -> plt.Figure:
    """
    Enhanced visualization of SSL training metrics with comprehensive subplot organization.
    
    Args:
        history: Dictionary containing training history metrics
        output_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
        title: Main title for the figure
        model_type: Type of SSL model ('fixmatch' or 'flexmatch')
        
    Returns:
        matplotlib Figure object
    """
    # Check available metrics
    has_val_loss = 'val_loss' in history and len(history['val_loss']) > 0
    has_sup_loss = 'sup_loss' in history and len(history['sup_loss']) > 0
    has_unsup_loss = 'unsup_loss' in history and len(history['unsup_loss']) > 0
    has_mask_ratio = 'mask_ratio' in history and len(history['mask_ratio']) > 0
    has_pseudo_accuracy = 'pseudo_accuracy' in history and len(history['pseudo_accuracy']) > 0
    has_accuracy = 'accuracy' in history and len(history['accuracy']) > 0
    has_val_accuracy = 'val_accuracy' in history and len(history['val_accuracy']) > 0

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # 1. Plot Accuracy metrics
    ax_acc = axes[0]
    if has_accuracy:
        ax_acc.plot(history['accuracy'], label='Training Accuracy', marker='o', color='blue')
    if has_val_accuracy:
        ax_acc.plot(history['val_accuracy'], label='Validation Accuracy', marker='s', color='skyblue')
    ax_acc.set_title('Model Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Plot Loss metrics
    ax_loss = axes[1]
    if has_sup_loss:
        ax_loss.plot(history['sup_loss'], label='Supervised Loss', marker='o', color='red')
    if has_unsup_loss:
        ax_loss.plot(history['unsup_loss'], label='Unsupervised Loss', marker='s', color='orange')
    if has_val_loss:
        ax_loss.plot(history['val_loss'], label='Validation Loss', marker='^', color='pink')
    ax_loss.set_title('Training Losses')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    
    # Add best validation loss marker
    if has_val_loss and len(history['val_loss']) > 0:
        best_idx = np.argmin(history['val_loss'])
        best_value = history['val_loss'][best_idx]
        ax_loss.axvline(x=best_idx, color='r', linestyle='--', alpha=0.3)
        ax_loss.text(best_idx, best_value, f'Best: {best_value:.4f}', 
                   verticalalignment='center', color='red')
    
    # 3. Plot mask ratio (confidence threshold)
    ax_mask = axes[2]
    if has_mask_ratio:
        ax_mask.plot(history['mask_ratio'], label='Mask Ratio', marker='o', color='green')
        ax_mask.set_title('Mask Ratio (Samples Above Confidence Threshold)')
        ax_mask.set_xlabel('Epoch')
        ax_mask.set_ylabel('Mask Ratio')
        ax_mask.legend(loc='best')
        ax_mask.grid(True, linestyle='--', alpha=0.6)
        
        # Add moving average trend line
        if len(history['mask_ratio']) > 5:
            window_size = min(5, len(history['mask_ratio']))
            mask_ratio_ma = np.convolve(history['mask_ratio'], 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
            epochs_ma = np.arange(window_size-1, len(history['mask_ratio']))
            ax_mask.plot(epochs_ma, mask_ratio_ma, 
                        label=f'Moving Avg (window={window_size})',
                        linestyle='--', color='darkgreen', alpha=0.7)
            ax_mask.legend(loc='best')
    
    # 4. Plot pseudo-label accuracy
    ax_pseudo = axes[3]
    if has_pseudo_accuracy:
        ax_pseudo.plot(history['pseudo_accuracy'], 
                      label='Pseudo-Label Accuracy', 
                      marker='o', color='purple')
        ax_pseudo.set_title('Pseudo-Label Accuracy')
        ax_pseudo.set_xlabel('Epoch')
        ax_pseudo.set_ylabel('Accuracy')
        ax_pseudo.legend(loc='best')
        ax_pseudo.grid(True, linestyle='--', alpha=0.6)
        
        # Add trend line
        if len(history['pseudo_accuracy']) > 5:
            window_size = min(5, len(history['pseudo_accuracy']))
            pseudo_acc_ma = np.convolve(history['pseudo_accuracy'], 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
            epochs_ma = np.arange(window_size-1, len(history['pseudo_accuracy']))
            ax_pseudo.plot(epochs_ma, pseudo_acc_ma, 
                         label=f'Moving Avg (window={window_size})',
                         linestyle='--', color='darkviolet', alpha=0.7)
            ax_pseudo.legend(loc='best')
    
    # 5. Plot total loss
    ax_total = axes[4]
    if 'total_loss' in history:
        ax_total.plot(history['total_loss'], 
                     label='Total Loss', 
                     marker='o', color='brown')
        ax_total.set_title('Total Training Loss')
        ax_total.set_xlabel('Epoch')
        ax_total.set_ylabel('Loss')
        ax_total.legend(loc='upper right')
        ax_total.grid(True, linestyle='--', alpha=0.6)
    
    # 6. Additional metrics specific to model type
    ax_extra = axes[5]
    if model_type == 'flexmatch' and 'class_threshold' in history:
        for class_idx in range(len(history['class_threshold'][0])):
            class_thresholds = [epoch_thresholds[class_idx] 
                              for epoch_thresholds in history['class_threshold']]
            ax_extra.plot(class_thresholds, 
                         label=f'Class {class_idx} Threshold',
                         marker='o')
        ax_extra.set_title('FlexMatch Class-wise Thresholds')
        ax_extra.set_xlabel('Epoch')
        ax_extra.set_ylabel('Threshold')
        ax_extra.legend(loc='best')
        ax_extra.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    return fig
