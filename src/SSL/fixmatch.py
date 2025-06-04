"""
Optimized FixMatch implementation with proper gradient flow and EMA handling.
"""

import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np
import logging
from typing import Tuple, Dict, Union, Optional, Any, List, cast

# Configure logger
logger = logging.getLogger(__name__)

def _var_key(var):
    """Get the key for storing optimizer slots."""
    return var.experimental_ref() if hasattr(var, 'experimental_ref') else var.ref()

class FixMatch:
    """
    FixMatch semi-supervised learning algorithm implementation.
    
    Key features:
    1. Uses weak and strong augmentations for consistency regularization
    2. Applies confidence thresholding for pseudo-labeling
    3. Combines supervised and unsupervised losses with unified gradient flow
    """
    
    def __init__(
        self, 
        model: tfk.Model, 
        num_classes: int, 
        confidence_threshold: float, 
        lambda_u: float = 1.0,
        T: float = 1.0,
        optimizer: Optional[tfk.optimizers.Optimizer] = None,
        ema_decay: float = 0.999
    ) -> None:
        """
        Initialize FixMatch trainer.
        
        Args:
            model: Neural network model for classification
            num_classes: Number of output classes
            confidence_threshold: Threshold for pseudo-labeling
            lambda_u: Weight for unsupervised loss
            T: Temperature for sharpening pseudo-labels
            optimizer: Optimizer for model updates (will be provided by trainer)
            ema_decay: Decay rate for EMA teacher model (if used)
        """
        self.model = model
        
        # Use provided optimizer or create default Adam
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            logger.warning("Using default optimizer. For best results, provide optimizer from trainer.")
            
        # Loss functions - standard categorical cross-entropy
        self.labeled_loss_fn = tfk.losses.CategoricalCrossentropy(from_logits=True)
        self.unlabeled_loss_fn = tfk.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tfk.losses.Reduction.NONE
        )
        
        # Core SSL parameters
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.T = T
        self.ema_decay_rate = ema_decay

        # Metrics for monitoring
        self.sup_loss_metric = tfk.metrics.Mean(name='sup_loss')
        self.unsup_loss_metric = tfk.metrics.Mean(name='unsup_loss')
        self.total_loss_metric = tfk.metrics.Mean(name='total_loss')
        self.mask_ratio_metric = tfk.metrics.Mean(name='mask_ratio')
        self.pseudo_label_accuracy_metric = tfk.metrics.Mean(name='pseudo_label_accuracy')
        
        logger.info(f"Initialized FixMatch with {num_classes} classes, threshold={confidence_threshold}, λ_u={lambda_u}")

    @tf.function
    def train_step(
        self,
        labeled_data: Tuple[tf.Tensor, tf.Tensor], 
        x_unlabeled_weak: tf.Tensor,
        x_unlabeled_strong: tf.Tensor,
        teacher_model: Optional[tfk.Model] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Execute a single training step with proper unified gradient flow.
        """
        x_labeled, y_labeled = labeled_data
        
        # Cast to float32 for consistency
        x_labeled = tf.cast(x_labeled, tf.float32)
        x_unlabeled_weak = tf.cast(x_unlabeled_weak, tf.float32)
        x_unlabeled_strong = tf.cast(x_unlabeled_strong, tf.float32)
        
        # Convert labels to one-hot
        y_labeled_one_hot = tf.one_hot(y_labeled, depth=self.num_classes)
        
        # UNIFIED gradient tape for proper backpropagation
        with tf.GradientTape() as tape:
            # Supervised loss from labeled samples
            logits_labeled = self.model(x_labeled, training=True)
            sup_loss = self.labeled_loss_fn(y_labeled_one_hot, logits_labeled)
            
            # Generate pseudo-labels using weak augmentations
            if teacher_model is not None:
                # Use EMA teacher for pseudo-labels (no gradients)
                logits_weak = teacher_model(x_unlabeled_weak, training=False)
            else:
                # Self-training: use current model for pseudo-labels
                logits_weak = self.model(x_unlabeled_weak, training=True)
            
            # Stop gradients for pseudo-label generation (critical for stability)
            logits_weak = tf.stop_gradient(logits_weak)
            
            # Process strongly augmented samples (with gradients)
            logits_strong = self.model(x_unlabeled_strong, training=True)
            
            # Calculate consistency loss with confidence thresholding
            unsup_loss, mask_ratio, pseudo_labels, mask, pseudo_accuracy = self._consistency_loss(
                logits_strong, logits_weak
            )
            
            # UNIFIED LOSS - combine supervised and unsupervised components
            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        # Calculate gradients from unified loss
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Optional gradient clipping
        if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        
        # Apply gradients to update model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.sup_loss_metric.update_state(sup_loss)
        self.unsup_loss_metric.update_state(unsup_loss)
        self.total_loss_metric.update_state(total_loss)
        self.mask_ratio_metric.update_state(mask_ratio)
        self.pseudo_label_accuracy_metric.update_state(pseudo_accuracy)

        return {
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "total_loss": total_loss,
            "mask_ratio": mask_ratio,
            "pseudo_accuracy": pseudo_accuracy,
            "learning_rate": self._get_optimizer_lr(),
        }
    
    def _get_optimizer_lr(self) -> tf.Tensor:
        """Safely extract current learning rate from optimizer."""
        lr = self.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            return lr
        elif callable(lr):
            return lr(self.optimizer.iterations)
        return tf.convert_to_tensor(lr)

    @tf.function  
    def _consistency_loss(
        self, 
        logits_strong: tf.Tensor, 
        logits_weak: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate consistency loss with confidence thresholding.
        """
        # Generate pseudo-labels with temperature sharpening
        pseudo_probs = tf.nn.softmax(logits_weak / self.T, axis=-1) 
        max_probs = tf.reduce_max(pseudo_probs, axis=-1)
        pseudo_labels = tf.argmax(pseudo_probs, axis=-1)
        
        # Get predictions from strong augmentations for accuracy calculation
        strong_probs = tf.nn.softmax(logits_strong, axis=-1)
        strong_predictions = tf.argmax(strong_probs, axis=-1)
        
        # Apply confidence threshold
        mask = tf.cast(max_probs >= self.confidence_threshold, tf.float32)
        
        # Convert pseudo-labels to one-hot
        pseudo_labels_one_hot = tf.one_hot(pseudo_labels, depth=self.num_classes, dtype=tf.float32)
        
        # Calculate per-sample loss
        per_sample_loss = self.unlabeled_loss_fn(pseudo_labels_one_hot, logits_strong)
        
        # Apply mask and calculate average with proper error handling
        masked_loss = per_sample_loss * mask
        mask_sum = tf.reduce_sum(mask)
        
        # Handle empty mask case to avoid NaN
        unsup_loss = tf.cond(
            mask_sum > 0,
            lambda: tf.reduce_sum(masked_loss) / mask_sum,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        # Calculate mask ratio and pseudo-accuracy
        mask_ratio = tf.reduce_mean(mask)
        pseudo_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pseudo_labels, strong_predictions), tf.float32)
        )
        
        return unsup_loss, mask_ratio, pseudo_labels, mask, pseudo_accuracy

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.sup_loss_metric.reset_states()
        self.unsup_loss_metric.reset_states()
        self.total_loss_metric.reset_states()
        self.mask_ratio_metric.reset_states()
        self.pseudo_label_accuracy_metric.reset_states()

    def get_metrics(self) -> Dict[str, tf.Tensor]:
        """Get current metrics values as a dictionary."""
        return {
            "sup_loss": self.sup_loss_metric.result(),
            "unsup_loss": self.unsup_loss_metric.result(),
            "total_loss": self.total_loss_metric.result(),
            "mask_ratio": self.mask_ratio_metric.result(),
            "pseudo_accuracy": self.pseudo_label_accuracy_metric.result(),
        }
    
    def on_epoch_end(self) -> None:
        """Reset metrics at the end of each epoch."""
        self.reset_metrics()