"""
FixMatch implementation for semi-supervised learning of mosquito wingbeat classification.
Implementation follows Sohn et al. "FixMatch: Simplifying Semi-Supervised Learning 
with Consistency and Confidence" (NeurIPS 2020).
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
    3. Combines supervised and unsupervised losses
    """
    
    def __init__(
        self, 
        model: tfk.Model, 
        num_classes: int, 
        confidence_threshold: float, 
        lambda_u: float = 1.0,
        T: float = 1.0,
        optimizer: Optional[tfk.optimizers.Optimizer] = None  # CHANGED: Accept optimizer instead of learning_rate
    ) -> None:
        """
        Initialize FixMatch trainer.
        
        Args:
            model: Neural network model for classification
            num_classes: Number of output classes
            confidence_threshold: Threshold for pseudo-labeling
            lambda_u: Weight for unsupervised loss
            T: Temperature for sharpening pseudo-labels
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        
        # Use provided optimizer or create default Adam
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
        # Rest of your initialization code...
        self.labeled_loss_fn = tfk.losses.CategoricalCrossentropy(from_logits=True)
        self.unlabeled_loss_fn = tfk.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tfk.losses.Reduction.NONE
        )
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.T = T

        # Metrics
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
        Execute a single training step implementing FixMatch with proper unified gradient flow.
        """
        x_labeled, y_labeled = labeled_data
        
        # Cast to float32 for consistency
        x_labeled = tf.cast(x_labeled, tf.float32)
        x_unlabeled_weak = tf.cast(x_unlabeled_weak, tf.float32)
        x_unlabeled_strong = tf.cast(x_unlabeled_strong, tf.float32)
        
        # Convert labels to one-hot
        y_labeled_one_hot = tf.one_hot(y_labeled, depth=self.num_classes)
        
        # SINGLE gradient tape for unified training (CRITICAL FIX)
        with tf.GradientTape() as tape:
            # Supervised loss
            logits_labeled = self.model(x_labeled, training=True)
            sup_loss = self.labeled_loss_fn(y_labeled_one_hot, logits_labeled)
            
            # Generate pseudo-labels using weak augmentations
            if teacher_model is not None:
                # Teacher-Student: use EMA teacher for pseudo-labels
                logits_weak = teacher_model(x_unlabeled_weak, training=False)
                logits_weak = tf.stop_gradient(logits_weak)
            else:
                # Self-training: use current model for pseudo-labels
                logits_weak = self.model(x_unlabeled_weak, training=True)
                logits_weak = tf.stop_gradient(logits_weak)
            
            # Process strongly augmented samples with student model
            logits_strong = self.model(x_unlabeled_strong, training=True)
            
            # Calculate consistency loss
            unsup_loss, mask_ratio, pseudo_labels, mask, pseudo_accuracy = self._consistency_loss(
                logits_strong, 
                logits_weak
            )
            
            # UNIFIED LOSS - This is crucial for proper SSL training
            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        # SINGLE gradient computation and application
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Optional gradient clipping
        if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update EMA teacher if provided
        if teacher_model is not None and hasattr(self, 'ema_decay_rate'):
            self._update_ema_teacher(teacher_model)
        
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
            "learning_rate": self.optimizer.learning_rate,
        }

    @tf.function  
    def _consistency_loss(
        self, 
        logits_strong: tf.Tensor, 
        logits_weak: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate FixMatch consistency loss with proper error handling.
        """
        # Generate pseudo-labels with temperature sharpening
        pseudo_probs = tf.nn.softmax(logits_weak / self.T, axis=-1) 
        max_probs = tf.reduce_max(pseudo_probs, axis=-1)
        pseudo_labels = tf.argmax(pseudo_probs, axis=-1)
        
        # Get predictions from strong augmentations
        strong_probs = tf.nn.softmax(logits_strong, axis=-1)
        strong_predictions = tf.argmax(strong_probs, axis=-1)
        
        # Apply confidence threshold
        mask = tf.cast(max_probs >= self.confidence_threshold, tf.float32)
        
        # Convert pseudo-labels to one-hot
        pseudo_labels_one_hot = tf.one_hot(pseudo_labels, depth=self.num_classes, dtype=tf.float32)
        
        # Calculate per-sample loss
        per_sample_loss = self.unlabeled_loss_fn(pseudo_labels_one_hot, logits_strong)
        
        # Apply mask and calculate average
        masked_loss = per_sample_loss * mask
        mask_sum = tf.reduce_sum(mask)
        
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

    def _update_ema_teacher(self, teacher_model: tfk.Model) -> None:
        """Update EMA teacher model weights."""
        if not hasattr(self, 'ema_decay_rate'):
            self.ema_decay_rate = 0.999
        
        for teacher_param, student_param in zip(teacher_model.trainable_variables, 
                                            self.model.trainable_variables):
            teacher_param.assign(
                self.ema_decay_rate * teacher_param + 
                (1.0 - self.ema_decay_rate) * student_param
            )

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.sup_loss_metric.reset_states()
        self.unsup_loss_metric.reset_states()
        self.total_loss_metric.reset_states()
        self.mask_ratio_metric.reset_states()

    def get_metrics(self) -> Dict[str, tf.Tensor]:
        """Get current metrics values as a dictionary."""
        return {
            "sup_loss": self.sup_loss_metric.result(),
            "unsup_loss": self.unsup_loss_metric.result(),
            "total_loss": self.total_loss_metric.result(),
            "mask_ratio": self.mask_ratio_metric.result(),
        }
    
    def on_epoch_end(self) -> None:
        """Reset metrics and optimizer state at the end of each epoch."""
        # Reset metrics
        self.reset_metrics()

