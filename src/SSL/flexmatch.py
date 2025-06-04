"""
Optimized FlexMatch implementation with correct curriculum thresholding and distribution alignment.
"""

import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np
import logging
from typing import Tuple, Dict, Union, Optional, Any, List, cast

# Configure logger
logger = logging.getLogger(__name__)

class FlexMatch:
    
    def __init__(
        self, 
        model: tfk.Model, 
        num_classes: int, 
        confidence_threshold: float, 
        lambda_u: float = 1.0,
        T: float = 1.0, 
        ema_decay: float = 0.999,
        use_DA: bool = False, 
        p_target_dist: Optional[tf.Tensor] = None,
        optimizer: Optional[tfk.optimizers.Optimizer] = None  
    ) -> None:
        """
        Initialize FlexMatch trainer with correct curriculum learning implementation.
        
        Args:
            model: Neural network model for classification
            num_classes: Number of output classes
            confidence_threshold: Base threshold (τ) for pseudo-labeling
            lambda_u: Weight for unsupervised loss
            T: Temperature for sharpening pseudo-labels
            ema_decay: Decay rate for EMA of class statistics
            use_DA: Whether to use Distribution Alignment
            p_target_dist: Target distribution for DA (uniform if None)
            optimizer: Optimizer for training (creates Adam if None)
        """
        self.model = model
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.T = T
        
        # Use provided optimizer or create default Adam
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            logger.warning("Using default optimizer. For best results, provide optimizer from trainer.")
            
        # Loss functions
        self.labeled_loss_fn = tfk.losses.CategoricalCrossentropy(from_logits=True)
        self.unlabeled_loss_fn = tfk.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tfk.losses.Reduction.NONE
        )
        
        # FlexMatch parameters
        self.ema_decay = ema_decay
        self.use_DA = use_DA
        
        # Distribution Alignment parameters
        self.p_target = p_target_dist if p_target_dist is not None else tf.ones(num_classes, dtype=tf.float32) / float(num_classes)
        self.p_model = tf.Variable(tf.zeros(num_classes, dtype=tf.float32), trainable=False, name="p_model_dist")

        # FIXED: Initialize class selection rates to a sensible value (half the confidence threshold)
        # This ensures curriculum learning starts with reasonable thresholds
        initial_rates = tf.ones([num_classes], dtype=tf.float32) * (self.confidence_threshold * 0.5)
        self.class_selection_rate = tf.Variable(
            initial_rates,
            trainable=False, 
            name="class_selection_rate"
        )
        
        # Metrics
        self.sup_loss_metric = tfk.metrics.Mean(name='sup_loss')
        self.unsup_loss_metric = tfk.metrics.Mean(name='unsup_loss')
        self.total_loss_metric = tfk.metrics.Mean(name='total_loss')
        self.mask_ratio_metric = tfk.metrics.Mean(name='mask_ratio')
        self.pseudo_label_accuracy_metric = tfk.metrics.Mean(name='pseudo_label_accuracy')
        
        # Curriculum learning counters
        self.pseudo_label_total_count = tf.Variable(
            tf.zeros(num_classes, dtype=tf.int32), 
            trainable=False, 
            name="pseudo_label_total_count"
        )
        self.pseudo_label_selected_count = tf.Variable(
            tf.zeros(num_classes, dtype=tf.int32), 
            trainable=False, 
            name="pseudo_label_selected_count"
        )
        
        logger.info(f"Initialized FlexMatch with {num_classes} classes, τ={confidence_threshold}, λ_u={lambda_u}")
        
    def update_p_model(self, pseudo_probs: tf.Tensor) -> None:
        """
        Update distribution alignment statistics with EMA.
        
        Args:
            pseudo_probs: Predicted probability distribution from the model [batch_size, num_classes]
        """
        if self.use_DA:
            # Calculate current batch distribution
            current_batch_dist = tf.reduce_mean(pseudo_probs, axis=0)
            
            # Ensure we don't have any zeros that would cause division issues
            current_batch_dist = tf.maximum(current_batch_dist, 1e-8)
            current_batch_dist = current_batch_dist / tf.reduce_sum(current_batch_dist)
            
            # Initialize or update with exponential moving average
            if tf.reduce_sum(tf.abs(self.p_model)) < 1e-8:
                # First update: initialize with current distribution
                self.p_model.assign(current_batch_dist)
                logger.info("Initialized p_model distribution for Distribution Alignment")
            else:
                # EMA update: p_model = α * p_model + (1 - α) * current_batch
                updated_dist = self.ema_decay * self.p_model + (1.0 - self.ema_decay) * current_batch_dist
                
                # Normalize to ensure it's a valid distribution
                updated_dist = updated_dist / tf.reduce_sum(updated_dist)
                self.p_model.assign(updated_dist)

    def update_curriculum(self) -> None:
        """
        FIXED: Update class selection rates based on accumulated counts.
        Formula: η_c = S_c / N_c (ratio of selected pseudo-labels per class)
        
        This follows the FlexMatch paper algorithm for dynamic thresholding.
        """
        N_counts = tf.cast(self.pseudo_label_total_count, tf.float32)
        S_counts = tf.cast(self.pseudo_label_selected_count, tf.float32)

        # Avoid division by zero for classes with no samples
        epsilon = 1e-7
        
        # Calculate η_c = S_c / N_c for each class
        new_selection_rates = S_counts / (N_counts + epsilon)
        
        # FIXED: Ensure sensible bounds on selection rates (0.05 to confidence_threshold)
        # Prevent extremely low rates that would accept noise
        new_selection_rates = tf.clip_by_value(
            new_selection_rates, 
            0.05,  # Minimum selection rate
            self.confidence_threshold  # Maximum is base threshold
        )
        
        # Assign to class_selection_rate with more stable update (EMA)
        # This prevents wild fluctuations between epochs
        updated_rates = 0.8 * self.class_selection_rate + 0.2 * new_selection_rates
        self.class_selection_rate.assign(updated_rates)
        
        logger.info(f"Updated class selection rates (η_c): {self.class_selection_rate.numpy()}")
        
        # Reset counters for next epoch
        self.pseudo_label_total_count.assign(tf.zeros(self.num_classes, dtype=tf.int32))
        self.pseudo_label_selected_count.assign(tf.zeros(self.num_classes, dtype=tf.int32))

    @tf.function
    def train_step(
        self,
        labeled_data: Tuple[tf.Tensor, tf.Tensor], 
        x_unlabeled_weak: tf.Tensor,
        x_unlabeled_strong: tf.Tensor,
        teacher_model: Optional[tfk.Model] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Execute a single training step with proper gradient flow.
        """
        x_labeled, y_labeled = labeled_data
        
        # Convert labels to one-hot
        y_labeled_one_hot = tf.one_hot(y_labeled, depth=self.num_classes)
        
        # UNIFIED gradient tape for proper backpropagation
        with tf.GradientTape() as tape:
            # Forward pass for labeled data (supervised loss)
            logits_labeled = self.model(x_labeled, training=True)
            sup_loss = self.labeled_loss_fn(y_labeled_one_hot, logits_labeled)
            
            # Forward pass for unlabeled data
            if teacher_model is not None:
                # Teacher-Student approach: use EMA teacher for pseudo-labels
                logits_weak = teacher_model(x_unlabeled_weak, training=False)
            else:
                # Self-training approach: use current model for pseudo-labels
                logits_weak = self.model(x_unlabeled_weak, training=True)
            
            # Stop gradients for pseudo-label generation
            logits_weak = tf.stop_gradient(logits_weak)
            
            # Forward pass for strongly augmented samples
            logits_strong = self.model(x_unlabeled_strong, training=True)
            
            # Calculate FlexMatch unsupervised loss
            unsup_loss, mask_ratio, pseudo_labels, selected_mask, pseudo_accuracy = self._flexmatch_loss(
                logits_strong, logits_weak
            )
            
            # Combined loss (unified gradient flow)
            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        # Calculate gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradient clipping if needed
        if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update distribution alignment statistics
        pseudo_probs = tf.nn.softmax(logits_weak / self.T, axis=-1)
        self.update_p_model(pseudo_probs)
        
        # Update class statistics for curriculum pseudo-labeling
        self._update_class_statistics(pseudo_labels, selected_mask)
        
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
            "class_selection_rates": self.class_selection_rate
        }
        
    def _get_optimizer_lr(self) -> tf.Tensor:
        """Safely extract current learning rate from optimizer."""
        lr = self.optimizer.learning_rate
        if isinstance(lr, tf.Variable):
            return lr
        elif callable(lr):
            return lr(self.optimizer.iterations)
        return tf.convert_to_tensor(lr)

    def _update_class_statistics(self, pseudo_labels: tf.Tensor, selected: tf.Tensor) -> None:
        """
        FIXED: Update counters for curriculum pseudo-labeling with proper atomic operations.
        
        Args:
            pseudo_labels: Pseudo-labels assigned to unlabeled samples (class indices)
            selected: Binary mask indicating which samples were selected
        """
        # Convert to int32 for bincount
        pseudo_labels = tf.cast(pseudo_labels, tf.int32)
        selected = tf.cast(selected, tf.int32)
        
        # Count total pseudo-labels per class
        total_counts = tf.math.bincount(
            pseudo_labels, 
            minlength=self.num_classes, 
            dtype=tf.int32
        )
        
        # Count selected pseudo-labels per class
        # Only count samples where selected == 1
        selected_indices = tf.boolean_mask(pseudo_labels, tf.cast(selected, tf.bool))
        
        # Handle empty case to avoid errors
        if tf.size(selected_indices) > 0:
            selected_counts = tf.math.bincount(
                selected_indices,
                minlength=self.num_classes,
                dtype=tf.int32
            )
        else:
            selected_counts = tf.zeros(self.num_classes, dtype=tf.int32)
        
        # Atomic updates to prevent race conditions
        self.pseudo_label_total_count.assign_add(total_counts)
        self.pseudo_label_selected_count.assign_add(selected_counts)
            
    @tf.function
    def _flexmatch_loss(
        self, 
        logits_strong: tf.Tensor, 
        logits_weak: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        FIXED: Calculate FlexMatch unsupervised loss with correct curriculum thresholding.
        
        Formula from paper: τ_c = τ * sqrt(η_c / η̄)
        Where:
        - τ_c is the class-specific threshold
        - τ is the base threshold
        - η_c is the class-specific selection rate
        - η̄ is the average selection rate
        """
        # Generate pseudo-labels from weak augmentations
        pseudo_probs = tf.nn.softmax(logits_weak / self.T, axis=-1)
        
        # Apply distribution alignment if enabled
        if self.use_DA and tf.reduce_sum(self.p_model) > 1e-6:
            # Ensure p_model is valid (no zeros)
            p_model_safe = tf.maximum(self.p_model, 1e-8)
            
            # Calculate alignment ratio: p_target / p_model
            alignment_ratio = self.p_target / p_model_safe
            
            # Apply alignment: p' = p * (p_target / p_model)
            pseudo_probs = pseudo_probs * alignment_ratio
            
            # Re-normalize to ensure valid probabilities
            pseudo_probs = pseudo_probs / (tf.reduce_sum(pseudo_probs, axis=-1, keepdims=True) + 1e-9)
        
        # Get confidence and predicted class
        max_probs = tf.reduce_max(pseudo_probs, axis=-1)
        pseudo_labels = tf.argmax(pseudo_probs, axis=-1)
        
        # FIXED: FlexMatch curriculum thresholding with correct formula
        # Get selection rate for each sample's class
        sample_selection_rates = tf.gather(self.class_selection_rate, pseudo_labels)
        
        # Calculate average selection rate across all classes
        avg_selection_rate = tf.reduce_mean(self.class_selection_rate)
        
        # Small constant to prevent division by zero
        delta = 0.05
        
        # Ensure both numerator and denominator are at least delta
        safe_sample_rates = tf.maximum(sample_selection_rates, delta)
        safe_avg_rate = tf.maximum(avg_selection_rate, delta)
        
        # CORRECT FORMULA: τ_c = τ * sqrt(η_c / η̄)
        # This adjusts threshold based on relative class difficulty
        flex_thresholds = self.confidence_threshold * tf.sqrt(safe_sample_rates / safe_avg_rate)
        
        # Clamp thresholds to reasonable bounds
        flex_thresholds = tf.clip_by_value(
            flex_thresholds, 
            0.5,  # Lower bound prevents accepting noise
            0.95  # Upper bound prevents rejecting everything
        )
        
        # Apply flexible thresholds for loss masking
        flex_mask = tf.cast(max_probs >= flex_thresholds, tf.float32)
        
        # Track samples selected by fixed threshold (for curriculum updates)
        fixed_threshold_mask = tf.cast(max_probs >= self.confidence_threshold, tf.float32)
        
        # Convert to one-hot for loss calculation
        pseudo_labels_one_hot = tf.one_hot(pseudo_labels, depth=self.num_classes, dtype=tf.float32)
        
        # Calculate per-sample cross-entropy loss
        per_sample_loss = self.unlabeled_loss_fn(pseudo_labels_one_hot, logits_strong)
        
        # Apply flexible mask
        masked_loss = per_sample_loss * flex_mask
        
        # Calculate mean loss (handle zero mask case)
        mask_sum = tf.reduce_sum(flex_mask)
        unsup_loss = tf.cond(
            mask_sum > 0.0,
            lambda: tf.reduce_sum(masked_loss) / mask_sum,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        # Calculate mask ratio for monitoring
        mask_ratio = tf.reduce_mean(flex_mask)
        
        # Calculate pseudo-label accuracy
        strong_probs = tf.nn.softmax(logits_strong, axis=-1)
        strong_predictions = tf.argmax(strong_probs, axis=-1)
        correct_predictions = tf.cast(tf.equal(pseudo_labels, strong_predictions), tf.float32)
        pseudo_accuracy = tf.reduce_mean(correct_predictions)
        
        return unsup_loss, mask_ratio, pseudo_labels, fixed_threshold_mask, pseudo_accuracy

    def reset_metrics(self) -> None:
        """Reset all metrics and counters."""
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
            "class_selection_rate": self.class_selection_rate,
        }
    
    def on_epoch_end(self) -> None:
        """Reset metrics and update curriculum at the end of each epoch."""
        # Reset metrics
        self.reset_metrics()
        
        # Update curriculum for next epoch
        self.update_curriculum()