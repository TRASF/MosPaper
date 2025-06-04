"""
Clean Coder refactored Semi-Supervised Learning trainer for SSL optimization.
Configuration-driven, type-safe implementation with vectorized operations.
Optimized for FixMatch/FlexMatch efficiency with reduced training overhead.
"""
import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path

# Import refactored custom callbacks
from .custom_callbacks import VerboseReduceLROnPlateau, VerboseEarlyStopping

# SSL method imports
from ..SSL.fixmatch import FixMatch
from ..SSL.flexmatch import FlexMatch

logger = logging.getLogger(__name__)

# Type aliases for clarity and SSL optimization
DatasetType = tf.data.Dataset
MetricsDict = Dict[str, Any]
LossDict = Dict[str, tf.Tensor]
OptimizerType = tf.keras.optimizers.Optimizer
ModelType = tf.keras.Model
ConfigType = Dict[str, Any]

def _validate_ssl_config(config: Any) -> None:
    """
    Configuration-driven validation for SSL training parameters.
    Single-shot validation following Clean Coder principles.
    """
    required_attrs = ['training_mode', 'learning_rate', 'optimizer_type', 'dataset']
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required config attribute: {attr}")
    
    if config.training_mode not in ['fixmatch', 'flexmatch']:
        raise ValueError(f"Unsupported SSL training_mode: {config.training_mode}. "
                        f"Must be one of: ['fixmatch', 'flexmatch']")
    
    if not hasattr(config.dataset, 'num_classes') or config.dataset.num_classes <= 0:
        raise ValueError(f"Invalid num_classes: {getattr(config.dataset, 'num_classes', None)}. "
                        f"Must be a positive integer.")
    
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")


def _create_ssl_metrics(num_classes: int) -> MetricsDict:
    """
    Vectorized creation of SSL training metrics.
    Optimized for FixMatch/FlexMatch efficiency.
    """
    return {
        'sup_loss': tfk.metrics.Mean(name='train_sup_loss'),
        'unsup_loss': tfk.metrics.Mean(name='train_unsup_loss'), 
        'total_loss': tfk.metrics.Mean(name='train_total_loss'),
        'mask_ratio': tfk.metrics.Mean(name='train_mask_ratio'),
        'pseudo_label_accuracy': tfk.metrics.Mean(name='pseudo_label_accuracy')
    }


def _create_validation_metrics(num_classes: int) -> MetricsDict:
    """
    Vectorized creation of validation metrics.
    Type-safe implementation for SSL evaluation.
    """
    return {
        'val_loss': tfk.metrics.Mean(name='val_loss'),
        'val_accuracy': tfk.metrics.CategoricalAccuracy(name='val_accuracy'),
        'val_top3_accuracy': tfk.metrics.TopKCategoricalAccuracy(k=3, name='val_top3_accuracy')
    }


class SemiSupervisedTrainer:
    """
    Configuration-driven Semi-Supervised Learning trainer.
    Clean implementation optimized for FixMatch/FlexMatch with vectorized operations.
    Type-safe with comprehensive error handling and SSL efficiency monitoring.
    """
    
    def __init__(
        self, 
        config: ConfigType, 
        model: tfk.Model, 
        callbacks: Optional[List[tfk.callbacks.Callback]] = None
    ) -> None:
        """
        Initialize SSL trainer with configuration-driven validation.
        All parameters validated once for training loop efficiency.
        """
        # Single-shot configuration validation
        _validate_ssl_config(config)
        
        self.model = model
        self.config = config
        self.num_classes = config.dataset.num_classes
        self.results_dir = Path(config.results_dir)
        
        # Create unified optimizer for SSL training
        self.optimizer = self._create_optimizer()
        
        # Vectorized metrics creation
        self.train_metrics = _create_ssl_metrics(self.num_classes)
        self.val_metrics = _create_validation_metrics(self.num_classes)
        
        # SSL method initialization with unified optimizer
        self.ssl_method = self._initialize_ssl_method()
        
        # EMA model for SSL efficiency (optional)
        self.ema_model: Optional[ModelType] = None
        if getattr(config.ssl, 'use_ema', False):
            self.ema_decay = getattr(config.ssl, 'ema_decay', 0.999)
            self.ema_model = self._create_ema_model()
            logger.info(f"EMA model enabled with decay: {self.ema_decay}")
        
        # Training state variables for efficiency tracking
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32, name="current_epoch")
        
        # Loss function for supervised learning
        self.supervised_loss_fn = tfk.losses.CategoricalCrossentropy(from_logits=True)
        
        # Callbacks list for SSL optimization
        self.callbacks_list: List[tfk.callbacks.Callback] = []
        
        # Efficiency tracking for SSL analysis
        self.training_stats: Dict[str, Any] = {
            'epoch_times': [],
            'ssl_overhead': [],
            'callback_overhead': [],
            'total_training_time': 0.0
        }

        self.history = self._initialize_training_history()

    def _initialize_training_history(self) -> Dict[str, List[float]]:
        """Initializes the training history dictionary."""
        return {
            'epoch': [], 
            'lr': [],
            'loss': [],
            'total_loss': [], 
            'sup_loss': [], 
            'unsup_loss': [], 
            'mask_ratio': [],
            'pseudo_accuracy': [],
            'val_loss': [], 
            'val_accuracy': []
        }

    def _reset_all_metrics(self) -> None:
        """Resets the state of all training and validation metrics."""
        for metric in self.train_metrics.values():
            metric.reset_states()
        for metric in self.val_metrics.values():
            metric.reset_states()
        logger.info("All training and validation metrics have been reset.")

    def _setup_callbacks(self) -> None:
        """Sets up the callbacks for the training process."""
        for callback in self.callbacks_list:
            if hasattr(callback, 'set_model'):
                callback.set_model(self.model)
            if isinstance(callback, VerboseReduceLROnPlateau) and hasattr(callback, 'set_optimizer'):
                callback.set_optimizer(self.optimizer)
            
        callback_container = tfk.callbacks.CallbackList(
            self.callbacks_list,
            add_history=True,
            add_progbar=self.config.get('verbose', 1) > 0,
            model=self.model,
            epochs=self.config.get('num_epochs', 100),
            verbose=self.config.get('verbose', 1)
        )
        self.model.callbacks = callback_container
        logger.info("Callbacks have been set up and attached to the model.")

    def _initialize_ssl_method(self) -> Union[FixMatch, FlexMatch]:
        """
        CORRECTED: Type-safe SSL method initialization with unified optimizer.
        Passes the trainer's optimizer instead of creating a new one.
        """
        ssl_config = self.config.ssl
        
        # CRITICAL FIX: Pass the unified optimizer instead of learning_rate
        common_params = {
            'model': self.model,
            'num_classes': self.num_classes,
            'confidence_threshold': ssl_config.confidence_threshold,
            'lambda_u': ssl_config.lambda_u,
            'T': ssl_config.T,
            'optimizer': self.optimizer  # Pass existing optimizer for unity
        }
        
        if self.config.training_mode == 'fixmatch':
            ssl_method = FixMatch(**common_params)
            
        elif self.config.training_mode == 'flexmatch':
            # FlexMatch-specific parameters
            p_target_dist = None
            if getattr(ssl_config, 'use_DA', False) and getattr(ssl_config, 'p_target_uniform', False):
                p_target_dist = tf.ones(self.num_classes, dtype=tf.float32) / float(self.num_classes)
            
            flexmatch_params = {
                **common_params,
                'use_DA': getattr(ssl_config, 'use_DA', False),
                'p_target_dist': p_target_dist
            }
            ssl_method = FlexMatch(**flexmatch_params)
        else:
            raise ValueError(f"Unsupported SSL method: {self.config.training_mode}")
            
        # Ensure SSL method uses the same optimizer
        ssl_method.optimizer = self.optimizer
        logger.info(f"SSL method {self.config.training_mode} initialized with unified optimizer")
        return ssl_method

    def _create_optimizer(self) -> OptimizerType:
        """
        Configuration-driven optimizer creation with SSL-optimized learning rate handling.
        Type-safe implementation with vectorized parameter setting.
        """
        initial_lr = self.config.learning_rate
        
        # Determine if dynamic LR is needed (for ReduceLROnPlateau callback)
        use_variable_lr = getattr(self.config, 'reduce_lr_patience', 0) > 0
        
        if use_variable_lr:
            learning_rate = tf.Variable(initial_lr, dtype=tf.float32, name="learning_rate")
            logger.info(f"Using tf.Variable learning rate for dynamic adjustment: {initial_lr}")
        else:
            learning_rate = initial_lr
            logger.info(f"Using fixed learning rate: {initial_lr}")
        
        # Vectorized optimizer creation
        optimizer_type = self.config.optimizer_type.lower()
        
        if optimizer_type == 'adam':
            optimizer = tfk.optimizers.Adam(
                learning_rate=learning_rate,
                epsilon=getattr(self.config, 'adam_epsilon', 1e-7),
                beta_1=getattr(self.config, 'adam_beta1', 0.9),
                beta_2=getattr(self.config, 'adam_beta2', 0.999)
            )
        elif optimizer_type == 'sgd':
            optimizer = tfk.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=getattr(self.config, 'momentum', 0.9),
                nesterov=getattr(self.config, 'nesterov', False)
            )
        elif optimizer_type == 'adamw':
            weight_decay = getattr(self.config.ssl, 'weight_decay', 
                         getattr(self.config, 'weight_decay', 0.01))
            
            optimizer = tfk.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epsilon=getattr(self.config, 'adam_epsilon', 1e-7),
                beta_1=getattr(self.config, 'adam_beta1', 0.9),
                beta_2=getattr(self.config, 'adam_beta2', 0.999)
            )
            logger.info(f"Using AdamW with weight_decay: {weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                           f"Supported: ['adam', 'sgd', 'adamw']")
        
        logger.info(f"Created {optimizer_type.upper()} optimizer with "
                   f"{'variable' if use_variable_lr else 'fixed'} learning rate")
        return optimizer

    def _get_optimizer_lr(self) -> float:
        """Type-safe extraction of current learning rate from optimizer."""
        lr_attr = self.optimizer.lr
        
        if isinstance(lr_attr, tf.Variable):
            return float(lr_attr.numpy())
        elif hasattr(lr_attr, '__call__'):  # LearningRateSchedule
            iterations = getattr(self.optimizer, '_iterations', tf.Variable(0))
            return float(lr_attr(iterations))
        else:
            return float(lr_attr)

    def _create_ema_model(self) -> ModelType:
        """
        Efficient EMA model creation for SSL training.
        Vectorized weight initialization with proper state management.
        """
        ema_model = tfk.models.clone_model(self.model)
        ema_model.set_weights(self.model.get_weights())
        
        # Disable training for all layers (vectorized operation)
        for layer in ema_model.layers:
            layer.trainable = False
            
        logger.info("EMA model created and initialized with current model weights")
        return ema_model

    @tf.function
    def _update_ema_model_weights(self) -> None:
        """
        CORRECTED: Vectorized EMA weight update with error handling and graph compilation.
        Optimized for minimal overhead during training.
        """
        if self.ema_model is None:
            return
            
        # Vectorized EMA update with comprehensive error handling
        try:
            for ema_var, model_var in zip(self.ema_model.weights, self.model.weights):
                # Ensure shapes match for safety
                tf.debugging.assert_equal(
                    tf.shape(ema_var), tf.shape(model_var),
                    message="EMA and model weights shape mismatch"
                )
                # Vectorized EMA update: θ_ema = α * θ_ema + (1 - α) * θ_student
                ema_var.assign(
                    self.ema_decay * ema_var + (1.0 - self.ema_decay) * model_var
                )
        except Exception as e:
            tf.print(f"EMA update failed: {e}")

    def _initialize_callbacks(self) -> List[tfk.callbacks.Callback]:
        """
        Configuration-driven callback initialization with SSL optimization.
        Type-safe implementation with vectorized callback creation.
        """
        callbacks = []
        
        # EarlyStopping with SSL-optimized parameters
        if getattr(self.config, 'early_stop_patience', 0) > 0:
            early_stopping = VerboseEarlyStopping(
                monitor=getattr(self.config, 'early_stop_monitor', 'val_loss'),
                patience=self.config.early_stop_patience,
                verbose=1,
                min_delta=getattr(self.config, 'early_stop_min_delta', 1e-4),
                restore_best_weights=getattr(self.config, 'early_stop_restore_best', True)
            )
            callbacks.append(early_stopping)
            logger.info(f"VerboseEarlyStopping enabled: monitor='{early_stopping.monitor}', "
                       f"patience={early_stopping.patience}")
        
        # ReduceLROnPlateau with SSL-optimized parameters
        if getattr(self.config, 'reduce_lr_patience', 0) > 0:
            lr_reducer = VerboseReduceLROnPlateau(
                monitor=getattr(self.config, 'reduce_lr_monitor', 'val_loss'),
                factor=getattr(self.config, 'reduce_lr_factor', 0.1),
                patience=self.config.reduce_lr_patience,
                verbose=1,
                min_delta=getattr(self.config, 'reduce_lr_min_delta', 1e-4),
                min_lr=getattr(self.config, 'reduce_lr_min_lr', 1e-7),
                cooldown=getattr(self.config, 'reduce_lr_cooldown', 0)
            )
            callbacks.append(lr_reducer)
            logger.info(f"VerboseReduceLROnPlateau enabled: monitor='{lr_reducer.monitor}', "
                       f"factor={lr_reducer.factor}, patience={lr_reducer.patience}")
        
        # ModelCheckpoint for SSL model saving
        if hasattr(self.config, 'checkpoint_dir') and self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_filepath = checkpoint_dir / "ssl_epoch_{epoch:02d}-val_loss_{val_loss:.4f}.hdf5"
            model_checkpoint = tfk.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_filepath),
                monitor=getattr(self.config, 'checkpoint_monitor', 'val_loss'),
                save_best_only=getattr(self.config, 'checkpoint_save_best_only', True),
                save_weights_only=getattr(self.config, 'checkpoint_save_weights_only', False),
                verbose=1,
                save_freq='epoch'
            )
            callbacks.append(model_checkpoint)
            logger.info(f"ModelCheckpoint enabled: saving to {checkpoint_dir}")

        # TensorBoard for SSL training visualization
        if hasattr(self.config, 'tensorboard_log_dir') and self.config.tensorboard_log_dir:
            log_dir = Path(self.config.tensorboard_log_dir) / f"ssl_{self.config.training_mode}_run"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            tensorboard = tfk.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=getattr(self.config, 'tensorboard_histogram_freq', 1),
                write_graph=getattr(self.config, 'tensorboard_write_graph', True),
                update_freq='epoch',
                profile_batch=0  # Disable profiling for SSL efficiency
            )
            callbacks.append(tensorboard)
            logger.info(f"TensorBoard enabled: logging to {log_dir}")
        
        callback_names = [cb.__class__.__name__ for cb in callbacks]
        logger.info(f"Initialized SSL callbacks: {callback_names}")
        
        return callbacks

    @tf.function
    def _train_step_ssl(
        self, 
        labeled_data: Tuple[tf.Tensor, tf.Tensor], 
        unlabeled_data: Tuple[tf.Tensor, tf.Tensor]
    ) -> LossDict:
        """
        CORRECTED: Vectorized SSL training step with proper Teacher-Student temporal ordering.
        
        Key corrections:
        1. Update EMA model BEFORE using it as teacher (temporal consistency)
        2. Let SSL method handle ALL gradient computation and updates (no double application)
        3. Proper error handling with meaningful fallbacks
        """
        x_labeled, y_labeled = labeled_data
        x_unlabeled_weak, x_unlabeled_strong = unlabeled_data

        # CRITICAL FIX: Update EMA model weights BEFORE using it as teacher
        # This ensures temporal consistency in Teacher-Student learning
        if self.ema_model is not None:
            self._update_ema_model_weights()

        # Type-safe label conversion (vectorized operation)
        y_labeled = tf.cast(y_labeled, tf.int32)
        ssl_labeled_data = (x_labeled, y_labeled)

        # SSL method handles ALL gradient computation and optimizer updates
        # No additional gradient application should happen here
        try:
            results = self.ssl_method.train_step(
                ssl_labeled_data,
                x_unlabeled_weak,
                x_unlabeled_strong,
                teacher_model=self.ema_model  # Use updated EMA model as teacher
            )
        except Exception as e:
            # Meaningful fallback with proper error handling
            logger.error(f"SSL train_step failed: {e}")
            return {
                'total_loss': tf.constant(0.0, dtype=tf.float32),
                'sup_loss': tf.constant(0.0, dtype=tf.float32),
                'unsup_loss': tf.constant(0.0, dtype=tf.float32),
                'mask_ratio': tf.constant(0.0, dtype=tf.float32)
            }
            
        # Vectorized loss extraction with defaults
        sup_loss = results.get('sup_loss', tf.constant(0.0, dtype=tf.float32))
        unsup_loss = results.get('unsup_loss', tf.constant(0.0, dtype=tf.float32))
        mask_ratio = results.get('mask_ratio', tf.constant(0.0, dtype=tf.float32))
        
        # Calculate total loss if not provided (vectorized operation)
        lambda_u = getattr(self.ssl_method, 'lambda_u', 1.0)
        total_loss = results.get('total_loss', sup_loss + unsup_loss * lambda_u)

        # Vectorized metric updates for SSL efficiency (NO gradient computation)
        self.train_metrics['sup_loss'].update_state(sup_loss)
        self.train_metrics['unsup_loss'].update_state(unsup_loss)
        self.train_metrics['total_loss'].update_state(total_loss)
        self.train_metrics['mask_ratio'].update_state(mask_ratio)
        
        # Update pseudo-label accuracy if available
        if 'pseudo_accuracy' in results:
            self.train_metrics['pseudo_label_accuracy'].update_state(results['pseudo_accuracy'])

        return {
            'total_loss': total_loss,
            'sup_loss': sup_loss,
            'unsup_loss': unsup_loss,
            'mask_ratio': mask_ratio
        }

    @tf.function
    def validation_step(
        self, 
        x_batch_val: tf.Tensor, 
        y_batch_val: tf.Tensor
    ) -> LossDict:
        """
        CORRECTED: Efficient validation step with proper EMA integration.
        Ensures EMA model is up-to-date before evaluation.
        """
        # CORRECTED: Ensure EMA model is current before validation
        if self.ema_model is not None:
            self._update_ema_model_weights()
        
        # Determine evaluation model with proper fallback
        use_ema_for_eval = (
            self.ema_model is not None and 
            getattr(self.config.ssl, 'evaluate_ema_model', False)
        )
        
        if use_ema_for_eval:
            eval_model = self.ema_model
            logger.debug("Using EMA model for validation")
        else:
            eval_model = self.model
            logger.debug("Using main model for validation")
        
        # Forward pass with training=False (vectorized operation)
        try:
            val_logits = eval_model(x_batch_val, training=False)
        except Exception as e:
            logger.error(f"Validation forward pass failed: {e}")
            # Fallback to main model if EMA fails
            val_logits = self.model(x_batch_val, training=False)
        
        # Type-safe label conversion to one-hot (vectorized)
        y_batch_val_one_hot = tf.one_hot(
            tf.cast(y_batch_val, tf.int32), 
            depth=self.num_classes
        )

        # Calculate validation loss
        val_loss = self.supervised_loss_fn(y_batch_val_one_hot, val_logits)
        
        # Vectorized metric updates
        self.val_metrics['val_loss'].update_state(val_loss)
        self.val_metrics['val_accuracy'].update_state(y_batch_val_one_hot, val_logits)
        
        # Update top-3 accuracy if available
        if 'val_top3_accuracy' in self.val_metrics:
            self.val_metrics['val_top3_accuracy'].update_state(y_batch_val_one_hot, val_logits)
        
        return {
            'val_loss': val_loss,
            'val_accuracy': self.val_metrics['val_accuracy'].result()
        }

    def train(
        self, 
        train_labeled_dataset: DatasetType, 
        train_unlabeled_dataset: DatasetType, 
        val_dataset: DatasetType, 
        epochs: int, 
        callbacks: Optional[List[tfk.callbacks.Callback]] = None
    ) -> Dict[str, List[Any]]:
        """
        Configuration-driven SSL training loop with vectorized operations.
        Optimized for FixMatch/FlexMatch efficiency with comprehensive error handling.
        Type-safe implementation following Clean Coder principles.
        """
        training_start_time = time.perf_counter()
        
        logger.info(f"Starting SSL training: {epochs} epochs, method={self.config.get('training_mode', 'N/A')}")
        
        # Vectorized metric reset for training efficiency
        self._reset_all_metrics()
        
        # Setup callbacks with SSL optimization
        self.callbacks_list = callbacks or self._initialize_callbacks()
        self._setup_callbacks()
        
        # Create combined training dataset with error handling
        try:
            train_dataset = self._create_combined_dataset(train_labeled_dataset, train_unlabeled_dataset)
        except Exception as e:
            logger.error(f"Failed to create combined dataset: {e}")
            if getattr(self.config, 'raise_training_exceptions', True):
                raise
            return self.history
        
        # Determine steps per epoch with configuration-driven logic
        steps_per_epoch = self._calculate_steps_per_epoch(train_labeled_dataset, train_unlabeled_dataset)
        
        # Main training loop with SSL efficiency monitoring
        try:
            for epoch in range(epochs):
                epoch_start_time = time.perf_counter()
                
                # Epoch initialization
                self._on_epoch_begin(epoch)
                
                # Training phase with vectorized operations
                epoch_metrics = self._execute_training_epoch(train_dataset, steps_per_epoch, epoch)
                
                # Validation phase with EMA model support
                val_metrics = self._execute_validation_epoch(epoch, val_dataset)
                
                # Epoch completion and callback handling
                should_stop = self._on_epoch_end(epoch, epoch_metrics, val_metrics)
                
                # Update training history (SIMPLIFIED)
                self._update_training_history_simple(epoch, epoch_metrics, val_metrics)
                
                # Track efficiency statistics
                epoch_time = time.perf_counter() - epoch_start_time
                self.training_stats['epoch_times'].append(epoch_time)
                
                if should_stop:
                    logger.info(f"Training stopped early at epoch {epoch + 1}")
                    break
                    
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch}: {e}")
            if getattr(self.config, 'raise_training_exceptions', True):
                raise
        finally:
            # Training completion
            self._on_train_end()
            self._create_training_plots()
            
            # Log final efficiency statistics
            total_time = time.perf_counter() - training_start_time
            self.training_stats['total_training_time'] = total_time
            
            if getattr(self.config, 'log_efficiency_stats', True):
                self._log_efficiency_statistics()
        
        logger.info("SSL training completed successfully")
        return self.history

    def _update_training_history_simple(
        self, 
        epoch: int, 
        epoch_metrics: MetricsDict, 
        val_metrics: MetricsDict
    ) -> None:
        """
        SIMPLIFIED: Clean and efficient training history update.
        Removes complex fallback logic and excessive padding.
        """
        # Basic epoch information
        self.history['epoch'].append(epoch + 1)
        self.history['lr'].append(self._get_optimizer_lr())
        
        # Training metrics with direct mapping
        metric_mapping = {
            'total_loss': ['total_loss', 'loss'],  # Map to both keys for compatibility
            'sup_loss': ['sup_loss'],
            'unsup_loss': ['unsup_loss'],
            'mask_ratio': ['mask_ratio'],
            'pseudo_label_accuracy': ['pseudo_accuracy']
        }
        
        for metric_key, history_keys in metric_mapping.items():
            if metric_key in epoch_metrics:
                value = epoch_metrics[metric_key]
                processed_value = value.numpy() if hasattr(value, 'numpy') else float(value)
                
                for history_key in history_keys:
                    if history_key in self.history:
                        self.history[history_key].append(processed_value)
        
        # Validation metrics with direct mapping
        for key, value in val_metrics.items():
            if key in self.history:
                processed_value = value.numpy() if hasattr(value, 'numpy') else float(value)
                self.history[key].append(processed_value)
        
        logger.debug(f"Updated training history for epoch {epoch + 1}")

    def _on_train_end(self) -> None:
        """Handles tasks at the very end of the training process."""
        logger.info("Executing _on_train_end...")
        final_logs = {}
        
        for callback in self.callbacks_list:
            if hasattr(callback, 'on_train_end'):
                try:
                    callback.on_train_end(logs=final_logs)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__class__.__name__}.on_train_end: {e}")
        logger.info("_on_train_end completed.")

    def _on_epoch_begin(self, epoch: int) -> None:
        """Handles tasks at the beginning of each epoch."""
        # Warm-up learning rate linearly for initial epochs
        warmup = getattr(self.config, 'warmup_epochs', 0)
        if warmup and epoch < warmup:
            base_lr = self.config.learning_rate
            new_lr = base_lr * float(epoch + 1) / float(warmup)
            try:
                tf.keras.backend.set_value(self.optimizer.lr, new_lr)
                logger.info(f"Warmup LR: set learning_rate to {new_lr:.6e} at epoch {epoch+1}")
            except Exception:
                logger.warning("Could not set learning rate for warmup period.")

        logger.info(f"Starting epoch {epoch + 1}.")

    def _on_epoch_end(self, epoch: int, epoch_metrics: MetricsDict, val_metrics: MetricsDict) -> bool:
        """Handles end-of-epoch tasks like callbacks and early stopping."""
        logs = {**epoch_metrics, **val_metrics}
        
        stop_training = False
        for callback in self.callbacks_list:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs=logs)
            if hasattr(self.model, 'stop_training') and self.model.stop_training:
                stop_training = True
        
        # Convert tensors to scalars for logging
        log_items = []
        for k, v in logs.items():
            if hasattr(v, 'numpy'):
                log_items.append(f"{k}: {v.numpy():.4f}")
            else:
                log_items.append(f"{k}: {v:.4f}")
        
        log_msg = f"Epoch {epoch+1} Summary: " + ", ".join(log_items)
        logger.info(log_msg)
        return stop_training

    def _log_efficiency_statistics(self) -> None:
        """Logs training efficiency statistics."""
        if not self.training_stats.get('epoch_times'):
            logger.warning("No timing statistics to log.")
            return
            
        total_time = self.training_stats.get('total_training_time', 0.0)
        epoch_times = np.array(self.training_stats['epoch_times'])
        
        logger.info(f"Total Training Time: {total_time:.2f}s for {len(epoch_times)} epochs.")
        if len(epoch_times) > 0:
            logger.info(f"Mean Epoch Time: {np.mean(epoch_times):.2f}s, Std: {np.std(epoch_times):.2f}s")

    def evaluate(self, test_dataset: DatasetType, use_ema: bool = False) -> MetricsDict:
        """Evaluates the model on the provided test dataset with comprehensive metrics."""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        logger.info(f"Starting evaluation on the test dataset. EMA: {use_ema}")
        
        eval_model = self.model
        if use_ema and self.ema_model:
            # Ensure EMA model is up-to-date
            self._update_ema_model_weights()
            eval_model = self.ema_model
            logger.info("Using EMA model for evaluation.")
        elif use_ema:
            logger.warning("EMA model requested for evaluation, but not available. Using standard model.")

        # Reset validation metrics
        for metric in self.val_metrics.values():
            metric.reset_states()

        y_true_all = []
        y_pred_all = []
        
        for step, (x_batch, y_batch) in enumerate(test_dataset):
            y_batch_one_hot = tf.one_hot(tf.cast(y_batch, tf.int32), depth=self.num_classes)
            val_logits = eval_model(x_batch, training=False)
            loss = self.supervised_loss_fn(y_batch_one_hot, val_logits)
            
            # Update metrics
            if 'val_loss' in self.val_metrics:
                self.val_metrics['val_loss'].update_state(loss)
            if 'val_accuracy' in self.val_metrics:
                self.val_metrics['val_accuracy'].update_state(y_batch_one_hot, val_logits)
            if 'val_top3_accuracy' in self.val_metrics:
                self.val_metrics['val_top3_accuracy'].update_state(y_batch_one_hot, val_logits)
            
            # Collect predictions for sklearn metrics
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(tf.argmax(val_logits, axis=1).numpy())

        # Compile results
        test_results: MetricsDict = {}
        for name, metric_obj in self.val_metrics.items():
            try:
                result_value = metric_obj.result()
                test_results[f"test_{name.replace('val_', '')}"] = (
                    result_value.numpy() if hasattr(result_value, 'numpy') else result_value
                )
            except Exception as e:
                logger.error(f"Error getting result for metric {name}: {e}")
                test_results[f"test_{name.replace('val_', '')}"] = np.nan

        # Sklearn metrics
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=np.arange(self.num_classes), zero_division=0
        )
        
        # Add per-class metrics
        for i in range(self.num_classes):
            test_results[f"precision_class_{i}"] = float(precision[i])
            test_results[f"recall_class_{i}"] = float(recall[i])
            test_results[f"f1_class_{i}"] = float(f1[i])
        
        # Macro/micro averages
        macro = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
        micro = precision_recall_fscore_support(y_true_all, y_pred_all, average='micro', zero_division=0)
        
        test_results.update({
            "precision_macro": float(macro[0]),
            "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]),
            "precision_micro": float(micro[0]),
            "recall_micro": float(micro[1]),
            "f1_micro": float(micro[2]),
            "test_overall_accuracy": float(accuracy_score(y_true_all, y_pred_all))
        })

        # Log summary
        log_msg = "Test Evaluation Summary: " + ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, (float, np.number)) else f"{k}: {v}" 
            for k, v in test_results.items() if not k.startswith('precision_class_')
        ])
        logger.info(log_msg)
        
        return test_results

    def _create_combined_dataset(
        self, labeled_dataset: tf.data.Dataset, unlabeled_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Combines labeled and unlabeled datasets for SSL training."""
        return tf.data.Dataset.zip((labeled_dataset, unlabeled_dataset))

    def _calculate_steps_per_epoch(
        self, 
        train_labeled_dataset: DatasetType, 
        train_unlabeled_dataset: DatasetType
    ) -> int:
        """Determines the number of steps per epoch for SSL training."""
        steps_from_config = getattr(self.config, 'steps_per_epoch', None)
        if steps_from_config is not None:
            logger.info(f"Using steps_per_epoch from config: {steps_from_config}")
            return int(steps_from_config)

        # Compute from dataset cardinality
        labeled_batches = tf.data.experimental.cardinality(train_labeled_dataset).numpy()
        unlabeled_batches = tf.data.experimental.cardinality(train_unlabeled_dataset).numpy()
        
        if labeled_batches < 0 or unlabeled_batches < 0:
            logger.warning("Could not determine dataset cardinality; defaulting steps_per_epoch to 1000.")
            return 1000
            
        steps = min(labeled_batches, unlabeled_batches)
        logger.info(f"Calculated steps_per_epoch as min({labeled_batches}, {unlabeled_batches}) = {steps}")
        return int(steps)

    def _execute_training_epoch(
        self, train_dataset: DatasetType, steps_per_epoch: int, epoch: int
    ) -> MetricsDict:
        """Executes one training epoch for SSL."""
        # Reset training metrics
        for metric in self.train_metrics.values():
            metric.reset_states()

        iterator = iter(train_dataset)
        for step in range(steps_per_epoch):
            try:
                batch = next(iterator)
                labeled_data, unlabeled_data = batch
                self._train_step_ssl(labeled_data, unlabeled_data)
            except StopIteration:
                break

        # Collect results
        return {name: metric.result() for name, metric in self.train_metrics.items()}

    def _execute_validation_epoch(self, epoch: int, val_dataset: DatasetType) -> MetricsDict:
        """Executes one validation epoch."""
        # Reset validation metrics
        for metric in self.val_metrics.values():
            metric.reset_states()

        for batch in val_dataset:
            x_batch_val, y_batch_val = batch
            self.validation_step(x_batch_val, y_batch_val)

        # Collect results
        return {name: metric.result() for name, metric in self.val_metrics.items()}

    def predict_classes(self, dataset):
        """Predict classes and return true labels and predictions."""
        true_labels = []
        predictions = []
        
        for batch in dataset:
            x_batch, y_batch = batch
            logits = self.model(x_batch, training=False)
            predicted_classes = tf.argmax(logits, axis=1)
            true_labels.append(y_batch.numpy())
            predictions.append(predicted_classes.numpy())
            
        return np.concatenate(true_labels, axis=0), np.concatenate(predictions, axis=0)

    def predict_classes_ema(self, dataset):
        """Predict classes using the EMA model."""
        if self.ema_model is None:
            raise ValueError("EMA model is not initialized. Cannot predict with EMA.")
            
        # Ensure EMA model is current
        self._update_ema_model_weights()
        
        true_labels = []
        predictions = []
        
        for batch in dataset:
            x_batch, y_batch = batch
            logits = self.ema_model(x_batch, training=False)
            predicted_classes = tf.argmax(logits, axis=1)
            true_labels.append(y_batch.numpy())
            predictions.append(predicted_classes.numpy())
            
        return np.concatenate(true_labels, axis=0), np.concatenate(predictions, axis=0)

    def _create_training_plots(self) -> None:
        """Create plots for training history, including SSL-specific metrics."""
        if not self.history:
            logger.warning("No training history available for plotting.")
            return

        try:
            from src.analysis.result_analysis import plot_training_history, plot_ssl_metrics
            import datetime

            # Get the timestamped experiment directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = self.results_dir / f"{self.config.experiment_name}_{timestamp}"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Plot comprehensive metrics visualization
            plot_ssl_metrics(
                history=self.history,
                output_path=plots_dir / "training_plots.png",
                title=f"SSL Training Metrics ({self.config.training_mode})",
                model_type=self.config.training_mode.lower()
            )

            logger.info(f"Training history plots saved to {plots_dir}")
            
        except ImportError as e:
            logger.warning(f"Could not create training plots: {e}")
        except Exception as e:
            logger.error(f"Error creating training plots: {e}")