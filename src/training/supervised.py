"""
Supervised training implementation for mosquito wingbeat classification.
"""

import tensorflow as tf
import numpy as np
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import analysis modules
# Ensure generate_classification_report is imported
from src.analysis.result_analysis import plot_confusion_matrix, plot_training_history, generate_classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import local modules
from src.utils.config import Config
from src.models.MosSongPlus import create_model_from_config

# Configure logger
logger = logging.getLogger(__name__)

# Disable global XLA JIT to avoid compiling IteratorGetNext for GPU
try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    logger.warning(f"Could not disable XLA JIT: {e}")


class SupervisedTrainer:
    """
    Trainer for supervised learning approaches.
    """
    model: tf.keras.Model
    history: Optional[Dict[str, List[Any]]]
    results_dir: Path
    steps_per_epoch: Optional[int]
    validation_steps: Optional[int]
    label_smoothing: float
    num_classes: int # Added for one-hot encoding depth
    _last_test_dataset: Optional[tf.data.Dataset] = None

    def __init__(self, model: tf.keras.Model, config: Config):
        """
        Initialize the trainer with a pre-created model.

        Args:
            model: Pre-built TensorFlow Keras model
            config: Configuration object
        """
        self.config = config
        self.model = model
        self.history = None

        # Get training parameters from config
        self.num_epochs: int = config.get('num_epochs', 100)
        self.learning_rate: float = config.get('learning_rate', 0.001)
        self.label_smoothing: float = config.get('labels_smoothing', 0.0) 
        self.early_stop_patience: int = config.get('early_stop_patience', 20)
        self.checkpoint_interval: int = config.get('checkpoint_interval', 10)
        self.model_type: str = config.get('model_type', 'MosSong+')
        self.num_classes: int = config.get('num_classes', 11) # Store num_classes

        # Initialize steps_per_epoch
        self.steps_per_epoch: Optional[int] = None
        data_type: str = config.get('data_type', 'raw')
        if data_type == 'stft':
            self.steps_per_epoch = config.get('stft_steps_per_epoch')
            if self.steps_per_epoch is not None:
                logger.info(f"Using STFT specific steps_per_epoch: {self.steps_per_epoch}")
        elif data_type == 'raw':
            self.steps_per_epoch = config.get('waveform_steps_per_epoch')
            if self.steps_per_epoch is not None:
                logger.info(f"Using Waveform specific steps_per_epoch: {self.steps_per_epoch}")
        
        if self.steps_per_epoch is None: # Fallback if not set by type-specific keys
            self.steps_per_epoch = config.get('steps_per_epoch') # Generic key
            if self.steps_per_epoch is not None:
                 logger.info(f"Using generic steps_per_epoch: {self.steps_per_epoch}")

        # Validation steps from config
        self.validation_steps: Optional[int] = config.get('validation_steps')
        if self.validation_steps is not None:
            logger.info(f"Using validation_steps: {self.validation_steps}")

        # Create checkpoint and results directories
        self.results_dir = Path(config.results_actual_dir) 
        self.checkpoint_dir = self.results_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Compile the passed model
        logger.info(f"Compiling model {self.model_type} with LR {self.learning_rate}, Label Smoothing {self.label_smoothing}")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            # Use CategoricalCrossentropy for label smoothing with one-hot labels
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.label_smoothing),
            metrics=['accuracy'],
        )
        self.model.summary(print_fn=logger.info)

    def _prepare_dataset_for_loss(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Converts labels to one-hot encoding for CategoricalCrossentropy loss."""
        def to_one_hot(x_batch: Any, y_batch: tf.Tensor) -> Tuple[Any, tf.Tensor]:
            y_one_hot = tf.one_hot(tf.cast(y_batch, dtype=tf.int32), depth=self.num_classes)
            return x_batch, y_one_hot
        return dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    def train(self, train_tf_dataset: tf.data.Dataset, val_tf_dataset: tf.data.Dataset) -> Dict[str, List[Any]]:
        """
        Train the model using provided TensorFlow datasets.
        Labels will be one-hot encoded for the loss function.
        
        Args:
            train_tf_dataset: TensorFlow dataset for training
            val_tf_dataset: TensorFlow dataset for validation
            
        Returns:
            Dictionary containing training history
        
        Note:
            steps_per_epoch is applied only to the training dataset;
            validation dataset is processed in full.
        """
        logger.info(f"Starting supervised training for {self.num_epochs} epochs")

        start_time = time.time()
        callbacks: List[tf.keras.callbacks.Callback] = self._create_callbacks()

        # Prepare datasets with one-hot encoded labels for the loss function
        train_dataset_one_hot = self._prepare_dataset_for_loss(train_tf_dataset)
        val_dataset_one_hot = self._prepare_dataset_for_loss(val_tf_dataset)

        history_obj = self.model.fit(
            train_dataset_one_hot,  
            validation_data=val_dataset_one_hot,  
            epochs=self.num_epochs,
            callbacks=callbacks,
            verbose=0, 
            steps_per_epoch=self.steps_per_epoch,
        )
        self.history = history_obj.history

        # Save final model - first save weights to make sure something is saved
        final_weights_path = self.checkpoint_dir / 'final_model_weights.keras'
        self.model.save_weights(str(final_weights_path))
        logger.info(f"Final model weights saved to {final_weights_path}")

        # Attempt to save full model
        final_model_path = self.checkpoint_dir / 'final_model.keras'
        try:

            tf.keras.models.save_model(
                self.model, 
                str(final_model_path),
                save_format='keras',
            )
            logger.info(f"Final model saved to {final_model_path}")
        except Exception as e:
            logger.error(f"Error saving full model: {e}")

        # Create training plots
        self._create_training_plots()

        # Log training results
        training_time = time.time() - start_time
        if self.history and 'val_accuracy' in self.history and self.history['val_accuracy']:
            best_val_accuracy = max(self.history['val_accuracy'])
            best_epoch = self.history['val_accuracy'].index(best_val_accuracy) + 1
            final_val_accuracy = float(self.history['val_accuracy'][-1])
            final_val_loss = float(self.history['val_loss'][-1])
        else:
            logger.warning("Validation accuracy not found in history. Setting metrics to default values.")
            best_val_accuracy = 0.0
            best_epoch = 0
            final_val_accuracy = 0.0
            final_val_loss = 0.0

        self.config.log_experiment({
            'training_time': training_time,
            'best_val_accuracy': float(best_val_accuracy),
            'best_epoch': best_epoch,
            'final_val_accuracy': final_val_accuracy,
            'final_val_loss': final_val_loss
        })

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")

        return self.history if self.history is not None else {}

    def _create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create callbacks for training."""
        callbacks: List[tf.keras.callbacks.Callback] = []
        
        # Custom progress display callback
        class EnhancedProgressBar(tf.keras.callbacks.Callback):
            def __init__(self, num_epochs: int):
                super().__init__()
                self.num_epochs = num_epochs
                self.green = '\\033[1;32m'
                self.blue = '\\033[1;34m'
                self.reset = '\\033[0m'
                
            def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
                print(f"\\n{self.blue}Epoch {epoch+1}/{self.num_epochs}{self.reset}") # Use print for direct console output
                
            def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
                metrics_str = []
                if logs: # Check if logs is not None
                    for metric, value in logs.items():
                        # Highlight accuracy metrics in green
                        if 'acc' in metric:
                            metrics_str.append(f"{self.green}{metric}: {value:.4f}{self.reset}")
                        else:
                            metrics_str.append(f"{metric}: {value:.4f}")
                    print(f"{self.blue}Results:{self.reset} " + " - ".join(metrics_str))
        
        # Add our enhanced progress bar
        callbacks.append(EnhancedProgressBar(self.num_epochs))
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.early_stop_patience,
            restore_best_weights=True,
            verbose=0  # Reduce verbosity since we have our own progress display
        )
        callbacks.append(early_stop)
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / 'model_epoch_{epoch:03d}.keras'
        
        # Use save_weights_only=True to avoid issues with the native Keras format
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,  # Save only weights to avoid 'options' parameter issues
            save_freq='epoch',       # Changed from 'period' which is deprecated
            verbose=0  # Reduce verbosity since we have our own progress display
        )
        callbacks.append(checkpoint)
        
        # Add a separate checkpoint for the full model at the end of training
        # We'll handle this manually in the train method
        
        # TensorBoard callback
        tensorboard_log_dir = self.results_dir / 'tensorboard' # Ensure this is 'results_dir' not 'tensorboard_dir'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_log_dir), # Convert Path to str
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler (optional)
        if self.config.get('use_lr_scheduler', False):
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0, # Changed to int(1e-7) which is 0, to satisfy linter if it expects int
                verbose=0  # Reduce verbosity since we have our own progress display
            )
            callbacks.append(lr_scheduler)
            
        return callbacks
        
    def _create_training_plots(self) -> None: # Return type hint
        """Create plots for training history."""
        if self.history is None:
            logger.warning("No training history available for plotting")
            return
            
        # Create plots directory
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the result_analysis module to create better plots
        from src.analysis.result_analysis import plot_training_history
        plot_training_history(
            history=self.history,
            output_path=plots_dir / 'training_history.png',
            metrics=['accuracy', 'loss']
        )
        
        # Defensive check for keys in history dictionary
        if 'accuracy' in self.history and 'val_accuracy' in self.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['accuracy'], label='Training')
            plt.plot(self.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'accuracy.png')
            plt.close() # Close plot to free memory
        else:
            logger.warning("Accuracy or val_accuracy not found in history for plotting.")

        if 'loss' in self.history and 'val_loss' in self.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['loss'], label='Training')
            plt.plot(self.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'loss.png')
            plt.close() # Close plot to free memory
        else:
            logger.warning("Loss or val_loss not found in history for plotting.")
        
    def evaluate(self, test_tf_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on the test set.
        Uses original test_tf_dataset for metrics calculation (integer labels)
        and a one-hot encoded version for model.evaluate().
        """
        logger.info("Evaluating model on test set")
        
        # Prepare dataset with one-hot labels for model.evaluate()
        test_dataset_one_hot = self._prepare_dataset_for_loss(test_tf_dataset)
        self._last_test_dataset = test_tf_dataset # Keep original for metrics

        # Evaluate model
        evaluation_output = self.model.evaluate(test_dataset_one_hot, verbose=1, return_dict=True)
        test_loss = float(evaluation_output.get('loss', 0.0))
        test_accuracy = float(evaluation_output.get('accuracy', 0.0))

        y_pred_list: List[np.ndarray] = []
        y_true_list: List[np.ndarray] = [] # Will store integer labels
        
        # Iterate over the ORIGINAL dataset for integer labels for sklearn metrics
        for x_batch, y_batch_sparse in test_tf_dataset: 
            logits = self.model(x_batch, training=False)
            predictions = tf.argmax(logits, axis=1)
            y_pred_list.append(predictions.numpy())
            y_true_list.append(y_batch_sparse.numpy()) # y_batch_sparse contains integer labels
            
        y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])
        y_true = np.concatenate(y_true_list) if y_true_list else np.array([]) # y_true now holds integer labels
            
        cm: np.ndarray = np.array([])
        if y_true.size > 0 and y_pred.size > 0:
            cm = confusion_matrix(y_true, y_pred)
            # Pass y_true (integer) and y_pred (integer) to plot_confusion_matrix
            self._plot_confusion_matrix(cm, y_true, y_pred) 
        else:
            logger.warning("Cannot compute confusion matrix: y_true or y_pred is empty.")

        class_names = [str(i) for i in range(self.num_classes)]
        report_dict: Dict[str, Any] = {} # Ensure report_dict is typed
        if y_true.size > 0 and y_pred.size > 0:
            # generate_classification_report expects integer labels
            report_dict = generate_classification_report(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                output_path=self.results_dir / 'classification_report.txt'
            )
        
        metrics: Dict[str, float] = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        for class_idx, class_name in enumerate(class_names):
            if class_name in report_dict and isinstance(report_dict[class_name], dict):
                class_metrics = report_dict[class_name]
                metrics[f'precision_class_{class_idx}'] = float(class_metrics.get('precision', 0.0))
                metrics[f'recall_class_{class_idx}'] = float(class_metrics.get('recall', 0.0))
                metrics[f'f1_class_{class_idx}'] = float(class_metrics.get('f1-score', 0.0))
        
        self.config.log_experiment(metrics)
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        if y_true.size > 0 and y_pred.size > 0:
            logger.info(f"Classification report saved to {self.results_dir / 'classification_report.txt'}")
        
        return metrics
        
    def _plot_confusion_matrix(self, cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix (numpy array)
        """
        # Create plots directory
        plots_dir: Path = self.results_dir / 'plots' # Type hint for plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        class_names = [str(i) for i in range(self.num_classes)]
        
        # Use y_true and y_pred directly for plotting if available and cm was generated from them
        if y_true.size > 0 and y_pred.size > 0:
            plot_confusion_matrix(
                y_true=y_true, 
                y_pred=y_pred,
                class_names=class_names,
                output_path=plots_dir / 'confusion_matrix.png',
                title='Confusion Matrix (Normalized)',
                normalize=True,
                hide_zeros=True,
                show_counts=True
            )
            plt.close() # Close plot
            plot_confusion_matrix(
                y_true=y_true, 
                y_pred=y_pred,
                class_names=class_names,
                output_path=plots_dir / 'confusion_matrix_raw.png',
                title='Confusion Matrix (Raw Counts)',
                normalize=False,
                hide_zeros=True,
                show_counts=True
            )
            plt.close() # Close plot
        elif cm.size > 0: # Fallback if only cm is available (less ideal for y_true/y_pred based plotting)
            logger.warning("Plotting confusion matrix from cm array only as y_true/y_pred were not suitable for direct use.")
            # Fallback to plot using cm if y_true/y_pred somehow failed but cm was generated
            # This requires plot_confusion_matrix to handle cm directly or reconstruct y_true/y_pred
            # For now, this branch indicates a potential issue if reached.
            # A simple plot from cm might be:
            # plt.figure(figsize=(10, 7))
            # import seaborn as sns
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            # plt.title('Raw Confusion Matrix (from cm array)')
            # plt.ylabel('Actual')
            # plt.xlabel('Predicted')
            # plt.savefig(plots_dir / 'confusion_matrix_raw_from_cm.png')
            # plt.close()
            pass # Current plot_confusion_matrix expects y_true, y_pred.

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model (string or Path object)
        """
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(str(model_path)) 
        
        logger.info("Recompiling loaded model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.label_smoothing),
            metrics=['accuracy']
        )
