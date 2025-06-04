#!/usr/bin/env python3
"""
Main entry point for mosquito wingbeat classification experiments.
Supports both supervised and SSL (FixMatch, FlexMatch) training approaches.
"""

import os
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Configure environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter out INFO and WARNING logs
os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '0'  # Turn off function JIT compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # Ensure XLA auto JIT is disabled

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger('MosquitoMain')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import TensorFlow after environment setup
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Further reduce TensorFlow logging verbosity
try:
    tf.config.optimizer.set_jit(False)  # Disable XLA JIT globally
except Exception:
    pass

# Import NumPy for random seed setting
import numpy as np

# Import project modules
from src.utils.config import Config
from src.training.supervised import SupervisedTrainer
from src.training.semi_supervised import SemiSupervisedTrainer
from src.data.dataset import DatasetManager

# Import analysis modules conditionally to avoid errors if not available
try:
    from src.analysis.result_analysis import (
        plot_training_history as plot_training_history_analysis,
        plot_confusion_matrix,
        generate_classification_report
    )
    analysis_available = True
except ImportError as e:
    logger.warning(f"Could not import analysis modules: {e}. Analysis features will be limited.")
    plot_training_history_analysis = None
    plot_confusion_matrix = None
    generate_classification_report = None
    analysis_available = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mosquito Wingbeat Classification")
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                        help='Path to configuration YAML file')
    
    # Main experiment options
    experiment_group = parser.add_argument_group('Experiment Configuration')
    experiment_group.add_argument('--training_mode', type=str, 
                            choices=['supervised', 'fixmatch', 'flexmatch', 'evaluate'], 
                            help='Training mode (overrides config file)')
    experiment_group.add_argument('--data_type', type=str, choices=['raw', 'stft'], 
                            help='Input data type (overrides config file)')
    experiment_group.add_argument('--model_type', type=str, choices=['MosSong+', 'PureWingbeat'], 
                            help='Model architecture (overrides config file)')
    experiment_group.add_argument('--experiment_name', type=str, 
                            help='Name for this experiment run (overrides config file)')
    experiment_group.add_argument('--seed', type=int, 
                            help='Global random seed (overrides config file)')
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_epochs', type=int, 
                          help='Number of training epochs (overrides config file)')
    training_group.add_argument('--batch_size', type=int, 
                          help='Batch size (overrides config file)')
    training_group.add_argument('--learning_rate', type=float, 
                          help='Learning rate (overrides config file)')
    training_group.add_argument('--checkpoint', type=str, default=None, 
                          help='Path to model checkpoint for evaluation or resuming training')

    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset.split_mode', type=str, 
                         choices=['fixed', 'ratio', 'max_train'],
                         help='Data splitting mode (overrides config file)')
    dataset_group.add_argument('--dataset.train_samples', type=int,
                         help='Number of training samples per class (fixed count split)')
    dataset_group.add_argument('--dataset.ssl_labels_per_class', type=int, 
                         help='Number of labeled examples per class for SSL')

    # SSL specific configuration
    ssl_group = parser.add_argument_group('SSL Configuration')
    ssl_group.add_argument('--ssl.confidence_threshold', type=float, 
                     help='Confidence threshold for FixMatch/FlexMatch')
    ssl_group.add_argument('--ssl.lambda_u', type=float, 
                     help='Weight for unsupervised loss in SSL')
    
    # Analysis mode
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument('--analysis', action='store_true',
                          help='Perform analysis on experiment results')
    analysis_group.add_argument('--results-dir-to-analyze', type=str, default=None,
                          help='Path to a specific experiment results directory to analyze')

    # Hyperparameter tuning
    tuning_group = parser.add_argument_group('Hyperparameter Tuning')
    tuning_group.add_argument('--tune', action='store_true', 
                        help='Run hyperparameter tuning')
    
    return parser.parse_args()

def create_model(config: Config, input_shape: Tuple[int, ...]) -> tf.keras.Model:
    """
    Create a model instance based on configuration with validated input shape.
    
    This function follows the "single source of truth" pattern where input shapes
    are validated once during preprocessing and passed to model creation.
    
    Args:
        config: Configuration object with model settings
        input_shape: Validated input shape from DatasetManager preprocessing
        
    Returns:
        Compiled TensorFlow model
    
    Raises:
        ValueError: If model_type is unsupported
        
    Note:
        input_shape is now required to enforce single-point validation.
        Use DatasetManager.get_preprocessing_stats()['shape_info']['input_shape']
        to get validated shapes.
    """
    logger.info(f"Creating {config.model_type} model with validated input shape: {input_shape}")

    # Update config with validated input shape for model creation
    original_input_shape = getattr(config, '_input_shape', None)
    config._input_shape = input_shape

    try:
        if config.model_type == 'MosSong+':
            from src.models.MosSongPlus import create_model_from_config
        elif config.model_type == 'PureWingbeat':
            from src.models.PureWingbeat import create_model_from_config
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        
        model = create_model_from_config(config)
        logger.info(f"Model {config.model_type} created successfully")
        return model
    finally:
        # Restore original input shape if it existed
        if original_input_shape is not None:
            config._input_shape = original_input_shape
        elif hasattr(config, '_input_shape'):
            delattr(config, '_input_shape')


def determine_input_shape(config: Config) -> Tuple[int, ...]:
    """
    Determine model input shape based on data type and configuration.
    
    DEPRECATED: This function is kept for backward compatibility only.
    For new code, use validated shapes from DatasetManager.get_preprocessing_stats()
    which implements the "single source of truth" pattern.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple specifying the input shape
        
    Raises:
        ValueError: If data_type is unknown
        
    Note:
        This function calculates shapes from configuration parameters but does
        not validate that the actual data matches these shapes. Use the validated
        approach via DatasetManager for production code.
    """
    logger.warning("determine_input_shape() is deprecated. Use validated shapes from DatasetManager.get_preprocessing_stats()")
    
    if config.data_type == 'raw':
        samples = int(config.audio.sample_rate * config.audio.audio_length)
        return (samples, 1)  # (length, channels=1 for raw audio)
    
    elif config.data_type == 'stft':
        # Shape is (time_frames, freq_bins, channels)
        return (config.stft.time_frames, config.stft.freq_bins, 1)
    
    else:
        raise ValueError(f"Unknown data_type: {config.data_type}")

def run_supervised_training(config: Config):
    """
    Run supervised training pipeline using validated datasets and single source of truth pattern.
    
    Shape validation occurs ONCE during preprocessing, all subsequent operations assume validated shapes.
    
    Args:
        config: Configuration object with training settings validated at initialization
        
    Returns:
        Tuple of (trained_model, training_history, test_metrics)
    """
    logger.info(f"Starting supervised training: {config.experiment_run_name}")
    logger.info(f"Data type: {config.data_type}, Model: {config.model_type}")
    logger.info(f"Dataset files resolved: {len(config.dataset.paths)} files.")

    # 1. Create datasets with single-point validation and preprocessing
    try:
        dataset_manager = DatasetManager(file_paths=config.dataset.paths, config=config)
        
        # Shape validation happens ONCE here - all subsequent operations assume validated shapes
        train_dataset, val_dataset, test_dataset = dataset_manager.get_validated_datasets(
            train_augment=True,
            train_repeat=config.get('repeat_training_dataset', True),
            train_shuffle=True,
        )
        
        # Get comprehensive preprocessing statistics (single source of truth for shapes)
        preprocessing_stats = dataset_manager.get_preprocessing_stats()
        validated_input_shape = preprocessing_stats['shape_info']['input_shape']
        
        # Log dataset statistics
        split_stats = preprocessing_stats['split_stats']
        logger.info("Validated dataset statistics:")
        logger.info(f"  Train: {split_stats['train']['segments']} segments from {split_stats['train']['files']} files")
        logger.info(f"  Val: {split_stats['val']['segments']} segments from {split_stats['val']['files']} files")
        logger.info(f"  Test: {split_stats['test']['segments']} segments from {split_stats['test']['files']} files")
        logger.info(f"  Validated input shape: {validated_input_shape}")
        
    except ValueError as e:
        logger.error(f"Dataset initialization failed: {e}")
        return None, None, {"error": "Dataset initialization failed"}

    if train_dataset is None or val_dataset is None or test_dataset is None:
        logger.error("Failed to create one or more TensorFlow datasets. Aborting training.")
        return None, None, {"error": "Dataset creation failed"}

    # 2. Create Model using validated input shape (single source of truth)
    try:
        model = create_model(config, input_shape=validated_input_shape)
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return None, None, {"error": "Model creation failed"}

    # 3. Create Trainer
    trainer = SupervisedTrainer(model, config)
    
    # 4. Train
    logger.info("Starting model training...")
    history = trainer.train(train_dataset, val_dataset)
    logger.info("Model training finished.")
    
    # 5. Evaluate
    logger.info("Evaluating model on the test set...")
    test_metrics = trainer.evaluate(test_dataset)
    logger.info(f"Test set evaluation complete. Metrics: {test_metrics}")
    
    # 6. Save results
    if history and plot_training_history_analysis:
        try:
            plot_training_history_analysis(history, config.results_actual_dir / "training_plots.png")
            logger.info(f"Training history plots saved to {config.results_actual_dir}")
            
            # Generate additional analysis if available
            if plot_confusion_matrix and generate_classification_report:
                # Get predictions for test set
                y_true, y_pred = trainer.predict_classes(test_dataset)
                
                # Save predictions for later analysis
                np.save(config.results_actual_dir / "y_true.npy", y_true)
                np.save(config.results_actual_dir / "y_pred.npy", y_pred)
                
                # Generate confusion matrix
                plot_confusion_matrix(
                    y_true, 
                    y_pred, 
                    class_names=list(config.dataset.class_dict.keys()),
                    output_path=config.results_actual_dir / "confusion_matrix.png"
                )
                
                # Generate classification report
                report = generate_classification_report(
                    y_true, 
                    y_pred, 
                    class_names=list(config.dataset.class_dict.keys()),
                    output_path=config.results_actual_dir / "classification_report.txt"
                )
                
                logger.info(f"Classification analysis saved to {config.results_actual_dir}")
        except Exception as e:
            logger.warning(f"Error saving analysis results: {e}")
    
    # Save test metrics as JSON (convert all values to native Python types)
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_native(v) for v in obj)
        return obj
    with open(config.results_actual_dir / f"{config.training_mode}_test_metrics.json", "w") as f:
        json.dump(_to_native(test_metrics), f, indent=4)
    
    # ---
    # Practical suggestions to improve accuracy (for user reference):
    # - Increase labeled data if possible
    # - Tune learning rate, batch size, and SSL hyperparameters (lambda_u, confidence_threshold)
    # - Use stronger data augmentation
    # - Ensure validation/test data is not leaking from training
    # - Try longer training (more epochs) or early stopping
    # ---
    return model, history, test_metrics

def run_ssl_training(config: Config):
    """
    Run semi-supervised learning training pipeline (FixMatch or FlexMatch) using validated datasets.
    
    Shape validation occurs ONCE during preprocessing, all subsequent operations assume validated shapes.
    
    Args:
        config: Configuration object with SSL training settings
    """
    logger.info(f"Starting {config.training_mode} training: {config.experiment_run_name}")
    logger.info(f"Data type: {config.data_type}, Model: {config.model_type}")
    logger.info(f"Dataset files resolved: {len(config.dataset.paths)} files.")
    logger.info(f"SSL labeled examples per class: {config.dataset.ssl_labels_per_class}")

    # 1. Create datasets with labeled and unlabeled splits using validated methods
    try:
        dataset_manager = DatasetManager(file_paths=config.dataset.paths, config=config)
        labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = dataset_manager.get_validated_ssl_datasets(
            labeled_samples_per_class=config.dataset.ssl_labels_per_class,
            batch_size=config.batch_size,
            unlabeled_batch_size=config.batch_size * config.ssl.mu,
            train_shuffle=True,
            train_augment_labeled=False,  # SSL trainer handles augmentation internally
            train_repeat=True,
            val_repeat=False,
            test_repeat=False
        )
        preprocessing_stats = dataset_manager.get_preprocessing_stats()
        validated_input_shape = preprocessing_stats['shape_info']['input_shape']
        split_stats = preprocessing_stats['split_stats']
        logger.info("Validated SSL dataset statistics:")
        logger.info(f"  Labeled train: {split_stats['labeled']['segments']} segments from {split_stats['labeled']['files']} files")
        logger.info(f"  Unlabeled train: {split_stats['unlabeled']['segments']} segments from {split_stats['unlabeled']['files']} files")
        logger.info(f"  Val: {split_stats['val']['segments']} segments from {split_stats['val']['files']} files")
        logger.info(f"  Test: {split_stats['test']['segments']} segments from {split_stats['test']['files']} files")
        logger.info(f"  Validated input shape: {validated_input_shape}")
        if not all([labeled_dataset, unlabeled_dataset, val_dataset, test_dataset]):
            logger.error("Failed to create one or more SSL datasets. Aborting training.")
            return None, None, {"error": "Dataset creation failed"}
    except ValueError as e:
        logger.error(f"Dataset initialization failed: {e}")
        return None, None, {"error": "Dataset initialization failed"}

    # 2. Create Model using validated input shape (single source of truth)
    try:
        model = create_model(config, input_shape=validated_input_shape)
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return None, None, {"error": "Model creation failed"}

    # 3. CORRECTED: Create SSL Trainer with simplified constructor
    # The trainer will internally create the SSL method and optimizer
    try:
        trainer = SemiSupervisedTrainer(
            config=config,
            model=model,
            callbacks=None  # Will be created internally based on config
        )
        logger.info(f"✓ {config.training_mode} trainer created successfully")
    except Exception as e:
        logger.error(f"SSL trainer creation failed: {e}")
        return None, None, {"error": "SSL trainer creation failed"}

    # 4. Train
    logger.info(f"Starting {config.training_mode} model training...")
    try:
        history = trainer.train(
            train_labeled_dataset=labeled_dataset,
            train_unlabeled_dataset=unlabeled_dataset,
            val_dataset=val_dataset,
            epochs=config.num_epochs
        )
        logger.info(f"{config.training_mode} model training finished.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, None, {"error": f"Training failed: {e}"}
    
    # 5. Evaluate (using EMA model if available and configured)
    logger.info("Evaluating model on the test set...")
    try:
        if config.ssl.evaluate_ema_model and config.ssl.use_ema:
            logger.info("Using EMA model for evaluation")
            test_metrics = trainer.evaluate(test_dataset, use_ema=True)
        else:
            test_metrics = trainer.evaluate(test_dataset)
        logger.info(f"Test set evaluation complete. Metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None, None, {"error": f"Evaluation failed: {e}"}
    
    # 6. Save results
    if history and plot_training_history_analysis:
        try:
            plot_training_history_analysis(history, config.results_actual_dir / "training_plots.png")
            logger.info(f"Training history plots saved to {config.results_actual_dir}")
            
            # Generate additional analysis if available
            if plot_confusion_matrix and generate_classification_report:
                # Get predictions for test set (using EMA model if configured)
                use_ema = config.ssl.evaluate_ema_model and config.ssl.use_ema
                if use_ema and hasattr(trainer, 'predict_classes_ema'):
                    y_true, y_pred = trainer.predict_classes_ema(test_dataset)
                else:
                    if use_ema:
                        logger.warning("'predict_classes_ema' not found, falling back to 'predict_classes'.")
                    y_true, y_pred = trainer.predict_classes(test_dataset)
                
                # Save predictions for later analysis
                np.save(config.results_actual_dir / "y_true.npy", y_true)
                np.save(config.results_actual_dir / "y_pred.npy", y_pred)
                
                # Generate confusion matrix
                plot_confusion_matrix(
                    y_true, 
                    y_pred, 
                    class_names=list(config.dataset.class_dict.keys()),
                    output_path=config.results_actual_dir / "confusion_matrix.png"
                )
                
                # Generate classification report
                report = generate_classification_report(
                    y_true, 
                    y_pred, 
                    class_names=list(config.dataset.class_dict.keys()),
                    output_path=config.results_actual_dir / "classification_report.txt"
                )
                
                # FlexMatch specific visualizations
                if config.training_mode == 'flexmatch' and 'flex_thresholds' in history:
                    try:
                        from src.training.flexmatch_monitor import plot_flex_threshold_history
                        plot_flex_threshold_history(
                            history['flex_thresholds'],
                            list(config.dataset.class_dict.keys()),
                            output_path=config.results_actual_dir / "flex_threshold_history.png"
                        )
                        logger.info("FlexMatch threshold history saved.")
                    except ImportError:
                        logger.warning("FlexMatch monitor module not available. Skipping threshold visualization.")
                
                logger.info(f"Classification analysis saved to {config.results_actual_dir}")
        except Exception as e:
            logger.warning(f"Error saving analysis results: {e}")
    
    # Save test metrics as JSON (convert all values to native Python types)
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_native(v) for v in obj)
        return obj
    
    try:
        with open(config.results_actual_dir / f"{config.training_mode}_test_metrics.json", "w") as f:
            json.dump(_to_native(test_metrics), f, indent=4)
        logger.info(f"Test metrics saved to {config.results_actual_dir}")
    except Exception as e:
        logger.warning(f"Error saving test metrics: {e}")
    
    return model, history, test_metrics

def run_evaluation(config: Config, checkpoint_path: str):
    """
    Run evaluation on a pre-trained model using validated data pipeline.
    
    Args:
        config: Configuration object with evaluation settings
        checkpoint_path: Path to the model checkpoint file
    """
    logger.info(f"Starting evaluation: {config.experiment_run_name}")
    logger.info(f"Checkpoint for evaluation: {checkpoint_path}")
    logger.info(f"Data type: {config.data_type}, Model: {config.model_type}")

    # 1. Create validated test dataset (single source of truth for data shape validation)
    try:
        dataset_manager = DatasetManager(file_paths=config.dataset.paths, config=config)
        
        # For evaluation, we only need the test dataset - use validated pipeline
        _, _, test_dataset = dataset_manager.get_validated_datasets(
            batch_size=config.get('batch_size', 32),
            train_shuffle=False,  # No need to shuffle for evaluation
            train_augment=False,  # No augmentation needed for test data
            train_repeat=False,   # No need to repeat for evaluation
            val_repeat=False,
            test_repeat=False
        )
        
        # Get validated preprocessing statistics (includes shape_info)
        preprocessing_stats = dataset_manager.get_preprocessing_stats()
        validated_input_shape = preprocessing_stats['shape_info']['input_shape']
        
        logger.info(f"✓ Validated test dataset created with input shape: {validated_input_shape}")
        logger.info(f"Test dataset: {preprocessing_stats['split_stats']['test']['segments']} segments from {preprocessing_stats['split_stats']['test']['files']} files")
        
    except ValueError as e:
        logger.error(f"Dataset validation failed: {e}")
        return None, {"error": "Dataset validation failed"}

    if test_dataset is None:
        logger.error("Failed to create test dataset. Aborting evaluation.")
        return None, {"error": "Test dataset creation failed"}

    # 2. Create model with validated input shape (single source of truth)
    try:
        model = create_model(config, validated_input_shape)
        logger.info(f"✓ Model created with validated input shape: {validated_input_shape}")
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return None, {"error": f"Model creation failed: {e}"}

    # 3. Load model weights from checkpoint
    try:
        model.load_weights(checkpoint_path)
        logger.info(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load weights from {checkpoint_path}: {e}")
        return None, {"error": f"Weight loading failed: {e}"}
    
    # 4. Create appropriate trainer based on training mode
    if config.training_mode in ['fixmatch', 'flexmatch']:
        trainer = SemiSupervisedTrainer(config, model)
        # Use EMA model if available and configured
        if config.ssl.evaluate_ema_model and config.ssl.use_ema:
            logger.info("Using EMA model for evaluation")
            trainer.initialize_ema_model()
            test_metrics = trainer.evaluate(test_dataset, use_ema=True)
        else:
            test_metrics = trainer.evaluate(test_dataset)
    else:
        # Default to supervised trainer for evaluation
        trainer = SupervisedTrainer(model, config)
        test_metrics = trainer.evaluate(test_dataset)
    
    logger.info(f"Evaluation complete. Metrics: {test_metrics}")
    
    # 5. Generate and save additional analysis if available
    try:
        if plot_confusion_matrix and generate_classification_report:
            # Get predictions for test set
            if hasattr(config, 'ssl') and config.ssl.evaluate_ema_model and config.ssl.use_ema:
                y_true, y_pred = trainer.predict_classes_ema(test_dataset)
            else:
                y_true, y_pred = trainer.predict_classes(test_dataset)
            
            # Save predictions for later analysis
            np.save(config.results_actual_dir / "eval_y_true.npy", y_true)
            np.save(config.results_actual_dir / "eval_y_pred.npy", y_pred)
            
            # Generate confusion matrix
            plot_confusion_matrix(
                y_true, 
                y_pred, 
                class_names=list(config.dataset.class_dict.keys()),
                output_path=config.results_actual_dir / "eval_confusion_matrix.png"
            )
            
            # Generate classification report
            report = generate_classification_report(
                y_true, 
                y_pred, 
                class_names=list(config.dataset.class_dict.keys()),
                output_path=config.results_actual_dir / "eval_classification_report.txt"
            )
            
            logger.info(f"Evaluation analysis saved to {config.results_actual_dir}")
    except Exception as e:
        logger.warning(f"Error generating evaluation analysis: {e}")
    
    # Save evaluation metrics as JSON
    with open(config.results_actual_dir / "evaluation_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
    
    return test_metrics

def run_analysis(config: Config, specific_experiment_dir: Optional[str] = None):
    """Perform analysis on saved experiment results."""
    if not analysis_available: # Simplified check
        logger.error("Analysis functions not available. Cannot run analysis.")
        return

    if specific_experiment_dir:
        analysis_target_dir = Path(specific_experiment_dir)
        if not analysis_target_dir.is_dir():
            logger.error(f"Specified experiment directory for analysis does not exist: {analysis_target_dir}")
            return
        logger.info(f"Analyzing results from specified directory: {analysis_target_dir}")
    else:
        analysis_target_dir = config.results_actual_dir # Analyze the current run if no specific dir given
        logger.info(f"Analyzing results from current experiment run: {analysis_target_dir}")

    logger.info(f"Attempting to perform analysis on directory: {analysis_target_dir}")

    # Example: Load test metrics if they exist
    test_metrics_path = analysis_target_dir / f"{config.training_mode}_test_metrics.json" # or a generic name
    if test_metrics_path.exists():
        with open(test_metrics_path, 'r') as f:
            test_metrics = json.load(f)
        logger.info(f"Loaded test metrics: {test_metrics}")
        # Further processing of test_metrics can be done here.
    else:
        logger.warning(f"Test metrics file not found at {test_metrics_path}")

    # Example: Generate and plot confusion matrix if predictions and true labels are saved
    # This would require predictions and labels to be saved during training/evaluation.
    # For demonstration, let's assume they are saved as numpy arrays.
    # y_true_path = analysis_target_dir / "y_true.npy"
    # y_pred_path = analysis_target_dir / "y_pred.npy"
    # if y_true_path.exists() and y_pred_path.exists() and plot_confusion_matrix:
    #     y_true = np.load(y_true_path)
    #     y_pred = np.load(y_pred_path)
    #     class_names = config.dataset.get('class_names') # Assuming class_names are in config
    #     plot_confusion_matrix(
    #         y_true, y_pred, 
    #         class_names=class_names, 
    #         output_path=analysis_target_dir / "confusion_matrix.png",
    #         title=f"Confusion Matrix for {config.experiment_run_name}"
    #     )
    #     logger.info(f"Confusion matrix plotted for {analysis_target_dir}")
    # else:
    #     logger.warning("Could not generate confusion matrix. y_true.npy or y_pred.npy not found, or plot_confusion_matrix not available.")

    logger.info(f"Basic analysis (loading metrics) complete for {analysis_target_dir}")


def main():
    """
    Main entry point for the mosquito classification program.
    Handles command-line arguments and runs the appropriate mode.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Create Config object from YAML and CLI overrides
    cli_overrides = {
        key: value for key, value in vars(args).items() 
        if value is not None and key not in [
            'config', 'analysis', 'tune', 'checkpoint', 
            'results_dir_to_analyze'
        ]
    }
    
    config = Config(
        config_path=args.config, 
        auto_save=True,  # Save the final config for reproducibility
        **cli_overrides
    )

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    
    # Determine which mode to run based on arguments
    if args.analysis:
        run_analysis(config, args.results_dir_to_analyze)
    elif args.tune:
        run_hyperparameter_tuning(config)
    elif config.training_mode == 'evaluate':
        if args.checkpoint:
            run_evaluation(config, args.checkpoint)
        else:
            logger.error("Evaluation mode requires a checkpoint path via --checkpoint argument.")
    elif config.training_mode == 'supervised':
        run_supervised_training(config)
    elif config.training_mode in ['fixmatch', 'flexmatch']:
        run_ssl_training(config)
    else:
        logger.error(f"Unsupported training_mode in config: {config.training_mode}")


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Integer seed value
    """
    logger.info(f"Setting global random seed to: {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Add other random generators if needed
    # import random
    # random.seed(seed)


def run_analysis(config: Config, results_dir: Optional[str] = None):
    """
    Run analysis on existing experiment results.
    
    Args:
        config: Configuration object
        results_dir: Optional path to specific results directory to analyze
    """
    if not analysis_available:
        logger.error("Analysis modules not available. Make sure analysis dependencies are installed.")
        return
    
    analysis_dir = Path(results_dir) if results_dir else config.results_actual_dir
    logger.info(f"Running analysis on results directory: {analysis_dir}")
    
    # Check if directory exists
    if not analysis_dir.exists():
        logger.error(f"Results directory does not exist: {analysis_dir}")
        return
    
    # TODO: Implement analysis logic
    logger.info("Analysis functionality is not fully implemented yet.")


def run_hyperparameter_tuning(config: Config):
    """
    Run hyperparameter tuning to find optimal model parameters.
    
    Args:
        config: Configuration object with tuning settings
    """
    logger.info("Starting hyperparameter tuning")
    
    if not config.hyperparameter_tuning.get('enabled', False):
        logger.warning("Hyperparameter tuning is disabled in config. Skipping.")
        return
    
    # Check tuning method
    method = config.hyperparameter_tuning.get('method', 'grid')
    logger.info(f"Tuning method: {method}")
    
    # TODO: Implement hyperparameter tuning logic
    logger.info("Hyperparameter tuning functionality is not fully implemented yet.")


if __name__ == '__main__':
    main()
