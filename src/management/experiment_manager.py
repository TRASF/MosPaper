"""
Experiment management module for coordinating training runs and result analysis.
This module provides tools for setting up, tracking, and comparing experiments.
"""

import os
import json
import logging
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import config
from src.utils.config import Config

# Configure logger
logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manager for coordinating multiple training experiments and comparing results.
    Supports parallel execution of different configurations for comparison.
    """
    
    def __init__(self, base_config: Config):
        """
        Initialize the experiment manager with a base configuration.
        
        Args:
            base_config: Base configuration object to derive experiment configs from
        """
        self.base_config = base_config
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.results_dir = Path(base_config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_results: Dict[str, Any] = {}
        
        # Set up thread-safe queue for logging
        self.log_queue = queue.Queue()
        self.is_running = False
        
        logger.info(f"Experiment manager initialized with base config: {base_config['experiment_id']}")
    
    def create_experiment_config(self, name: str, overrides: Dict[str, Any]) -> Config:
        """
        Create a new experiment configuration by overriding base config values.
        
        Args:
            name: Name for the experiment
            overrides: Dictionary of configuration values to override
        
        Returns:
            New Config object for the experiment
        """
        # Create a new config starting from the base config file path
        config_path = self.base_config.config_dict.get('config_path')
        exp_config = Config(config_path)
        
        # Apply base config's non-generated values first
        for key, value in self.base_config.config_dict.items():
            if key not in ['experiment_id', 'results_dir', 'timestamp']:
                exp_config[key] = value
                
        # Then apply the overrides
        for key, value in overrides.items():
            exp_config[key] = value
            
        # Set experiment name
        exp_config['experiment_name'] = name
        
        return exp_config
    
    def add_experiment(self, name: str, overrides: Dict[str, Any]) -> str:
        """
        Add an experiment to the manager.
        
        Args:
            name: Name for the experiment
            overrides: Dictionary of configuration values to override
        
        Returns:
            Experiment ID
        """
        config = self.create_experiment_config(name, overrides)
        experiment_id = config['experiment_id']
        
        self.experiments[experiment_id] = {
            'name': name,
            'config': config,
            'status': 'pending',
            'result': None
        }
        
        logger.info(f"Added experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def run_experiment(self, experiment_id: str, training_func: Callable) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            experiment_id: ID of the experiment to run
            training_func: Function to run the training (should take config as input)
        
        Returns:
            Experiment results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment ID {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        config = experiment['config']
        
        logger.info(f"Starting experiment: {experiment['name']} ({experiment_id})")
        experiment['status'] = 'running'
        
        try:
            # Run the training function
            result = training_func(config)
            experiment['status'] = 'completed'
            experiment['result'] = result
            
            # Log completion
            logger.info(f"Experiment {experiment['name']} completed successfully")
            return result
        except Exception as e:
            experiment['status'] = 'failed'
            experiment['error'] = str(e)
            logger.error(f"Experiment {experiment['name']} failed: {e}")
            raise
    
    def run_all_experiments(self, training_func: Callable, max_workers: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Run all pending experiments, optionally in parallel.
        
        Args:
            training_func: Function to run the training (should take config as input)
            max_workers: Maximum number of parallel workers
        
        Returns:
            Dictionary of experiment results
        """
        self.is_running = True
        results = {}
        pending_experiments = {exp_id: exp for exp_id, exp in self.experiments.items() 
                               if exp['status'] == 'pending'}
        
        # Start log monitor thread
        log_monitor = threading.Thread(target=self._log_monitor)
        log_monitor.daemon = True
        log_monitor.start()
        
        try:
            # Run experiments in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_exp_id = {
                    executor.submit(self._run_experiment_wrapper, exp_id, training_func): exp_id
                    for exp_id in pending_experiments
                }
                
                # Process results as they complete
                for future in as_completed(future_to_exp_id):
                    exp_id = future_to_exp_id[future]
                    exp_name = self.experiments[exp_id]['name']
                    
                    try:
                        result = future.result()
                        results[exp_id] = result
                        logger.info(f"Experiment '{exp_name}' completed")
                    except Exception as e:
                        logger.error(f"Experiment '{exp_name}' failed: {e}")
        
        finally:
            self.is_running = False
            # Wait for log monitor to process remaining logs
            log_monitor.join(timeout=1.0)
        
        return results
    
    def _run_experiment_wrapper(self, experiment_id: str, training_func: Callable) -> Dict[str, Any]:
        """
        Wrapper to run experiment with proper logging context.
        
        Args:
            experiment_id: ID of the experiment to run
            training_func: Function to run the training
        
        Returns:
            Experiment result
        """
        experiment = self.experiments[experiment_id]
        
        # Add experiment context to thread-local logger
        thread_name = threading.current_thread().name
        exp_name = experiment['name']
        self.log_queue.put(f"Thread {thread_name} starting experiment '{exp_name}'")
        
        try:
            return self.run_experiment(experiment_id, training_func)
        finally:
            self.log_queue.put(f"Thread {thread_name} finished experiment '{exp_name}'")
    
    def _log_monitor(self):
        """Monitor and process logs from the experiment threads."""
        while self.is_running or not self.log_queue.empty():
            try:
                message = self.log_queue.get(block=True, timeout=0.5)
                logger.info(message)
                self.log_queue.task_done()
            except queue.Empty:
                pass
    
    def compare_experiments(self, metric_name: str = 'test_accuracy') -> pd.DataFrame:
        """
        Compare experiments based on a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
        
        Returns:
            DataFrame with experiment comparison
        """
        comparison_data = []
        
        for exp_id, experiment in self.experiments.items():
            if experiment['status'] == 'completed' and experiment['result'] is not None:
                result = experiment['result']
                
                # Extract the history and metrics
                if isinstance(result, tuple) and len(result) >= 3:
                    _, history, metrics = result[:3]
                else:
                    history = {}
                    metrics = result if isinstance(result, dict) else {}
                
                # Extract the metric value
                metric_value = metrics.get(metric_name, None)
                
                comparison_data.append({
                    'experiment_id': exp_id,
                    'name': experiment['name'],
                    'training_mode': experiment['config'].get('training_mode'),
                    'data_type': experiment['config'].get('data_type'),
                    'model_type': experiment['config'].get('model_type'),
                    metric_name: metric_value
                })
        
        # Create DataFrame and sort by metric
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values(by=metric_name, ascending=False)
            
        self.comparison_results = df
        return df
    
    def plot_comparison(self, metric_name: str = 'test_accuracy', output_path: Optional[str] = None) -> None:
        """
        Plot experiment comparison based on a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
            output_path: Optional path to save the figure
        """
        if self.comparison_results.empty:
            self.compare_experiments(metric_name)
            
        if self.comparison_results.empty:
            logger.warning("No experiment results to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create barplot with seaborn
        ax = sns.barplot(
            x='name',
            y=metric_name,
            hue='training_mode',
            data=self.comparison_results
        )
        
        plt.title(f'Experiment Comparison - {metric_name}')
        plt.xlabel('Experiment')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Comparison plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save experiment results to a JSON file.
        
        Args:
            output_path: Optional path to save the results
        """
        if output_path is None:
            output_path = self.results_dir / 'experiment_comparisons.json'
        else:
            output_path = Path(output_path)
            
        # Create serializable results dictionary
        results_dict = {
            'experiments': {},
            'comparison': self.comparison_results.to_dict('records') if not self.comparison_results.empty else []
        }
        
        for exp_id, experiment in self.experiments.items():
            exp_data = {
                'name': experiment['name'],
                'status': experiment['status'],
                'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                          for k, v in experiment['config'].config_dict.items()},
            }
            
            if experiment['result'] is not None:
                if isinstance(experiment['result'], tuple):
                    try:
                        _, history, metrics = experiment['result'][:3]
                        exp_data['metrics'] = metrics
                    except Exception:
                        exp_data['result'] = "Result format not recognized"
                elif isinstance(experiment['result'], dict):
                    exp_data['metrics'] = experiment['result']
            
            results_dict['experiments'][exp_id] = exp_data
        
        # Save to JSON
        try:
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Experiment results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")


def create_experiment_variations(base_config: Config, 
                                variation_configs: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create experiment variations for comparing different configurations.
    
    Args:
        base_config: Base configuration
        variation_configs: List of dictionaries with configuration variations
        
    Returns:
        List of (name, config_overrides) tuples
    """
    variations = []
    
    for i, variation in enumerate(variation_configs):
        name = variation.pop('name', f"experiment_{i}")
        variations.append((name, variation))
    
    return variations