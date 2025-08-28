import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Optional, Callable, List, Union
from flax.training import checkpoints
import time
import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil
import wandb
import mlflow
from datetime import datetime
import hashlib
import pickle
import gzip

@dataclass
class TrainingState:
    epoch: int
    step: int
    loss: float
    metrics: Dict[str, float]
    learning_rate: float
    throughput: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    disk_io: Dict[str, float] = None
    network_io: Dict[str, float] = None
    model_hash: str = ""
    training_duration: float = 0.0
    batch_processing_time: float = 0.0
    data_loading_time: float = 0.0
    gradient_computation_time: float = 0.0
    optimizer_step_time: float = 0.0

class Callback(ABC):
    @abstractmethod
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        pass

class AdvancedModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', 
                 save_best_only: bool = True, save_freq: int = 1, 
                 save_weights_only: bool = False, keep_checkpoint_max: int = 5,
                 save_optimizer_state: bool = True, save_model_metadata: bool = True,
                 compression: bool = False, encryption: bool = False):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_optimizer_state = save_optimizer_state
        self.save_model_metadata = save_model_metadata
        self.compression = compression
        self.encryption = encryption
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.monitor_op = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)
        self.checkpoint_history = []
        self.metadata = {}
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.metadata = {
            'training_start_time': datetime.now().isoformat(),
            'model_config': logs.get('config', {}),
            'framework_versions': {
                'jax': jax.__version__,
                'flax': flax.__version__ if 'flax' in globals() else 'unknown',
                'optax': optax.__version__ if 'optax' in globals() else 'unknown'
            }
        }
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if epoch % self.save_freq != 0:
            return
            
        current_metric = logs.get(self.monitor, None)
        if current_metric is None:
            return
            
        checkpoint_path = f"{self.filepath}_epoch_{epoch}"
        
        if not self.save_best_only:
            self._save_checkpoint(logs, checkpoint_path, epoch)
            self._manage_checkpoint_history(checkpoint_path, epoch)
            return
            
        if self.monitor_op(current_metric, self.best_metric):
            self.best_metric = current_metric
            self._save_checkpoint(logs, checkpoint_path, epoch)
            self._manage_checkpoint_history(checkpoint_path, epoch)
    
    def _save_checkpoint(self, logs: Dict[str, Any], checkpoint_path: str, epoch: int) -> None:
        state = logs.get('state')
        if state is None:
            return
        
        # Save checkpoint
        if self.save_weights_only:
            checkpoints.save_checkpoint(checkpoint_path, state.params, epoch, overwrite=True)
        else:
            checkpoints.save_checkpoint(checkpoint_path, state, epoch, overwrite=True)
        
        # Save metadata if enabled
        if self.save_model_metadata:
            metadata = self.metadata.copy()
            metadata.update({
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'metrics': logs.get('training_state', {}).metrics if logs.get('training_state') else {},
                'model_hash': self._compute_model_hash(state.params)
            })
            
            metadata_path = f"{checkpoint_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _manage_checkpoint_history(self, checkpoint_path: str, epoch: int) -> None:
        self.checkpoint_history.append((epoch, checkpoint_path))
        if len(self.checkpoint_history) > self.keep_checkpoint_max:
            oldest_epoch, oldest_path = self.checkpoint_history.pop(0)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
            # Also remove metadata file if it exists
            metadata_path = f"{oldest_path}_metadata.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
    
    def _compute_model_hash(self, params) -> str:
        """Compute a hash of the model parameters for integrity verification."""
        param_bytes = pickle.dumps(params)
        return hashlib.sha256(param_bytes).hexdigest()

class AdvancedEarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0, 
                 patience: int = 0, mode: str = 'min', restore_best_weights: bool = True,
                 start_from_epoch: int = 0, baseline: Optional[float] = None,
                 verbose: bool = False, min_epochs: int = 0, 
                 restore_best_weights_on_interrupt: bool = True):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.baseline = baseline
        self.verbose = verbose
        self.min_epochs = min_epochs
        self.restore_best_weights_on_interrupt = restore_best_weights_on_interrupt
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        self.monitor_op = (lambda a, b: a < b - min_delta) if mode == 'min' else (lambda a, b: a > b + min_delta)
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        if self.baseline is not None:
            self.best_metric = self.baseline
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if epoch < self.start_from_epoch:
            return
            
        current_metric = logs.get(self.monitor, None)
        if current_metric is None:
            return
            
        if self.monitor_op(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = logs.get('state')
                if self.verbose:
                    print(f'Epoch {epoch}: {self.monitor} improved to {current_metric:.6f}')
        else:
            self.wait += 1
            if self.verbose:
                print(f'Epoch {epoch}: {self.monitor} did not improve from {self.best_metric:.6f}')
            if self.wait >= self.patience and epoch >= self.min_epochs:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                if self.verbose:
                    print(f'Epoch {epoch}: early stopping')
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            logs['state'] = self.best_weights
            if self.verbose:
                print(f'Restoring model weights from epoch {self.best_epoch}')

class AdvancedReduceLROnPlateau(Callback):
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.1, 
                 patience: int = 10, min_lr: float = 1e-6, mode: str = 'min',
                 cooldown: int = 0, verbose: bool = False, 
                 min_delta: float = 1e-4, threshold_mode: str = 'rel'):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.cooldown = cooldown
        self.verbose = verbose
        self.min_delta = min_delta
        self.threshold_mode = threshold_mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.monitor_op = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)
        self.threshold_op = (lambda a, b: a < b * (1 - min_delta)) if threshold_mode == 'rel' else (lambda a, b: a < b - min_delta)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current_metric = logs.get(self.monitor, None)
        if current_metric is None:
            return
            
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            
        if self.monitor_op(current_metric, self.best_metric) and \
           self.threshold_op(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
            if self.verbose:
                print(f'Epoch {epoch}: {self.monitor} improved to {current_metric:.6f}')
        else:
            self.wait += 1
            if self.wait >= self.patience and self.cooldown_counter <= 0:
                current_lr = logs.get('learning_rate', 1e-3)
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr:
                    logs['learning_rate'] = new_lr
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    if self.verbose:
                        print(f'Epoch {epoch}: reducing learning rate to {new_lr:.6f}')

class AdvancedCSVLogger(Callback):
    def __init__(self, filename: str, separator: str = ',', append: bool = False,
                 flush_every: int = 100, include_timestamp: bool = True):
        self.filename = filename
        self.separator = separator
        self.append = append
        self.flush_every = flush_every
        self.include_timestamp = include_timestamp
        self.keys = None
        self.file = None
        self.writer = None
        self.append_header = True
        self.row_count = 0
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if self.append:
            if os.path.exists(self.filename):
                self.append_header = False
        
        self.file = open(self.filename, 'w' if not self.append else 'a')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.keys is None:
            self.keys = ['timestamp'] + list(logs.keys()) if self.include_timestamp else list(logs.keys())
            if self.append_header:
                self.file.write(self.separator.join(self.keys) + '\n')
        
        values = []
        if self.include_timestamp:
            values.append(datetime.now().isoformat())
        
        for k in self.keys:
            if k == 'timestamp':
                continue
            values.append(str(logs.get(k, '')))
        
        self.file.write(self.separator.join(values) + '\n')
        self.row_count += 1
        
        if self.row_count % self.flush_every == 0:
            self.file.flush()
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.file:
            self.file.flush()
            self.file.close()

class AdvancedTensorBoardCallback(Callback):
    def __init__(self, log_dir: str, update_freq: Union[str, int] = 'epoch',
                 profile_batch: int = 2, histogram_freq: int = 0,
                 write_graph: bool = True, write_images: bool = False,
                 write_grads: bool = False, write_lr: bool = True):
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.profile_batch = profile_batch
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.write_grads = write_grads
        self.write_lr = write_lr
        os.makedirs(log_dir, exist_ok=True)
        self.writer = None
        self._train_step = 0
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            
            # Log model graph if requested
            if self.write_graph and logs.get('model'):
                # This would require actual model tracing, which is complex
                pass
        except ImportError:
            print("TensorBoard not available. Install torch to use this callback.")
            self.writer = None
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.writer is None:
            return
            
        if self.update_freq == 'batch' or (isinstance(self.update_freq, int) and batch % self.update_freq == 0):
            self._train_step += 1
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'batch/{key}', value, self._train_step)
            
            # Log learning rate if requested
            if self.write_lr and 'learning_rate' in logs:
                self.writer.add_scalar('learning_rate', logs['learning_rate'], self._train_step)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.writer is None:
            return
            
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
        
        # Log learning rate if requested
        if self.write_lr and 'learning_rate' in logs:
            self.writer.add_scalar('epoch/learning_rate', logs['learning_rate'], epoch)
        
        # Log histograms if requested
        if self.histogram_freq > 0 and epoch % self.histogram_freq == 0:
            # Add histogram logging if needed
            pass
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.writer:
            self.writer.close()

class AdvancedWandbCallback(Callback):
    def __init__(self, project: str = "deep-learning-hpc", entity: str = None,
                 name: str = None, config: Dict = None, 
                 log_freq: Union[str, int] = 'epoch',
                 save_code: bool = True, sync_tensorboard: bool = False):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.log_freq = log_freq
        self.save_code = save_code
        self.sync_tensorboard = sync_tensorboard
        self.run = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                save_code=self.save_code,
                sync_tensorboard=self.sync_tensorboard
            )
        except Exception as e:
            print(f"WandB initialization failed: {e}")
            self.run = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.run is None:
            return
            
        if self.log_freq == 'epoch' or (isinstance(self.log_freq, int) and epoch % self.log_freq == 0):
            wandb.log(logs, step=epoch)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.run is None:
            return
            
        if self.log_freq == 'batch' or (isinstance(self.log_freq, int) and batch % self.log_freq == 0):
            wandb.log(logs, step=batch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.run:
            wandb.finish()

class AdvancedMLflowCallback(Callback):
    def __init__(self, experiment_name: str = "deep-learning-hpc",
                 tracking_uri: str = None, log_freq: Union[str, int] = 'epoch'):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_freq = log_freq
        self.run = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run()
            
            # Log config parameters
            config = logs.get('config', {})
            if config:
                mlflow.log_params(config)
        except Exception as e:
            print(f"MLflow initialization failed: {e}")
            self.run = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.run is None:
            return
            
        if self.log_freq == 'epoch' or (isinstance(self.log_freq, int) and epoch % self.log_freq == 0):
            # Log metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=epoch)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.run is None:
            return
            
        if self.log_freq == 'batch' or (isinstance(self.log_freq, int) and batch % self.log_freq == 0):
            # Log metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"batch_{key}", value, step=batch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.run:
            mlflow.end_run()

class AdvancedLearningRateScheduler(Callback):
    def __init__(self, schedule: Callable[[int], float], verbose: bool = False,
                 update_optimizer: bool = True):
        self.schedule = schedule
        self.verbose = verbose
        self.update_optimizer = update_optimizer
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        new_lr = self.schedule(epoch)
        logs['learning_rate'] = new_lr
        if self.verbose:
            print(f'Epoch {epoch}: learning rate set to {new_lr:.6f}')

class SystemMonitor(Callback):
    def __init__(self, log_freq: int = 10, collect_disk_io: bool = True,
                 collect_network_io: bool = True, collect_process_stats: bool = True):
        self.log_freq = log_freq
        self.collect_disk_io = collect_disk_io
        self.collect_network_io = collect_network_io
        self.collect_process_stats = collect_process_stats
        self.start_time = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prev_disk_io = None
        self.prev_network_io = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        self.start_time = time.time()
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_network_io = psutil.net_io_counters()
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if batch % self.log_freq == 0:
            self._log_system_metrics(logs)
    
    def _log_system_metrics(self, logs: Dict[str, Any]) -> None:
        def collect_metrics():
            metrics = {}
            
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent()
            cpu_times = psutil.cpu_times_percent()
            metrics['cpu_user_time'] = cpu_times.user
            metrics['cpu_system_time'] = cpu_times.system
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available'] = memory.available
            metrics['memory_used'] = memory.used
            
            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                gpu_metrics = []
                for gpu in gpus:
                    gpu_metrics.append({
                        'id': gpu.id,
                        'utilization': gpu.load * 100,
                        'memory_utilization': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
                metrics['gpu_metrics'] = gpu_metrics
            except:
                pass
            
            # Disk I/O metrics
            if self.collect_disk_io:
                current_disk_io = psutil.disk_io_counters()
                if self.prev_disk_io:
                    metrics['disk_read_bytes_per_sec'] = (current_disk_io.read_bytes - self.prev_disk_io.read_bytes)
                    metrics['disk_write_bytes_per_sec'] = (current_disk_io.write_bytes - self.prev_disk_io.write_bytes)
                self.prev_disk_io = current_disk_io
            
            # Network I/O metrics
            if self.collect_network_io:
                current_network_io = psutil.net_io_counters()
                if self.prev_network_io:
                    metrics['network_bytes_sent_per_sec'] = (current_network_io.bytes_sent - self.prev_network_io.bytes_sent)
                    metrics['network_bytes_recv_per_sec'] = (current_network_io.bytes_recv - self.prev_network_io.bytes_recv)
                self.prev_network_io = current_network_io
            
            # Process metrics
            if self.collect_process_stats:
                process = psutil.Process()
                metrics['process_cpu_percent'] = process.cpu_percent()
                process_memory = process.memory_info()
                metrics['process_memory_rss'] = process_memory.rss
                metrics['process_memory_vms'] = process_memory.vms
            
            metrics['elapsed_time'] = time.time() - self.start_time
            return metrics
        
        future = self.executor.submit(collect_metrics)
        try:
            metrics = future.result(timeout=1.0)
            logs.update(metrics)
        except:
            pass

class PerformanceProfiler(Callback):
    def __init__(self, profile_freq: int = 100, log_detailed_stats: bool = True):
        self.profile_freq = profile_freq
        self.log_detailed_stats = log_detailed_stats
        self.batch_times = []
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.start_batch_time = None
        self.start_data_loading_time = None
        self.start_forward_time = None
        self.start_backward_time = None
        self.start_optimizer_time = None
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]) -> None:
        if batch % self.profile_freq == 0:
            self.start_batch_time = time.time()
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if batch % self.profile_freq == 0 and self.start_batch_time is not None:
            batch_time = time.time() - self.start_batch_time
            self.batch_times.append(batch_time)
            
            if len(self.batch_times) > 1000:
                self.batch_times.pop(0)
            
            logs['batch_time'] = batch_time
            logs['avg_batch_time'] = np.mean(self.batch_times)
            logs['batch_time_std'] = np.std(self.batch_times)
            
            if self.log_detailed_stats:
                # Log detailed timing stats
                if self.data_loading_times:
                    logs['avg_data_loading_time'] = np.mean(self.data_loading_times)
                if self.forward_times:
                    logs['avg_forward_time'] = np.mean(self.forward_times)
                if self.backward_times:
                    logs['avg_backward_time'] = np.mean(self.backward_times)
                if self.optimizer_times:
                    logs['avg_optimizer_time'] = np.mean(self.optimizer_times)

class ModelValidationCallback(Callback):
    def __init__(self, validation_data, validation_freq: int = 1,
                 validation_metric: str = 'val_accuracy', mode: str = 'max'):
        self.validation_data = validation_data
        self.validation_freq = validation_freq
        self.validation_metric = validation_metric
        self.mode = mode
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.monitor_op = (lambda a, b: a > b) if mode == 'max' else (lambda a, b: a < b)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if epoch % self.validation_freq != 0:
            return
            
        # Perform validation
        model = logs.get('model')
        if model is None:
            return
            
        # This would require implementing actual validation logic
        # For now, we'll just simulate it
        val_metric = np.random.random()  # Replace with actual validation
        
        if self.monitor_op(val_metric, self.best_metric):
            self.best_metric = val_metric
            logs[f'best_{self.validation_metric}'] = self.best_metric

class CallbackList:
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)