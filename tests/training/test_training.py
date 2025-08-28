# Tests for training module

import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.training.callbacks import (
    AdvancedEarlyStopping,
    AdvancedReduceLROnPlateau,
    AdvancedCSVLogger,
    SystemMonitor,
    PerformanceProfiler
)
from src.training.train_flax import FlaxTrainerConfig
from src.training.train_torch import TorchTrainerConfig

class TestFlaxTrainerConfig:
    """Test suite for FlaxTrainerConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = FlaxTrainerConfig(
            num_epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            warmup_steps=1000,
            weight_decay=0.01
        )
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.01
    
    def test_config_default_values(self):
        """Test configuration default values."""
        config = FlaxTrainerConfig()
        
        assert config.num_epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.0

class TestTorchTrainerConfig:
    """Test suite for TorchTrainerConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = TorchTrainerConfig(
            num_epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            warmup_steps=1000,
            weight_decay=0.01
        )
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.01
    
    def test_config_default_values(self):
        """Test configuration default values."""
        config = TorchTrainerConfig()
        
        assert config.num_epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.0

class TestAdvancedEarlyStopping:
    """Test suite for AdvancedEarlyStopping callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = AdvancedEarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=10,
            verbose=True,
            mode='min',
            restore_best_weights=True
        )
        
        assert callback.monitor == 'val_loss'
        assert callback.min_delta == 0.01
        assert callback.patience == 10
        assert callback.verbose is True
        assert callback.mode == 'min'
        assert callback.restore_best_weights is True
        assert callback.wait == 0
        assert callback.stopped_epoch == 0
        assert callback.best_weights is None
    
    def test_callback_on_epoch_end_improvement(self):
        """Test callback behavior when metric improves."""
        callback = AdvancedEarlyStopping(monitor='val_loss', patience=2)
        
        # Simulate epoch end with improving metric
        logs = {'val_loss': 0.5, 'epoch': 1}
        callback.on_epoch_end(1, logs)
        
        assert callback.best_metric == 0.5
        assert callback.wait == 0
    
    def test_callback_on_epoch_end_no_improvement(self):
        """Test callback behavior when metric doesn't improve."""
        callback = AdvancedEarlyStopping(monitor='val_loss', patience=2)
        
        # First epoch with good metric
        logs1 = {'val_loss': 0.5, 'epoch': 1}
        callback.on_epoch_end(1, logs1)
        
        # Second epoch with worse metric
        logs2 = {'val_loss': 0.6, 'epoch': 2}
        callback.on_epoch_end(2, logs2)
        
        assert callback.best_metric == 0.5
        assert callback.wait == 1
    
    def test_callback_early_stopping(self):
        """Test early stopping condition."""
        callback = AdvancedEarlyStopping(monitor='val_loss', patience=2)
        
        # Set initial best metric
        callback.best_metric = 0.5
        
        # Simulate epochs without improvement
        for epoch in range(1, 4):
            logs = {'val_loss': 0.6, 'epoch': epoch}
            callback.on_epoch_end(epoch, logs)
        
        # Check that early stopping was triggered
        assert callback.wait == 3
        assert callback.stopped_epoch == 3
        assert logs.get('stop_training') is True

class TestAdvancedReduceLROnPlateau:
    """Test suite for AdvancedReduceLROnPlateau callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = AdvancedReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        assert callback.monitor == 'val_loss'
        assert callback.factor == 0.5
        assert callback.patience == 5
        assert callback.min_lr == 1e-6
        assert callback.verbose is True
        assert callback.wait == 0
    
    def test_callback_lr_reduction(self):
        """Test learning rate reduction."""
        callback = AdvancedReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        # Set initial best metric and learning rate
        callback.best_metric = 0.5
        logs = {'val_loss': 0.6, 'learning_rate': 1e-3, 'epoch': 1}
        
        # Simulate epochs without improvement
        for epoch in range(1, 4):
            logs['epoch'] = epoch
            callback.on_epoch_end(epoch, logs)
        
        # Check that learning rate was reduced
        assert logs['learning_rate'] == 5e-4  # 1e-3 * 0.5

class TestAdvancedCSVLogger:
    """Test suite for AdvancedCSVLogger callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = AdvancedCSVLogger(
            filename='test_log.csv',
            separator=',',
            append=False,
            flush_every=100
        )
        
        assert callback.filename == 'test_log.csv'
        assert callback.separator == ','
        assert callback.append is False
        assert callback.flush_every == 100
        assert callback.keys is None
        assert callback.row_count == 0
    
    def test_callback_logging(self, temp_dir):
        """Test logging to CSV file."""
        import os
        log_file = os.path.join(temp_dir, 'test_log.csv')
        
        callback = AdvancedCSVLogger(
            filename=log_file,
            separator=',',
            append=False,
            flush_every=1
        )
        
        # Simulate training begin
        callback.on_train_begin({})
        
        # Simulate epoch end with logs
        logs = {'epoch': 1, 'loss': 0.5, 'accuracy': 0.8}
        callback.on_epoch_end(1, logs)
        
        # Simulate training end
        callback.on_train_end({})
        
        # Check that file was created and has content
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert 'epoch,loss,accuracy' in content
            assert '1,0.5,0.8' in content

class TestSystemMonitor:
    """Test suite for SystemMonitor callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = SystemMonitor(
            log_freq=10,
            collect_disk_io=True,
            collect_network_io=True
        )
        
        assert callback.log_freq == 10
        assert callback.collect_disk_io is True
        assert callback.collect_network_io is True
        assert callback.start_time is None
    
    def test_callback_system_metrics(self):
        """Test system metrics collection."""
        callback = SystemMonitor(log_freq=1)
        
        # Simulate training begin
        callback.on_train_begin({})
        
        # Simulate batch end with logs
        logs = {'batch': 1, 'loss': 0.5}
        callback.on_batch_end(1, logs)
        
        # Check that system metrics were added to logs
        assert 'cpu_percent' in logs
        assert 'memory_percent' in logs

class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = PerformanceProfiler(
            profile_freq=100,
            log_detailed_stats=True
        )
        
        assert callback.profile_freq == 100
        assert callback.log_detailed_stats is True
        assert len(callback.batch_times) == 0
    
    def test_callback_profiling(self):
        """Test performance profiling."""
        callback = PerformanceProfiler(profile_freq=1)
        
        # Simulate batch begin and end
        logs = {'batch': 1}
        callback.on_batch_begin(1, logs)
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        callback.on_batch_end(1, logs)
        
        # Check that timing information was recorded
        assert len(callback.batch_times) >= 0