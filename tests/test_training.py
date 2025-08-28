import unittest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
from src.training.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, CallbackList
from src.optimizers.optax_utils import lion, ranger, lamb, adabelief
from src.optimizers.torch_optimizers import Lion

class DummyCallback(Callback):
    def __init__(self):
        self.calls = []
    
    def on_train_begin(self, logs):
        self.calls.append('on_train_begin')
    
    def on_train_end(self, logs):
        self.calls.append('on_train_end')
    
    def on_epoch_begin(self, epoch, logs):
        self.calls.append(f'on_epoch_begin_{epoch}')
    
    def on_epoch_end(self, epoch, logs):
        self.calls.append(f'on_epoch_end_{epoch}')
    
    def on_batch_begin(self, batch, logs):
        self.calls.append(f'on_batch_begin_{batch}')
    
    def on_batch_end(self, batch, logs):
        self.calls.append(f'on_batch_end_{batch}')

class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_checkpoint(self):
        filepath = os.path.join(self.temp_dir, "checkpoint")
        checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_freq=1)
        
        logs = {'state': {'params': jnp.ones((2, 2)), 'step': 0}}
        checkpoint.on_train_begin(logs)
        checkpoint.on_epoch_end(0, logs)
        
        self.assertTrue(os.path.exists(os.path.dirname(filepath)))
    
    def test_early_stopping(self):
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        logs = {'val_loss': 1.0}
        early_stopping.on_epoch_end(0, logs)
        
        logs = {'val_loss': 1.05}
        early_stopping.on_epoch_end(1, logs)
        
        logs = {'val_loss': 1.1}
        early_stopping.on_epoch_end(2, logs)
        
        self.assertEqual(early_stopping.wait, 2)
    
    def test_reduce_lr_on_plateau(self):
        reduce_lr = ReduceLROnPlateau(patience=1, factor=0.5)
        
        logs = {'val_loss': 1.0, 'learning_rate': 0.01}
        reduce_lr.on_epoch_end(0, logs)
        
        logs = {'val_loss': 1.05, 'learning_rate': 0.01}
        reduce_lr.on_epoch_end(1, logs)
        
        self.assertEqual(logs['learning_rate'], 0.005)
    
    def test_csv_logger(self):
        filepath = os.path.join(self.temp_dir, "logs.csv")
        csv_logger = CSVLogger(filepath)
        
        logs = {'loss': 1.0, 'accuracy': 0.5}
        csv_logger.on_train_begin(logs)
        csv_logger.on_epoch_end(0, logs)
        csv_logger.on_train_end(logs)
        
        self.assertTrue(os.path.exists(filepath))
    
    def test_callback_list(self):
        callback1 = DummyCallback()
        callback2 = DummyCallback()
        
        callback_list = CallbackList([callback1, callback2])
        
        logs = {}
        callback_list.on_train_begin(logs)
        callback_list.on_epoch_begin(0, logs)
        callback_list.on_epoch_end(0, logs)
        callback_list.on_train_end(logs)
        
        expected_calls = ['on_train_begin', 'on_epoch_begin_0', 'on_epoch_end_0', 'on_train_end']
        for call in expected_calls:
            self.assertIn(call, callback1.calls)
            self.assertIn(call, callback2.calls)

class TestOptaxOptimizers(unittest.TestCase):
    def test_lion_optimizer(self):
        optimizer = lion(learning_rate=0.001, b1=0.9, b2=0.99)
        self.assertIsNotNone(optimizer)
    
    def test_ranger_optimizer(self):
        optimizer = ranger(learning_rate=0.001)
        self.assertIsNotNone(optimizer)
    
    def test_lamb_optimizer(self):
        optimizer = lamb(learning_rate=0.001, weight_decay=0.01)
        self.assertIsNotNone(optimizer)
    
    def test_adabelief_optimizer(self):
        optimizer = adabelief(learning_rate=0.001)
        self.assertIsNotNone(optimizer)

class TestTorchOptimizers(unittest.TestCase):
    def test_lion_torch_optimizer(self):
        import torch
        import torch.nn as nn
        
        model = nn.Linear(2, 2)
        optimizer = Lion(model.parameters(), lr=0.001)
        self.assertIsNotNone(optimizer)
        
        # Test optimizer step
        x = torch.randn(1, 2)
        y = model(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    unittest.main()