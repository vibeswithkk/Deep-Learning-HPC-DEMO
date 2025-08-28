# Tests for optimizers

import pytest
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import numpy as np
from src.optimizers.optax_utils import create_advanced_optax_optimizer
from src.optimizers.torch_optimizers import create_advanced_torch_optimizer

class TestOptaxOptimizers:
    """Test suite for Optax optimizers."""
    
    @pytest.fixture
    def sample_params(self):
        """Fixture providing sample parameters for testing."""
        params = {
            'layer1': {
                'kernel': jnp.ones((10, 5)),
                'bias': jnp.zeros(5)
            },
            'layer2': {
                'kernel': jnp.ones((5, 1)),
                'bias': jnp.zeros(1)
            }
        }
        return params
    
    @pytest.fixture
    def sample_gradients(self):
        """Fixture providing sample gradients for testing."""
        grads = {
            'layer1': {
                'kernel': jnp.ones((10, 5)) * 0.1,
                'bias': jnp.ones(5) * 0.1
            },
            'layer2': {
                'kernel': jnp.ones((5, 1)) * 0.1,
                'bias': jnp.ones(1) * 0.1
            }
        }
        return grads
    
    @pytest.mark.parametrize("optimizer_name", [
        "adam", "adamw", "sgd", "rmsprop", "adabelief", "lion"
    ])
    def test_optimizer_initialization(self, optimizer_name, sample_params):
        """Test optimizer initialization."""
        optimizer = create_advanced_optax_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=1e-3
        )
        
        # Initialize optimizer state
        opt_state = optimizer.init(sample_params)
        
        # Check that state is properly initialized
        assert opt_state is not None
    
    def test_adam_optimizer_step(self, sample_params, sample_gradients):
        """Test Adam optimizer step."""
        optimizer = create_advanced_optax_optimizer(
            optimizer_name="adam",
            learning_rate=1e-3
        )
        
        # Initialize optimizer state
        opt_state = optimizer.init(sample_params)
        
        # Perform optimizer step
        updates, new_opt_state = optimizer.update(sample_gradients, opt_state, sample_params)
        new_params = jax.tree_map(lambda p, u: p + u, sample_params, updates)
        
        # Check that parameters are updated
        for layer_name in sample_params:
            for param_name in sample_params[layer_name]:
                original_param = sample_params[layer_name][param_name]
                updated_param = new_params[layer_name][param_name]
                assert not jnp.allclose(original_param, updated_param, rtol=1e-5)
    
    def test_adabelief_optimizer_step(self, sample_params, sample_gradients):
        """Test AdaBelief optimizer step."""
        optimizer = create_advanced_optax_optimizer(
            optimizer_name="adabelief",
            learning_rate=1e-3
        )
        
        # Initialize optimizer state
        opt_state = optimizer.init(sample_params)
        
        # Perform optimizer step
        updates, new_opt_state = optimizer.update(sample_gradients, opt_state, sample_params)
        new_params = jax.tree_map(lambda p, u: p + u, sample_params, updates)
        
        # Check that parameters are updated
        for layer_name in sample_params:
            for param_name in sample_params[layer_name]:
                original_param = sample_params[layer_name][param_name]
                updated_param = new_params[layer_name][param_name]
                assert not jnp.allclose(original_param, updated_param, rtol=1e-5)
    
    def test_gradient_clipping(self, sample_params, sample_gradients):
        """Test optimizer with gradient clipping."""
        optimizer = create_advanced_optax_optimizer(
            optimizer_name="adam",
            learning_rate=1e-3,
            gradient_clipping=0.01
        )
        
        # Initialize optimizer state
        opt_state = optimizer.init(sample_params)
        
        # Perform optimizer step
        updates, new_opt_state = optimizer.update(sample_gradients, opt_state, sample_params)
        
        # Check that gradients are clipped
        for layer_name in updates:
            for param_name in updates[layer_name]:
                update = updates[layer_name][param_name]
                assert jnp.all(jnp.abs(update) <= 0.01 + 1e-6)

class TestTorchOptimizers:
    """Test suite for PyTorch optimizers."""
    
    @pytest.fixture
    def sample_model(self):
        """Fixture providing a sample model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        return model
    
    @pytest.fixture
    def sample_parameters(self, sample_model):
        """Fixture providing sample parameters for testing."""
        return list(sample_model.parameters())
    
    @pytest.mark.parametrize("optimizer_name", [
        "adam", "sgd", "rmsprop", "adabelief", "lion", "adan", "sophia"
    ])
    def test_optimizer_initialization(self, optimizer_name, sample_parameters):
        """Test optimizer initialization."""
        optimizer = create_advanced_torch_optimizer(
            model_parameters=sample_parameters,
            optimizer_name=optimizer_name,
            learning_rate=1e-3
        )
        
        # Check that optimizer is properly initialized
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0
    
    def test_adam_optimizer_step(self, sample_model, sample_parameters):
        """Test Adam optimizer step."""
        optimizer = create_advanced_torch_optimizer(
            model_parameters=sample_parameters,
            optimizer_name="adam",
            learning_rate=1e-3
        )
        
        # Create sample input and target
        input_data = torch.randn(4, 10)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = sample_model(input_data)
        loss = torch.mean((output - target) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Store original parameter values
        original_params = [p.clone() for p in sample_model.parameters()]
        
        # Optimizer step
        optimizer.step()
        
        # Check that parameters are updated
        for orig_param, new_param in zip(original_params, sample_model.parameters()):
            assert not torch.allclose(orig_param, new_param, rtol=1e-5)
    
    def test_sophia_optimizer_step(self, sample_model, sample_parameters):
        """Test Sophia optimizer step."""
        optimizer = create_advanced_torch_optimizer(
            model_parameters=sample_parameters,
            optimizer_name="sophia",
            learning_rate=1e-3
        )
        
        # Create sample input and target
        input_data = torch.randn(4, 10)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = sample_model(input_data)
        loss = torch.mean((output - target) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Store original parameter values
        original_params = [p.clone() for p in sample_model.parameters()]
        
        # Optimizer step
        optimizer.step()
        
        # Check that parameters are updated
        for orig_param, new_param in zip(original_params, sample_model.parameters()):
            assert not torch.allclose(orig_param, new_param, rtol=1e-5)
    
    def test_gradient_clipping(self, sample_model, sample_parameters):
        """Test optimizer with gradient clipping."""
        optimizer = create_advanced_torch_optimizer(
            model_parameters=sample_parameters,
            optimizer_name="adam",
            learning_rate=1e-3,
            gradient_clipping=0.01
        )
        
        # Create sample input and target
        input_data = torch.randn(4, 10)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = sample_model(input_data)
        loss = torch.mean((output - target) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are clipped
        for param in sample_model.parameters():
            if param.grad is not None:
                assert torch.all(torch.abs(param.grad) <= 0.01 + 1e-6)
        
        # Optimizer step
        optimizer.step()