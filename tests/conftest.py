# Configuration file for pytest

import pytest
import jax
import torch
import numpy as np
import tempfile
import os
from typing import Dict, Any
from dataclasses import dataclass

# Enable float64 for JAX tests
jax.config.update("jax_enable_x64", True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

@dataclass
class TestConfig:
    """Test configuration class."""
    batch_size: int = 4
    image_size: tuple = (32, 32, 3)
    num_classes: int = 10
    learning_rate: float = 1e-3
    num_epochs: int = 2
    temp_dir: str = tempfile.gettempdir()

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TestConfig()

@pytest.fixture
def sample_image_batch(test_config):
    """Fixture providing a sample image batch."""
    batch_size, image_size = test_config.batch_size, test_config.image_size
    return np.random.rand(batch_size, *image_size).astype(np.float32)

@pytest.fixture
def sample_label_batch(test_config):
    """Fixture providing a sample label batch."""
    batch_size, num_classes = test_config.batch_size, test_config.num_classes
    labels = np.random.randint(0, num_classes, batch_size)
    return np.eye(num_classes)[labels].astype(np.float32)

@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run GPU tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
            
    if config.getoption("--rungpu"):
        # --rungpu given in cli: do not skip GPU tests
        return
    skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)