# Tests for utilities module

import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.utils.dataset import (
    DatasetConfig,
    AdvancedImageAugmentation,
    AdvancedDatasetProcessor,
    create_advanced_image_dataset,
    jax_collate_fn,
    calculate_advanced_metrics,
    normalize_image,
    denormalize_image
)

class TestDatasetConfig:
    """Test suite for DatasetConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = DatasetConfig(
            dataset_name="cifar10",
            image_size=(32, 32),
            num_classes=10,
            batch_size=32,
            augment=True
        )
        
        assert config.dataset_name == "cifar10"
        assert config.image_size == (32, 32)
        assert config.num_classes == 10
        assert config.batch_size == 32
        assert config.augment is True
    
    def test_config_default_values(self):
        """Test configuration default values."""
        config = DatasetConfig()
        
        assert config.dataset_name == "mnist"
        assert config.image_size == (224, 224)
        assert config.num_classes == 1000
        assert config.batch_size == 64
        assert config.augment is False

class TestAdvancedImageAugmentation:
    """Test suite for AdvancedImageAugmentation."""
    
    def test_augmentation_initialization(self):
        """Test augmentation initialization."""
        config = DatasetConfig(augment=True)
        augmentation = AdvancedImageAugmentation(config)
        
        assert augmentation.config == config
        assert augmentation.transform is not None
    
    def test_augment_image(self):
        """Test image augmentation."""
        config = DatasetConfig(augment=True)
        augmentation = AdvancedImageAugmentation(config)
        
        # Create a test image
        test_image = np.random.rand(32, 32, 3).astype(np.float32)
        
        # Apply augmentation
        augmented_image = augmentation.augment_image(test_image)
        
        # Check that output has the same shape
        assert augmented_image.shape == test_image.shape
        assert augmented_image.dtype == test_image.dtype

class TestAdvancedDatasetProcessor:
    """Test suite for AdvancedDatasetProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = DatasetConfig()
        processor = AdvancedDatasetProcessor(config)
        
        assert processor.config == config
        assert processor.executor is not None
    
    def test_decode_image_from_bytes(self):
        """Test decoding image from bytes."""
        config = DatasetConfig()
        processor = AdvancedDatasetProcessor(config)
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Convert to bytes (simulating TF image encoding)
        import tensorflow as tf
        img_tensor = tf.convert_to_tensor(test_image)
        img_bytes = tf.io.encode_png(img_tensor).numpy()
        
        # Decode the image
        decoded_image = processor.decode_image(img_bytes)
        
        # Check that the decoded image has the correct shape
        assert decoded_image.shape == (32, 32, 3)
        assert decoded_image.dtype == np.uint8
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        config = DatasetConfig(
            image_size=(32, 32),
            normalize=True,
            dtype="float32"
        )
        processor = AdvancedDatasetProcessor(config)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Preprocess the image
        processed_image = processor.preprocess_image(test_image)
        
        # Check output shape and type
        assert processed_image.shape == (32, 32, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= -10.0) and np.all(processed_image <= 10.0)
    
    def test_preprocess_example(self):
        """Test example preprocessing."""
        config = DatasetConfig(
            num_classes=10,
            image_size=(32, 32),
            normalize=True
        )
        processor = AdvancedDatasetProcessor(config)
        
        # Create a test example
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        example = {
            'image': test_image,
            'label': 5
        }
        
        # Preprocess the example
        processed_example = processor.preprocess_example(example)
        
        # Check output shapes and types
        assert processed_example['image'].shape == (32, 32, 3)
        assert processed_example['image'].dtype == np.float32
        assert processed_example['label'].shape == (10,)
        assert processed_example['label'].dtype == np.float32

class TestMetricFunctions:
    """Test suite for metric functions."""
    
    def test_calculate_advanced_metrics(self):
        """Test advanced metrics calculation."""
        config = DatasetConfig(num_classes=3)
        
        # Create test predictions and labels
        predictions = jnp.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
        labels = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        
        # Calculate metrics
        metrics = calculate_advanced_metrics(predictions, labels, config)
        
        # Check that all expected metrics are present
        expected_metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create a test image with known values
        test_image = jnp.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]])
        
        # Normalize the image
        normalized_image = normalize_image(test_image)
        
        # Check output shape and type
        assert normalized_image.shape == test_image.shape
        assert normalized_image.dtype == test_image.dtype
    
    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create a normalized test image
        test_image = jnp.array([[[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]], 
                               [[-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]])
        
        # Denormalize the image
        denormalized_image = denormalize_image(test_image)
        
        # Check output shape and type
        assert denormalized_image.shape == test_image.shape
        assert denormalized_image.dtype == test_image.dtype

class TestCollateFunctions:
    """Test suite for collate functions."""
    
    def test_jax_collate_fn(self):
        """Test JAX collate function."""
        # Create test batch
        batch = [
            {'image': np.random.rand(32, 32, 3), 'label': np.array([1, 0, 0])},
            {'image': np.random.rand(32, 32, 3), 'label': np.array([0, 1, 0])},
            {'image': np.random.rand(32, 32, 3), 'label': np.array([0, 0, 1])}
        ]
        
        # Collate the batch
        collated_batch = jax_collate_fn(batch)
        
        # Check output shapes
        assert collated_batch['image'].shape == (3, 32, 32, 3)
        assert collated_batch['label'].shape == (3, 3)
        assert isinstance(collated_batch['image'], jnp.ndarray)
        assert isinstance(collated_batch['label'], jnp.ndarray)

class TestDatasetCreation:
    """Test suite for dataset creation functions."""
    
    @patch('tensorflow_datasets.load')
    def test_create_advanced_image_dataset(self, mock_tfds_load):
        """Test advanced image dataset creation."""
        # Mock tfds.load to return a simple dataset
        import tensorflow as tf
        mock_dataset = tf.data.Dataset.from_tensor_slices({
            'image': [np.random.rand(32, 32, 3).astype(np.float32)],
            'label': [5]
        })
        mock_tfds_load.return_value = mock_dataset
        
        config = DatasetConfig(
            dataset_name="test_dataset",
            image_size=(32, 32),
            batch_size=1,
            shuffle_buffer_size=0,
            repeat=False,
            prefetch=False,
            cache=False
        )
        
        # Create the dataset
        dataset = create_advanced_image_dataset(config)
        
        # Check that the dataset was created
        assert dataset is not None
        
        # Verify that tfds.load was called with correct parameters
        mock_tfds_load.assert_called_once_with("test_dataset", split='train')