# Tests for model registry

import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
import tempfile
import os
import hashlib
from unittest.mock import patch, MagicMock
from src.registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion,
    ModelRegistryError
)

class TestModelMetadata:
    """Test suite for ModelMetadata."""
    
    def test_metadata_initialization(self):
        """Test metadata initialization."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            framework="flax",
            num_classes=1000,
            input_shape=[224, 224, 3],
            description="Test model for testing"
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.framework == "flax"
        assert metadata.num_classes == 1000
        assert metadata.input_shape == [224, 224, 3]
        assert metadata.description == "Test model for testing"
    
    def test_metadata_default_values(self):
        """Test metadata default values."""
        metadata = ModelMetadata(name="test_model")
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.framework == "flax"
        assert metadata.num_classes == 1000
        assert metadata.input_shape == [224, 224, 3]
        assert metadata.description == ""
        assert metadata.created_at is not None
        assert metadata.updated_at is not None

class TestModelVersion:
    """Test suite for ModelVersion."""
    
    def test_version_initialization(self):
        """Test version initialization."""
        version = ModelVersion(
            version="1.0.0",
            path="/models/test_model/1.0.0",
            hash="abc123",
            size=1024,
            metrics={"accuracy": 0.95}
        )
        
        assert version.version == "1.0.0"
        assert version.path == "/models/test_model/1.0.0"
        assert version.hash == "abc123"
        assert version.size == 1024
        assert version.metrics == {"accuracy": 0.95}
        assert version.created_at is not None
    
    def test_version_default_values(self):
        """Test version default values."""
        version = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        
        assert version.version == "1.0.0"
        assert version.path == "/models/test_model/1.0.0"
        assert version.hash == ""
        assert version.size == 0
        assert version.metrics == {}
        assert version.created_at is not None

class TestModelRegistry:
    """Test suite for ModelRegistry."""
    
    @pytest.fixture
    def registry(self, temp_dir):
        """Fixture providing a model registry instance."""
        registry_path = os.path.join(temp_dir, "registry")
        return ModelRegistry(registry_path=registry_path)
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization."""
        registry_path = os.path.join(temp_dir, "registry")
        registry = ModelRegistry(registry_path=registry_path)
        
        assert registry.registry_path == registry_path
        assert os.path.exists(registry_path)
        assert registry.models == {}
    
    def test_register_model(self, registry):
        """Test registering a model."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            framework="flax"
        )
        
        # Register the model
        registry.register_model(metadata)
        
        # Check that the model was registered
        assert "test_model" in registry.models
        assert registry.models["test_model"]["metadata"] == metadata
        assert registry.models["test_model"]["versions"] == {}
    
    def test_register_model_version(self, registry):
        """Test registering a model version."""
        metadata = ModelMetadata(name="test_model")
        version = ModelVersion(
            version="1.0.0",
            path="/models/test_model/1.0.0",
            hash="abc123",
            size=1024
        )
        
        # Register the model and version
        registry.register_model(metadata)
        registry.register_model_version("test_model", version)
        
        # Check that the version was registered
        assert "1.0.0" in registry.models["test_model"]["versions"]
        assert registry.models["test_model"]["versions"]["1.0.0"] == version
    
    def test_get_model_metadata(self, registry):
        """Test getting model metadata."""
        metadata = ModelMetadata(name="test_model")
        
        # Register the model
        registry.register_model(metadata)
        
        # Get the model metadata
        retrieved_metadata = registry.get_model_metadata("test_model")
        
        # Check that the metadata matches
        assert retrieved_metadata == metadata
    
    def test_get_model_version(self, registry):
        """Test getting model version."""
        metadata = ModelMetadata(name="test_model")
        version = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        
        # Register the model and version
        registry.register_model(metadata)
        registry.register_model_version("test_model", version)
        
        # Get the model version
        retrieved_version = registry.get_model_version("test_model", "1.0.0")
        
        # Check that the version matches
        assert retrieved_version == version
    
    def test_list_models(self, registry):
        """Test listing models."""
        # Register multiple models
        metadata1 = ModelMetadata(name="model1")
        metadata2 = ModelMetadata(name="model2")
        
        registry.register_model(metadata1)
        registry.register_model(metadata2)
        
        # List models
        models = registry.list_models()
        
        # Check that both models are listed
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2
    
    def test_list_model_versions(self, registry):
        """Test listing model versions."""
        metadata = ModelMetadata(name="test_model")
        version1 = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        version2 = ModelVersion(version="2.0.0", path="/models/test_model/2.0.0")
        
        # Register the model and versions
        registry.register_model(metadata)
        registry.register_model_version("test_model", version1)
        registry.register_model_version("test_model", version2)
        
        # List model versions
        versions = registry.list_model_versions("test_model")
        
        # Check that both versions are listed
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert len(versions) == 2
    
    def test_delete_model(self, registry):
        """Test deleting a model."""
        metadata = ModelMetadata(name="test_model")
        
        # Register the model
        registry.register_model(metadata)
        assert "test_model" in registry.models
        
        # Delete the model
        registry.delete_model("test_model")
        
        # Check that the model was deleted
        assert "test_model" not in registry.models
    
    def test_delete_model_version(self, registry):
        """Test deleting a model version."""
        metadata = ModelMetadata(name="test_model")
        version = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        
        # Register the model and version
        registry.register_model(metadata)
        registry.register_model_version("test_model", version)
        assert "1.0.0" in registry.models["test_model"]["versions"]
        
        # Delete the model version
        registry.delete_model_version("test_model", "1.0.0")
        
        # Check that the version was deleted
        assert "1.0.0" not in registry.models["test_model"]["versions"]
    
    def test_save_and_load_registry(self, registry):
        """Test saving and loading registry."""
        metadata = ModelMetadata(name="test_model")
        version = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        
        # Register the model and version
        registry.register_model(metadata)
        registry.register_model_version("test_model", version)
        
        # Save the registry
        registry.save()
        
        # Create a new registry instance and load from disk
        new_registry = ModelRegistry(registry_path=registry.registry_path)
        new_registry.load()
        
        # Check that the models were loaded correctly
        assert "test_model" in new_registry.models
        assert new_registry.models["test_model"]["metadata"] == metadata
        assert "1.0.0" in new_registry.models["test_model"]["versions"]
        assert new_registry.models["test_model"]["versions"]["1.0.0"] == version
    
    def test_model_hash_verification(self, registry, temp_dir):
        """Test model hash verification."""
        # Create a temporary model file
        model_path = os.path.join(temp_dir, "test_model.pkl")
        with open(model_path, 'w') as f:
            f.write("test model content")
        
        # Calculate the hash of the file
        with open(model_path, 'rb') as f:
            content = f.read()
            expected_hash = hashlib.md5(content).hexdigest()
        
        # Verify the hash
        calculated_hash = registry._calculate_file_hash(model_path)
        assert calculated_hash == expected_hash
    
    def test_model_integrity_check(self, registry, temp_dir):
        """Test model integrity check."""
        # Create a temporary model file
        model_path = os.path.join(temp_dir, "test_model.pkl")
        with open(model_path, 'w') as f:
            f.write("test model content")
        
        # Calculate the hash of the file
        with open(model_path, 'rb') as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()
        
        metadata = ModelMetadata(name="test_model")
        version = ModelVersion(
            version="1.0.0",
            path=model_path,
            hash=file_hash
        )
        
        # Register the model and version
        registry.register_model(metadata)
        registry.register_model_version("test_model", version)
        
        # Check model integrity (should pass)
        assert registry.check_model_integrity("test_model", "1.0.0") is True
        
        # Modify the file to break integrity
        with open(model_path, 'w') as f:
            f.write("modified content")
        
        # Check model integrity (should fail)
        assert registry.check_model_integrity("test_model", "1.0.0") is False
    
    def test_get_latest_version(self, registry):
        """Test getting the latest version of a model."""
        metadata = ModelMetadata(name="test_model")
        version1 = ModelVersion(version="1.0.0", path="/models/test_model/1.0.0")
        version2 = ModelVersion(version="2.0.0", path="/models/test_model/2.0.0")
        version3 = ModelVersion(version="1.5.0", path="/models/test_model/1.5.0")
        
        # Register the model and versions
        registry.register_model(metadata)
        registry.register_model_version("test_model", version1)
        registry.register_model_version("test_model", version2)
        registry.register_model_version("test_model", version3)
        
        # Get the latest version
        latest_version = registry.get_latest_version("test_model")
        
        # Check that the latest version is correct (semantic versioning)
        assert latest_version == "2.0.0"
    
    def test_model_not_found_error(self, registry):
        """Test error when model is not found."""
        with pytest.raises(ModelRegistryError, match="Model non_existent_model not found"):
            registry.get_model_metadata("non_existent_model")
    
    def test_version_not_found_error(self, registry):
        """Test error when version is not found."""
        metadata = ModelMetadata(name="test_model")
        registry.register_model(metadata)
        
        with pytest.raises(ModelRegistryError, match="Version 1.0.0 not found for model test_model"):
            registry.get_model_version("test_model", "1.0.0")