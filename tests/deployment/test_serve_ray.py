# Tests for deployment module

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch, MagicMock
from src.deployment.serve_ray import (
    ModelConfigDTO, 
    ImageProcessor, 
    CacheManager,
    RequestLimiter,
    RateLimiter,
    CircuitBreaker
)

class TestModelConfigDTO:
    """Test suite for ModelConfigDTO."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = ModelConfigDTO(
            num_classes=1000,
            model_path="/models/latest",
            input_shape=[224, 224, 3]
        )
        
        assert config.num_classes == 1000
        assert config.model_path == "/models/latest"
        assert config.input_shape == [224, 224, 3]
    
    def test_config_default_values(self):
        """Test configuration default values."""
        config = ModelConfigDTO()
        
        assert config.num_classes == 1000
        assert config.model_path == "./models/latest"
        assert config.input_shape == [224, 224, 3]
        assert config.max_concurrent_requests == 100
        assert config.cache_size == 1000

class TestImageProcessor:
    """Test suite for ImageProcessor."""
    
    def test_decode_image_from_base64(self):
        """Test decoding image from base64 string."""
        # Create a simple test image
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Convert to base64
        import base64
        from PIL import Image
        from io import BytesIO
        
        pil_image = Image.fromarray(test_image)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Decode the image
        decoded_image = ImageProcessor.decode_image(img_str)
        
        # Check that the decoded image has the correct shape
        assert decoded_image.shape == (32, 32, 3)
        assert decoded_image.dtype == np.uint8
    
    def test_decode_image_from_list(self):
        """Test decoding image from list."""
        test_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        test_array = np.array(test_list, dtype=np.float32)
        
        decoded_image = ImageProcessor.decode_image(test_list)
        
        assert np.allclose(decoded_image, test_array)
        assert decoded_image.dtype == np.float32
    
    def test_decode_image_from_array(self):
        """Test decoding image from numpy array."""
        test_array = np.random.rand(32, 32, 3).astype(np.float32)
        
        decoded_image = ImageProcessor.decode_image(test_array)
        
        assert np.allclose(decoded_image, test_array)
        assert decoded_image.dtype == np.float32
    
    def test_preprocess_image_rgb(self):
        """Test preprocessing RGB image."""
        test_image = np.random.rand(64, 64, 3).astype(np.float32)
        target_shape = [32, 32, 3]
        
        processed_image = ImageProcessor.preprocess_image(test_image, target_shape)
        
        assert processed_image.shape == (32, 32, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0)
    
    def test_preprocess_image_grayscale(self):
        """Test preprocessing grayscale image."""
        test_image = np.random.rand(64, 64).astype(np.float32)
        target_shape = [32, 32, 3]
        
        processed_image = ImageProcessor.preprocess_image(test_image, target_shape)
        
        assert processed_image.shape == (32, 32, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0)

class TestCacheManager:
    """Test suite for CacheManager."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = CacheManager(cache_size=100)
        
        assert cache.cache_size == 100
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_cache_put_and_get(self):
        """Test putting and getting items from cache."""
        cache = CacheManager(cache_size=100)
        
        # Put an item in cache
        cache.put("key1", "value1")
        
        # Get the item from cache
        result = cache.get("key1")
        
        assert result == "value1"
        assert len(cache.cache) == 1
        assert len(cache.access_order) == 1
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = CacheManager(cache_size=100)
        
        # Try to get an item that's not in cache
        result = cache.get("key1")
        
        assert result is None
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(cache_size=2)
        
        # Fill the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add a new item, which should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        # Check that key2 was evicted and key1 and key3 are still there
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

class TestRequestLimiter:
    """Test suite for RequestLimiter."""
    
    def test_limiter_initialization(self):
        """Test limiter initialization."""
        limiter = RequestLimiter(max_concurrent_requests=10)
        
        assert limiter.max_concurrent_requests == 10
        assert limiter.active_requests == 0
    
    def test_acquire_success(self):
        """Test successful acquire."""
        limiter = RequestLimiter(max_concurrent_requests=2)
        
        # Acquire first request
        result1 = limiter.acquire()
        assert result1 is True
        assert limiter.active_requests == 1
        
        # Acquire second request
        result2 = limiter.acquire()
        assert result2 is True
        assert limiter.active_requests == 2
    
    def test_acquire_failure(self):
        """Test failed acquire when limit is reached."""
        limiter = RequestLimiter(max_concurrent_requests=1)
        
        # Acquire first request
        result1 = limiter.acquire()
        assert result1 is True
        assert limiter.active_requests == 1
        
        # Try to acquire second request (should fail)
        result2 = limiter.acquire()
        assert result2 is False
        assert limiter.active_requests == 1
    
    def test_release(self):
        """Test releasing requests."""
        limiter = RequestLimiter(max_concurrent_requests=2)
        
        # Acquire requests
        limiter.acquire()
        limiter.acquire()
        assert limiter.active_requests == 2
        
        # Release one request
        limiter.release()
        assert limiter.active_requests == 1
        
        # Release another request
        limiter.release()
        assert limiter.active_requests == 0

class TestRateLimiter:
    """Test suite for RateLimiter."""
    
    def test_limiter_initialization(self):
        """Test limiter initialization."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        
        assert limiter.requests_per_second == 10
        assert limiter.burst_size == 5
        assert limiter.tokens == 5  # Should start with burst size tokens
    
    def test_acquire_success(self):
        """Test successful acquire."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        
        # Acquire a request
        result = limiter.acquire()
        assert result is True
        assert limiter.tokens == 4  # Should have one less token
    
    def test_acquire_failure(self):
        """Test failed acquire when no tokens available."""
        limiter = RateLimiter(requests_per_second=10, burst_size=1)
        
        # Acquire the only token
        result1 = limiter.acquire()
        assert result1 is True
        assert limiter.tokens == 0
        
        # Try to acquire another request (should fail)
        result2 = limiter.acquire()
        assert result2 is False
        assert limiter.tokens == 0

class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""
    
    def test_breaker_initialization(self):
        """Test breaker initialization."""
        breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        
        assert breaker.failure_threshold == 5
        assert breaker.timeout_seconds == 60
        assert breaker.failure_count == 0
        assert breaker.state == "CLOSED"
    
    def test_call_success(self):
        """Test successful call."""
        breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        
        # Define a simple function to call
        def test_function(x):
            return x * 2
        
        # Call the function through the circuit breaker
        result = breaker.call(test_function, 5)
        
        assert result == 10
        assert breaker.failure_count == 0
        assert breaker.state == "CLOSED"
    
    def test_call_failure_and_recovery(self):
        """Test failure handling and recovery."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=1)
        
        # Define a function that raises an exception
        def failing_function():
            raise Exception("Test failure")
        
        # Call the function multiple times to trigger circuit breaker
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        assert breaker.failure_count == 1
        assert breaker.state == "CLOSED"
        
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        assert breaker.failure_count == 2
        assert breaker.state == "OPEN"
        
        # Try to call again while circuit is open (should raise CircuitBreakerOpenException)
        with pytest.raises(Exception, match="Circuit breaker is open"):
            breaker.call(failing_function)