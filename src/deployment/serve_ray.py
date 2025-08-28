import ray
from ray import serve
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from src.models.flax_cnn import create_model, ModelConfig
import numpy as np
from typing import Dict, Any, List, Union, Optional
import json
import time
import logging
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import asyncio
import psutil
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import base64
from PIL import Image
import io
import gzip
import pickle
from datetime import datetime, timedelta
import uuid

@dataclass
class ModelConfigDTO:
    model_path: str = "./checkpoints"
    model_version: str = "1.0"
    input_shape: List[int] = None
    num_classes: int = 1000
    batch_size: int = 32
    max_payload_size: int = 10 * 1024 * 1024
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30
    enable_caching: bool = True
    cache_size: int = 1000
    enable_compression: bool = True
    compression_threshold: int = 1024
    enable_batching: bool = True
    max_batch_size: int = 64
    batch_wait_timeout_s: float = 0.1
    enable_metrics: bool = True
    metrics_port: int = 8001
    enable_health_checks: bool = True
    health_check_path: str = "/healthz"
    enable_model_switching: bool = True
    supported_models: List[str] = None
    enable_request_queuing: bool = True
    max_queue_size: int = 1000
    queue_timeout_seconds: int = 60
    enable_auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    scale_period_seconds: int = 60
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    enable_request_tracing: bool = True
    trace_sampling_rate: float = 0.1
    enable_auditing: bool = True
    audit_log_path: str = "./logs/audit.log"
    enable_rate_limiting: bool = True
    rate_limit_requests_per_second: int = 1000
    rate_limit_burst_size: int = 100
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    enable_fallback_model: bool = True
    fallback_model_path: str = "./checkpoints/fallback"
    enable_model_ensemble: bool = False
    ensemble_model_paths: List[str] = None
    enable_adversarial_detection: bool = False
    adversarial_detection_threshold: float = 0.1
    enable_model_explanation: bool = False
    explanation_method: str = "integrated_gradients"
    enable_model_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    enable_model_drift_detection: bool = True
    drift_detection_threshold: float = 0.05
    enable_model_bias_detection: bool = True
    bias_detection_threshold: float = 0.05
    enable_model_fairness: bool = True
    fairness_metrics: List[str] = None
    enable_model_lineage: bool = True
    lineage_tracking_depth: int = 10
    enable_model_governance: bool = True
    governance_policies: Dict[str, Any] = None

@dataclass
class PredictionResult:
    predicted_class: int
    confidence: float
    model_version: str
    inference_time: float
    request_id: str = ""
    cached: bool = False
    trace_id: str = ""
    model_ensemble_results: List[Dict] = None
    model_explanation: Dict = None
    adversarial_score: float = 0.0
    bias_metrics: Dict = None
    fairness_metrics: Dict = None

logger = logging.getLogger("ModelDeployment")
logging.basicConfig(level=logging.INFO)

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests', ['model_version', 'status'])
INFERENCE_TIME = Histogram('model_inference_seconds', 'Inference time in seconds', ['model_version'])
ERROR_COUNT = Counter('model_errors_total', 'Total number of errors', ['model_version', 'error_type'])
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('model_cpu_usage_percent', 'CPU usage percentage')
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Number of active requests')
CACHE_HITS = Counter('model_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('model_cache_misses_total', 'Total cache misses')
QUEUE_SIZE = Gauge('model_queue_size', 'Current queue size')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency in seconds')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_version'])
MODEL_DRIFT = Gauge('model_drift', 'Model drift score', ['model_version'])
MODEL_BIAS = Gauge('model_bias', 'Model bias score', ['model_version'])

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.is_open():
                raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                raise e
    
    def is_open(self):
        if self.failure_count < self.failure_threshold:
            return False
        
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time < self.timeout_seconds

class RateLimiter:
    def __init__(self, requests_per_second: int = 1000, burst_size: int = 100):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill_time
            new_tokens = int(time_passed * self.requests_per_second)
            
            if new_tokens > 0:
                self.tokens = min(self.burst_size, self.tokens + new_tokens)
                self.last_refill_time = now
            
            if self.tokens > 0:
                self.tokens -= 1
                return True
            
            return False

class RequestQueue:
    def __init__(self, max_queue_size: int = 1000, timeout_seconds: int = 60):
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds
        self.queue = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def enqueue(self, request) -> bool:
        with self.condition:
            if len(self.queue) >= self.max_queue_size:
                return False
            
            self.queue.append((request, time.time()))
            self.condition.notify()
            return True
    
    def dequeue(self, timeout: float = None):
        with self.condition:
            while not self.queue:
                if not self.condition.wait(timeout):
                    return None
            
            request, timestamp = self.queue.pop(0)
            if time.time() - timestamp > self.timeout_seconds:
                return None
            
            return request

class RequestLimiter:
    def __init__(self, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        self.active_requests = 0
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        with self.lock:
            if self.active_requests < self.max_concurrent_requests:
                self.active_requests += 1
                return True
            return False
    
    def release(self):
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)

class ImageProcessor:
    @staticmethod
    def decode_image(image_data: Union[str, List, np.ndarray]) -> np.ndarray:
        if isinstance(image_data, str):
            # Handle base64 encoded images
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)
        elif isinstance(image_data, list):
            return np.array(image_data, dtype=np.float32)
        elif isinstance(image_data, np.ndarray):
            return image_data
        else:
            raise ValueError("Unsupported image format")
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_shape: List[int]) -> np.ndarray:
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel to RGB
            image = np.concatenate([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            # RGBA or other multi-channel to RGB
            image = image[:, :, :3]
        
        # Resize if needed
        if image.shape[:2] != target_shape[:2]:
            from skimage.transform import resize
            image = resize(image, target_shape[:2], anti_aliasing=True)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / image.max()
        
        # Ensure correct shape
        if len(image.shape) == 3:
            if image.shape[2] != target_shape[2]:
                # Add batch dimension if missing
                image = np.expand_dims(image, axis=0)
        else:
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            if image.shape[-1] != target_shape[2]:
                image = np.concatenate([image] * target_shape[2], axis=-1)
        
        return image.astype(np.float32)

class CacheManager:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end to show it was recently used
                self.access_order.remove(key)
                self.access_order.append(key)
                CACHE_HITS.inc()
                return self.cache[key]
            CACHE_MISSES.inc()
            return None
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new entry
                if len(self.cache) >= self.cache_size:
                    # Remove least recently used
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

@serve.deployment(
    route_prefix="/predict",
    ray_actor_options={"num_cpus": 2, "num_gpus": 1 if jax.devices()[0].platform == "gpu" else 0}
)
class ModelDeployment:
    def __init__(self, config: ModelConfigDTO = None):
        if config is None:
            config = ModelConfigDTO()
        self.config = config
        self.model_config = ModelConfig(num_classes=config.num_classes)
        self.model = create_model(self.model_config)
        self.load_model()
        self.jitted_apply = jax.jit(self.model.apply)
        self.request_limiter = RequestLimiter(config.max_concurrent_requests)
        self.cache_manager = CacheManager(config.cache_size) if config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.circuit_breaker = CircuitBreaker(
            config.circuit_breaker_failure_threshold,
            config.circuit_breaker_timeout_seconds
        ) if config.enable_circuit_breaker else None
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests_per_second,
            config.rate_limit_burst_size
        ) if config.enable_rate_limiting else None
        self.request_queue = RequestQueue(
            config.max_queue_size,
            config.queue_timeout_seconds
        ) if config.enable_request_queuing else None
        
        if config.enable_metrics:
            start_http_server(config.metrics_port)
        
        # Initialize metrics
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent())
        
        # Initialize audit logging
        if config.enable_auditing:
            os.makedirs(os.path.dirname(config.audit_log_path), exist_ok=True)
            self.audit_logger = logging.getLogger("ModelAudit")
            audit_handler = logging.FileHandler(config.audit_log_path)
            audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.audit_logger.addHandler(audit_handler)
            self.audit_logger.setLevel(logging.INFO)
    
    def load_model(self):
        try:
            variables = checkpoints.restore_checkpoint(self.config.model_path, None)
            self.params = variables['params']
            if 'batch_stats' in variables:
                self.batch_stats = variables['batch_stats']
            logger.info(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.config.model_path}: {str(e)}")
            raise
    
    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.1)
    async def __call__(self, requests: List[Dict]):
        REQUEST_COUNT.labels(model_version=self.config.model_version, status='received').inc(len(requests))
        ACTIVE_REQUESTS.inc(len(requests))
        QUEUE_SIZE.dec(len(requests))
        
        start_time = time.time()
        
        try:
            # Rate limiting
            if self.rate_limiter:
                for _ in requests:
                    if not self.rate_limiter.acquire():
                        error_response = {
                            "error": "Rate limit exceeded",
                            "message": "Too many requests per second"
                        }
                        REQUEST_COUNT.labels(model_version=self.config.model_version, status='rejected').inc(len(requests))
                        ACTIVE_REQUESTS.dec(len(requests))
                        return [error_response for _ in requests]
            
            if not self.request_limiter.acquire():
                # Queue requests if enabled
                if self.request_queue:
                    queued_requests = []
                    for req in requests:
                        if self.request_queue.enqueue(req):
                            queued_requests.append(req)
                        else:
                            error_response = {
                                "error": "Queue full",
                                "message": "Maximum queue size exceeded"
                            }
                            REQUEST_COUNT.labels(model_version=self.config.model_version, status='rejected').inc(1)
                            ACTIVE_REQUESTS.dec(1)
                            return [error_response if r == req else {"error": "Success"} for r in requests]
                    
                    # Wait for dequeuing
                    processed_requests = []
                    for req in queued_requests:
                        dequeued_req = self.request_queue.dequeue(self.config.queue_timeout_seconds)
                        if dequeued_req:
                            processed_requests.append(dequeued_req)
                        else:
                            error_response = {
                                "error": "Request timeout",
                                "message": "Request timed out in queue"
                            }
                            REQUEST_COUNT.labels(model_version=self.config.model_version, status='timeout').inc(1)
                            ACTIVE_REQUESTS.dec(1)
                            return [error_response if r == req else {"error": "Success"} for r in requests]
                    
                    requests = processed_requests
                else:
                    error_response = {
                        "error": "Too many concurrent requests",
                        "message": f"Maximum concurrent requests ({self.config.max_concurrent_requests}) exceeded"
                    }
                    REQUEST_COUNT.labels(model_version=self.config.model_version, status='rejected').inc(len(requests))
                    ACTIVE_REQUESTS.dec(len(requests))
                    return [error_response for _ in requests]
            
            images = []
            request_ids = []
            trace_ids = []
            
            for req in requests:
                json_input = await req.json()
                request_id = json_input.get("request_id", str(uuid.uuid4()))
                trace_id = json_input.get("trace_id", str(uuid.uuid4())) if self.config.enable_request_tracing else ""
                request_ids.append(request_id)
                trace_ids.append(trace_id)
                
                # Check cache if enabled
                cached_result = None
                if self.cache_manager:
                    cache_key = hashlib.md5(str(json_input).encode()).hexdigest()
                    cached_result = self.cache_manager.get(cache_key)
                    if cached_result:
                        cached_result["cached"] = True
                        cached_result["request_id"] = request_id
                        cached_result["trace_id"] = trace_id
                
                if cached_result:
                    images.append(None)  # Placeholder
                    continue
                
                self.validate_input(json_input)
                
                # Process image
                image_data = json_input["image"]
                image_array = ImageProcessor.decode_image(image_data)
                processed_image = ImageProcessor.preprocess_image(image_array, self.config.input_shape or [224, 224, 3])
                images.append(processed_image)
            
            # Filter out cached requests
            non_cached_indices = [i for i, img in enumerate(images) if img is not None]
            non_cached_images = [images[i] for i in non_cached_indices]
            
            results = []
            
            # Handle cached results first
            for i, (img, req_id, trace_id) in enumerate(zip(images, request_ids, trace_ids)):
                if img is None:  # Cached result
                    # Find the cached result in the original requests
                    for j, orig_req in enumerate(requests):
                        orig_json = asyncio.run(orig_req.json())
                        if orig_json.get("request_id", str(uuid.uuid4())) == req_id:
                            cache_key = hashlib.md5(str(orig_json).encode()).hexdigest()
                            cached_result = self.cache_manager.get(cache_key)
                            if cached_result:
                                results.append(cached_result)
                            break
            
            # Process non-cached images
            if non_cached_images:
                batch_images = np.stack(non_cached_images, axis=0)
                batch_images = jnp.array(batch_images)
                
                # Apply circuit breaker if enabled
                if self.circuit_breaker:
                    predictions = self.circuit_breaker.call(
                        self.jitted_apply,
                        {'params': self.params, 'batch_stats': self.batch_stats} if hasattr(self, 'batch_stats') else {'params': self.params},
                        batch_images,
                        train=False
                    )
                else:
                    if hasattr(self, 'batch_stats'):
                        predictions = self.jitted_apply(
                            {'params': self.params, 'batch_stats': self.batch_stats}, 
                            batch_images, 
                            train=False,
                            mutable=False
                        )
                    else:
                        predictions = self.jitted_apply(
                            {'params': self.params}, 
                            batch_images, 
                            train=False
                        )
                
                # Process predictions
                for i, pred in enumerate(predictions):
                    predicted_class = int(jnp.argmax(pred, axis=-1))
                    softmax_probs = jax.nn.softmax(pred, axis=-1)
                    confidence = float(jnp.max(softmax_probs, axis=-1))
                    inference_time = time.time() - start_time
                    
                    result = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "model_version": self.config.model_version,
                        "inference_time": inference_time,
                        "request_id": request_ids[non_cached_indices[i]],
                        "trace_id": trace_ids[non_cached_indices[i]],
                        "cached": False
                    }
                    
                    # Add adversarial detection if enabled
                    if self.config.enable_adversarial_detection:
                        # Simple adversarial detection based on confidence
                        adversarial_score = 1.0 - confidence
                        result["adversarial_score"] = adversarial_score
                    
                    # Cache result if enabled
                    if self.cache_manager:
                        orig_req = requests[non_cached_indices[i]]
                        orig_json = asyncio.run(orig_req.json())
                        cache_key = hashlib.md5(str(orig_json).encode()).hexdigest()
                        self.cache_manager.put(cache_key, result)
                    
                    results.append(result)
            
            # Update metrics
            inference_duration = time.time() - start_time
            INFERENCE_TIME.labels(model_version=self.config.model_version).observe(inference_duration)
            REQUEST_COUNT.labels(model_version=self.config.model_version, status='success').inc(len(requests))
            REQUEST_LATENCY.observe(inference_duration)
            
            # Update system metrics
            MEMORY_USAGE.set(psutil.Process().memory_info().rss)
            CPU_USAGE.set(psutil.cpu_percent())
            
            # Audit logging
            if self.config.enable_auditing:
                for result in results:
                    self.audit_logger.info(f"Prediction: {json.dumps(result)}")
            
            return results
            
        except Exception as e:
            ERROR_COUNT.labels(model_version=self.config.model_version, error_type=type(e).__name__).inc(len(requests))
            logger.error(f"Inference error: {str(e)}")
            error_response = {
                "error": "Invalid input or inference failed",
                "message": str(e) if len(str(e)) < 200 else "Internal server error"
            }
            return [error_response for _ in requests]
        finally:
            self.request_limiter.release()
            ACTIVE_REQUESTS.dec(len(requests))
    
    def validate_input(self, json_input: Dict):
        if not self.config.enable_input_validation:
            return
            
        if "image" not in json_input:
            raise ValueError("Missing 'image' field in request")
        
        image_data = json_input["image"]
        if not isinstance(image_data, (str, list, np.ndarray)):
            raise ValueError("Image data must be a string (base64), list, or array")
        
        if isinstance(image_data, str):
            # Check if it's base64 encoded
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                base64.b64decode(image_data)
            except Exception:
                raise ValueError("Invalid base64 encoded image")
        elif isinstance(image_data, list):
            image_array = np.array(image_data)
            if image_array.dtype not in [np.float32, np.float64, np.uint8]:
                raise ValueError("Image data must be numeric")
        elif isinstance(image_data, np.ndarray):
            if image_data.dtype not in [np.float32, np.float64, np.uint8]:
                raise ValueError("Image data must be numeric")
        
        # Check payload size
        if isinstance(image_data, str):
            payload_size = len(image_data.encode('utf-8'))
        else:
            payload_size = np.array(image_data).nbytes
        
        if payload_size > self.config.max_payload_size:
            raise ValueError(f"Image payload too large: {payload_size} bytes > {self.config.max_payload_size} bytes")
    
    def reconfigure(self, config: Dict[Any, Any]):
        try:
            new_config = ModelConfigDTO(**config)
            if new_config.model_path != self.config.model_path:
                self.config = new_config
                self.model_config = ModelConfig(num_classes=new_config.num_classes)
                self.model = create_model(self.model_config)
                self.load_model()
                self.jitted_apply = jax.jit(self.model.apply)
                logger.info(f"Model reconfigured with new path: {new_config.model_path}")
            else:
                self.config = new_config
                logger.info("Model configuration updated")
            
            # Update cache if needed
            if new_config.enable_caching and (not self.cache_manager or new_config.cache_size != self.cache_manager.cache_size):
                self.cache_manager = CacheManager(new_config.cache_size)
            elif not new_config.enable_caching:
                self.cache_manager = None
                
            # Update circuit breaker
            if new_config.enable_circuit_breaker:
                self.circuit_breaker = CircuitBreaker(
                    new_config.circuit_breaker_failure_threshold,
                    new_config.circuit_breaker_timeout_seconds
                )
            else:
                self.circuit_breaker = None
                
            # Update rate limiter
            if new_config.enable_rate_limiting:
                self.rate_limiter = RateLimiter(
                    new_config.rate_limit_requests_per_second,
                    new_config.rate_limit_burst_size
                )
            else:
                self.rate_limiter = None
                
            # Update request queue
            if new_config.enable_request_queuing:
                self.request_queue = RequestQueue(
                    new_config.max_queue_size,
                    new_config.queue_timeout_seconds
                )
            else:
                self.request_queue = None
                
        except Exception as e:
            logger.error(f"Failed to reconfigure model: {str(e)}")
            raise

model_deployment = ModelDeployment.bind(ModelConfigDTO())