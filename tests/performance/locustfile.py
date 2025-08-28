# Performance testing script for Deep Learning HPC DEMO

import base64
import json
import numpy as np
from locust import HttpUser, task, between, events
from PIL import Image
from io import BytesIO

class ModelServingUser(HttpUser):
    """
    Locust user class for testing model serving performance.
    
    This class simulates users making requests to the model serving endpoint
    to test performance under various load conditions.
    """
    
    # Wait time between requests (in seconds)
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user with sample data."""
        # Create a sample image for testing
        self.sample_image = self._create_sample_image()
        self.sample_image_base64 = self._image_to_base64(self.sample_image)
    
    def _create_sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image (32x32 RGB)
        image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    def _image_to_base64(self, image):
        """Convert PIL image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    @task(10)
    def predict_image(self):
        """Task to make prediction requests."""
        # Prepare request payload
        payload = {
            "image": self.sample_image_base64,
            "request_id": f"locust_test_{np.random.randint(1000000)}"
        }
        
        # Make POST request to prediction endpoint
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="POST /predict"
        ) as response:
            # Check response status
            if response.status_code == 200:
                try:
                    # Parse response
                    result = response.json()
                    
                    # Validate response structure
                    if "predicted_class" in result and "confidence" in result:
                        response.success()
                    else:
                        response.failure(f"Invalid response structure: {result}")
                except json.JSONDecodeError:
                    response.failure("Response is not valid JSON")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(2)
    def predict_batch_images(self):
        """Task to make batch prediction requests."""
        # Create a batch of sample images
        batch_size = 4
        batch_payload = []
        
        for i in range(batch_size):
            payload = {
                "image": self.sample_image_base64,
                "request_id": f"locust_batch_test_{np.random.randint(1000000)}_{i}"
            }
            batch_payload.append(payload)
        
        # Make batch POST requests
        for payload in batch_payload:
            with self.client.post(
                "/predict",
                json=payload,
                catch_response=True,
                name="POST /predict (batch)"
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def health_check(self):
        """Task to perform health checks."""
        with self.client.get("/health", catch_response=True, name="GET /health") as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Health check failed: {result}")
                except json.JSONDecodeError:
                    response.failure("Health check response is not valid JSON")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def ready_check(self):
        """Task to perform readiness checks."""
        with self.client.get("/ready", catch_response=True, name="GET /ready") as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("status") == "ready":
                        response.success()
                    else:
                        response.failure(f"Readiness check failed: {result}")
                except json.JSONDecodeError:
                    response.failure("Readiness check response is not valid JSON")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

# Event hooks for custom metrics and logging
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is started."""
    print("Starting performance test for Deep Learning HPC DEMO")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is stopped."""
    print("Performance test completed for Deep Learning HPC DEMO")

@events.request.add_listener
def on_request(context, **kwargs):
    """Called for each request made."""
    # Custom request processing can be added here
    pass

# Custom statistics
@events.report_to_master.add_listener
def on_report_to_master(client_id, data):
    """Called when a worker reports to master."""
    # Add custom statistics to report to master
    data["custom_stat"] = 42

@events.worker_report.add_listener
def on_worker_report(client_id, data):
    """Called when master receives report from worker."""
    # Process custom statistics from worker
    custom_stat = data.get("custom_stat", 0)
    if custom_stat > 0:
        print(f"Worker {client_id} reported custom stat: {custom_stat}")