"""
Enterprise Client for Deep Learning HPC DEMO
This script provides a comprehensive client implementation for interacting with
the deployed model serving infrastructure with enterprise-grade features.
"""

import requests
import numpy as np
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
import base64
from PIL import Image
import io

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnterpriseClient")

class EnterpriseModelClient:
    """Enterprise-grade client for interacting with the deployed model service."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the enterprise client.
        
        Args:
            base_url: Base URL of the model serving endpoint
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "DeepLearningHPC-EnterpriseClient/1.0"
        })
        
        # Add API key if provided
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })
        
        logger.info(f"Enterprise client initialized with base URL: {self.base_url}")
    
    def predict_single(self, image: Union[np.ndarray, List[List[List[float]]]], 
                      request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a single prediction request.
        
        Args:
            image: Input image as numpy array or nested list
            request_id: Optional request identifier for tracing
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(image, np.ndarray):
                image_data = image.tolist()
            else:
                image_data = image
            
            payload = {
                "image": image_data
            }
            
            # Add request ID if provided
            if request_id:
                payload["request_id"] = request_id
            
            logger.info(f"Making prediction request with request_id: {request_id}")
            start_time = time.time()
            
            response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
            response_time = time.time() - start_time
            
            logger.info(f"Prediction completed in {response_time:.2f}s with status {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                result["response_time"] = response_time
                return result
            else:
                error_msg = f"Prediction failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.HTTPError(error_msg)
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, images: List[Union[np.ndarray, List[List[List[float]]]]], 
                     request_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Make batch prediction requests.
        
        Args:
            images: List of input images
            request_ids: Optional list of request identifiers
            
        Returns:
            List of dictionaries containing prediction results
        """
        results = []
        
        # Process images in batch
        for i, image in enumerate(images):
            request_id = request_ids[i] if request_ids and i < len(request_ids) else f"batch_{i}"
            try:
                result = self.predict_single(image, request_id)
                results.append(result)
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "request_id": request_id,
                    "status": "failed"
                }
                results.append(error_result)
                logger.warning(f"Batch prediction failed for request {request_id}: {str(e)}")
        
        return results
    
    def predict_from_image_file(self, image_path: str, 
                               request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a prediction from an image file.
        
        Args:
            image_path: Path to the image file
            request_id: Optional request identifier
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard size (224x224)
                img = img.resize((224, 224))
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
            return self.predict_single(img_array, request_id)
            
        except Exception as e:
            logger.error(f"Error processing image file {image_path}: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the model service.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            logger.info("Performing health check")
            response = self.session.get(f"{self.base_url}/healthz", timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "details": response.json() if response.content else {},
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "details": response.text,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the deployed model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            logger.info("Retrieving model information")
            response = self.session.get(f"{self.base_url}/model/info", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to retrieve model info: {response.status_code}"
                logger.error(error_msg)
                raise requests.exceptions.HTTPError(error_msg)
                
        except Exception as e:
            logger.error(f"Error retrieving model information: {str(e)}")
            raise

def test_single_prediction():
    """Test single prediction functionality."""
    url = "http://localhost:8000"
    
    # Create client
    client = EnterpriseModelClient(base_url=url)
    
    # Generate sample image
    sample_image = np.random.rand(224, 224, 3).tolist()
    
    try:
        result = client.predict_single(sample_image, request_id="test_single_001")
        print(f"Single prediction result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"Error in single prediction: {e}")
        return None

def test_batch_prediction():
    """Test batch prediction functionality."""
    url = "http://localhost:8000"
    
    # Create client
    client = EnterpriseModelClient(base_url=url)
    
    # Generate batch of sample images
    batch_images = [np.random.rand(224, 224, 3).tolist() for _ in range(4)]
    request_ids = [f"batch_test_{i}" for i in range(4)]
    
    try:
        results = client.predict_batch(batch_images, request_ids)
        print(f"Batch prediction results:")
        for i, result in enumerate(results):
            print(f"  Item {i}: {json.dumps(result, indent=2)}")
        return results
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return None

def test_invalid_input():
    """Test handling of invalid input."""
    url = "http://localhost:8000"
    
    # Create client
    client = EnterpriseModelClient(base_url=url)
    
    # Test with invalid payload
    invalid_payload = {
        "data": [1, 2, 3]
    }
    
    try:
        response = client.session.post(f"{url}/predict", json=invalid_payload)
        print(f"Invalid input test - Status Code: {response.status_code}")
        print(f"Invalid input test - Response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error in invalid input test: {e}")
        return None

def test_health_check():
    """Test health check functionality."""
    url = "http://localhost:8000"
    
    # Create client
    client = EnterpriseModelClient(base_url=url)
    
    try:
        health_status = client.health_check()
        print(f"Health check result: {json.dumps(health_status, indent=2)}")
        return health_status
    except Exception as e:
        print(f"Error in health check: {e}")
        return None

if __name__ == "__main__":
    print("Testing Enterprise Model Client")
    print("=" * 40)
    
    print("\n1. Testing health check:")
    test_health_check()
    
    print("\n2. Testing single prediction:")
    test_single_prediction()
    
    print("\n3. Testing invalid input:")
    test_invalid_input()
    
    print("\n4. Testing batch prediction:")
    test_batch_prediction()
    
    print("\nEnterprise client testing completed!")