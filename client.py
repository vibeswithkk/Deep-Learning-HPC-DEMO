import requests
import numpy as np
import json
import time

def test_single_prediction():
    url = "http://localhost:8000/predict"
    
    sample_image = np.random.rand(224, 224, 3).tolist()
    
    payload = {
        "image": sample_image
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_batch_prediction():
    url = "http://localhost:8000/predict"
    
    batch_images = [np.random.rand(224, 224, 3).tolist() for _ in range(4)]
    
    results = []
    for image in batch_images:
        payload = {"image": image}
        try:
            response = requests.post(url, json=payload)
            results.append(response.json())
        except Exception as e:
            print(f"Error in batch request: {e}")
            results.append(None)
    
    return results

def test_invalid_input():
    url = "http://localhost:8000/predict"
    
    invalid_payload = {
        "data": [1, 2, 3]
    }
    
    try:
        response = requests.post(url, json=invalid_payload)
        print(f"Invalid input test - Status Code: {response.status_code}")
        print(f"Invalid input test - Response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error in invalid input test: {e}")
        return None

if __name__ == "__main__":
    print("Testing Ray Serve Deployment")
    print("=" * 40)
    
    print("\n1. Testing single prediction:")
    result = test_single_prediction()
    
    print("\n2. Testing invalid input:")
    test_invalid_input()
    
    print("\n3. Testing batch prediction:")
    batch_results = test_batch_prediction()
    for i, result in enumerate(batch_results):
        print(f"Batch item {i+1}: {result}")