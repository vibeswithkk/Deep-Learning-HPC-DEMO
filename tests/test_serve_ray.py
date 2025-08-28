import unittest
import numpy as np
import jax
import jax.numpy as jnp
from src.deployment.serve_ray import ModelConfigDTO, ModelDeployment
from src.models.flax_cnn import create_model, ModelConfig
import tempfile
import os

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        self.model_config = ModelConfig(num_classes=1000)
        self.model = create_model(self.model_config)
        self.rng = jax.random.PRNGKey(0)
        self.sample_input = jnp.ones((1, 224, 224, 3))
        self.variables = self.model.init(self.rng, self.sample_input)
        
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint")
        
        from flax.training import checkpoints
        checkpoints.save_checkpoint(self.checkpoint_path, self.variables, 0, overwrite=True)
        
        config = ModelConfigDTO(
            model_path=self.checkpoint_path,
            num_classes=1000,
            model_version="1.0-test"
        )
        self.deployment = ModelDeployment(config)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        config = ModelConfigDTO()
        self.assertEqual(config.model_path, "./checkpoints")
        self.assertEqual(config.model_version, "1.0")
        self.assertEqual(config.num_classes, 1000)
    
    def test_model_loading(self):
        self.assertIsNotNone(self.deployment.params)
        self.assertTrue(hasattr(self.deployment, 'jitted_apply'))
    
    def test_input_validation_valid(self):
        valid_input = {"image": np.random.rand(1, 224, 224, 3).tolist()}
        try:
            self.deployment.validate_input(valid_input)
        except Exception as e:
            self.fail(f"Valid input validation failed: {e}")
    
    def test_input_validation_missing_image(self):
        invalid_input = {"data": [1, 2, 3]}
        with self.assertRaises(ValueError):
            self.deployment.validate_input(invalid_input)
    
    def test_input_validation_wrong_type(self):
        invalid_input = {"image": "not_an_array"}
        with self.assertRaises(ValueError):
            self.deployment.validate_input(invalid_input)
    
    def test_input_validation_wrong_shape(self):
        invalid_input = {"image": np.random.rand(10, 10).tolist()}
        with self.assertRaises(ValueError):
            self.deployment.validate_input(invalid_input)
    
    def test_input_validation_payload_too_large(self):
        config = ModelConfigDTO(max_payload_size=100)
        deployment = ModelDeployment(config)
        large_image = {"image": np.random.rand(100, 100, 3).tolist()}
        with self.assertRaises(ValueError):
            deployment.validate_input(large_image)
    
    def test_reconfigure(self):
        new_config = {
            "model_path": self.checkpoint_path,
            "model_version": "2.0",
            "num_classes": 500
        }
        try:
            self.deployment.reconfigure(new_config)
            self.assertEqual(self.deployment.config.model_version, "2.0")
            self.assertEqual(self.deployment.config.num_classes, 500)
        except Exception as e:
            self.fail(f"Reconfigure failed: {e}")
    
    def test_prediction_result_structure(self):
        from dataclasses import is_dataclass
        from src.deployment.serve_ray import PredictionResult
        self.assertTrue(is_dataclass(PredictionResult))
        result = PredictionResult(
            predicted_class=1,
            confidence=0.95,
            model_version="1.0",
            inference_time=0.1
        )
        self.assertEqual(result.predicted_class, 1)
        self.assertEqual(result.confidence, 0.95)
    
    def test_jit_compilation(self):
        self.assertTrue(hasattr(self.deployment, 'jitted_apply'))
        self.assertTrue(callable(self.deployment.jitted_apply))

if __name__ == '__main__':
    unittest.main()