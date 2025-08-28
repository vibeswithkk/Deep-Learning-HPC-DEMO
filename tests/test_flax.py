import unittest
import jax
import jax.numpy as jnp
from src.models.flax_cnn import create_model

class TestFlaxCNN(unittest.TestCase):
    def setUp(self):
        self.model = create_model(num_classes=1000)
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 3, 224, 224)
        self.sample_input = jnp.ones(self.input_shape)
    
    def test_model_creation(self):
        variables = self.model.init(self.rng, self.sample_input)
        self.assertIn('params', variables)
    
    def test_forward_pass(self):
        variables = self.model.init(self.rng, self.sample_input)
        output = self.model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_training_mode(self):
        variables = self.model.init(self.rng, self.sample_input)
        output_train = self.model.apply(variables, self.sample_input, train=True, rngs={'dropout': self.rng})
        output_eval = self.model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output_train.shape, (1, 1000))
        self.assertEqual(output_eval.shape, (1, 1000))
    
    def test_parameter_count(self):
        variables = self.model.init(self.rng, self.sample_input)
        params = variables['params']
        param_count = sum(x.size for x in jax.tree_leaves(params))
        self.assertGreater(param_count, 1000000)

if __name__ == '__main__':
    unittest.main()