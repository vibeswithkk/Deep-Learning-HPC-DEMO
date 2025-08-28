import unittest
import jax
import jax.numpy as jnp
from flax.core import freeze
from src.models.flax_cnn import create_model, ModelConfig, create_train_state

class TestAdvancedFlaxCNN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 224, 224, 3)
        self.sample_input = jnp.ones(self.input_shape)
    
    def test_resnet18_creation(self):
        config = ModelConfig(
            num_classes=1000,
            backbone="resnet18",
            block_sizes=[2, 2, 2, 2],
            use_bottleneck=False
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        self.assertIn('params', variables)
    
    def test_resnet50_creation(self):
        config = ModelConfig(
            num_classes=1000,
            backbone="resnet50",
            block_sizes=[3, 4, 6, 3],
            use_bottleneck=True
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        self.assertIn('params', variables)
    
    def test_forward_pass_resnet18(self):
        config = ModelConfig(
            num_classes=1000,
            backbone="resnet18",
            block_sizes=[2, 2, 2, 2],
            use_bottleneck=False
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        output = model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_forward_pass_resnet50(self):
        config = ModelConfig(
            num_classes=1000,
            backbone="resnet50",
            block_sizes=[3, 4, 6, 3],
            use_bottleneck=True
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        output = model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_logits_output(self):
        config = ModelConfig(num_classes=1000)
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        output = model.apply(variables, self.sample_input, train=False)
        # Check that output is logits (not softmax probabilities)
        self.assertFalse(jnp.allclose(jnp.sum(jax.nn.softmax(output, axis=-1), axis=-1), 1.0))
    
    def test_group_normalization(self):
        config = ModelConfig(
            num_classes=1000,
            normalization="groupnorm"
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        output = model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_layer_normalization(self):
        config = ModelConfig(
            num_classes=1000,
            normalization="layernorm"
        )
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        output = model.apply(variables, self.sample_input, train=False)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_different_activations(self):
        for activation in ["relu", "gelu", "swish"]:
            with self.subTest(activation=activation):
                config = ModelConfig(
                    num_classes=1000,
                    activation=activation
                )
                model = create_model(config)
                variables = model.init(self.rng, self.sample_input)
                output = model.apply(variables, self.sample_input, train=False)
                self.assertEqual(output.shape, (1, 1000))
    
    def test_train_state_creation(self):
        config = ModelConfig(num_classes=1000)
        state = create_train_state(self.rng, config)
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.params)
        self.assertIsNotNone(state.tx)
    
    def test_parameter_count(self):
        config = ModelConfig(num_classes=1000)
        model = create_model(config)
        variables = model.init(self.rng, self.sample_input)
        params = variables['params']
        param_count = sum(x.size for x in jax.tree_leaves(params))
        self.assertGreater(param_count, 1000000)

if __name__ == '__main__':
    unittest.main()