import jax
import jax.numpy as jnp
from src.models.flax_cnn import create_model, ModelConfig

def main():
    print("Advanced Flax CNN Demo")
    print("======================")
    
    # Demo 1: ResNet-18
    print("\n1. Creating ResNet-18 model:")
    config_resnet18 = ModelConfig(
        num_classes=1000,
        backbone="resnet18",
        block_sizes=[2, 2, 2, 2],
        use_bottleneck=False,
        normalization="batchnorm"
    )
    model_resnet18 = create_model(config_resnet18)
    rng = jax.random.PRNGKey(0)
    sample_input = jnp.ones((1, 224, 224, 3))
    variables_resnet18 = model_resnet18.init(rng, sample_input)
    output_resnet18 = model_resnet18.apply(variables_resnet18, sample_input, train=False)
    print(f"   ResNet-18 output shape: {output_resnet18.shape}")
    
    # Demo 2: ResNet-50 with GroupNorm
    print("\n2. Creating ResNet-50 model with GroupNorm:")
    config_resnet50 = ModelConfig(
        num_classes=1000,
        backbone="resnet50",
        block_sizes=[3, 4, 6, 3],
        use_bottleneck=True,
        normalization="groupnorm"
    )
    model_resnet50 = create_model(config_resnet50)
    variables_resnet50 = model_resnet50.init(rng, sample_input)
    output_resnet50 = model_resnet50.apply(variables_resnet50, sample_input, train=False)
    print(f"   ResNet-50 output shape: {output_resnet50.shape}")
    
    # Demo 3: Custom architecture
    print("\n3. Creating custom architecture:")
    config_custom = ModelConfig(
        num_classes=100,
        backbone="custom",
        block_sizes=[2, 2, 2],
        features=[32, 64, 128],
        use_bottleneck=False,
        normalization="layernorm",
        activation="gelu",
        dropout_rate=0.1
    )
    model_custom = create_model(config_custom)
    sample_input_small = jnp.ones((1, 224, 224, 3))
    variables_custom = model_custom.init(rng, sample_input_small)
    output_custom = model_custom.apply(variables_custom, sample_input_small, train=False)
    print(f"   Custom model output shape: {output_custom.shape}")
    
    print("\nAll models created successfully!")
    print("Features demonstrated:")
    print("- Dynamic configuration")
    print("- Multiple backbones (ResNet-18, ResNet-50, custom)")
    print("- Different normalization layers (BatchNorm, GroupNorm, LayerNorm)")
    print("- Different activation functions (ReLU, GELU)")
    print("- Dropout and stochastic depth support")
    print("- Output logits (no softmax applied)")

if __name__ == "__main__":
    main()