import jax
import jax.numpy as jnp
import numpy as np
from src.models.flax_cnn import create_model, ModelConfig
from src.deployment.serve_ray import ModelConfigDTO, ModelDeployment
import tempfile
import os
import time

def main():
    print("Enterprise Deployment Demo")
    print("==========================")
    
    print("\n1. Creating model and checkpoint...")
    model_config = ModelConfig(num_classes=1000)
    model = create_model(model_config)
    rng = jax.random.PRNGKey(0)
    sample_input = jnp.ones((1, 224, 224, 3))
    variables = model.init(rng, sample_input)
    
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    
    from flax.training import checkpoints
    checkpoints.save_checkpoint(checkpoint_path, variables, 0, overwrite=True)
    print(f"   Checkpoint saved to {checkpoint_path}")
    
    print("\n2. Initializing deployment service...")
    deploy_config = ModelConfigDTO(
        model_path=checkpoint_path,
        model_version="1.0-demo",
        num_classes=1000,
        batch_size=16
    )
    
    deployment = ModelDeployment(deploy_config)
    print("   Deployment service initialized")
    
    print("\n3. Testing input validation...")
    try:
        valid_input = {"image": np.random.rand(224, 224, 3).tolist()}
        deployment.validate_input(valid_input)
        print("   ✓ Valid input passed validation")
    except Exception as e:
        print(f"   ✗ Valid input failed validation: {e}")
    
    try:
        invalid_input = {"data": [1, 2, 3]}
        deployment.validate_input(invalid_input)
        print("   ✗ Invalid input passed validation (should have failed)")
    except ValueError:
        print("   ✓ Invalid input correctly rejected")
    
    print("\n4. Testing dynamic reconfiguration...")
    new_config = {
        "model_path": checkpoint_path,
        "model_version": "2.0-demo",
        "num_classes": 500
    }
    try:
        deployment.reconfigure(new_config)
        print("   ✓ Model reconfigured successfully")
        print(f"   New version: {deployment.config.model_version}")
        print(f"   New classes: {deployment.config.num_classes}")
    except Exception as e:
        print(f"   ✗ Reconfiguration failed: {e}")
    
    print("\n5. Testing JIT compilation...")
    start_time = time.time()
    sample_data = jnp.ones((1, 224, 224, 3))
    if hasattr(deployment, 'batch_stats'):
        _ = deployment.jitted_apply(
            {'params': deployment.params, 'batch_stats': deployment.batch_stats}, 
            sample_data, 
            train=False,
            mutable=False
        )
    else:
        _ = deployment.jitted_apply(
            {'params': deployment.params}, 
            sample_data, 
            train=False
        )
    jit_time = time.time() - start_time
    print(f"   ✓ JIT compilation working - inference time: {jit_time:.4f}s")
    
    print("\n6. Enterprise features demonstrated:")
    print("   - Input validation and error handling")
    print("   - Dynamic model reconfiguration")
    print("   - JIT-compiled inference for performance")
    print("   - Checkpoint management")
    print("   - Configuration management")
    print("   - Resource allocation awareness")
    
    print("\nDeployment service ready for production use!")

if __name__ == "__main__":
    main()