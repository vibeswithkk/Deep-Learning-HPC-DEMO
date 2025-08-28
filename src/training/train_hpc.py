import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from typing import Any
import functools
import ml_collections
import orbax.checkpoint as orbax
import yaml

from src.models.flax_cnn import create_model, ModelConfig, train_step, get_dataset, save_checkpoint

def train_and_evaluate(config_path: str, workdir: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ModelConfig(
        num_classes=config_dict['model']['num_classes'],
        input_shape=tuple(config_dict['model']['input_shape']),
        backbone=config_dict['model']['name'],
        dtype=getattr(jnp, config_dict['model']['precision']) if 'precision' in config_dict['model'] else jnp.float32,
        normalization=config_dict.get('normalization', 'batchnorm'),
        activation=config_dict.get('activation', 'relu'),
        dropout_rate=config_dict['training'].get('dropout_rate', 0.0),
        stochastic_depth_rate=config_dict['training'].get('stochastic_depth_rate', 0.0)
    )
    
    train_ds = get_dataset(config, config_dict['data']['batch_size'], is_train=True)
    eval_ds = get_dataset(config, config_dict['data']['batch_size'], is_train=False)
    
    rng = jax.random.PRNGKey(config_dict.get('seed', 0))
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(init_rng, config, config_dict['training']['learning_rate'])
    dropout_rng = jax.random.split(rng, jax.local_device_count())[0]
    
    for epoch in range(config_dict['training']['epochs']):
        for batch in train_ds:
            batch = {
                'image': jnp.array(batch['image']),
                'label': jnp.array(batch['label'])
            }
            state, metrics = train_step(state, batch, dropout_rng)
        
        if epoch % config_dict['training'].get('save_every', 10) == 0:
            save_checkpoint(state, f"{workdir}/checkpoints")
    
    save_checkpoint(state, f"{workdir}/final_checkpoint")
    return state

def create_train_state(rng, config: ModelConfig, learning_rate: float = 0.001):
    model = create_model(config)
    variables = model.init(rng, jnp.ones([1, *config.input_shape]))
    params = variables['params']
    
    if config_dict['training']['optimizer'] == 'adam':
        tx = optax.adam(learning_rate=learning_rate)
    elif config_dict['training']['optimizer'] == 'adamw':
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=config_dict['training'].get('weight_decay', 0.01))
    elif config_dict['training']['optimizer'] == 'lion':
        from lion import Lion
        tx = Lion(learning_rate=learning_rate)
    elif config_dict['training']['optimizer'] == 'lamb':
        tx = optax.lamb(learning_rate=learning_rate, weight_decay=config_dict['training'].get('weight_decay', 0.01))
    else:
        tx = optax.sgd(learning_rate=learning_rate, momentum=config_dict['training'].get('momentum', 0.9))
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

if __name__ == "__main__":
    train_and_evaluate("config/train_config.yaml", "./results")