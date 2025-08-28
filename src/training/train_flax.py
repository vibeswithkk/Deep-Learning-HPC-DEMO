import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any, Callable, Dict, Tuple, Optional, Union
import functools
import time
import os
import numpy as np
from src.models.flax_cnn import create_model, ModelConfig
from src.training.callbacks import CallbackList, TrainingState
import orbax.checkpoint as orbax
from clu import metrics
import ml_collections
import wandb
import json
import torch
from torch.utils.tensorboard import SummaryWriter

def create_advanced_train_state(rng, config: ModelConfig, learning_rate: float):
    model = create_model(config)
    variables = model.init(rng, jnp.ones([1, *config.input_shape]))
    params = variables['params']
    
    # Create learning rate schedule
    if hasattr(config, 'lr_schedule') and config.lr_schedule == 'cosine':
        schedule_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=config.get('lr_decay_steps', 10000),
            alpha=config.get('lr_min_factor', 0.01)
        )
    elif hasattr(config, 'lr_schedule') and config.lr_schedule == 'warmup_cosine':
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=config.get('lr_warmup_steps', 1000),
            decay_steps=config.get('lr_decay_steps', 10000),
            end_value=learning_rate * config.get('lr_min_factor', 0.01)
        )
    else:
        schedule_fn = learning_rate
    
    # Create optimizer
    if hasattr(config, 'optimizer') and config.optimizer == 'lion':
        from src.optimizers.optax_utils import lion
        tx = lion(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.9),
            b2=config.get('optimizer_beta2', 0.99)
        )
    elif hasattr(config, 'optimizer') and config.optimizer == 'ranger':
        from src.optimizers.optax_utils import ranger
        tx = ranger(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.95),
            b2=config.get('optimizer_beta2', 0.999)
        )
    elif hasattr(config, 'optimizer') and config.optimizer == 'lamb':
        from src.optimizers.optax_utils import lamb
        tx = lamb(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.9),
            b2=config.get('optimizer_beta2', 0.999),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif hasattr(config, 'optimizer') and config.optimizer == 'sophia':
        from src.optimizers.optax_utils import sophia
        tx = sophia(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.9),
            b2=config.get('optimizer_beta2', 0.999),
            rho=config.get('optimizer_rho', 0.05),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif hasattr(config, 'optimizer') and config.optimizer == 'adan':
        from src.optimizers.optax_utils import adan
        tx = adan(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.98),
            b2=config.get('optimizer_beta2', 0.92),
            b3=config.get('optimizer_beta3', 0.99),
            weight_decay=config.get('weight_decay', 0.01),
            eps=config.get('optimizer_eps', 1e-8)
        )
    else:
        tx = optax.adamw(
            learning_rate=schedule_fn,
            b1=config.get('optimizer_beta1', 0.9),
            b2=config.get('optimizer_beta2', 0.999),
            eps=config.get('optimizer_eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
    
    # Add gradient clipping if specified
    if config.get('gradient_clipping', 0.0) > 0.0:
        tx = optax.chain(
            optax.clip_by_global_norm(config.gradient_clipping),
            tx
        )
    
    # Add EMA if specified
    if config.get('use_ema', False):
        tx = optax.chain(
            tx,
            optax.ema(decay=config.get('ema_decay', 0.999))
        )
    
    # Add Lookahead if specified
    if config.get('use_lookahead', False):
        tx = optax.chain(
            tx,
            optax.lookahead(
                tx,
                sync_period=config.get('lookahead_sync_period', 5),
                slow_step_size=config.get('lookahead_slow_step_size', 0.5)
            )
        )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, batch, dropout_rng, config):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    
    def loss_fn(params):
        variables = {'params': params}
        if config.get('use_batch_stats', False):
            variables['batch_stats'] = batch.get('batch_stats', {})
        
        logits = state.apply_fn(
            variables,
            batch['image'],
            train=True,
            rngs={'dropout': dropout_rng}
        )
        
        labels = batch['label']
        if labels.ndim == 1:
            labels = jax.nn.one_hot(labels, logits.shape[-1])
        
        # Apply label smoothing if specified
        if config.get('label_smoothing', 0.0) > 0.0:
            labels = optax.smooth_labels(labels, config.label_smoothing)
        
        # Calculate focal loss if specified
        if config.get('use_focal_loss', False):
            probs = jax.nn.softmax(logits)
            ce_loss = optax.softmax_cross_entropy(logits, labels)
            focal_weight = (1 - probs) ** config.get('focal_loss_gamma', 2.0)
            focal_loss = focal_weight * ce_loss
            loss = jnp.mean(focal_loss)
        else:
            loss = optax.softmax_cross_entropy(logits, labels).mean()
        
        # Add L1 regularization if specified
        if config.get('l1_regularization', 0.0) > 0.0:
            l1_penalty = config.l1_regularization * sum(
                jnp.sum(jnp.abs(param)) for param in jax.tree_leaves(params)
            )
            loss = loss + l1_penalty
        
        # Add L2 regularization if specified
        if config.get('l2_regularization', 0.0) > 0.0:
            l2_penalty = config.l2_regularization * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params)
            )
            loss = loss + l2_penalty
        
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    
    # Calculate additional metrics
    predictions = jnp.argmax(logits, -1)
    targets = jnp.argmax(labels, -1) if labels.ndim > 1 else labels
    
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(predictions == targets),
        'top5_accuracy': jnp.mean(jnp.top_k(logits, k=5).indices == targets[:, None]),
        'learning_rate': state.opt_state[-1].hyperparams['learning_rate'] if hasattr(state.opt_state[-1], 'hyperparams') else 0.0
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    return state, metrics, new_dropout_rng

@functools.partial(jax.pmap, axis_name='batch')
def eval_step(state, batch, config):
    variables = {'params': state.params}
    if config.get('use_batch_stats', False):
        variables['batch_stats'] = batch.get('batch_stats', {})
    
    logits = state.apply_fn(
        variables,
        batch['image'],
        train=False
    )
    
    labels = batch['label']
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, logits.shape[-1])
    
    predictions = jnp.argmax(logits, -1)
    targets = jnp.argmax(labels, -1)
    
    metrics = {
        'loss': optax.softmax_cross_entropy(logits, labels).mean(),
        'accuracy': jnp.mean(predictions == targets),
        'top5_accuracy': jnp.mean(jnp.top_k(logits, k=5).indices == targets[:, None])
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics

def get_advanced_dataset(config: ModelConfig, batch_size: int, is_train: bool):
    dataset_name = config.get('dataset_name', 'imagenet2012')
    dataset = tfds.load(dataset_name, split='train' if is_train else 'validation')
    
    def preprocess_data(data):
        image = tf.cast(data['image'], tf.float32) / 255.0
        image = tf.image.resize(image, config.input_shape[:2])
        
        if is_train:
            # Advanced augmentations
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)
            
            # Random cropping
            if config.get('random_crop', False):
                image = tf.image.random_crop(image, [*config.input_shape[:2], config.input_shape[2]])
                image = tf.image.resize(image, config.input_shape[:2])
            
            # Random rotation
            if config.get('random_rotation', 0.0) > 0.0:
                angle = tf.random.uniform([], -config.random_rotation, config.random_rotation)
                image = tf.contrib.image.rotate(image, angle)
        
        label = tf.one_hot(data['label'], config.num_classes)
        return {'image': image, 'label': label}
    
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle if training
    if is_train and config.get('shuffle_buffer_size', 10000) > 0:
        dataset = dataset.shuffle(config.shuffle_buffer_size)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Apply mixup augmentation if specified
    if is_train and config.get('augmentation_mixup_alpha', 0.0) > 0.0:
        def mixup_augmentation(batch):
            alpha = config.augmentation_mixup_alpha
            batch_size = tf.shape(batch['image'])[0]
            lam = tf.random.gamma([], alpha, alpha)
            indices = tf.random.shuffle(tf.range(batch_size))
            
            mixed_images = lam * batch['image'] + (1 - lam) * tf.gather(batch['image'], indices)
            mixed_labels = lam * batch['label'] + (1 - lam) * tf.gather(batch['label'], indices)
            
            batch['image'] = mixed_images
            batch['label'] = mixed_labels
            return batch
        
        dataset = dataset.map(mixup_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply cutmix augmentation if specified
    if is_train and config.get('augmentation_cutmix_alpha', 0.0) > 0.0:
        def cutmix_augmentation(batch):
            alpha = config.augmentation_cutmix_alpha
            batch_size = tf.shape(batch['image'])[0]
            lam = tf.random.gamma([], alpha, alpha)
            indices = tf.random.shuffle(tf.range(batch_size))
            
            # Generate random box
            img_height, img_width = config.input_shape[0], config.input_shape[1]
            cut_rat = tf.sqrt(1. - lam)
            cut_w = tf.cast(img_width * cut_rat, tf.int32)
            cut_h = tf.cast(img_height * cut_rat, tf.int32)
            cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
            cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)
            
            bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width)
            bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height)
            bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_width)
            bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_height)
            
            # Apply cutmix
            image1 = batch['image']
            image2 = tf.gather(batch['image'], indices)
            
            mask = tf.ones((img_height, img_width, 1))
            mask = tf.pad(mask, [[bby1, img_height - bby2], [bbx1, img_width - bbx2], [0, 0]], constant_values=0)
            mask = tf.expand_dims(mask, 0)
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
            
            mixed_images = image1 * (1 - mask) + image2 * mask
            mixed_labels = lam * batch['label'] + (1 - lam) * tf.gather(batch['label'], indices)
            
            batch['image'] = mixed_images
            batch['label'] = mixed_labels
            return batch
        
        dataset = dataset.map(cutmix_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch for performance
    if config.get('prefetch_buffer_size', tf.data.AUTOTUNE) != 0:
        dataset = dataset.prefetch(config.prefetch_buffer_size)
    
    return dataset

def save_checkpoint(state, path: str, step: int, keep: int = 3):
    checkpoint_manager = orbax.CheckpointManager(
        path,
        orbax.PyTreeCheckpointer(),
        options=orbax.CheckpointManagerOptions(
            max_to_keep=keep,
            create=True
        )
    )
    checkpoint_manager.save(step, state)
    return checkpoint_manager

def load_checkpoint(path: str, state=None):
    checkpoint_manager = orbax.CheckpointManager(
        path,
        orbax.PyTreeCheckpointer()
    )
    if checkpoint_manager.latest_step() is not None:
        return checkpoint_manager.restore(checkpoint_manager.latest_step(), items=state)
    return None

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int):
    def schedule(step):
        cos_inner = (step % total_steps) / total_steps
        return min_lr + (base_lr - min_lr) * (1 + jnp.cos(jnp.pi * cos_inner)) / 2
    return schedule

def warmup_cosine_decay_schedule(warmup_steps: int, base_lr: float, min_lr: float, total_steps: int):
    warmup_fn = optax.linear_schedule(0.0, base_lr, warmup_steps)
    cosine_fn = cosine_annealing_schedule(base_lr, min_lr, total_steps - warmup_steps)
    
    def schedule(step):
        return jnp.where(step < warmup_steps, 
                        warmup_fn(step), 
                        cosine_fn(step - warmup_steps))
    return schedule

def train_and_evaluate(config_path: str, workdir: str):
    import yaml
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
        stochastic_depth_rate=config_dict['training'].get('stochastic_depth_rate', 0.0),
        optimizer=config_dict['training'].get('optimizer', 'adamw'),
        optimizer_beta1=config_dict['training'].get('optimizer_beta1', 0.9),
        optimizer_beta2=config_dict['training'].get('optimizer_beta2', 0.999),
        optimizer_eps=config_dict['training'].get('optimizer_eps', 1e-8),
        weight_decay=config_dict['training'].get('weight_decay', 0.01),
        gradient_clipping=config_dict['training'].get('gradient_clipping', 0.0),
        use_ema=config_dict['training'].get('use_ema', False),
        ema_decay=config_dict['training'].get('ema_decay', 0.9999),
        label_smoothing=config_dict['training'].get('label_smoothing', 0.0),
        l1_regularization=config_dict['training'].get('l1_regularization', 0.0),
        l2_regularization=config_dict['training'].get('l2_regularization', 0.0),
        use_focal_loss=config_dict['training'].get('use_focal_loss', False),
        focal_loss_gamma=config_dict['training'].get('focal_loss_gamma', 2.0),
        augmentation_mixup_alpha=config_dict['data'].get('augmentation_mixup_alpha', 0.0),
        augmentation_cutmix_alpha=config_dict['data'].get('augmentation_cutmix_alpha', 0.0),
        shuffle_buffer_size=config_dict['data'].get('shuffle_buffer_size', 10000),
        prefetch_buffer_size=config_dict['data'].get('prefetch_buffer_size', tf.data.AUTOTUNE),
        dataset_name=config_dict['data'].get('dataset_name', 'imagenet2012'),
        random_crop=config_dict['data'].get('random_crop', False),
        random_rotation=config_dict['data'].get('random_rotation', 0.0),
        lr_schedule=config_dict['training'].get('lr_schedule', 'constant'),
        lr_decay_steps=config_dict['training'].get('lr_decay_steps', 10000),
        lr_min_factor=config_dict['training'].get('lr_min_factor', 0.01),
        lr_warmup_steps=config_dict['training'].get('lr_warmup_steps', 1000),
        use_lookahead=config_dict['training'].get('use_lookahead', False),
        lookahead_sync_period=config_dict['training'].get('lookahead_sync_period', 5),
        lookahead_slow_step_size=config_dict['training'].get('lookahead_slow_step_size', 0.5)
    )
    
    # Initialize logging
    if config_dict.get('logging', {}).get('use_wandb', False):
        wandb.init(
            project=config_dict['logging'].get('project_name', 'deep-learning-hpc'),
            config=config_dict
        )
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(workdir, 'logs'))
    
    batch_size = config_dict['data']['batch_size'] * jax.local_device_count()
    train_ds = get_advanced_dataset(config, batch_size, is_train=True)
    eval_ds = get_advanced_dataset(config, batch_size, is_train=False)
    
    rng = jax.random.PRNGKey(config_dict.get('seed', 0))
    rng, init_rng = jax.random.split(rng)
    
    state = create_advanced_train_state(init_rng, config, config_dict['training']['learning_rate'])
    state = flax.jax_utils.replicate(state)
    
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    
    callbacks = CallbackList()
    
    training_state = TrainingState(
        epoch=0,
        step=0,
        loss=0.0,
        metrics={},
        learning_rate=config_dict['training']['learning_rate']
    )
    
    callbacks.on_train_begin({'state': state, 'config': config_dict})
    
    for epoch in range(config_dict['training']['epochs']):
        training_state.epoch = epoch
        callbacks.on_epoch_begin(epoch, {'state': state, 'training_state': training_state})
        
        epoch_start_time = time.time()
        train_metrics = []
        
        for batch_idx, batch in enumerate(train_ds):
            training_state.step = batch_idx
            callbacks.on_batch_begin(batch_idx, {'state': state, 'training_state': training_state})
            
            batch = {
                'image': jnp.array(batch['image']),
                'label': jnp.array(batch['label'])
            }
            
            state, metrics, dropout_rngs = train_step(state, batch, dropout_rngs, config_dict)
            train_metrics.append(metrics)
            
            training_state.loss = float(flax.jax_utils.unreplicate(metrics)['loss'])
            training_state.metrics = {k: float(v) for k, v in flax.jax_utils.unreplicate(metrics).items()}
            
            # Log metrics to TensorBoard
            if batch_idx % config_dict.get('logging', {}).get('log_interval', 100) == 0:
                for key, value in training_state.metrics.items():
                    writer.add_scalar(f'train/{key}', value, epoch * len(train_ds) + batch_idx)
            
            callbacks.on_batch_end(batch_idx, {
                'state': flax.jax_utils.unreplicate(state),
                'training_state': training_state,
                'learning_rate': training_state.learning_rate
            })
        
        epoch_time = time.time() - epoch_start_time
        training_state.metrics['epoch_time'] = epoch_time
        training_state.throughput = len(train_metrics) / epoch_time
        
        train_metrics = flax.jax_utils.unreplicate(train_metrics[-1])
        training_state.loss = float(train_metrics['loss'])
        training_state.metrics = {k: float(v) for k, v in train_metrics.items()}
        
        # Log epoch metrics to TensorBoard
        for key, value in training_state.metrics.items():
            writer.add_scalar(f'epoch/train_{key}', value, epoch)
        
        eval_metrics = []
        for batch in eval_ds:
            batch = {
                'image': jnp.array(batch['image']),
                'label': jnp.array(batch['label'])
            }
            metrics = eval_step(state, batch, config_dict)
            eval_metrics.append(metrics)
        
        eval_metrics = flax.jax_utils.unreplicate(eval_metrics[-1])
        training_state.metrics.update({f"val_{k}": float(v) for k, v in eval_metrics.items()})
        
        # Log validation metrics to TensorBoard
        for key, value in eval_metrics.items():
            writer.add_scalar(f'epoch/val_{key}', float(value), epoch)
        
        callbacks.on_epoch_end(epoch, {
            'state': flax.jax_utils.unreplicate(state),
            'training_state': training_state,
            'learning_rate': training_state.learning_rate
        })
        
        # Log to wandb if enabled
        if config_dict.get('logging', {}).get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': training_state.loss,
                'train_accuracy': training_state.metrics.get('accuracy', 0.0),
                'val_loss': training_state.metrics.get('val_loss', 0.0),
                'val_accuracy': training_state.metrics.get('val_accuracy', 0.0),
                'learning_rate': training_state.learning_rate
            })
    
    callbacks.on_train_end({'state': flax.jax_utils.unreplicate(state)})
    
    # Close TensorBoard writer
    writer.close()
    
    # Finish wandb if enabled
    if config_dict.get('logging', {}).get('use_wandb', False):
        wandb.finish()
    
    return flax.jax_utils.unreplicate(state)

if __name__ == "__main__":
    train_and_evaluate("config/train_config.yaml", "./results")