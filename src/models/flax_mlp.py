import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import numpy as np
from dataclasses import dataclass, field
import optax
from jax import lax
from jax.scipy.special import logsumexp
import math

@dataclass
class MLPConfig:
    input_dim: int = 784
    hidden_dims: Sequence[int] = field(default_factory=lambda: [512, 256, 128])
    output_dim: int = 10
    activation: str = "gelu"
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_layer_norm: bool = True
    use_residual_connections: bool = True
    attention_heads: int = 8
    attention_dim: int = 64
    use_attention: bool = False
    moe_layers: bool = False
    num_experts: int = 8
    top_k_experts: int = 2
    use_reversible: bool = False
    use_gradient_checkpointing: bool = True
    use_adaptive_normalization: bool = True
    adaptive_norm_momentum: float = 0.99
    use_stochastic_depth: float = 0.0
    use_squeeze_excite: bool = False
    se_ratio: float = 0.25
    use_highway_connections: bool = False
    use_fourier_features: bool = False
    fourier_frequency: int = 16
    use_positional_encoding: bool = False
    max_sequence_length: int = 1024
    use_conditional_scaling: bool = False
    conditional_scaling_dim: int = 128
    use_router_bias: bool = True
    router_jitter_noise: float = 0.0
    use_expert_parallelism: bool = False
    expert_parallel_size: int = 1
    use_token_dropout: bool = False
    token_dropout_rate: float = 0.1
    use_layer_scale: bool = False
    layer_scale_init_value: float = 1e-6
    use_dynamic_layer_scaling: bool = False
    dynamic_layer_scaling_min: float = 1e-7
    dynamic_layer_scaling_max: float = 1e-5
    use_adaptive_dropout: bool = False
    adaptive_dropout_min: float = 0.05
    adaptive_dropout_max: float = 0.3
    use_stochastic_depth_per_layer: bool = True
    stochastic_depth_per_layer_prob: float = 0.05
    use_flash_attention: bool = False
    use_causal_mask: bool = False
    attention_dropout_rate: float = 0.0
    use_relative_position_encoding: bool = False
    relative_position_max_distance: int = 128
    use_rotary_position_embedding: bool = False
    use_alibi_bias: bool = False
    use_attention_bias: bool = False
    use_bias_dropout_fusion: bool = False
    use_fused_ops: bool = True
    use_tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    use_sequence_parallel: bool = False
    sequence_parallel_size: int = 1
    use_pipeline_parallel: bool = False
    pipeline_parallel_size: int = 1
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"
    use_gradient_scaling: bool = False
    gradient_scaling_factor: float = 1.0
    use_gradient_noise: bool = False
    gradient_noise_eta: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clipping_norm: float = 1.0
    use_weight_decay: bool = True
    weight_decay_rate: float = 0.01
    use_lion_optimizer: bool = False
    use_adamw_optimizer: bool = True
    use_sophia_optimizer: bool = False
    use_adan_optimizer: bool = False
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_beta3: float = 0.999
    optimizer_epsilon: float = 1e-8
    use_lookahead_optimizer: bool = False
    lookahead_steps: int = 5
    lookahead_alpha: float = 0.5
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_warmup_scheduler: bool = True
    warmup_steps: int = 1000
    use_cosine_scheduler: bool = True
    cosine_min_lr_ratio: float = 0.01
    use_linear_scheduler: bool = False
    use_constant_scheduler: bool = False
    use_polynomial_scheduler: bool = False
    polynomial_power: float = 1.0
    use_inverse_sqrt_scheduler: bool = False
    inverse_sqrt_warmup_steps: int = 4000
    use_plateau_scheduler: bool = False
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-7
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1
    use_focal_loss: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    use_auto_augment: bool = False
    auto_augment_policy: str = "v0"
    use_rand_augment: bool = False
    rand_augment_num_layers: int = 2
    rand_augment_magnitude: float = 9.0
    use_cutout: bool = False
    cutout_prob: float = 0.5
    cutout_size: float = 0.2
    use_random_erasing: bool = False
    random_erasing_prob: float = 0.25
    random_erasing_mode: str = "pixel"
    use_color_jitter: bool = False
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    use_random_grayscale: bool = False
    random_grayscale_prob: float = 0.2
    use_gaussian_noise: bool = False
    gaussian_noise_std: float = 0.1
    use_solarization: bool = False
    solarization_threshold: float = 0.5
    use_posterization: bool = False
    posterization_bits: int = 4
    use_equalization: bool = False
    use_sharpening: bool = False
    sharpening_alpha: float = 0.5
    use_blurring: bool = False
    blurring_sigma: float = 1.0
    use_elastic_transform: bool = False
    elastic_transform_alpha: float = 50.0
    elastic_transform_sigma: float = 5.0
    use_optical_distortion: bool = False
    optical_distortion_distort_limit: float = 0.05
    optical_distortion_shift_limit: float = 0.05
    use_grid_distortion: bool = False
    grid_distortion_num_steps: int = 5
    grid_distortion_distort_limit: float = 0.3
    use_coarse_dropout: bool = False
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height: int = 32
    coarse_dropout_max_width: int = 32
    use_channel_shuffle: bool = False
    channel_shuffle_groups: int = 1
    use_channel_dropout: bool = False
    channel_dropout_prob: float = 0.01
    use_channel_dropout_mode: str = "uniform"
    use_spatial_dropout: bool = False
    spatial_dropout_prob: float = 0.1
    use_spatial_dropout_mode: str = "2d"
    use_batch_dropout: bool = False
    batch_dropout_prob: float = 0.1
    use_feature_dropout: bool = False
    feature_dropout_prob: float = 0.1
    use_path_dropout: bool = False
    path_dropout_prob: float = 0.1
    use_layer_dropout: bool = False
    layer_dropout_prob: float = 0.1
    use_time_dropout: bool = False
    time_dropout_prob: float = 0.1
    use_frequency_dropout: bool = False
    frequency_dropout_prob: float = 0.1
    use_spectral_dropout: bool = False
    spectral_dropout_prob: float = 0.1
    use_instance_dropout: bool = False
    instance_dropout_prob: float = 0.1
    use_patch_dropout: bool = False
    patch_dropout_prob: float = 0.1
    use_block_dropout: bool = False
    block_dropout_prob: float = 0.1
    use_region_dropout: bool = False
    region_dropout_prob: float = 0.1
    use_object_dropout: bool = False
    object_dropout_prob: float = 0.1
    use_context_dropout: bool = False
    context_dropout_prob: float = 0.1
    use_attention_dropout: bool = False
    attention_dropout_prob: float = 0.1
    use_query_dropout: bool = False
    query_dropout_prob: float = 0.1
    use_key_dropout: bool = False
    key_dropout_prob: float = 0.1
    use_value_dropout: bool = False
    value_dropout_prob: float = 0.1
    use_output_dropout: bool = False
    output_dropout_prob: float = 0.1
    use_feedforward_dropout: bool = False
    feedforward_dropout_prob: float = 0.1
    use_residual_dropout: bool = False
    residual_dropout_prob: float = 0.1
    use_hidden_dropout: bool = False
    hidden_dropout_prob: float = 0.1
    use_embedding_dropout: bool = False
    embedding_dropout_prob: float = 0.1
    use_position_dropout: bool = False
    position_dropout_prob: float = 0.1
    use_token_dropout_mode: str = "uniform"
    use_token_dropout_schedule: str = "constant"
    token_dropout_schedule_params: Dict[str, Any] = field(default_factory=dict)
    use_layer_adaptive_dropout: bool = False
    layer_adaptive_dropout_schedule: str = "linear"
    layer_adaptive_dropout_params: Dict[str, Any] = field(default_factory=dict)
    use_temporal_dropout: bool = False
    temporal_dropout_prob: float = 0.1
    use_spatial_temporal_dropout: bool = False
    spatial_temporal_dropout_prob: float = 0.1
    use_channel_temporal_dropout: bool = False
    channel_temporal_dropout_prob: float = 0.1
    use_feature_temporal_dropout: bool = False
    feature_temporal_dropout_prob: float = 0.1
    use_attention_temporal_dropout: bool = False
    attention_temporal_dropout_prob: float = 0.1
    use_output_temporal_dropout: bool = False
    output_temporal_dropout_prob: float = 0.1
    use_feedforward_temporal_dropout: bool = False
    feedforward_temporal_dropout_prob: float = 0.1
    use_residual_temporal_dropout: bool = False
    residual_temporal_dropout_prob: float = 0.1
    use_hidden_temporal_dropout: bool = False
    hidden_temporal_dropout_prob: float = 0.1
    use_embedding_temporal_dropout: bool = False
    embedding_temporal_dropout_prob: float = 0.1
    use_position_temporal_dropout: bool = False
    position_temporal_dropout_prob: float = 0.1
    use_token_temporal_dropout: bool = False
    token_temporal_dropout_prob: float = 0.1
    use_layer_temporal_dropout: bool = False
    layer_temporal_dropout_prob: float = 0.1
    use_path_temporal_dropout: bool = False
    path_temporal_dropout_prob: float = 0.1
    use_block_temporal_dropout: bool = False
    block_temporal_dropout_prob: float = 0.1
    use_region_temporal_dropout: bool = False
    region_temporal_dropout_prob: float = 0.1
    use_object_temporal_dropout: bool = False
    object_temporal_dropout_prob: float = 0.1
    use_context_temporal_dropout: bool = False
    context_temporal_dropout_prob: float = 0.1
    use_query_temporal_dropout: bool = False
    query_temporal_dropout_prob: float = 0.1
    use_key_temporal_dropout: bool = False
    key_temporal_dropout_prob: float = 0.1
    use_value_temporal_dropout: bool = False
    value_temporal_dropout_prob: float = 0.1
    use_adversarial_training: bool = False
    adversarial_epsilon: float = 0.03
    adversarial_alpha: float = 0.01
    adversarial_num_steps: int = 1
    adversarial_clip_min: float = -1.0
    adversarial_clip_max: float = 1.0
    use_virtual_adversarial_training: bool = False
    vat_epsilon: float = 1e-6
    vat_xi: float = 1e-6
    vat_power_iteration: int = 1
    use_consistency_regularization: bool = False
    consistency_alpha: float = 0.1
    consistency_temperature: float = 0.5
    use_mean_teacher: bool = False
    mean_teacher_alpha: float = 0.999
    use_dual_batchnorm: bool = False
    dual_batchnorm_alpha: float = 0.1
    use_shake_shake: bool = False
    shake_shake_alpha: float = 0.1
    use_shake_drop: bool = False
    shake_drop_alpha: float = 0.1
    shake_drop_beta: float = 0.1
    use_stochastic_depth_schedule: str = "linear"
    stochastic_depth_schedule_params: Dict[str, Any] = field(default_factory=dict)
    use_layer_scale_schedule: str = "constant"
    layer_scale_schedule_params: Dict[str, Any] = field(default_factory=dict)
    use_adaptive_layer_scaling: bool = False
    adaptive_layer_scaling_min: float = 1e-7
    adaptive_layer_scaling_max: float = 1e-5
    use_dynamic_layer_scaling_schedule: str = "linear"
    dynamic_layer_scaling_schedule_params: Dict[str, Any] = field(default_factory=dict)
    use_adaptive_dropout_schedule: str = "linear"
    adaptive_dropout_schedule_params: Dict[str, Any] = field(default_factory=dict)
    use_token_dropout_adaptive: bool = False
    token_dropout_adaptive_threshold: float = 0.1
    use_layer_dropout_adaptive: bool = False
    layer_dropout_adaptive_threshold: float = 0.1
    use_path_dropout_adaptive: bool = False
    path_dropout_adaptive_threshold: float = 0.1
    use_block_dropout_adaptive: bool = False
    block_dropout_adaptive_threshold: float = 0.1
    use_region_dropout_adaptive: bool = False
    region_dropout_adaptive_threshold: float = 0.1
    use_object_dropout_adaptive: bool = False
    object_dropout_adaptive_threshold: float = 0.1
    use_context_dropout_adaptive: bool = False
    context_dropout_adaptive_threshold: float = 0.1
    use_attention_dropout_adaptive: bool = False
    attention_dropout_adaptive_threshold: float = 0.1
    use_query_dropout_adaptive: bool = False
    query_dropout_adaptive_threshold: float = 0.1
    use_key_dropout_adaptive: bool = False
    key_dropout_adaptive_threshold: float = 0.1
    use_value_dropout_adaptive: bool = False
    value_dropout_adaptive_threshold: float = 0.1
    use_output_dropout_adaptive: bool = False
    output_dropout_adaptive_threshold: float = 0.1
    use_feedforward_dropout_adaptive: bool = False
    feedforward_dropout_adaptive_threshold: float = 0.1
    use_residual_dropout_adaptive: bool = False
    residual_dropout_adaptive_threshold: float = 0.1
    use_hidden_dropout_adaptive: bool = False
    hidden_dropout_adaptive_threshold: float = 0.1
    use_embedding_dropout_adaptive: bool = False
    embedding_dropout_adaptive_threshold: float = 0.1
    use_position_dropout_adaptive: bool = False
    position_dropout_adaptive_threshold: float = 0.1
    use_temporal_dropout_adaptive: bool = False
    temporal_dropout_adaptive_threshold: float = 0.1
    use_spatial_temporal_dropout_adaptive: bool = False
    spatial_temporal_dropout_adaptive_threshold: float = 0.1
    use_channel_temporal_dropout_adaptive: bool = False
    channel_temporal_dropout_adaptive_threshold: float = 0.1
    use_feature_temporal_dropout_adaptive: bool = False
    feature_temporal_dropout_adaptive_threshold: float = 0.1
    use_attention_temporal_dropout_adaptive: bool = False
    attention_temporal_dropout_adaptive_threshold: float = 0.1
    use_output_temporal_dropout_adaptive: bool = False
    output_temporal_dropout_adaptive_threshold: float = 0.1
    use_feedforward_temporal_dropout_adaptive: bool = False
    feedforward_temporal_dropout_adaptive_threshold: float = 0.1
    use_residual_temporal_dropout_adaptive: bool = False
    residual_temporal_dropout_adaptive_threshold: float = 0.1
    use_hidden_temporal_dropout_adaptive: bool = False
    hidden_temporal_dropout_adaptive_threshold: float = 0.1
    use_embedding_temporal_dropout_adaptive: bool = False
    embedding_temporal_dropout_adaptive_threshold: float = 0.1
    use_position_temporal_dropout_adaptive: bool = False
    position_temporal_dropout_adaptive_threshold: float = 0.1
    use_token_temporal_dropout_adaptive: bool = False
    token_temporal_dropout_adaptive_threshold: float = 0.1
    use_layer_temporal_dropout_adaptive: bool = False
    layer_temporal_dropout_adaptive_threshold: float = 0.1
    use_path_temporal_dropout_adaptive: bool = False
    path_temporal_dropout_adaptive_threshold: float = 0.1
    use_block_temporal_dropout_adaptive: bool = False
    block_temporal_dropout_adaptive_threshold: float = 0.1
    use_region_temporal_dropout_adaptive: bool = False
    region_temporal_dropout_adaptive_threshold: float = 0.1
    use_object_temporal_dropout_adaptive: bool = False
    object_temporal_dropout_adaptive_threshold: float = 0.1
    use_context_temporal_dropout_adaptive: bool = False
    context_temporal_dropout_adaptive_threshold: float = 0.1
    use_query_temporal_dropout_adaptive: bool = False
    query_temporal_dropout_adaptive_threshold: float = 0.1
    use_key_temporal_dropout_adaptive: bool = False
    key_temporal_dropout_adaptive_threshold: float = 0.1
    use_value_temporal_dropout_adaptive: bool = False
    value_temporal_dropout_adaptive_threshold: float = 0.1
    use_ema_temporal: bool = False
    ema_temporal_decay: float = 0.999
    use_ema_spatial: bool = False
    ema_spatial_decay: float = 0.999
    use_ema_channel: bool = False
    ema_channel_decay: float = 0.999
    use_ema_feature: bool = False
    ema_feature_decay: float = 0.999
    use_ema_attention: bool = False
    ema_attention_decay: float = 0.999
    use_ema_query: bool = False
    ema_query_decay: float = 0.999
    use_ema_key: bool = False
    ema_key_decay: float = 0.999
    use_ema_value: bool = False
    ema_value_decay: float = 0.999
    use_ema_output: bool = False
    ema_output_decay: float = 0.999
    use_ema_feedforward: bool = False
    ema_feedforward_decay: float = 0.999
    use_ema_residual: bool = False
    ema_residual_decay: float = 0.999
    use_ema_hidden: bool = False
    ema_hidden_decay: float = 0.999
    use_ema_embedding: bool = False
    ema_embedding_decay: float = 0.999
    use_ema_position: bool = False
    ema_position_decay: float = 0.999
    use_ema_token: bool = False
    ema_token_decay: float = 0.999
    use_ema_layer: bool = False
    ema_layer_decay: float = 0.999
    use_ema_path: bool = False
    ema_path_decay: float = 0.999
    use_ema_block: bool = False
    ema_block_decay: float = 0.999
    use_ema_region: bool = False
    ema_region_decay: float = 0.999
    use_ema_object: bool = False
    ema_object_decay: float = 0.999
    use_ema_context: bool = False
    ema_context_decay: float = 0.999
    use_ema_temporal_spatial: bool = False
    ema_temporal_spatial_decay: float = 0.999
    use_ema_temporal_channel: bool = False
    ema_temporal_channel_decay: float = 0.999
    use_ema_temporal_feature: bool = False
    ema_temporal_feature_decay: float = 0.999
    use_ema_temporal_attention: bool = False
    ema_temporal_attention_decay: float = 0.999
    use_ema_temporal_query: bool = False
    ema_temporal_query_decay: float = 0.999
    use_ema_temporal_key: bool = False
    ema_temporal_key_decay: float = 0.999
    use_ema_temporal_value: bool = False
    ema_temporal_value_decay: float = 0.999
    use_ema_temporal_output: bool = False
    ema_temporal_output_decay: float = 0.999
    use_ema_temporal_feedforward: bool = False
    ema_temporal_feedforward_decay: float = 0.999
    use_ema_temporal_residual: bool = False
    ema_temporal_residual_decay: float = 0.999
    use_ema_temporal_hidden: bool = False
    ema_temporal_hidden_decay: float = 0.999
    use_ema_temporal_embedding: bool = False
    ema_temporal_embedding_decay: float = 0.999
    use_ema_temporal_position: bool = False
    ema_temporal_position_decay: float = 0.999
    use_ema_temporal_token: bool = False
    ema_temporal_token_decay: float = 0.999
    use_ema_temporal_layer: bool = False
    ema_temporal_layer_decay: float = 0.999
    use_ema_temporal_path: bool = False
    ema_temporal_path_decay: float = 0.999
    use_ema_temporal_block: bool = False
    ema_temporal_block_decay: float = 0.999
    use_ema_temporal_region: bool = False
    ema_temporal_region_decay: float = 0.999
    use_ema_temporal_object: bool = False
    ema_temporal_object_decay: float = 0.999
    use_ema_temporal_context: bool = False
    ema_temporal_context_decay: float = 0.999

class FourierFeatures(nn.Module):
    features: int
    frequency: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        positions = jnp.arange(x.shape[1], dtype=self.dtype)
        frequencies = jnp.linspace(0, self.frequency, self.features // 2)
        angles = positions[:, None] * frequencies[None, :]
        fourier_features = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return jnp.broadcast_to(fourier_features[None, :, :], (batch_size, x.shape[1], self.features))

class PositionalEncoding(nn.Module):
    max_len: int
    features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        position = jnp.arange(self.max_len, dtype=self.dtype)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.features, 2, dtype=self.dtype) * 
                          -(jnp.log(10000.0) / self.features))
        pe = jnp.zeros((self.max_len, self.features), dtype=self.dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[:x.shape[1], :]

class AdaptiveLayerNorm(nn.Module):
    momentum: float = 0.99
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = True):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + 1e-5)
        
        if training:
            adaptive_scale = self.param('adaptive_scale', nn.initializers.ones, (1,), self.dtype)
            adaptive_bias = self.param('adaptive_bias', nn.initializers.zeros, (1,), self.dtype)
            return normalized * adaptive_scale + adaptive_bias
        else:
            return normalized

class SqueezeExcite(nn.Module):
    ratio: float = 0.25
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST

    @nn.compact
    def __call__(self, x, deterministic=True):
        se_features = max(1, int(x.shape[-1] * self.ratio))
        squeeze = jnp.mean(x, axis=(1, 2), keepdims=True) if len(x.shape) == 4 else jnp.mean(x, axis=1, keepdims=True)
        excitation = nn.Dense(se_features, dtype=self.dtype, precision=self.precision)(squeeze)
        excitation = nn.relu(excitation)
        excitation = nn.Dense(x.shape[-1], dtype=self.dtype, precision=self.precision)(excitation)
        excitation = nn.sigmoid(excitation)
        return x * excitation

class ConditionalScaling(nn.Module):
    condition_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST

    @nn.compact
    def __call__(self, x, condition, deterministic=True):
        scale = nn.Dense(x.shape[-1], dtype=self.dtype, precision=self.precision)(condition)
        scale = nn.sigmoid(scale)
        bias = nn.Dense(x.shape[-1], dtype=self.dtype, precision=self.precision)(condition)
        return x * scale + bias

class RelativePositionEncoding(nn.Module):
    max_distance: int = 128
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, seq_len):
        range_vec = jnp.arange(seq_len)
        range_mat = range_vec[None, :] - range_vec[:, None]
        range_mat_clipped = jnp.clip(range_mat, -self.max_distance, self.max_distance)
        return range_mat_clipped + self.max_distance

class RotaryPositionEmbedding(nn.Module):
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[-2]
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))
        sinusoid_inp = jnp.einsum('i,j->ij', jnp.arange(seq_len, dtype=self.dtype), inv_freq)
        sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
        sin, cos = jnp.repeat(sin[..., None], 2, axis=-1), jnp.repeat(cos[..., None], 2, axis=-1)
        sin, cos = jnp.reshape(sin, (seq_len, -1)), jnp.reshape(cos, (seq_len, -1))
        return sin, cos

class AlibiBias(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, seq_len):
        slopes = jnp.array([2**(-8 * (h + 1) / self.num_heads) for h in range(self.num_heads)], dtype=self.dtype)
        positions = jnp.arange(seq_len, dtype=self.dtype)
        relative_positions = positions[None, :] - positions[:, None]
        relative_positions = jnp.abs(relative_positions)
        alibi_bias = slopes[:, None, None] * (-relative_positions[None, :, :])
        return alibi_bias

class FlashAttention(nn.Module):
    num_heads: int
    head_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    dropout_rate: float = 0.0
    use_causal_mask: bool = False
    use_alibi_bias: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):
        seq_len = x.shape[-2]
        d_model = x.shape[-1]
        
        qkv = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,
            dtype=self.dtype,
            precision=self.precision
        )(x)
        
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Apply rotary position embedding if enabled
        if self.has_variable('params', 'rotary_emb'):
            rotary_emb = RotaryPositionEmbedding(self.head_dim, dtype=self.dtype)
            sin, cos = rotary_emb(seq_len)
            q = self.apply_rotary_pos_emb(q, sin, cos)
            k = self.apply_rotary_pos_emb(k, sin, cos)
        
        # Apply ALiBi bias if enabled
        if self.use_alibi_bias:
            alibi_bias = AlibiBias(self.num_heads, dtype=self.dtype)(seq_len)
        else:
            alibi_bias = None
        
        # Flash attention implementation
        block_size = min(64, seq_len)
        output = jnp.zeros_like(q)
        
        for i in range(0, seq_len, block_size):
            q_block = q[..., i:i+block_size, :]
            k_block = k[..., :i+block_size, :]
            v_block = v[..., :i+block_size, :]
            
            block_scores = jnp.einsum('...hqd,...hkd->...hqk', q_block, k_block)
            block_scores = block_scores / jnp.sqrt(self.head_dim)
            
            if self.use_causal_mask:
                mask = jnp.tril(jnp.ones((block_size, i+block_size), dtype=self.dtype))
                block_scores = jnp.where(mask == 0, -1e9, block_scores)
            
            if alibi_bias is not None:
                block_scores = block_scores + alibi_bias[:, i:i+block_size, :i+block_size]
            
            if self.dropout_rate > 0.0 and not deterministic:
                dropout_rng = self.make_rng('dropout')
                block_scores = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(block_scores, rng=dropout_rng)
            
            block_weights = jax.nn.softmax(block_scores, axis=-1)
            block_output = jnp.einsum('...hqk,...hvd->...hqd', block_weights, v_block)
            output = output.at[..., i:i+block_size, :].set(block_output)
        
        return output

    def apply_rotary_pos_emb(self, x, sin, cos):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

class AttentionLayer(nn.Module):
    num_heads: int
    head_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_causal_mask: bool = False
    use_flash_attention: bool = False
    dropout_rate: float = 0.0
    use_relative_position_encoding: bool = False
    relative_position_max_distance: int = 128
    use_rotary_position_embedding: bool = False
    use_alibi_bias: bool = False
    use_attention_bias: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):
        seq_len = x.shape[-2]
        d_model = x.shape[-1]
        
        qkv = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,
            dtype=self.dtype,
            precision=self.precision
        )(x)
        
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Apply rotary position embedding if enabled
        if self.use_rotary_position_embedding:
            rotary_emb = RotaryPositionEmbedding(self.head_dim, dtype=self.dtype)
            sin, cos = rotary_emb(seq_len)
            q = self.apply_rotary_pos_emb(q, sin, cos)
            k = self.apply_rotary_pos_emb(k, sin, cos)
        
        if self.use_flash_attention:
            attention_output = FlashAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                precision=self.precision,
                dropout_rate=self.dropout_rate,
                use_causal_mask=self.use_causal_mask,
                use_alibi_bias=self.use_alibi_bias
            )(x, deterministic=deterministic)
        else:
            attention_scores = jnp.einsum('...hqd,...hkd->...hqk', q, k)
            attention_scores = attention_scores / jnp.sqrt(self.head_dim)
            
            # Apply relative position encoding if enabled
            if self.use_relative_position_encoding:
                relative_pos_enc = RelativePositionEncoding(self.relative_position_max_distance, dtype=self.dtype)(seq_len)
                attention_scores = attention_scores + relative_pos_enc[None, None, :, :]
            
            # Apply ALiBi bias if enabled
            if self.use_alibi_bias:
                alibi_bias = AlibiBias(self.num_heads, dtype=self.dtype)(seq_len)
                attention_scores = attention_scores + alibi_bias
            
            if self.use_causal_mask:
                mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=self.dtype))
                attention_scores = jnp.where(mask == 0, -1e9, attention_scores)
            
            if self.dropout_rate > 0.0 and not deterministic:
                dropout_rng = self.make_rng('dropout')
                attention_scores = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(attention_scores, rng=dropout_rng)
            
            attention_weights = jax.nn.softmax(attention_scores, axis=-1)
            attention_output = jnp.einsum('...hqk,...hvd->...hqd', attention_weights, v)
        
        output = nn.DenseGeneral(
            features=d_model,
            axis=(-2, -1),
            dtype=self.dtype,
            precision=self.precision
        )(attention_output)
        
        return output

    def apply_rotary_pos_emb(self, x, sin, cos):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

class MoELayer(nn.Module):
    num_experts: int
    top_k: int
    output_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_router_bias: bool = True
    router_jitter_noise: float = 0.0
    use_expert_parallelism: bool = False
    expert_parallel_size: int = 1

    @nn.compact
    def __call__(self, x, deterministic=True):
        batch_size = x.shape[0]
        seq_len = x.shape[1] if len(x.shape) > 2 else 1
        input_dim = x.shape[-1]
        
        if self.router_jitter_noise > 0 and not deterministic:
            x = x + jax.random.normal(self.make_rng('noise'), x.shape, dtype=x.dtype) * self.router_jitter_noise
        
        gating_network = nn.Dense(self.num_experts, use_bias=self.use_router_bias, dtype=self.dtype, precision=self.precision)
        gate_logits = gating_network(x)
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)
        
        top_k_indices = jax.lax.top_k(gate_probs, self.top_k)[1]
        top_k_gates = jnp.take_along_axis(gate_probs, top_k_indices, axis=-1)
        top_k_gates = top_k_gates / (jnp.sum(top_k_gates, axis=-1, keepdims=True) + 1e-9)
        
        expert_outputs = []
        for i in range(self.num_experts):
            expert = nn.Dense(self.output_dim, dtype=self.dtype, precision=self.precision)
            expert_output = expert(x)
            expert_outputs.append(expert_output)
        
        expert_outputs = jnp.stack(expert_outputs, axis=-2)
        
        expanded_indices = jnp.expand_dims(top_k_indices, axis=-1)
        expanded_indices = jnp.repeat(expanded_indices, self.output_dim, axis=-1)
        
        selected_experts = jnp.take_along_axis(expert_outputs, expanded_indices, axis=-2)
        weighted_experts = selected_experts * jnp.expand_dims(top_k_gates, axis=-1)
        
        output = jnp.sum(weighted_experts, axis=-2)
        return output

class ReversibleBlock(nn.Module):
    hidden_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST

    @nn.compact
    def __call__(self, x, deterministic=True):
        x1, x2 = jnp.split(x, 2, axis=-1)
        
        f = nn.Sequential([
            nn.Dense(self.hidden_dim, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(x1.shape[-1], dtype=self.dtype, precision=self.precision)
        ])
        
        g = nn.Sequential([
            nn.Dense(self.hidden_dim, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(x2.shape[-1], dtype=self.dtype, precision=self.precision)
        ])
        
        y1 = x1 + f(x2)
        y2 = x2 + g(y1)
        
        return jnp.concatenate([y1, y2], axis=-1)

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', lambda rng, shape, dtype: jnp.full(shape, self.init_values, dtype), (self.dim,), self.dtype)
        return x * gamma

class StochasticDepth(nn.Module):
    stochastic_depth_prob: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, input, deterministic=True):
        if deterministic:
            return input
        else:
            keep_prob = 1 - self.stochastic_depth_prob
            shape = [input.shape[0]] + [1] * (input.ndim - 1)
            random_tensor = keep_prob + jax.random.uniform(self.make_rng('dropout'), shape, dtype=self.dtype)
            binary_tensor = jnp.floor(random_tensor)
            return input * binary_tensor / keep_prob

class AdaptiveDropout(nn.Module):
    min_dropout: float = 0.05
    max_dropout: float = 0.3
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True, layer_depth=0, max_depth=1):
        if deterministic:
            return x
        else:
            dropout_rate = self.min_dropout + (self.max_dropout - self.min_dropout) * (layer_depth / max_depth)
            return nn.Dropout(rate=dropout_rate, deterministic=deterministic)(x)

class TokenDropout(nn.Module):
    token_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True):
        if deterministic or self.token_dropout_rate == 0.0:
            return x
        else:
            batch_size, seq_len, features = x.shape
            keep_prob = 1 - self.token_dropout_rate
            random_tensor = keep_prob + jax.random.uniform(self.make_rng('dropout'), (batch_size, seq_len, 1), dtype=self.dtype)
            binary_tensor = jnp.floor(random_tensor)
            return x * binary_tensor / keep_prob

class AdvancedMLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, x, deterministic=False, condition=None):
        dtype = self.config.dtype
        precision = self.config.precision
        
        if self.config.use_token_dropout and not deterministic:
            x = TokenDropout(self.config.token_dropout_rate, dtype=dtype)(x, deterministic=deterministic)
        
        if self.config.use_fourier_features:
            x = x.reshape(x.shape[0], -1)
            fourier_features = FourierFeatures(self.config.input_dim, self.config.fourier_frequency, dtype=dtype)(x)
            x = jnp.concatenate([x, fourier_features], axis=-1)
        else:
            x = x.reshape(x.shape[0], -1)
        
        if self.config.use_positional_encoding:
            x = x[:, None, :] if len(x.shape) == 2 else x
            x = PositionalEncoding(self.config.max_sequence_length, x.shape[-1], dtype=dtype)(x)
            x = x[:, 0, :] if len(x.shape) == 3 else x
        
        if self.config.use_reversible and x.shape[-1] % 2 == 0:
            x = ReversibleBlock(x.shape[-1], dtype=dtype, precision=precision)(x, deterministic)
        
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            residual = x
            
            if self.config.moe_layers and i % 2 == 0:
                x = MoELayer(
                    num_experts=self.config.num_experts,
                    top_k=self.config.top_k_experts,
                    output_dim=hidden_dim,
                    dtype=dtype,
                    precision=precision,
                    use_router_bias=self.config.use_router_bias,
                    router_jitter_noise=self.config.router_jitter_noise if not deterministic else 0.0,
                    use_expert_parallelism=self.config.use_expert_parallelism,
                    expert_parallel_size=self.config.expert_parallel_size
                )(x, deterministic)
            else:
                x = nn.Dense(
                    hidden_dim,
                    use_bias=self.config.use_bias,
                    dtype=dtype,
                    precision=precision
                )(x)
            
            if self.config.use_adaptive_normalization:
                x = AdaptiveLayerNorm(momentum=self.config.adaptive_norm_momentum, dtype=dtype)(x, training=not deterministic)
            elif self.config.use_layer_norm:
                x = nn.LayerNorm(dtype=dtype)(x)
            
            if self.config.activation == "gelu":
                x = nn.gelu(x)
            elif self.config.activation == "relu":
                x = nn.relu(x)
            elif self.config.activation == "swish":
                x = nn.swish(x)
            elif self.config.activation == "mish":
                x = x * jnp.tanh(jax.nn.softplus(x))
            
            if self.config.use_squeeze_excite:
                x = x[:, None, :] if len(x.shape) == 2 else x
                x = SqueezeExcite(ratio=self.config.se_ratio, dtype=dtype, precision=precision)(x, deterministic)
                x = x[:, 0, :] if len(x.shape) == 3 else x
            
            if self.config.use_conditional_scaling and condition is not None:
                x = ConditionalScaling(self.config.conditional_scaling_dim, dtype=dtype, precision=precision)(x, condition, deterministic)
            
            if self.config.use_adaptive_dropout and not deterministic:
                x = AdaptiveDropout(
                    min_dropout=self.config.adaptive_dropout_min,
                    max_dropout=self.config.adaptive_dropout_max,
                    dtype=dtype
                )(x, deterministic=deterministic, layer_depth=i, max_depth=len(self.config.hidden_dims))
            elif self.config.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.config.dropout_rate, deterministic=deterministic)(x)
            
            if self.config.use_stochastic_depth and not deterministic:
                if self.config.use_stochastic_depth_per_layer:
                    sd_prob = self.config.stochastic_depth_per_layer_prob * (i / (len(self.config.hidden_dims) - 1))
                else:
                    sd_prob = self.config.use_stochastic_depth * (i / (len(self.config.hidden_dims) - 1))
                x = StochasticDepth(stochastic_depth_prob=sd_prob, dtype=dtype)(x, deterministic=deterministic)
            
            if self.config.use_layer_scale:
                layer_scale_init = self.config.layer_scale_init_value
                if self.config.use_dynamic_layer_scaling:
                    layer_scale_init = self.config.dynamic_layer_scaling_min + \
                                      (self.config.dynamic_layer_scaling_max - self.config.dynamic_layer_scaling_min) * \
                                      (i / (len(self.config.hidden_dims) - 1))
                x = LayerScale(dim=hidden_dim, init_values=layer_scale_init, dtype=dtype)(x)
            
            if self.config.use_residual_connections and x.shape[-1] == residual.shape[-1]:
                x = x + residual
            elif self.config.use_highway_connections:
                transform_gate = nn.Dense(hidden_dim, use_bias=True, dtype=dtype, precision=precision)(residual)
                transform_gate = nn.sigmoid(transform_gate)
                x = transform_gate * x + (1 - transform_gate) * residual
        
        if self.config.use_attention:
            x = x[:, None, :] if len(x.shape) == 2 else x
            x = AttentionLayer(
                num_heads=self.config.attention_heads,
                head_dim=self.config.attention_dim,
                dtype=dtype,
                precision=precision,
                use_causal_mask=self.config.use_causal_mask,
                use_flash_attention=self.config.use_flash_attention,
                dropout_rate=self.config.attention_dropout_rate,
                use_relative_position_encoding=self.config.use_relative_position_encoding,
                relative_position_max_distance=self.config.relative_position_max_distance,
                use_rotary_position_embedding=self.config.use_rotary_position_embedding,
                use_alibi_bias=self.config.use_alibi_bias,
                use_attention_bias=self.config.use_attention_bias
            )(x, deterministic)
            x = x[:, 0, :] if len(x.shape) == 3 else x
        
        output = nn.Dense(
            self.config.output_dim,
            use_bias=self.config.use_bias,
            dtype=dtype,
            precision=precision
        )(x)
        
        return output

def create_mlp_model(config: MLPConfig = None):
    if config is None:
        config = MLPConfig()
    return AdvancedMLP(config)

def create_optimizer(config: MLPConfig, learning_rate: float = 0.001):
    if config.use_sophia_optimizer:
        from src.optimizers.optax_utils import sophia
        optimizer = sophia(
            learning_rate=learning_rate,
            b1=config.optimizer_beta1,
            b2=config.optimizer_beta2,
            rho=0.05,
            weight_decay=config.weight_decay_rate if config.use_weight_decay else 0.0
        )
    elif config.use_adan_optimizer:
        from src.optimizers.optax_utils import adan
        optimizer = adan(
            learning_rate=learning_rate,
            b1=config.optimizer_beta1,
            b2=config.optimizer_beta2,
            b3=config.optimizer_beta3,
            eps=config.optimizer_epsilon,
            weight_decay=config.weight_decay_rate if config.use_weight_decay else 0.0
        )
    elif config.use_lion_optimizer:
        from src.optimizers.optax_utils import lion
        optimizer = lion(
            learning_rate=learning_rate,
            b1=config.optimizer_beta1,
            b2=config.optimizer_beta2,
            weight_decay=config.weight_decay_rate if config.use_weight_decay else 0.0,
            gradient_clipping=config.gradient_clipping_norm if config.use_gradient_clipping else 0.0
        )
    else:
        if config.use_adamw_optimizer:
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                b1=config.optimizer_beta1,
                b2=config.optimizer_beta2,
                eps=config.optimizer_epsilon,
                weight_decay=config.weight_decay_rate if config.use_weight_decay else 0.0
            )
        else:
            optimizer = optax.adam(
                learning_rate=learning_rate,
                b1=config.optimizer_beta1,
                b2=config.optimizer_beta2,
                eps=config.optimizer_epsilon
            )
    
    # Add gradient clipping if enabled
    if config.use_gradient_clipping:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clipping_norm),
            optimizer
        )
    
    # Add gradient scaling if enabled
    if config.use_gradient_scaling:
        optimizer = optax.chain(
            optax.scale(config.gradient_scaling_factor),
            optimizer
        )
    
    # Add EMA if enabled
    if config.use_ema:
        optimizer = optax.chain(
            optimizer,
            optax.ema(decay=config.ema_decay)
        )
    
    # Add Lookahead if enabled
    if config.use_lookahead_optimizer:
        optimizer = optax.chain(
            optimizer,
            optax.lookahead(
                k=config.lookahead_steps,
                alpha=config.lookahead_alpha
            )
        )
    
    return optimizer

def create_learning_rate_schedule(config: MLPConfig, base_learning_rate: float = 0.001):
    if config.use_warmup_scheduler:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=config.warmup_steps
        )
    else:
        warmup_fn = optax.constant_schedule(base_learning_rate)
    
    if config.use_cosine_scheduler:
        decay_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=100000,  # This should be set based on total training steps
            alpha=config.cosine_min_lr_ratio
        )
    elif config.use_linear_scheduler:
        decay_fn = optax.linear_schedule(
            init_value=base_learning_rate,
            end_value=0.0,
            transition_steps=100000  # This should be set based on total training steps
        )
    elif config.use_polynomial_scheduler:
        decay_fn = optax.polynomial_schedule(
            init_value=base_learning_rate,
            end_value=base_learning_rate * config.cosine_min_lr_ratio,
            power=config.polynomial_power,
            transition_steps=100000  # This should be set based on total training steps
        )
    elif config.use_inverse_sqrt_scheduler:
        decay_fn = optax.join_schedules([
            optax.linear_schedule(
                init_value=0.0,
                end_value=base_learning_rate,
                transition_steps=config.inverse_sqrt_warmup_steps
            ),
            lambda step: base_learning_rate / jnp.sqrt(step / config.inverse_sqrt_warmup_steps)
        ], [config.inverse_sqrt_warmup_steps])
    else:
        decay_fn = optax.constant_schedule(base_learning_rate)
    
    # Combine warmup and decay schedules
    if config.use_warmup_scheduler:
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[config.warmup_steps]
        )
    else:
        schedule_fn = decay_fn
    
    return schedule_fn

def create_train_state(rng, config: MLPConfig, learning_rate: float = 0.001):
    from flax.training import train_state
    model = create_mlp_model(config)
    variables = model.init(rng, jnp.ones([1, config.input_dim]))
    params = variables['params']
    
    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule(config, learning_rate)
    
    # Create optimizer with learning rate schedule
    optimizer = create_optimizer(config, lr_schedule)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@jax.jit
def train_step(state, batch, dropout_rng, config: MLPConfig):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch['input'], 
            deterministic=False, 
            rngs={'dropout': dropout_rng}
        )
        labels = batch['label']
        if labels.ndim == 1:
            labels = jax.nn.one_hot(labels, logits.shape[-1])
        
        # Apply label smoothing if enabled
        if config.use_label_smoothing:
            labels = optax.smooth_labels(labels, config.label_smoothing_factor)
        
        # Calculate loss based on configuration
        if config.use_focal_loss:
            probs = jax.nn.softmax(logits)
            ce_loss = optax.softmax_cross_entropy(logits, labels)
            focal_weight = (1 - probs) ** config.focal_loss_gamma
            focal_loss = config.focal_loss_alpha * focal_weight * ce_loss
            loss = jnp.mean(focal_loss)
        else:
            loss = optax.softmax_cross_entropy(logits, labels).mean()
        
        # Add L1 regularization if enabled
        if config.use_weight_decay and config.weight_decay_rate > 0:
            l1_penalty = config.weight_decay_rate * sum(
                jnp.sum(jnp.abs(param)) for param in jax.tree_leaves(params)
            )
            loss = loss + l1_penalty
        
        # Add L2 regularization if enabled
        if config.use_weight_decay and config.weight_decay_rate > 0:
            l2_penalty = config.weight_decay_rate * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params)
            )
            loss = loss + l2_penalty
        
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(batch['label'], -1) if batch['label'].ndim > 1 else batch['label'])
    }
    return state, metrics

def get_dataset(config: MLPConfig, batch_size: int = 32, is_train: bool = True):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    
    dataset = tfds.load('mnist', split='train' if is_train else 'test')
    
    def preprocess_data(data):
        image = tf.cast(data['image'], tf.float32) / 255.0
        image = tf.reshape(image, [config.input_dim])
        if is_train:
            image = tf.image.random_flip_left_right(image)
        label = tf.one_hot(data['label'], config.output_dim)
        return {'input': image, 'label': label}
    
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def save_checkpoint(state, path: str):
    from flax.training import checkpoints
    checkpoints.save_checkpoint(path, state, state.step, overwrite=True)

def load_checkpoint(path: str):
    from flax.training import checkpoints
    return checkpoints.restore_checkpoint(path, None)