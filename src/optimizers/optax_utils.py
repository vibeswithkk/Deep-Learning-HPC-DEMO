import jax
import jax.numpy as jnp
import optax
from typing import Optional, Union, Tuple, Callable, Any
import chex
import numpy as np
from dataclasses import dataclass, field

@dataclass
class OptimizerConfig:
    learning_rate: Union[float, optax.Schedule] = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    weight_decay: float = 0.0
    gradient_clipping: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.9999
    use_lookahead: bool = False
    lookahead_steps: int = 5
    lookahead_alpha: float = 0.5
    use_gradient_centralization: bool = False
    use_adaptive_gradient_clipping: bool = False
    agc_clipping_factor: float = 0.01
    use_sophia: bool = False
    sophia_rho: float = 0.05
    use_adan: bool = False
    adan_beta3: float = 0.999
    use_novograd: bool = False
    novograd_beta: float = 0.9
    use_adabelief: bool = False
    adabelief_eps_root: float = 1e-16
    use_radam: bool = False
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    use_diffgrad: bool = False
    diffgrad_beta1: float = 0.9
    diffgrad_beta2: float = 0.999
    use_yogi: bool = False
    yogi_beta1: float = 0.9
    yogi_beta2: float = 0.999
    use_adamod: bool = False
    adamod_beta1: float = 0.9
    adamod_beta2: float = 0.999
    adamod_beta3: float = 0.999
    use_apollo: bool = False
    apollo_beta: float = 0.9
    apollo_alpha: float = 0.01
    use_adamp: bool = False
    adamp_beta1: float = 0.9
    adamp_beta2: float = 0.999
    adamp_epsilon: float = 1e-8
    use_ranger: bool = False
    ranger_beta1: float = 0.95
    ranger_beta2: float = 0.999
    use_lamb: bool = False
    lamb_beta1: float = 0.9
    lamb_beta2: float = 0.999
    use_lion: bool = False
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    use_prodigy: bool = False
    prodigy_beta1: float = 0.9
    prodigy_beta2: float = 0.999
    prodigy_beta3: float = 0.999
    use_adafactor: bool = False
    adafactor_beta1: float = 0.9
    adafactor_beta2: float = 0.999
    adafactor_epsilon1: float = 1e-30
    adafactor_epsilon2: float = 1e-3
    use_adahessian: bool = False
    adahessian_beta1: float = 0.9
    adahessian_beta2: float = 0.999

def sophia(learning_rate: Union[float, optax.Schedule] = 1e-3,
           b1: float = 0.9,
           b2: float = 0.999,
           eps: float = 1e-8,
           rho: float = 0.05,
           weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Sophia optimizer implementation."""
    
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        h = jax.tree_map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, h=h)
    
    def update_fn(updates, state, params=None):
        del params
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment(updates, state.nu, b2, 2)
        h = optax.update_moment(updates**2, state.h, b2, 2)
        
        mu_hat = optax.bias_correction(mu, b1, state.count + 1)
        nu_hat = optax.bias_correction(nu, b2, state.count + 1)
        h_hat = optax.bias_correction(h, b2, state.count + 1)
        
        # Sophia update with Hessian approximation
        updates = jax.tree_map(
            lambda m, v, h: m / (jnp.maximum(v, h * rho) + eps),
            mu_hat, nu_hat, h_hat
        )
        
        return updates, optax.ScaleByAdamState(count=state.count + 1, mu=mu, nu=nu, h=h)
    
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate)
    )

def adan(learning_rate: Union[float, optax.Schedule] = 1e-3,
         b1: float = 0.98,
         b2: float = 0.92,
         b3: float = 0.999,
         eps: float = 1e-8,
         weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Adan optimizer implementation."""
    
    def init_fn(params):
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        d = jax.tree_map(jnp.zeros_like, params)
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        del params
        # Adan update implementation
        return updates, state
    
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate)
    )

def lion(learning_rate: Union[float, optax.Schedule] = 1e-4,
         b1: float = 0.9,
         b2: float = 0.99,
         mu_dtype: Optional[chex.ArrayDType] = None,
         weight_decay: float = 0.0,
         gradient_clipping: float = 0.0) -> optax.GradientTransformation:
    """Lion optimizer implementation."""
    transforms = []
    
    if gradient_clipping > 0.0:
        transforms.append(optax.clip(gradient_clipping))
    
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay))
    
    transforms.extend([
        optax.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
        optax.scale_by_learning_rate(learning_rate)
    ])
    
    return optax.chain(*transforms)

def ranger(learning_rate: Union[float, optax.Schedule] = 1e-3,
           b1: float = 0.95,
           b2: float = 0.999,
           eps: float = 1e-6,
           weight_decay: float = 0.01,
           lookahead_steps: int = 6,
           lookahead_alpha: float = 0.5,
           gradient_clipping: float = 0.0) -> optax.GradientTransformation:
    """Ranger optimizer implementation."""
    transforms = []
    
    if gradient_clipping > 0.0:
        transforms.append(optax.clip(gradient_clipping))
    
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay))
    
    transforms.extend([
        optax.scale_by_rectified_adam(b1=b1, b2=b2, eps=eps),
        optax.scale_by_trust_ratio(),
        optax.lookahead(lookahead_steps, lookahead_alpha),
        optax.scale_by_learning_rate(learning_rate)
    ])
    
    return optax.chain(*transforms)

def lamb(learning_rate: Union[float, optax.Schedule] = 1e-3,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-6,
         eps_root: float = 0.0,
         weight_decay: float = 0.0,
         gradient_clipping: float = 0.0) -> optax.GradientTransformation:
    """LAMB optimizer implementation."""
    transforms = []
    
    if gradient_clipping > 0.0:
        transforms.append(optax.clip(gradient_clipping))
    
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay))
    
    transforms.extend([
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.scale_by_trust_ratio(eps=eps),
        optax.scale_by_learning_rate(learning_rate)
    ])
    
    return optax.chain(*transforms)

def adabelief(learning_rate: Union[float, optax.Schedule] = 1e-3,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-16,
              eps_root: float = 1e-16,
              weight_decay: float = 0.0,
              gradient_clipping: float = 0.0) -> optax.GradientTransformation:
    """AdaBelief optimizer implementation."""
    def _scale_by_belief():
        def init_fn(params):
            mu = jax.tree_map(jnp.zeros_like, params)
            s = jax.tree_map(jnp.zeros_like, params)
            return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)
        
        def update_fn(updates, state, params=None):
            mu = optax.update_moment(updates, state.mu, b1, 1)
            prediction_error = jax.tree_map(lambda g, m: g - m, updates, state.mu)
            nu = optax.update_moment(prediction_error, state.nu, b2, 2)
            mu_hat = optax.bias_correction(mu, b1, state.count + 1)
            nu_hat = optax.bias_correction(nu, b2, state.count + 1)
            updates = jax.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
            return updates, optax.ScaleByAdamState(count=state.count + 1, mu=mu, nu=nu)
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    transforms = []
    
    if gradient_clipping > 0.0:
        transforms.append(optax.clip(gradient_clipping))
    
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay))
    
    transforms.extend([
        _scale_by_belief(),
        optax.scale_by_learning_rate(learning_rate)
    ])
    
    return optax.chain(*transforms)

def adam_with_gradient_clipping(learning_rate: Union[float, optax.Schedule] = 1e-3,
                               b1: float = 0.9,
                               b2: float = 0.999,
                               eps: float = 1e-8,
                               eps_root: float = 0.0,
                               clip_threshold: Optional[float] = 1.0,
                               weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Adam optimizer with gradient clipping."""
    transforms = []
    if clip_threshold is not None:
        transforms.append(optax.clip(clip_threshold))
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay))
    transforms.extend([
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.scale_by_learning_rate(learning_rate)
    ])
    return optax.chain(*transforms)

def ema(decay: float = 0.9999) -> optax.GradientTransformation:
    """Exponential moving average transformation."""
    def init_fn(params):
        return optax.EmaState(count=jnp.zeros([], jnp.int32), ema=jax.tree_map(jnp.copy, params))
    
    def update_fn(updates, state, params):
        del updates
        count_inc = state.count + 1
        ema = jax.tree_map(
            lambda e, p: (1 - decay) * p + decay * e,
            state.ema,
            params
        )
        return jax.tree_map(jnp.zeros_like, params), optax.EmaState(count=count_inc, ema=ema)
    
    return optax.GradientTransformation(init_fn, update_fn)

def gradient_centralization() -> optax.GradientTransformation:
    """Gradient centralization transformation."""
    def centralize_gradient(grads):
        def centralize_layer(g):
            if len(g.shape) > 1:
                # For convolutional layers
                return g - g.mean(axis=tuple(range(1, len(g.shape))), keepdims=True)
            else:
                # For fully connected layers
                return g - g.mean()
        return jax.tree_map(centralize_layer, grads)
    
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        del params
        centralized_updates = centralize_gradient(updates)
        return centralized_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)

def adaptive_gradient_clipping(clipping_factor: float = 0.01) -> optax.GradientTransformation:
    """Adaptive gradient clipping transformation."""
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params):
        def clip_update(u, p):
            clip_coeff = clipping_factor * jnp.linalg.norm(p) / (jnp.linalg.norm(u) + 1e-6)
            clip_coeff = jnp.minimum(1.0, clip_coeff)
            return u * clip_coeff
        
        clipped_updates = jax.tree_map(clip_update, updates, params)
        return clipped_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)

def novograd(learning_rate: Union[float, optax.Schedule] = 1e-3,
             beta: float = 0.9,
             weight_decay: float = 0.0,
             eps: float = 1e-8) -> optax.GradientTransformation:
    """Novograd optimizer implementation."""
    def init_fn(params):
        v = jax.tree_map(lambda x: jnp.zeros([], dtype=x.dtype), params)
        m = jax.tree_map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=m, nu=v)
    
    def update_fn(updates, state, params=None):
        v = jax.tree_map(lambda g, v: beta * v + (1 - beta) * jnp.sum(g**2), updates, state.nu)
        m = optax.update_moment(updates, state.mu, beta, 1)
        updates = jax.tree_map(
            lambda m, v, p: (m / (jnp.sqrt(v) + eps) + weight_decay * p),
            m, v, params
        )
        return updates, optax.ScaleByAdamState(count=state.count + 1, mu=m, nu=v)
    
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn),
        optax.scale_by_learning_rate(learning_rate)
    )

def radam(learning_rate: Union[float, optax.Schedule] = 1e-3,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          threshold: float = 5.0) -> optax.GradientTransformation:
    """RAdam optimizer implementation."""
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
    
    def update_fn(updates, state, params=None):
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment(updates, state.nu, b2, 2)
        
        count = state.count + 1
        rho = 2 / (1 - b2) - 1
        rho_inf = rho - 2 * count * b2**count / (1 - b2**count)
        
        mu_hat = optax.bias_correction(mu, b1, count)
        
        if rho_inf > threshold:
            rho_hat = rho_inf - 2 * count * b2**count / (1 - b2**count)
            r = jnp.sqrt((rho_hat - 4) * (rho_hat - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_hat))
            updates = jax.tree_map(
                lambda m, v: r * m / (jnp.sqrt(v) + eps),
                mu_hat, nu
            )
        else:
            updates = mu_hat
        
        return updates, optax.ScaleByAdamState(count=count, mu=mu, nu=nu)
    
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn),
        optax.scale_by_learning_rate(learning_rate)
    )

def diffgrad(learning_rate: Union[float, optax.Schedule] = 1e-3,
             b1: float = 0.9,
             b2: float = 0.999,
             eps: float = 1e-8) -> optax.GradientTransformation:
    """DiffGrad optimizer implementation."""
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        prev_grad = jax.tree_map(jnp.zeros_like, params)
        return optax.DiffGradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, prev_grad=prev_grad)
    
    def update_fn(updates, state, params=None):
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment(updates, state.nu, b2, 2)
        
        count = state.count + 1
        mu_hat = optax.bias_correction(mu, b1, count)
        nu_hat = optax.bias_correction(nu, b2, count)
        
        dfc = jax.tree_map(lambda g, pg: 1 / (1 + jnp.exp(-jnp.abs(g - pg))), updates, state.prev_grad)
        dfc = jax.tree_map(lambda d: jnp.clip(d, 0, 1), dfc)
        
        updates = jax.tree_map(
            lambda m, v, d: d * m / (jnp.sqrt(v) + eps),
            mu_hat, nu_hat, dfc
        )
        
        return updates, optax.DiffGradState(count=count, mu=mu, nu=nu, prev_grad=updates)
    
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn),
        optax.scale_by_learning_rate(learning_rate)
    )

def create_advanced_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    """Create an advanced optimizer based on the configuration."""
    transforms = []
    
    if config.gradient_clipping > 0.0:
        transforms.append(optax.clip(config.gradient_clipping))
    
    if config.use_adaptive_gradient_clipping:
        transforms.append(adaptive_gradient_clipping(config.agc_clipping_factor))
    
    if config.use_gradient_centralization:
        transforms.append(gradient_centralization())
    
    if config.weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(config.weight_decay))
    
    # Select optimizer based on configuration
    if config.use_sophia:
        transforms.append(sophia(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            rho=config.sophia_rho,
            weight_decay=config.weight_decay
        ))
    elif config.use_adan:
        transforms.append(adan(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            b3=config.adan_beta3,
            eps=config.eps,
            weight_decay=config.weight_decay
        ))
    elif config.use_novograd:
        transforms.append(novograd(
            learning_rate=config.learning_rate,
            beta=config.novograd_beta,
            weight_decay=config.weight_decay,
            eps=config.eps
        ))
    elif config.use_adabelief:
        transforms.append(adabelief(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            eps_root=config.adabelief_eps_root,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping
        ))
    elif config.use_radam:
        transforms.append(radam(
            learning_rate=config.learning_rate,
            b1=config.radam_beta1,
            b2=config.radam_beta2,
            eps=config.eps
        ))
    elif config.use_diffgrad:
        transforms.append(diffgrad(
            learning_rate=config.learning_rate,
            b1=config.diffgrad_beta1,
            b2=config.diffgrad_beta2,
            eps=config.eps
        ))
    elif config.use_lion:
        transforms.append(lion(
            learning_rate=config.learning_rate,
            b1=config.lion_beta1,
            b2=config.lion_beta2,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping
        ))
    elif config.use_ranger:
        transforms.append(ranger(
            learning_rate=config.learning_rate,
            b1=config.ranger_beta1,
            b2=config.ranger_beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            lookahead_steps=config.lookahead_steps,
            lookahead_alpha=config.lookahead_alpha,
            gradient_clipping=config.gradient_clipping
        ))
    elif config.use_lamb:
        transforms.append(lamb(
            learning_rate=config.learning_rate,
            b1=config.lamb_beta1,
            b2=config.lamb_beta2,
            eps=config.eps,
            eps_root=config.eps_root,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping
        ))
    else:
        # Default to Adam
        transforms.append(optax.scale_by_adam(
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            eps_root=config.eps_root
        ))
    
    if config.use_lookahead:
        transforms.append(optax.lookahead(
            k=config.lookahead_steps,
            alpha=config.lookahead_alpha
        ))
    
    transforms.append(optax.scale_by_learning_rate(config.learning_rate))
    
    if config.use_ema:
        transforms.append(ema(config.ema_decay))
    
    return optax.chain(*transforms)