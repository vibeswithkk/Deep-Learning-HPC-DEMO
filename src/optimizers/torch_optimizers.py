import torch
import torch.optim as optim
from torch.optim import Optimizer
from typing import Dict, Any, Optional, Callable, Iterable, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TorchOptimizerConfig:
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.999
    eps: float = 1e-8
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

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, 
                 gradient_clipping=0.0, use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       gradient_clipping=gradient_clipping, use_ema=use_ema,
                       ema_decay=ema_decay)
        super(Lion, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                if group['gradient_clipping'] > 0.0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                update = exp_avg.clone().mul_(beta2).add_(grad, alpha=1 - beta2).sign_()
                p.add_(update, alpha=-group['lr'])
                
                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])
        
        return loss

class Ranger(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), 
                 eps=1e-5, weight_decay=0, amsgrad=True, transformer='softplus', smooth=50,
                 grad_transformer='square', gradient_clipping=0.0, use_ema=False, ema_decay=0.9999):
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                       N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay,
                       smooth=smooth, transformer=transformer, grad_transformer=grad_transformer,
                       gradient_clipping=gradient_clipping, use_ema=use_ema, ema_decay=ema_decay)
        super(Ranger, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')
                
                if group['gradient_clipping'] > 0.0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])
                
                p_data_fp32 = p.data.float()
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad.add_(p_data_fp32, alpha=group['weight_decay'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                buffered = getattr(group, 'buffer', [[None, None, None] for _ in range(10)])
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    if N_sma >= group['N_sma_threshhold']:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif group['transformer'] == 'softplus':
                        step_size = torch.log(torch.exp(1 - beta1 ** state['step']) + group['smooth'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['lr'] * group['weight_decay'])
                
                if N_sma >= group['N_sma_threshhold']:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                
                p.data.copy_(p_data_fp32)
                
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=group['alpha'])
                    p.data.copy_(slow_p)
                
                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p_data_fp32, alpha=1 - group['ema_decay'])
        
        return loss

class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0,
                 gradient_clipping=0.0, use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       gradient_clipping=gradient_clipping, use_ema=use_ema, ema_decay=ema_decay)
        super(AdaBelief, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_var = state['exp_avg_var']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if group['gradient_clipping'] > 0.0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])

                # AdaBelief update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                pred_grad = exp_avg - grad
                exp_avg_var.mul_(beta2).addcmul_(pred_grad, pred_grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class Adan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0,
                 use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       use_ema=use_ema, ema_decay=ema_decay)
        super(Adan, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['diff'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                diff = state['diff']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Adan update
                diff.mul_(beta2).add_(grad - diff, alpha=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta3).add_(diff ** 2, alpha=1 - beta3)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])

                p.addcdiv_(exp_avg + beta2 * diff, denom, value=-step_size)

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class Sophia(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), rho=0.04, eps=1e-8, weight_decay=0,
                 use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps, weight_decay=weight_decay,
                       use_ema=use_ema, ema_decay=ema_decay)
        super(Sophia, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Hessian estimation
                if state['step'] % 10 == 1:  # Update Hessian every 10 steps
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                else:
                    hessian.copy_(grad.abs())

                # Sophia update
                denom = hessian.add_(group['eps']).sqrt()
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                p.addcdiv_(exp_avg, denom, value=-step_size * group['rho'])

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class Novograd(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, weight_decay=0,
                 use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                       use_ema=use_ema, ema_decay=ema_decay)
        super(Novograd, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = 0.0
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                beta = group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Novograd update
                exp_avg_sq = beta * state['exp_avg_sq'] + (1 - beta) * grad.norm().item() ** 2
                state['exp_avg_sq'] = exp_avg_sq

                exp_avg.mul_(beta).add_(grad / math.sqrt(exp_avg_sq + group['eps']), alpha=1 - beta)

                p.add_(exp_avg, alpha=-group['lr'])

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       use_ema=use_ema, ema_decay=ema_decay)
        super(RAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # RAdam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Rectified adaptive learning rate
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / bias_correction2

                if rho_t > 4:
                    l = math.sqrt((1 - bias_correction2) * (rho_t - 4) / (rho_inf - 4) * (rho_t - 2) / (rho_inf - 2) * rho_inf / (rho_inf - 2))
                    step_size = group['lr'] * l / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group['lr'] / bias_correction1
                    p.add_(exp_avg, alpha=-step_size)

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class DiffGrad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       use_ema=use_ema, ema_decay=ema_decay)
        super(DiffGrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                previous_grad = state['previous_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # DiffGrad update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # DiffGrad coefficient
                diff = abs(previous_grad - grad)
                dfc = 1. / (1. + torch.exp(-diff))
                state['previous_grad'] = grad.clone()

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] * dfc / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])

        return loss

class Yogi(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3, initial_accumulator=1e-6,
                 weight_decay=0, use_ema=False, ema_decay=0.9999):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= initial_accumulator:
            raise ValueError("Invalid initial_accumulator value: {}".format(initial_accumulator))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, initial_accumulator=initial_accumulator,
                       weight_decay=weight_decay, use_ema=use_ema, ema_decay=ema_decay)
        super(Yogi, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(Yogi, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.full_like(p, group['initial_accumulator'], memory_format=torch.preserve_format)
                    if group['use_ema']:
                        state['ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                grad_squared = grad**2
                exp_avg_sq.add_(torch.sign(exp_avg_sq - grad_squared) * grad_squared, alpha=-(1 - beta2))
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                if group['use_ema']:
                    state['ema'].mul_(group['ema_decay']).add_(p, alpha=1 - group['ema_decay'])
        
        return loss

def create_advanced_torch_optimizer(model_parameters, config: TorchOptimizerConfig) -> Optimizer:
    if config.use_sophia:
        return Sophia(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            rho=config.sophia_rho,
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_adan:
        return Adan(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2, config.beta3),
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_novograd:
        return Novograd(
            model_parameters,
            lr=config.learning_rate,
            beta=config.novograd_beta,
            weight_decay=config.weight_decay,
            eps=config.eps,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_adabelief:
        return AdaBelief(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_radam:
        return RAdam(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.radam_beta1, config.radam_beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_diffgrad:
        return DiffGrad(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.diffgrad_beta1, config.diffgrad_beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_lion:
        return Lion(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.lion_beta1, config.lion_beta2),
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    elif config.use_ranger:
        return Ranger(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.ranger_beta1, config.ranger_beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            gradient_clipping=config.gradient_clipping,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay
        )
    else:
        return optim.AdamW(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
