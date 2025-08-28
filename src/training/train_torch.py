import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, Callable, Union
import deepspeed
import time
import os
import json
import numpy as np
from src.models.torch_deepspeed_cnn import create_torch_model, DeepSpeedCNNConfig
from src.training.callbacks import CallbackList, TrainingState
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import wandb
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil

class AdvancedTorchTrainState:
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.loss = 0.0
        self.metrics = {}
        self.learning_rate = 0.0
        self.throughput = 0.0
        self.memory_usage = 0.0
        self.gpu_utilization = 0.0
        self.cpu_utilization = 0.0

def get_advanced_deepspeed_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train_batch_size": config_dict['data']['batch_size'] * torch.cuda.device_count() if torch.cuda.is_available() else 1,
        "gradient_accumulation_steps": config_dict['training'].get('gradient_accumulation_steps', 1),
        "optimizer": {
            "type": config_dict['training'].get('optimizer', 'Adam'),
            "params": {
                "lr": config_dict['training']['learning_rate'],
                "betas": [
                    config_dict['training'].get('optimizer_beta1', 0.9),
                    config_dict['training'].get('optimizer_beta2', 0.999)
                ],
                "eps": config_dict['training'].get('optimizer_eps', 1e-8),
                "weight_decay": config_dict['training'].get('weight_decay', 0.01)
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config_dict['training']['learning_rate'],
                "warmup_num_steps": config_dict['training'].get('warmup_steps', 1000),
                "total_num_steps": config_dict['training'].get('total_steps', 100000)
            }
        },
        "zero_optimization": {
            "stage": config_dict['training'].get('zero_stage', 3),
            "offload_optimizer": {
                "device": config_dict['training'].get('offload_optimizer_device', "cpu"),
                "pin_memory": config_dict['training'].get('offload_optimizer_pin_memory', True)
            },
            "offload_param": {
                "device": config_dict['training'].get('offload_param_device', "cpu"),
                "pin_memory": config_dict['training'].get('offload_param_pin_memory', True)
            },
            "overlap_comm": config_dict['training'].get('zero_overlap_comm', True),
            "contiguous_gradients": config_dict['training'].get('zero_contiguous_gradients', True),
            "sub_group_size": config_dict['training'].get('zero_sub_group_size', 1e9),
            "reduce_bucket_size": config_dict['training'].get('zero_reduce_bucket_size', "auto"),
            "stage3_prefetch_bucket_size": config_dict['training'].get('zero_stage3_prefetch_bucket_size', "auto"),
            "stage3_param_persistence_threshold": config_dict['training'].get('zero_stage3_param_persistence_threshold', "auto"),
            "stage3_max_live_parameters": config_dict['training'].get('zero_stage3_max_live_parameters', 1e9),
            "stage3_max_reuse_distance": config_dict['training'].get('zero_stage3_max_reuse_distance', 1e9),
            "stage3_gather_16bit_weights_on_model_save": config_dict['training'].get('zero_stage3_gather_16bit_weights_on_model_save', True)
        },
        "gradient_clipping": config_dict['training'].get('gradient_clipping', 1.0),
        "fp16": {
            "enabled": config_dict['training'].get('fp16_enabled', True),
            "loss_scale": config_dict['training'].get('fp16_loss_scale', 0),
            "loss_scale_window": config_dict['training'].get('fp16_loss_scale_window', 1000),
            "initial_scale_power": config_dict['training'].get('fp16_initial_scale_power', 16),
            "hysteresis": config_dict['training'].get('fp16_hysteresis', 2),
            "min_loss_scale": config_dict['training'].get('fp16_min_loss_scale', 1)
        },
        "bf16": {
            "enabled": config_dict['training'].get('bf16_enabled', False)
        },
        "wall_clock_breakdown": config_dict['training'].get('wall_clock_breakdown', False),
        "flops_profiler": {
            "enabled": config_dict['training'].get('flops_profiler_enabled', False),
            "profile_step": config_dict['training'].get('flops_profiler_profile_step', 1),
            "module_depth": config_dict['training'].get('flops_profiler_module_depth', -1),
            "top_modules": config_dict['training'].get('flops_profiler_top_modules', 1),
            "detailed": config_dict['training'].get('flops_profiler_detailed', True),
            "output_file": config_dict['training'].get('flops_profiler_output_file', None)
        },
        "tensor_parallel": {
            "enabled": config_dict['training'].get('tensor_parallel_enabled', False),
            "tp_size": config_dict['training'].get('tensor_parallel_size', 1)
        },
        "pipeline": {
            "enabled": config_dict['training'].get('pipeline_parallel_enabled', False),
            "stages": config_dict['training'].get('pipeline_stages', 1),
            "partition": config_dict['training'].get('pipeline_partition', "uniform")
        }
    }

def get_advanced_dataset(config_dict: Dict[str, Any], is_train: bool):
    transform_list = [
        transforms.Resize((config_dict['model'].get('input_height', 224), 
                          config_dict['model'].get('input_width', 224)))
    ]
    
    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=config_dict['data'].get('random_horizontal_flip_prob', 0.5)),
            transforms.RandomVerticalFlip(p=config_dict['data'].get('random_vertical_flip_prob', 0.0)),
            transforms.RandomRotation(degrees=config_dict['data'].get('random_rotation_degrees', 0)),
            transforms.ColorJitter(
                brightness=config_dict['data'].get('color_jitter_brightness', 0.0),
                contrast=config_dict['data'].get('color_jitter_contrast', 0.0),
                saturation=config_dict['data'].get('color_jitter_saturation', 0.0),
                hue=config_dict['data'].get('color_jitter_hue', 0.0)
            ),
            transforms.RandomAffine(
                degrees=config_dict['data'].get('random_affine_degrees', 0),
                translate=config_dict['data'].get('random_affine_translate', None),
                scale=config_dict['data'].get('random_affine_scale', None),
                shear=config_dict['data'].get('random_affine_shear', None)
            ),
            transforms.RandomPerspective(
                distortion_scale=config_dict['data'].get('random_perspective_distortion', 0.0),
                p=config_dict['data'].get('random_perspective_prob', 0.0)
            ),
            transforms.RandomErasing(
                p=config_dict['data'].get('random_erasing_prob', 0.0),
                scale=config_dict['data'].get('random_erasing_scale', (0.02, 0.33)),
                ratio=config_dict['data'].get('random_erasing_ratio', (0.3, 3.3))
            )
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if config_dict['data'].get('normalize', True):
        transform_list.append(
            transforms.Normalize(
                mean=config_dict['data'].get('normalize_mean', [0.485, 0.456, 0.406]),
                std=config_dict['data'].get('normalize_std', [0.229, 0.224, 0.225])
            )
        )
    
    transform = transforms.Compose(transform_list)
    
    dataset_name = config_dict['data'].get('dataset_name', 'CIFAR10')
    
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root=config_dict['data'].get('data_root', './data'),
            train=is_train,
            download=config_dict['data'].get('download', True),
            transform=transform
        )
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            root=config_dict['data'].get('data_root', './data'),
            train=is_train,
            download=config_dict['data'].get('download', True),
            transform=transform
        )
    elif dataset_name == 'ImageNet':
        dataset = torchvision.datasets.ImageNet(
            root=config_dict['data'].get('data_root', './data'),
            split='train' if is_train else 'val',
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    # Generate random box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Uniform distribution of bbox center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def label_smoothing_criterion(predictions, targets, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    ce_loss = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def get_system_metrics():
    # Get CPU utilization
    cpu_percent = psutil.cpu_percent()
    
    # Get memory utilization
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Get GPU utilization if available
    gpu_percent = 0.0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
    except:
        pass
    
    return {
        'cpu_utilization': cpu_percent,
        'memory_utilization': memory_percent,
        'gpu_utilization': gpu_percent
    }

def train_and_evaluate(config_path: str, workdir: str):
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Initialize logging
    if config_dict.get('logging', {}).get('use_wandb', False):
        wandb.init(
            project=config_dict['logging'].get('project_name', 'deep-learning-hpc'),
            config=config_dict
        )
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(workdir, 'logs'))
    
    model_config = DeepSpeedCNNConfig(
        num_classes=config_dict['model']['num_classes'],
        input_channels=config_dict['model'].get('input_channels', 3),
        base_channels=config_dict['model'].get('base_channels', 64),
        growth_rate=config_dict['model'].get('growth_rate', 32),
        num_layers=config_dict['model'].get('num_layers', 4),
        compression_ratio=config_dict['model'].get('compression_ratio', 0.5),
        use_attention=config_dict['model'].get('use_attention', True),
        attention_heads=config_dict['model'].get('attention_heads', 8),
        use_mixed_precision=config_dict['model'].get('use_mixed_precision', True),
        precision=config_dict['model'].get('precision', "bf16"),
        use_gradient_checkpointing=config_dict['model'].get('use_gradient_checkpointing', True),
        use_layer_scaling=config_dict['model'].get('use_layer_scaling', True),
        layer_scale_init_value=config_dict['model'].get('layer_scale_init_value', 1e-6),
        use_stochastic_depth=config_dict['model'].get('use_stochastic_depth', True),
        stochastic_depth_prob=config_dict['model'].get('stochastic_depth_prob', 0.1),
        use_squeeze_excite=config_dict['model'].get('use_squeeze_excite', True),
        se_ratio=config_dict['model'].get('se_ratio', 16),
        use_drop_path=config_dict['model'].get('use_drop_path', True),
        drop_path_rate=config_dict['model'].get('drop_path_rate', 0.0),
        use_ema=config_dict['model'].get('use_ema', True),
        ema_decay=config_dict['model'].get('ema_decay', 0.9999),
        use_fused_ops=config_dict['model'].get('use_fused_ops', True),
        use_tensor_parallel=config_dict['model'].get('use_tensor_parallel', False),
        tensor_parallel_size=config_dict['model'].get('tensor_parallel_size', 1),
        use_adaptive_normalization=config_dict['model'].get('use_adaptive_normalization', True),
        adaptive_norm_momentum=config_dict['model'].get('adaptive_norm_momentum', 0.99),
        use_highway_connections=config_dict['model'].get('use_highway_connections', False),
        use_fourier_features=config_dict['model'].get('use_fourier_features', False),
        fourier_frequency=config_dict['model'].get('fourier_frequency', 16),
        use_positional_encoding=config_dict['model'].get('use_positional_encoding', False),
        max_sequence_length=config_dict['model'].get('max_sequence_length', 1024),
        use_conditional_scaling=config_dict['model'].get('use_conditional_scaling', False),
        conditional_scaling_dim=config_dict['model'].get('conditional_scaling_dim', 128),
        use_flash_attention=config_dict['model'].get('use_flash_attention', False),
        use_causal_mask=config_dict['model'].get('use_causal_mask', False),
        use_router_bias=config_dict['model'].get('use_router_bias', True),
        router_jitter_noise=config_dict['model'].get('router_jitter_noise', 0.0),
        use_stochastic_depth_per_layer=config_dict['model'].get('use_stochastic_depth_per_layer', True),
        stochastic_depth_per_layer_prob=config_dict['model'].get('stochastic_depth_per_layer_prob', 0.05),
        use_layer_adaptive_dropout=config_dict['model'].get('use_layer_adaptive_dropout', False),
        adaptive_dropout_min=config_dict['model'].get('adaptive_dropout_min', 0.05),
        adaptive_dropout_max=config_dict['model'].get('adaptive_dropout_max', 0.3),
        use_dynamic_layer_scaling=config_dict['model'].get('use_dynamic_layer_scaling', False),
        dynamic_layer_scaling_min=config_dict['model'].get('dynamic_layer_scaling_min', 1e-7),
        dynamic_layer_scaling_max=config_dict['model'].get('dynamic_layer_scaling_max', 1e-5),
        use_moe=config_dict['model'].get('use_moe', False),
        num_experts=config_dict['model'].get('num_experts', 8),
        top_k_experts=config_dict['model'].get('top_k_experts', 2),
        use_reversible_blocks=config_dict['model'].get('use_reversible_blocks', False),
        use_gradient_checkpointing_per_block=config_dict['model'].get('use_gradient_checkpointing_per_block', True),
        activation=config_dict['model'].get('activation', 'gelu'),
        dropout_rate=config_dict['model'].get('dropout_rate', 0.1),
        use_residual_connections=config_dict['model'].get('use_residual_connections', True)
    )
    
    model = create_torch_model(model_config)
    
    deepspeed_config = get_advanced_deepspeed_config(config_dict)
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config
    )
    
    train_dataset = get_advanced_dataset(config_dict, is_train=True)
    val_dataset = get_advanced_dataset(config_dict, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict['data']['batch_size'],
        shuffle=config_dict['data'].get('shuffle', True),
        num_workers=config_dict['data'].get('num_workers', 4),
        pin_memory=config_dict['data'].get('pin_memory', True),
        persistent_workers=config_dict['data'].get('persistent_workers', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_dict['data']['batch_size'],
        shuffle=False,
        num_workers=config_dict['data'].get('num_workers', 4),
        pin_memory=config_dict['data'].get('pin_memory', True),
        persistent_workers=config_dict['data'].get('persistent_workers', True)
    )
    
    # Select criterion based on configuration
    if config_dict['training'].get('use_focal_loss', False):
        criterion = focal_loss
    else:
        criterion = nn.CrossEntropyLoss()
    
    callbacks = CallbackList()
    
    train_state = AdvancedTorchTrainState()
    callbacks.on_train_begin({'model': model, 'config': config_dict})
    
    scaler = GradScaler() if config_dict['training'].get('use_amp', True) else None
    
    for epoch in range(config_dict['training']['epochs']):
        train_state.epoch = epoch
        callbacks.on_epoch_begin(epoch, {'model': model, 'train_state': train_state})
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            train_state.step = batch_idx
            callbacks.on_batch_begin(batch_idx, {'model': model, 'train_state': train_state})
            
            data, target = data.to(model.device), target.to(model.device)
            
            # Apply mixup if specified
            if config_dict['training'].get('use_mixup', False) and np.random.rand() < config_dict['training'].get('mixup_prob', 0.5):
                data, target_a, target_b, lam = mixup_data(
                    data, target, 
                    config_dict['training'].get('mixup_alpha', 1.0), 
                    model.device
                )
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                else:
                    output = model(data)
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            # Apply cutmix if specified
            elif config_dict['training'].get('use_cutmix', False) and np.random.rand() < config_dict['training'].get('cutmix_prob', 0.5):
                data, target_a, target_b, lam = cutmix_data(
                    data, target, 
                    config_dict['training'].get('cutmix_alpha', 1.0), 
                    model.device
                )
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                else:
                    output = model(data)
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        if config_dict['training'].get('label_smoothing', 0.0) > 0.0:
                            loss = label_smoothing_criterion(output, target, config_dict['training']['label_smoothing'])
                        elif config_dict['training'].get('use_focal_loss', False):
                            loss = focal_loss(output, target, 
                                            config_dict['training'].get('focal_alpha', 1.0),
                                            config_dict['training'].get('focal_gamma', 2.0))
                        else:
                            loss = criterion(output, target)
                else:
                    output = model(data)
                    if config_dict['training'].get('label_smoothing', 0.0) > 0.0:
                        loss = label_smoothing_criterion(output, target, config_dict['training']['label_smoothing'])
                    elif config_dict['training'].get('use_focal_loss', False):
                        loss = focal_loss(output, target, 
                                        config_dict['training'].get('focal_alpha', 1.0),
                                        config_dict['training'].get('focal_gamma', 2.0))
                    else:
                        loss = criterion(output, target)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if config_dict['training'].get('gradient_clipping', 0.0) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict['training']['gradient_clipping'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config_dict['training'].get('gradient_clipping', 0.0) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict['training']['gradient_clipping'])
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            if config_dict['training'].get('use_mixup', False) and 'target_a' in locals():
                correct += (lam * predicted.eq(target_a).sum().item() + (1 - lam) * predicted.eq(target_b).sum().item())
            else:
                correct += predicted.eq(target).sum().item()
            
            train_state.loss = loss.item()
            train_state.metrics = {
                'loss': train_loss / (batch_idx + 1),
                'accuracy': 100. * correct / total,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Get system metrics
            system_metrics = get_system_metrics()
            train_state.cpu_utilization = system_metrics['cpu_utilization']
            train_state.memory_usage = system_metrics['memory_utilization']
            train_state.gpu_utilization = system_metrics['gpu_utilization']
            
            # Log metrics to TensorBoard
            if batch_idx % config_dict.get('logging', {}).get('log_interval', 100) == 0:
                for key, value in train_state.metrics.items():
                    writer.add_scalar(f'train/{key}', value, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('system/cpu_utilization', train_state.cpu_utilization, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('system/memory_utilization', train_state.memory_usage, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('system/gpu_utilization', train_state.gpu_utilization, epoch * len(train_loader) + batch_idx)
            
            callbacks.on_batch_end(batch_idx, {'model': model, 'train_state': train_state})
        
        epoch_time = time.time() - epoch_start_time
        train_state.metrics['epoch_time'] = epoch_time
        train_state.throughput = len(train_loader) / epoch_time
        
        # Log epoch metrics to TensorBoard
        for key, value in train_state.metrics.items():
            writer.add_scalar(f'epoch/train_{key}', value, epoch)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_top5_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Calculate top-5 accuracy
                _, top5_pred = output.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 = top5_pred.eq(target.view(1, -1).expand_as(top5_pred))
                val_top5_correct += correct_top5.sum().item()
        
        train_state.metrics.update({
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': 100. * val_correct / val_total,
            'val_top5_accuracy': 100. * val_top5_correct / val_total
        })
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('epoch/val_loss', train_state.metrics['val_loss'], epoch)
        writer.add_scalar('epoch/val_accuracy', train_state.metrics['val_accuracy'], epoch)
        writer.add_scalar('epoch/val_top5_accuracy', train_state.metrics['val_top5_accuracy'], epoch)
        
        callbacks.on_epoch_end(epoch, {
            'model': model,
            'train_state': train_state,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb if enabled
        if config_dict.get('logging', {}).get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_state.loss,
                'train_accuracy': train_state.metrics.get('accuracy', 0.0),
                'val_loss': train_state.metrics.get('val_loss', 0.0),
                'val_accuracy': train_state.metrics.get('val_accuracy', 0.0),
                'val_top5_accuracy': train_state.metrics.get('val_top5_accuracy', 0.0),
                'learning_rate': train_state.metrics.get('learning_rate', 0.0),
                'cpu_utilization': train_state.cpu_utilization,
                'memory_utilization': train_state.memory_usage,
                'gpu_utilization': train_state.gpu_utilization
            })
    
    callbacks.on_train_end({'model': model})
    
    # Close TensorBoard writer
    writer.close()
    
    # Finish wandb if enabled
    if config_dict.get('logging', {}).get('use_wandb', False):
        wandb.finish()
    
    return model

if __name__ == "__main__":
    train_and_evaluate("config/train_config.yaml", "./results")