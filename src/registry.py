import jax
import jax.numpy as jnp
import json
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from flax.training import checkpoints
import hashlib

@dataclass
class ModelMetadata:
    model_name: str
    version: str
    framework: str
    input_shape: List[int]
    num_classes: int
    training_dataset: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: float
    checkpoint_path: str
    hash: str

class ModelRegistry:
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.registry_file = os.path.join(registry_path, "registry.json")
        os.makedirs(registry_path, exist_ok=True)
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                return {k: ModelMetadata(**v) for k, v in data.items()}
        return {}
    
    def _save_registry(self) -> None:
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_hash(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def register_model(self, 
                      model_name: str,
                      version: str,
                      framework: str,
                      input_shape: List[int],
                      num_classes: int,
                      training_dataset: str,
                      training_config: Dict[str, Any],
                      metrics: Dict[str, float],
                      checkpoint_path: str) -> str:
        model_id = f"{model_name}_{version}"
        
        if os.path.exists(checkpoint_path):
            model_hash = self._calculate_hash(checkpoint_path)
        else:
            model_hash = ""
        
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            framework=framework,
            input_shape=input_shape,
            num_classes=num_classes,
            training_dataset=training_dataset,
            training_config=training_config,
            metrics=metrics,
            created_at=time.time(),
            checkpoint_path=checkpoint_path,
            hash=model_hash
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        
        return model_id
    
    def get_model(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        model_id = f"{model_name}_{version}"
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelMetadata]:
        return list(self.models.values())
    
    def delete_model(self, model_name: str, version: str) -> bool:
        model_id = f"{model_name}_{version}"
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            return True
        return False
    
    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        model_id = f"{model_name}_{version}"
        if model_id in self.models:
            if not hasattr(self.models[model_id], 'stages'):
                self.models[model_id].stages = []
            if stage not in self.models[model_id].stages:
                self.models[model_id].stages.append(stage)
            self._save_registry()
            return True
        return False

def load_model_from_registry(registry: ModelRegistry, 
                           model_name: str, 
                           version: str) -> Any:
    model_info = registry.get_model(model_name, version)
    if model_info is None:
        raise ValueError(f"Model {model_name} version {version} not found in registry")
    
    return checkpoints.restore_checkpoint(model_info.checkpoint_path, None)

def save_model_to_registry(model: Any,
                          registry: ModelRegistry,
                          model_name: str,
                          version: str,
                          framework: str,
                          input_shape: List[int],
                          num_classes: int,
                          training_dataset: str,
                          training_config: Dict[str, Any],
                          metrics: Dict[str, float],
                          checkpoint_path: str) -> str:
    checkpoints.save_checkpoint(checkpoint_path, model, 0, overwrite=True)
    
    model_id = registry.register_model(
        model_name=model_name,
        version=version,
        framework=framework,
        input_shape=input_shape,
        num_classes=num_classes,
        training_dataset=training_dataset,
        training_config=training_config,
        metrics=metrics,
        checkpoint_path=checkpoint_path
    )
    
    return model_id