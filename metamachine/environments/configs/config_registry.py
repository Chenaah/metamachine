"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from omegaconf import OmegaConf
import yaml

class ConfigRegistry:
    """Registry for environment configurations."""
    
    _configs: Dict[str, Dict[str, Any]] = {}
    _default_configs_dir = os.path.join(os.path.dirname(__file__), "default_configs")
    
    @classmethod
    def register(cls, name: str, config: Dict[str, Any]) -> None:
        """Register a new environment configuration schema.
        
        Args:
            name: Name of the environment configuration
            config: Configuration dictionary or path to YAML file
        """
        if isinstance(config, str) and os.path.exists(config):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        cls._configs[name] = config
    
    @classmethod
    def get_schema(cls, name: str) -> Dict[str, Any]:
        """Get a registered configuration schema by name."""
        if name not in cls._configs:
            # Try to load from default configs
            default_path = os.path.join(cls._default_configs_dir, f"{name}.yaml")
            if os.path.exists(default_path):
                with open(default_path, 'r') as f:
                    cls._configs[name] = yaml.safe_load(f)
            else:
                raise KeyError(f"Environment config '{name}' not found in registry or default configs")
        return cls._configs[name]
    
    @classmethod
    def _resolve_base_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve base config inheritance.
        
        If config has a '_base' field, merge it with the base config.
        The current config's values take precedence over base values.
        """
        if '_base' in config:
            base_name = config.pop('_base')
            base_config = cls.get_schema(base_name)
            # Convert to OmegaConf for proper merging
            base_conf = OmegaConf.create(base_config)
            current_conf = OmegaConf.create(config)
            # Merge with current config taking precedence
            merged = OmegaConf.merge(base_conf, current_conf)
            return OmegaConf.to_container(merged, resolve=True)
        return config
    
    @classmethod
    def create_from_name(cls, name: str, **kwargs) -> Any:
        """Create a configuration instance from a registered name with optional overrides."""
        schema = cls.get_schema(name)
        # Resolve any base config inheritance
        schema = cls._resolve_base_config(schema)
        config = OmegaConf.create(schema)
        
        if kwargs:
            # Convert kwargs to OmegaConf for proper merging
            override_conf = OmegaConf.create(kwargs)
            config = OmegaConf.merge(config, override_conf)
        
        return config
    
    @classmethod
    def create_from_file(cls, config_path: str, **kwargs) -> Any:
        """Create a configuration instance from a YAML file with optional overrides."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Resolve any base config inheritance
        config_dict = cls._resolve_base_config(config_dict)
        config = OmegaConf.create(config_dict)
        
        if kwargs:
            # Convert kwargs to OmegaConf for proper merging
            override_conf = OmegaConf.create(kwargs)
            config = OmegaConf.merge(config, override_conf)
            
        return config
    
    @classmethod
    def create_empty(cls, **kwargs) -> Any:
        """Create a configuration instance from scratch using kwargs."""
        return OmegaConf.create(kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered environment configurations."""
        registered = set(cls._configs.keys())
        if os.path.exists(cls._default_configs_dir):
            default_configs = {f.stem for f in Path(cls._default_configs_dir).glob("*.yaml")}
            registered.update(default_configs)
        return sorted(list(registered))
    
    @classmethod
    def validate_config(cls, config: Any, schema_name: Optional[str] = None) -> bool:
        """Validate a config against a schema if provided.
        
        Args:
            config: Configuration to validate
            schema_name: Optional name of schema to validate against
            
        Returns:
            bool: True if valid, raises error if invalid
        """
        if schema_name:
            schema = cls.get_schema(schema_name)
            schema_conf = OmegaConf.create(schema)
            # This will raise an error if validation fails
            OmegaConf.merge(schema_conf, config)
        return True 