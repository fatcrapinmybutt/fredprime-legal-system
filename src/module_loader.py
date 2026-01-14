"""
Dynamic Module Discovery & Auto-Loading System

Provides:
- Automatic module detection from directories
- Interface validation via ABC/Protocol
- Dynamic registration without manual imports
- Lazy loading for performance
- Namespace/plugin organization
"""

import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Protocol
import logging

logger = logging.getLogger(__name__)


class ModuleInterface(ABC):
    """Base interface for discoverable modules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Module version."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize module with configuration."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up module resources."""
        pass


class ModuleRegistry:
    """Registry for dynamically loaded modules."""
    
    def __init__(self):
        self._modules: Dict[str, ModuleInterface] = {}
        self._namespaces: Dict[str, Dict[str, ModuleInterface]] = {}
        self._lazy_loaders: Dict[str, Path] = {}
    
    def register(self, module: ModuleInterface, namespace: str = "default") -> None:
        """Register a module."""
        key = f"{namespace}:{module.name}"
        self._modules[key] = module
        
        if namespace not in self._namespaces:
            self._namespaces[namespace] = {}
        self._namespaces[namespace][module.name] = module
        
        logger.info(f"Registered module: {key} (v{module.version})")
    
    def get(self, name: str, namespace: str = "default") -> Optional[ModuleInterface]:
        """Get a registered module."""
        key = f"{namespace}:{name}"
        return self._modules.get(key)
    
    def get_all(self, namespace: str = "default") -> Dict[str, ModuleInterface]:
        """Get all modules in a namespace."""
        return self._namespaces.get(namespace, {})
    
    def list_modules(self) -> List[str]:
        """List all registered module keys."""
        return list(self._modules.keys())
    
    def unregister(self, name: str, namespace: str = "default") -> bool:
        """Unregister a module."""
        key = f"{namespace}:{name}"
        if key in self._modules:
            module = self._modules.pop(key)
            module.shutdown()
            if namespace in self._namespaces and name in self._namespaces[namespace]:
                del self._namespaces[namespace][name]
            logger.info(f"Unregistered module: {key}")
            return True
        return False


class ModuleLoader:
    """Dynamically load modules from directories."""
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry()
    
    def discover_modules(self, directory: Path, pattern: str = "*.py") -> List[Path]:
        """Discover module files in a directory."""
        if not directory.exists():
            logger.warning(f"Module directory does not exist: {directory}")
            return []
        
        modules = []
        for file in directory.glob(pattern):
            if file.name.startswith("_"):
                continue
            modules.append(file)
        
        logger.info(f"Discovered {len(modules)} modules in {directory}")
        return modules
    
    def load_module(self, path: Path, namespace: str = "default") -> Optional[ModuleInterface]:
        """Load a module from a file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for: {path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            
            # Find ModuleInterface implementations
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, ModuleInterface) and obj != ModuleInterface:
                    instance = obj()
                    self.registry.register(instance, namespace)
                    logger.info(f"Loaded module: {instance.name} from {path}")
                    return instance
            
            logger.warning(f"No ModuleInterface implementations found in: {path}")
            return None
        
        except Exception as e:
            logger.error(f"Error loading module {path}: {e}", exc_info=True)
            return None
    
    def load_all_modules(self, directory: Path, namespace: str = "default") -> int:
        """Load all modules from a directory."""
        modules = self.discover_modules(directory)
        loaded = 0
        
        for module_path in modules:
            if self.load_module(module_path, namespace):
                loaded += 1
        
        logger.info(f"Loaded {loaded}/{len(modules)} modules from {directory}")
        return loaded
    
    def lazy_load(self, module_path: Path, namespace: str = "default") -> None:
        """Register a module for lazy loading."""
        key = f"{namespace}:{module_path.stem}"
        self.registry._lazy_loaders[key] = module_path
        logger.info(f"Registered lazy loader for: {key}")


# Global registry
_global_registry = ModuleRegistry()
_global_loader = ModuleLoader(_global_registry)


def get_registry() -> ModuleRegistry:
    """Get the global module registry."""
    return _global_registry


def get_loader() -> ModuleLoader:
    """Get the global module loader."""
    return _global_loader


def discover_and_load(directory: Path, namespace: str = "default") -> int:
    """Convenience function to discover and load all modules."""
    return _global_loader.load_all_modules(directory, namespace)
