"""
Environment Factory for CloudSimPy DRL Scheduler

This module provides a factory pattern for creating appropriate environment
instances based on configuration parameters.
"""

from typing import Type, Dict, Any, List
import importlib
import logging

from .abstract_env import AbstractSchedulingEnvironment, EnvironmentType, EnvironmentConfig

logger = logging.getLogger(__name__)


class EnvironmentFactory:
    """
    Factory for creating training and testing environments.
    
    This factory uses a registry pattern to manage different environment
    implementations and provides a clean interface for environment creation.
    """
    
    _registry: Dict[EnvironmentType, Type[AbstractSchedulingEnvironment]] = {}
    _initialized = False
    
    @classmethod
    def register(cls, env_type: EnvironmentType, env_class: Type[AbstractSchedulingEnvironment]):
        """
        Register an environment implementation.
        
        Args:
            env_type: Type of environment (TRAINING or TESTING)
            env_class: Environment class to register
        """
        if not issubclass(env_class, AbstractSchedulingEnvironment):
            raise ValueError(f"Environment class must inherit from AbstractSchedulingEnvironment")
        
        cls._registry[env_type] = env_class
        logger.info(f"Registered environment: {env_type.value} -> {env_class.__name__}")
    
    @classmethod
    def create(cls, config: EnvironmentConfig) -> AbstractSchedulingEnvironment:
        """
        Create environment instance based on configuration.
        
        Args:
            config: Environment configuration
            
        Returns:
            Environment instance
            
        Raises:
            ValueError: If environment type is not registered
        """
        cls._ensure_initialized()
        
        if config.env_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Environment type {config.env_type} not registered. "
                f"Available types: {[t.value for t in available_types]}"
            )
        
        env_class = cls._registry[config.env_type]
        logger.info(f"Creating environment: {config.env_type.value} using {env_class.__name__}")
        
        try:
            return env_class(config)
        except Exception as e:
            logger.error(f"Failed to create environment {config.env_type.value}: {e}")
            raise
    
    @classmethod
    def list_available(cls) -> List[EnvironmentType]:
        """
        List available environment types.
        
        Returns:
            List of registered environment types
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, env_type: EnvironmentType) -> bool:
        """
        Check if environment type is registered.
        
        Args:
            env_type: Environment type to check
            
        Returns:
            True if registered, False otherwise
        """
        cls._ensure_initialized()
        return env_type in cls._registry
    
    @classmethod
    def unregister(cls, env_type: EnvironmentType):
        """
        Unregister an environment type.
        
        Args:
            env_type: Environment type to unregister
        """
        if env_type in cls._registry:
            del cls._registry[env_type]
            logger.info(f"Unregistered environment: {env_type.value}")
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered environments"""
        cls._registry.clear()
        cls._initialized = False
        logger.info("Cleared environment registry")
    
    @classmethod
    def _ensure_initialized(cls):
        """Ensure factory is initialized with default environments"""
        if cls._initialized:
            return
        
        try:
            # Try to import and register default environments
            cls._register_default_environments()
            cls._initialized = True
        except ImportError as e:
            logger.warning(f"Could not register default environments: {e}")
            # Continue without default registration
            cls._initialized = True
    
    @classmethod
    def _register_default_environments(cls):
        """Register default environment implementations"""
        try:
            # Import lightweight training environment
            from .training_env import LightweightTrainingEnvironment
            cls.register(EnvironmentType.TRAINING, LightweightTrainingEnvironment)
        except ImportError:
            logger.warning("Could not import LightweightTrainingEnvironment")
        
        try:
            # Import enhanced testing environment
            from .testing_env import EnhancedTestingEnvironment
            cls.register(EnvironmentType.TESTING, EnhancedTestingEnvironment)
        except ImportError:
            logger.warning("Could not import EnhancedTestingEnvironment")


def create_environment(
    env_type: str, 
    dataset_args: Dict[str, Any], 
    **kwargs
) -> AbstractSchedulingEnvironment:
    """
    Convenience function for environment creation.
    
    Args:
        env_type: Environment type ("training" or "testing")
        dataset_args: Dataset configuration arguments
        **kwargs: Additional configuration parameters
        
    Returns:
        Environment instance
    """
    config = EnvironmentConfig(
        env_type=EnvironmentType(env_type),
        dataset_args=dataset_args,
        **kwargs
    )
    return EnvironmentFactory.create(config)


def create_training_environment(dataset_args: Dict[str, Any], **kwargs) -> AbstractSchedulingEnvironment:
    """
    Convenience function for creating training environment.
    
    Args:
        dataset_args: Dataset configuration arguments
        **kwargs: Additional configuration parameters
        
    Returns:
        Training environment instance
    """
    return create_environment("training", dataset_args, **kwargs)


def create_testing_environment(dataset_args: Dict[str, Any], **kwargs) -> AbstractSchedulingEnvironment:
    """
    Convenience function for creating testing environment.
    
    Args:
        dataset_args: Dataset configuration arguments
        **kwargs: Additional configuration parameters
        
    Returns:
        Testing environment instance
    """
    return create_environment("testing", dataset_args, **kwargs)


class EnvironmentBuilder:
    """
    Builder pattern for complex environment configuration.
    
    This class provides a fluent interface for building environment
    configurations with validation and defaults.
    """
    
    def __init__(self):
        self._env_type = None
        self._dataset_args = {}
        self._reward_weights = None
        self._performance_mode = "balanced"
        self._debug_mode = False
        self._seed = None
    
    def training(self):
        """Set environment type to training"""
        self._env_type = EnvironmentType.TRAINING
        return self
    
    def testing(self):
        """Set environment type to testing"""
        self._env_type = EnvironmentType.TESTING
        return self
    
    def with_dataset_args(self, **kwargs):
        """Set dataset arguments"""
        self._dataset_args.update(kwargs)
        return self
    
    def with_reward_weights(self, makespan: float = 0.7, energy: float = 0.3):
        """Set reward weights"""
        self._reward_weights = {"makespan": makespan, "energy": energy}
        return self
    
    def with_performance_mode(self, mode: str):
        """Set performance mode"""
        if mode not in ["speed", "accuracy", "balanced"]:
            raise ValueError("Performance mode must be 'speed', 'accuracy', or 'balanced'")
        self._performance_mode = mode
        return self
    
    def with_debug(self, debug: bool = True):
        """Enable/disable debug mode"""
        self._debug_mode = debug
        return self
    
    def with_seed(self, seed: int):
        """Set random seed"""
        self._seed = seed
        return self
    
    def build(self) -> AbstractSchedulingEnvironment:
        """
        Build and return the configured environment.
        
        Returns:
            Configured environment instance
            
        Raises:
            ValueError: If required configuration is missing
        """
        if self._env_type is None:
            raise ValueError("Environment type must be specified")
        
        if not self._dataset_args:
            raise ValueError("Dataset arguments must be specified")
        
        config = EnvironmentConfig(
            env_type=self._env_type,
            dataset_args=self._dataset_args,
            reward_weights=self._reward_weights,
            performance_mode=self._performance_mode,
            debug_mode=self._debug_mode,
            seed=self._seed
        )
        
        return EnvironmentFactory.create(config)

