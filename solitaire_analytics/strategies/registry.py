"""Registry system for strategy management and discovery."""

from typing import Dict, Type, Optional

from solitaire_analytics.strategies.base import Strategy, StrategyConfig


class StrategyRegistry:
    """Registry for managing available strategies.
    
    This provides a centralized way to register, discover, and instantiate
    strategies. It follows a factory pattern for easy extensibility.
    """
    
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy]) -> None:
        """Register a strategy class.
        
        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a strategy.
        
        Args:
            name: Name of strategy to unregister
        """
        cls._strategies.pop(name.lower(), None)
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[Strategy]]:
        """Get a strategy class by name.
        
        Args:
            name: Name of the strategy
            
        Returns:
            Strategy class, or None if not found
        """
        return cls._strategies.get(name.lower())
    
    @classmethod
    def create_strategy(
        cls,
        name: str,
        config: Optional[StrategyConfig] = None
    ) -> Optional[Strategy]:
        """Create a strategy instance by name.
        
        Args:
            name: Name of the strategy
            config: Optional configuration for the strategy
            
        Returns:
            Strategy instance, or None if not found
        """
        strategy_class = cls.get_strategy_class(name)
        if strategy_class is None:
            return None
        
        return strategy_class(config)
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Type[Strategy]]:
        """List all registered strategies.
        
        Returns:
            Dictionary mapping strategy names to their classes
        """
        return cls._strategies.copy()
    
    @classmethod
    def get_strategy_names(cls) -> list:
        """Get names of all registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())


def get_strategy(name: str, config: Optional[StrategyConfig] = None) -> Optional[Strategy]:
    """Convenience function to get a strategy instance.
    
    Args:
        name: Name of the strategy
        config: Optional configuration for the strategy
        
    Returns:
        Strategy instance, or None if not found
        
    Example:
        >>> config = StrategyConfig(max_depth=5)
        >>> strategy = get_strategy("lookahead", config)
        >>> best_move = strategy.select_best_move(game_state)
    """
    return StrategyRegistry.create_strategy(name, config)


# Auto-register built-in strategies
def _register_builtin_strategies():
    """Register all built-in strategies."""
    from solitaire_analytics.strategies.simple import SimpleStrategy
    from solitaire_analytics.strategies.weighted import WeightedStrategy
    from solitaire_analytics.strategies.lookahead import LookaheadStrategy
    from solitaire_analytics.strategies.llm import LLMStrategy
    
    StrategyRegistry.register("simple", SimpleStrategy)
    StrategyRegistry.register("weighted", WeightedStrategy)
    StrategyRegistry.register("lookahead", LookaheadStrategy)
    StrategyRegistry.register("llm", LLMStrategy)


# Register on module import
_register_builtin_strategies()
