"""
Cost tracking service for OpenAI API usage.

Tracks token usage and calculates costs based on model pricing.
"""
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostEntry:
    """Cost entry for a single API call."""
    agent_name: str
    operation: str
    model: str
    tokens: TokenUsage
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CostTracker:
    """
    Thread-safe cost tracker for OpenAI API calls.
    
    Pricing (as of 2024, per 1M tokens):
    - gpt-4o-mini: $0.15 input, $0.60 output
    - gpt-4o: $2.50 input, $10.00 output
    - gpt-4-turbo: $10.00 input, $30.00 output
    """
    
    # Model pricing per 1M tokens (input, output)
    MODEL_PRICING = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    }
    
    def __init__(self):
        self._entries: list[CostEntry] = []
        self._lock = threading.Lock()
    
    def record_usage(
        self,
        agent_name: str,
        operation: str,
        model: str,
        usage: Optional[Dict] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0
    ) -> CostEntry:
        """
        Record token usage and calculate cost.
        
        Args:
            agent_name: Name of the agent making the call
            operation: Description of the operation (e.g., "evaluate", "extract_actions")
            model: Model name (e.g., "gpt-4o-mini")
            usage: Usage dict from OpenAI response (preferred)
            prompt_tokens: Prompt tokens (if usage not provided)
            completion_tokens: Completion tokens (if usage not provided)
            total_tokens: Total tokens (if usage not provided)
        
        Returns:
            CostEntry with calculated cost
        """
        with self._lock:
            # Extract from usage dict if provided
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            
            tokens = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            # Calculate cost
            cost_usd = self._calculate_cost(model, tokens)
            
            entry = CostEntry(
                agent_name=agent_name,
                operation=operation,
                model=model,
                tokens=tokens,
                cost_usd=cost_usd
            )
            
            self._entries.append(entry)
            return entry
    
    def _calculate_cost(self, model: str, tokens: TokenUsage) -> float:
        """Calculate cost in USD based on model pricing."""
        # Get pricing for model (default to gpt-4o-mini if unknown)
        input_price, output_price = self.MODEL_PRICING.get(
            model.lower(),
            self.MODEL_PRICING["gpt-4o-mini"]
        )
        
        # Convert per 1M tokens to per token
        input_cost = (tokens.prompt_tokens / 1_000_000) * input_price
        output_cost = (tokens.completion_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict:
        """
        Get cost summary aggregated by agent and total.
        
        Returns:
            Dict with summary statistics
        """
        with self._lock:
            if not self._entries:
                return {
                    "total_calls": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "by_agent": {},
                    "by_operation": {},
                    "by_model": {}
                }
            
            total_cost = sum(entry.cost_usd for entry in self._entries)
            total_tokens = sum(entry.tokens.total_tokens for entry in self._entries)
            
            # Aggregate by agent
            by_agent: Dict[str, Dict] = {}
            for entry in self._entries:
                if entry.agent_name not in by_agent:
                    by_agent[entry.agent_name] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "operations": []
                    }
                by_agent[entry.agent_name]["calls"] += 1
                by_agent[entry.agent_name]["tokens"] += entry.tokens.total_tokens
                by_agent[entry.agent_name]["cost_usd"] += entry.cost_usd
                by_agent[entry.agent_name]["operations"].append(entry.operation)
            
            # Aggregate by operation
            by_operation: Dict[str, Dict] = {}
            for entry in self._entries:
                if entry.operation not in by_operation:
                    by_operation[entry.operation] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0
                    }
                by_operation[entry.operation]["calls"] += 1
                by_operation[entry.operation]["tokens"] += entry.tokens.total_tokens
                by_operation[entry.operation]["cost_usd"] += entry.cost_usd
            
            # Aggregate by model
            by_model: Dict[str, Dict] = {}
            for entry in self._entries:
                if entry.model not in by_model:
                    by_model[entry.model] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0
                    }
                by_model[entry.model]["calls"] += 1
                by_model[entry.model]["tokens"] += entry.tokens.total_tokens
                by_model[entry.model]["cost_usd"] += entry.cost_usd
            
            return {
                "total_calls": len(self._entries),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "by_agent": {
                    agent: {
                        "calls": stats["calls"],
                        "tokens": stats["tokens"],
                        "cost_usd": round(stats["cost_usd"], 6),
                        "operations": list(set(stats["operations"]))
                    }
                    for agent, stats in by_agent.items()
                },
                "by_operation": {
                    op: {
                        "calls": stats["calls"],
                        "tokens": stats["tokens"],
                        "cost_usd": round(stats["cost_usd"], 6)
                    }
                    for op, stats in by_operation.items()
                },
                "by_model": {
                    model: {
                        "calls": stats["calls"],
                        "tokens": stats["tokens"],
                        "cost_usd": round(stats["cost_usd"], 6)
                    }
                    for model, stats in by_model.items()
                }
            }
    
    def get_entries(self) -> list[CostEntry]:
        """Get all cost entries."""
        with self._lock:
            return self._entries.copy()
    
    def reset(self):
        """Reset all tracked costs."""
        with self._lock:
            self._entries.clear()
    
    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("💰 COST SUMMARY")
        print("=" * 80)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost_usd']:.6f} USD")
        print()
        
        print("By Agent:")
        for agent, stats in summary['by_agent'].items():
            print(f"  {agent}:")
            print(f"    Calls: {stats['calls']}")
            print(f"    Tokens: {stats['tokens']:,}")
            print(f"    Cost: ${stats['cost_usd']:.6f} USD")
            print(f"    Operations: {', '.join(stats['operations'])}")
        print()
        
        print("By Model:")
        for model, stats in summary['by_model'].items():
            print(f"  {model}:")
            print(f"    Calls: {stats['calls']}")
            print(f"    Tokens: {stats['tokens']:,}")
            print(f"    Cost: ${stats['cost_usd']:.6f} USD")
        print()
        
        print("By Operation:")
        for op, stats in summary['by_operation'].items():
            print(f"  {op}:")
            print(f"    Calls: {stats['calls']}")
            print(f"    Tokens: {stats['tokens']:,}")
            print(f"    Cost: ${stats['cost_usd']:.6f} USD")
        print("=" * 80)


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()

