"""
Token usage and cost tracking service for LLM API calls.

Tracks token usage across all agents and calculates costs based on OpenAI pricing.
"""
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

# Try to import tiktoken for token estimation fallback
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    agent_name: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = ""  # e.g., "extract_actions", "evaluate", "rerank"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation
        }


class TokenTracker:
    """
    Singleton token tracker for monitoring LLM usage and costs.
    """
    _instance: Optional['TokenTracker'] = None
    
    # OpenAI pricing per 1M tokens (as of 2024)
    # Prices in USD per 1M tokens
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.15,   # $0.15 per 1M input tokens
            "output": 0.60   # $0.60 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        # Default fallback pricing (gpt-4o-mini)
        "default": {
            "input": 0.15,
            "output": 0.60
        }
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._usage_history: List[TokenUsage] = []
        self._session_usage: Dict[str, TokenUsage] = {}  # For aggregating per session
        self._initialized = True
    
    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a model, with fallback to default."""
        # Normalize model name (remove version suffixes if any)
        model_key = model.lower()
        for key in self.PRICING:
            if key != "default" and model_key.startswith(key):
                return self.PRICING[key]
        return self.PRICING["default"]
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for a given token usage."""
        pricing = self._get_pricing(model)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def record_usage(
        self,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        operation: str = "",
        session_id: Optional[str] = None
    ) -> TokenUsage:
        """
        Record token usage for an API call.
        
        Args:
            agent_name: Name of the agent making the call
            model: Model name (e.g., "gpt-4o-mini")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            operation: Description of the operation (e.g., "extract_actions")
            session_id: Optional session ID for aggregating usage
        
        Returns:
            TokenUsage object with cost calculated
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        usage = TokenUsage(
            agent_name=agent_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            operation=operation
        )
        
        self._usage_history.append(usage)
        
        # Aggregate by session if provided
        if session_id:
            if session_id not in self._session_usage:
                self._session_usage[session_id] = TokenUsage(
                    agent_name=agent_name,
                    model=model,
                    operation=f"session_{session_id}"
                )
            session = self._session_usage[session_id]
            session.prompt_tokens += prompt_tokens
            session.completion_tokens += completion_tokens
            session.total_tokens += total_tokens
            session.cost_usd += cost
        
        return usage
    
    def record_from_openai_response(
        self,
        agent_name: str,
        model: str,
        response,
        operation: str = "",
        session_id: Optional[str] = None,
        prompt_text: Optional[str] = None
    ) -> TokenUsage:
        """
        Record usage from an OpenAI API response object (including autogen wrappers).
        
        Args:
            agent_name: Name of the agent
            model: Model name
            response: OpenAI response object (OpenAI client or autogen AssistantAgent response)
            operation: Description of operation
            session_id: Optional session ID
        
        Returns:
            TokenUsage object
        """
        # Try multiple ways to extract usage from response
        usage_data = None
        prompt_tokens = 0
        completion_tokens = 0
        
        # Method 1: Direct usage attribute (OpenAI client response)
        if hasattr(response, 'usage'):
            usage_data = response.usage
        
        # Method 2: Autogen response - check inner_messages for OpenAI response
        if hasattr(response, 'inner_messages') and response.inner_messages:
            # inner_messages contains the actual OpenAI API responses
            for msg in response.inner_messages:
                if hasattr(msg, 'usage'):
                    usage_data = msg.usage
                    break
                elif hasattr(msg, 'response') and hasattr(msg.response, 'usage'):
                    usage_data = msg.response.usage
                    break
                elif isinstance(msg, dict) and 'usage' in msg:
                    usage_data = msg['usage']
                    break
                # Also check if msg itself is a response object
                elif hasattr(msg, 'choices') and hasattr(msg, 'usage'):
                    # This might be the actual OpenAI response
                    usage_data = msg.usage
                    break
        
        # Method 2b: Try to extract from autogen response structure
        if not usage_data:
            extracted = self._extract_usage_from_autogen_response(response)
            if extracted:
                # Convert to usage-like object
                class UsageObj:
                    def __init__(self, d):
                        self.prompt_tokens = d['prompt_tokens']
                        self.completion_tokens = d['completion_tokens']
                        self.total_tokens = d.get('total_tokens', self.prompt_tokens + self.completion_tokens)
                usage_data = UsageObj(extracted)
        
        # Method 3: Autogen response - check chat_message.usage
        if not usage_data and hasattr(response, 'chat_message'):
            if hasattr(response.chat_message, 'usage'):
                usage_data = response.chat_message.usage
            # Also check if chat_message has response_info or similar
            elif hasattr(response.chat_message, 'response_metadata'):
                metadata = response.chat_message.response_metadata
                if metadata and 'usage' in metadata:
                    usage_data = metadata['usage']
        
        # Method 4: Check for underlying OpenAI response in autogen
        if not usage_data and hasattr(response, 'response'):
            # Autogen might wrap the response
            if hasattr(response.response, 'usage'):
                usage_data = response.response.usage
        
        # Method 4: Dictionary format
        elif isinstance(response, dict):
            if 'usage' in response:
                usage_data = response['usage']
            elif 'response' in response and isinstance(response['response'], dict):
                if 'usage' in response['response']:
                    usage_data = response['response']['usage']
        
        # Method 5: Check for autogen's response_info or similar attributes
        if not usage_data:
            for attr in ['response_info', 'metadata', 'info', '_response']:
                if hasattr(response, attr):
                    attr_value = getattr(response, attr)
                    if isinstance(attr_value, dict) and 'usage' in attr_value:
                        usage_data = attr_value['usage']
                    elif hasattr(attr_value, 'usage'):
                        usage_data = attr_value.usage
        
        # Extract token counts from usage_data
        if usage_data:
            if hasattr(usage_data, 'prompt_tokens'):
                prompt_tokens = usage_data.prompt_tokens or 0
                completion_tokens = usage_data.completion_tokens or 0
            elif isinstance(usage_data, dict):
                prompt_tokens = usage_data.get('prompt_tokens', 0) or 0
                completion_tokens = usage_data.get('completion_tokens', 0) or 0
            else:
                # Try to access as attributes even if it's a dict-like object
                try:
                    prompt_tokens = getattr(usage_data, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(usage_data, 'completion_tokens', 0) or 0
                except:
                    pass
        
        if prompt_tokens > 0 or completion_tokens > 0:
            return self.record_usage(
                agent_name=agent_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                operation=operation,
                session_id=session_id
            )
        else:
            # No usage data available - try to estimate using tiktoken as fallback
            # But be conservative - only estimate if we really can't get actual data
            estimated_tokens = self._estimate_tokens_fallback(response, model, operation, prompt_text)
            
            if estimated_tokens and estimated_tokens['total_tokens'] > 0:
                # Apply conservative caps to prevent overestimation
                # Cap completion tokens at reasonable limits based on model
                max_completion_tokens = self._get_max_completion_tokens_for_model(model)
                if estimated_tokens['completion_tokens'] > max_completion_tokens:
                    print(f"[TokenTracker] ⚠️  Capping estimated completion tokens from {estimated_tokens['completion_tokens']} to {max_completion_tokens}")
                    estimated_tokens['completion_tokens'] = max_completion_tokens
                    estimated_tokens['total_tokens'] = estimated_tokens['prompt_tokens'] + estimated_tokens['completion_tokens']
                
                print(f"[TokenTracker] ⚠️  No usage data found, using estimated tokens for {agent_name}/{operation}")
                print(f"[TokenTracker]    Estimated: {estimated_tokens['prompt_tokens']} prompt + {estimated_tokens['completion_tokens']} completion = {estimated_tokens['total_tokens']} total")
                return self.record_usage(
                    agent_name=agent_name,
                    model=model,
                    prompt_tokens=estimated_tokens['prompt_tokens'],
                    completion_tokens=estimated_tokens['completion_tokens'],
                    operation=operation,
                    session_id=session_id
                )
            else:
                # No usage data and couldn't estimate
                print(f"[TokenTracker] ⚠️  No usage data found in response for {agent_name}/{operation}")
                return TokenUsage(
                    agent_name=agent_name,
                    model=model,
                    operation=operation
                )
    
    def _extract_usage_from_autogen_response(self, response) -> Optional[Dict[str, int]]:
        """
        Try to extract usage data from an autogen AssistantAgent response.
        Returns dict with prompt_tokens, completion_tokens, total_tokens or None.
        """
        usage_data = None
        
        # Try to access the model client's last response
        try:
            # Check if response has a reference to the model client
            if hasattr(response, '_agent') and hasattr(response._agent, 'model_client'):
                model_client = response._agent.model_client
            elif hasattr(response, '_model_client'):
                model_client = response._model_client
            else:
                model_client = None
            
            if model_client:
                # Try to get the last completion response
                if hasattr(model_client, '_last_completion'):
                    last_comp = model_client._last_completion
                    if hasattr(last_comp, 'usage'):
                        usage_data = last_comp.usage
                elif hasattr(model_client, 'last_completion'):
                    last_comp = model_client.last_completion
                    if hasattr(last_comp, 'usage'):
                        usage_data = last_comp.usage
        except:
            pass
        
        if usage_data:
            if hasattr(usage_data, 'prompt_tokens'):
                return {
                    'prompt_tokens': usage_data.prompt_tokens,
                    'completion_tokens': usage_data.completion_tokens,
                    'total_tokens': usage_data.total_tokens if hasattr(usage_data, 'total_tokens') else usage_data.prompt_tokens + usage_data.completion_tokens
                }
            elif isinstance(usage_data, dict):
                return {
                    'prompt_tokens': usage_data.get('prompt_tokens', 0),
                    'completion_tokens': usage_data.get('completion_tokens', 0),
                    'total_tokens': usage_data.get('total_tokens', usage_data.get('prompt_tokens', 0) + usage_data.get('completion_tokens', 0))
                }
        
        return None
    
    def _get_max_completion_tokens_for_model(self, model: str) -> int:
        """
        Get reasonable max completion tokens for a model to cap estimates.
        These are conservative limits to prevent overestimation.
        """
        # Common max_tokens values used in the codebase
        model_limits = {
            'gpt-4o-mini': 2000,  # Common limit used in evaluator
            'gpt-4o': 4000,
            'gpt-4': 4000,
            'gpt-3.5-turbo': 2000,
        }
        
        # Default to a conservative limit
        for model_key, limit in model_limits.items():
            if model_key in model.lower():
                return limit
        
        # Default conservative limit
        return 2000
    
    def _estimate_tokens_fallback(self, response, model: str, operation: str, prompt_text: Optional[str] = None) -> Optional[Dict[str, int]]:
        """
        Fallback: Estimate tokens using tiktoken if actual usage data is not available.
        Uses conservative estimation to avoid overcounting.
        
        Args:
            response: The response object
            model: Model name
            operation: Operation name
            prompt_text: Optional prompt text (if provided, use this instead of extracting)
        
        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens or None
        """
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            # Get encoding for the model
            encoding_name = "cl100k_base"  # Used by gpt-4o-mini and most OpenAI models
            try:
                encoding = tiktoken.get_encoding(encoding_name)
            except:
                return None
            
            # Extract text content
            extracted_prompt = prompt_text or ""
            completion_text = ""
            
            # If prompt_text not provided, try to extract from response
            if not extracted_prompt and hasattr(response, 'inner_messages') and response.inner_messages:
                # Look for user messages (prompt)
                for msg in response.inner_messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'role'):
                        if getattr(msg, 'role', None) == 'user':
                            extracted_prompt += str(getattr(msg, 'content', ''))
                    elif isinstance(msg, dict):
                        if msg.get('role') == 'user':
                            extracted_prompt += str(msg.get('content', ''))
            
            # Get completion from chat_message
            if hasattr(response, 'chat_message'):
                if hasattr(response.chat_message, 'content'):
                    completion_text = str(response.chat_message.content)
                elif isinstance(response.chat_message, dict):
                    completion_text = str(response.chat_message.get('content', ''))
            
            # Estimate tokens - be conservative
            # Tiktoken can slightly overestimate, so we'll use it but cap at reasonable limits
            prompt_tokens = len(encoding.encode(extracted_prompt)) if extracted_prompt else 0
            completion_tokens = len(encoding.encode(completion_text)) if completion_text else 0
            
            # Apply a small safety margin (reduce by 5% to account for tokenization differences)
            # This helps prevent overestimation
            prompt_tokens = int(prompt_tokens * 0.95) if prompt_tokens > 0 else 0
            completion_tokens = int(completion_tokens * 0.95) if completion_tokens > 0 else 0
            
            # Cap completion tokens at model-specific limit
            max_completion = self._get_max_completion_tokens_for_model(model)
            if completion_tokens > max_completion:
                completion_tokens = max_completion
            
            # Only return if we got some text
            if prompt_tokens > 0 or completion_tokens > 0:
                return {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }
        
        except Exception as e:
            # Silently fail estimation
            pass
        
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of all token usage."""
        if not self._usage_history:
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "by_agent": {},
                "by_model": {},
                "by_operation": {}
            }
        
        total_prompt = sum(u.prompt_tokens for u in self._usage_history)
        total_completion = sum(u.completion_tokens for u in self._usage_history)
        total_tokens = sum(u.total_tokens for u in self._usage_history)
        total_cost = sum(u.cost_usd for u in self._usage_history)
        
        # Aggregate by agent
        by_agent = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        })
        
        # Aggregate by model
        by_model = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        })
        
        # Aggregate by operation
        by_operation = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        })
        
        for usage in self._usage_history:
            # By agent
            agent_stats = by_agent[usage.agent_name]
            agent_stats["calls"] += 1
            agent_stats["prompt_tokens"] += usage.prompt_tokens
            agent_stats["completion_tokens"] += usage.completion_tokens
            agent_stats["total_tokens"] += usage.total_tokens
            agent_stats["cost_usd"] += usage.cost_usd
            
            # By model
            model_stats = by_model[usage.model]
            model_stats["calls"] += 1
            model_stats["prompt_tokens"] += usage.prompt_tokens
            model_stats["completion_tokens"] += usage.completion_tokens
            model_stats["total_tokens"] += usage.total_tokens
            model_stats["cost_usd"] += usage.cost_usd
            
            # By operation
            op_stats = by_operation[usage.operation or "unknown"]
            op_stats["calls"] += 1
            op_stats["prompt_tokens"] += usage.prompt_tokens
            op_stats["completion_tokens"] += usage.completion_tokens
            op_stats["total_tokens"] += usage.total_tokens
            op_stats["cost_usd"] += usage.cost_usd
        
        # Round costs
        for stats in by_agent.values():
            stats["cost_usd"] = round(stats["cost_usd"], 6)
        for stats in by_model.values():
            stats["cost_usd"] = round(stats["cost_usd"], 6)
        for stats in by_operation.values():
            stats["cost_usd"] = round(stats["cost_usd"], 6)
        
        return {
            "total_calls": len(self._usage_history),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "by_agent": dict(by_agent),
            "by_model": dict(by_model),
            "by_operation": dict(by_operation)
        }
    
    def print_summary(self):
        """Print a formatted summary of token usage."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("TOKEN USAGE SUMMARY")
        print("=" * 80)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - Prompt: {summary['total_prompt_tokens']:,}")
        print(f"  - Completion: {summary['total_completion_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
        print()
        
        if summary['by_agent']:
            print("By Agent:")
            for agent, stats in sorted(summary['by_agent'].items()):
                print(f"  {agent}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Tokens: {stats['total_tokens']:,} (${stats['cost_usd']:.6f})")
            print()
        
        if summary['by_model']:
            print("By Model:")
            for model, stats in sorted(summary['by_model'].items()):
                print(f"  {model}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Tokens: {stats['total_tokens']:,} (${stats['cost_usd']:.6f})")
            print()
        
        if summary['by_operation']:
            print("By Operation:")
            for op, stats in sorted(summary['by_operation'].items()):
                if op != "unknown":
                    print(f"  {op}:")
                    print(f"    Calls: {stats['calls']}")
                    print(f"    Tokens: {stats['total_tokens']:,} (${stats['cost_usd']:.6f})")
        
        print("=" * 80)
    
    def save_to_file(self, filepath: Optional[str] = None):
        """Save usage history to a JSON file."""
        if filepath is None:
            filepath = "token_usage.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "summary": self.get_summary(),
            "history": [u.to_dict() for u in self._usage_history],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[TokenTracker] 💾 Saved token usage to: {filepath}")
    
    def reset(self):
        """Reset all tracking data."""
        self._usage_history.clear()
        self._session_usage.clear()


# Global singleton instance
tracker = TokenTracker()

