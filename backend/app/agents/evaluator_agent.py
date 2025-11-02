from dotenv import load_dotenv
load_dotenv(override=True)

from autogen_core import (
    SingleThreadedAgentRuntime,
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
)

from app.models.message_model import Message
from app.models.action_plan_models import ActionPlanResponse
from openai import OpenAI
from typing import Dict
import json
import os


class ActionPlanEvaluatorAgent(RoutedAgent):
    """
    Agent that evaluates action plans with single revision capability.
    
    Sequential Architecture:
    1. Evaluate original plan
    2. If REVISE: call Revision once and re-evaluate
    3. Compare versions and select the better one
    4. Return final result (no loops)
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("ActionPlanEvaluator")
        self._runtime = runtime
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
class ActionPlanEvaluatorAgent(RoutedAgent):
    """
    Agent that evaluates action plans and routes based on recommendation.
    
    Sequential Architecture:
    1. Evaluate original plan
    2. Route based on recommendation:
       - APPROVE → return original
       - REVISE → call RevisionAgent (which handles selection)
       - BLOCK → return error
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("ActionPlanEvaluator")
        self._runtime = runtime
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    @message_handler
    async def on_evaluation_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Evaluate action plan and route appropriately.
        """
        try:
            input_data = json.loads(message.content)
            action_plan_json = input_data.get("action_plan")
            govdoc_data = input_data.get("govdoc_data")
            location = input_data.get("location", "Unknown")
            risk_level = "Warning"

            # Check if ActionPlan failed
            if action_plan_json is None or "error" in input_data:
                error_msg = input_data.get("error", "Action plan generation failed")
                print(f"[EvaluatorAgent] ❌ Received error from upstream: {error_msg}")
                return Message(content=json.dumps({
                    "status": "error",
                    "error": f"Cannot evaluate: {error_msg}",
                    "location": location
                }))
            
            original_plan = ActionPlanResponse(**action_plan_json)
            
            print(f"\n[EvaluatorAgent] ═══════════════════════════════════════")
            print(f"[EvaluatorAgent] Evaluating plan for {location}")
            print(f"[EvaluatorAgent] ═══════════════════════════════════════")
            
            # ========== Evaluate Original Plan ==========
            print(f"\n[EvaluatorAgent] 📊 Evaluating original plan...")
            
            coverage_data = self._check_category_coverage(original_plan)
            eval_result = await self._llm_evaluate(original_plan, location, risk_level, coverage_data, ctx)
            eval_result = self._calculate_final_scores(eval_result, coverage_data)
            recommendation = self._determine_recommendation(eval_result)
            eval_result["recommendation"] = recommendation
            
            print(f"[EvaluatorAgent] ✅ Evaluation complete:")
            print(f"[EvaluatorAgent]    Recommendation: {recommendation}")
            print(f"[EvaluatorAgent]    Overall Score: {eval_result['overall_score']:.3f}")
            print(f"[EvaluatorAgent]    Accuracy: {eval_result['accuracy']['score']:.2f}")
            print(f"[EvaluatorAgent]    Clarity: {eval_result['clarity']['score']:.2f}")
            print(f"[EvaluatorAgent]    Completeness: {eval_result['completeness']['score']:.2f}")
            print(f"[EvaluatorAgent]    Relevance: {eval_result['relevance']['score']:.2f}")
            print(f"[EvaluatorAgent]    Coherence: {eval_result['coherence']['score']:.2f}")
            
            # ========== Route Based on Recommendation ==========
            
            if recommendation == "REVISE":
                print(f"\n[EvaluatorAgent] 🔄 Routing to RevisionAgent...")
                
                # Pass to RevisionAgent (which will revise + select)
                revision_request = {
                    "original_plan": action_plan_json,
                    "evaluation": eval_result,
                    "govdoc_data": govdoc_data,
                    "location": location
                }
                
                # RevisionAgent returns final selected plan
                final_response = await self._runtime.send_message(
                    Message(content=json.dumps(revision_request)),
                    AgentId("Revision", "default")
                )
                
                # Add original evaluation to response
                final_data = json.loads(final_response.content)
                final_data["original_evaluation"] = eval_result
                
                return Message(content=json.dumps(final_data, indent=2, ensure_ascii=False))
            
            elif recommendation == "APPROVE":
                print(f"\n[EvaluatorAgent] ✅ Plan approved without revision")
                
                return Message(content=json.dumps({
                    "status": "approved",
                    "selected_version": "original",
                    "final_plan": action_plan_json,
                    "evaluation": eval_result
                }, indent=2, ensure_ascii=False))
            
            else:  # BLOCK
                print(f"\n[EvaluatorAgent] ❌ Plan blocked (quality too low)")
                
                return Message(content=json.dumps({
                    "status": "blocked",
                    "evaluation": eval_result,
                    "reason": "Plan quality below minimum acceptable threshold"
                }, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[EvaluatorAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "status": "error"
            }))
    
    def _print_action_plan(self, plan: ActionPlanResponse, version: str):
        """Print action plan details."""
        print(f"\n[EvaluatorAgent] 📋 {version} ACTION PLAN:")
        print(f"[EvaluatorAgent]    Location: {plan.location}")
        print(f"[EvaluatorAgent]    Total Actions: {plan.total_actions()}")
        print(f"[EvaluatorAgent]    Before Flood: {len(plan.before_flood)} actions")
        print(f"[EvaluatorAgent]    During Flood: {len(plan.during_flood)} actions")
        print(f"[EvaluatorAgent]    After Flood: {len(plan.after_flood)} actions")
        
        # Show sample actions
        if plan.before_flood:
            print(f"[EvaluatorAgent]    Sample Before Actions:")
            for i, action in enumerate(plan.before_flood[:3], 1):
                print(f"[EvaluatorAgent]      {i}. {action.title} ({action.category})")
        
        if plan.during_flood:
            print(f"[EvaluatorAgent]    Sample During Actions:")
            for i, action in enumerate(plan.during_flood[:3], 1):
                print(f"[EvaluatorAgent]      {i}. {action.title} ({action.category})")
        
        if plan.after_flood:
            print(f"[EvaluatorAgent]    Sample After Actions:")
            for i, action in enumerate(plan.after_flood[:3], 1):
                print(f"[EvaluatorAgent]      {i}. {action.title} ({action.category})")
    
    def _compare_versions(
        self, 
        eval_v1: Dict, 
        eval_v2: Dict,
        plan_v1: ActionPlanResponse,
        plan_v2: ActionPlanResponse
    ) -> Dict:
        """
        Compare two versions and determine which is better.
        
        Returns:
        {
            "better_version": "original" or "revised",
            "score_delta": float,
            "improvements": [...],
            "regressions": [...]
        }
        """
        score_v1 = eval_v1['overall_score']
        score_v2 = eval_v2['overall_score']
        score_delta = score_v2 - score_v1
        
        improvements = []
        regressions = []
        
        # Compare dimension scores
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            score1 = eval_v1[dim]['score']
            score2 = eval_v2[dim]['score']
            delta = score2 - score1
            
            if delta > 0.05:  # Significant improvement
                improvements.append(f"{dim.capitalize()} improved by {delta:.2f}")
            elif delta < -0.05:  # Significant regression
                regressions.append(f"{dim.capitalize()} decreased by {abs(delta):.2f}")
        
        # Compare action counts
        if plan_v2.total_actions() > plan_v1.total_actions():
            improvements.append(f"Added {plan_v2.total_actions() - plan_v1.total_actions()} more actions")
        elif plan_v2.total_actions() < plan_v1.total_actions():
            regressions.append(f"Removed {plan_v1.total_actions() - plan_v2.total_actions()} actions")
        
        # Compare phase distribution
        if len(plan_v2.during_flood) > len(plan_v1.during_flood):
            improvements.append(f"Added {len(plan_v2.during_flood) - len(plan_v1.during_flood)} 'during' actions")
        if len(plan_v2.after_flood) > len(plan_v1.after_flood):
            improvements.append(f"Added {len(plan_v2.after_flood) - len(plan_v1.after_flood)} 'after' actions")
        
        # Decision: revised is better if score improved OR equal score but more improvements
        if score_delta > 0.01:
            better_version = "revised"
        elif score_delta < -0.01:
            better_version = "original"
        else:
            # Tie-breaker: more improvements than regressions
            better_version = "revised" if len(improvements) > len(regressions) else "original"
        
        return {
            "better_version": better_version,
            "score_delta": round(score_delta, 3),
            "original_score": round(score_v1, 3),
            "revised_score": round(score_v2, 3),
            "improvements": improvements,
            "regressions": regressions
        }
    
    def _check_category_coverage(self, action_plan: ActionPlanResponse) -> Dict:
        """Check category coverage."""
        all_actions = (action_plan.before_flood + 
                       action_plan.during_flood + 
                       action_plan.after_flood)
        
        categories_found = set()
        for action in all_actions:
            if action.category:
                categories_found.add(action.category.lower().strip())
        
        essential_categories = {
            'evacuation', 'property_protection', 'emergency_kit',
            'communication', 'insurance', 'family_plan'
        }
        
        matched = set()
        for essential in essential_categories:
            for found in categories_found:
                if essential in found or found in essential:
                    matched.add(essential)
                    break
        
        total = len(all_actions)
        before_pct = (len(action_plan.before_flood) / total * 100) if total > 0 else 0
        during_pct = (len(action_plan.during_flood) / total * 100) if total > 0 else 0
        after_pct = (len(action_plan.after_flood) / total * 100) if total > 0 else 0
        
        return {
            'coverage_ratio': len(matched) / len(essential_categories),
            'categories_matched': list(matched),
            'categories_found': list(categories_found),
            'missing_essential': list(essential_categories - matched),
            'total_actions': total,
            'before_count': len(action_plan.before_flood),
            'during_count': len(action_plan.during_flood),
            'after_count': len(action_plan.after_flood),
            'before_pct': round(before_pct, 1),
            'during_pct': round(during_pct, 1),
            'after_pct': round(after_pct, 1),
        }
    
    async def _llm_evaluate(
        self, 
        action_plan: ActionPlanResponse,
        location: str,
        risk_level: str,
        coverage_data: Dict,
        ctx: MessageContext
    ) -> Dict:
        """LLM evaluation."""
        
        prompt = f"""Evaluate this flood action plan for {location}.

SUMMARY:
- Before: {coverage_data['before_count']} ({coverage_data['before_pct']}%)
- During: {coverage_data['during_count']} ({coverage_data['during_pct']}%)
- After: {coverage_data['after_count']} ({coverage_data['after_pct']}%)
- Coverage: {coverage_data['coverage_ratio']:.0%}, Missing: {coverage_data['missing_essential']}

PLAN:
{action_plan.model_dump_json(indent=2, exclude_none=True)}

Evaluate on 5 dimensions (0.0-1.0):

1. ACCURACY: Factually correct for {location}, no hallucinations
2. CLARITY: Plain language, specific details, actionable
3. COMPLETENESS: Category coverage, phase distribution
4. RELEVANCE: Location-specific, risk-aligned
5. COHERENCE: Correct phases, no duplicates, no contradictions

JSON output:
{{
  "accuracy": {{"score": 0.0, "justification": "...", "issues": []}},
  "clarity": {{"score": 0.0, "justification": "...", "issues": []}},
  "completeness": {{"score": 0.0, "justification": "...", "issues": []}},
  "relevance": {{"score": 0.0, "justification": "...", "issues": []}},
  "coherence": {{"score": 0.0, "phase_errors": [], "duplicate_actions": [], "contradictions": [], "justification": "...", "issues": []}}
}}
"""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You evaluate emergency plans objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            return json.loads(content)
        
        except Exception as e:
            print(f"[EvaluatorAgent] ❌ LLM error: {e}")
            return self._fallback_scores()
    
    def _fallback_scores(self) -> Dict:
        """Fallback scores if LLM fails."""
        return {
            "accuracy": {"score": 0.5, "justification": "Evaluation failed", "issues": []},
            "clarity": {"score": 0.5, "justification": "Evaluation failed", "issues": []},
            "completeness": {"score": 0.5, "justification": "Evaluation failed", "issues": []},
            "relevance": {"score": 0.5, "justification": "Evaluation failed", "issues": []},
            "coherence": {"score": 0.5, "justification": "Evaluation failed", "issues": [], 
                         "phase_errors": [], "duplicate_actions": [], "contradictions": []},
        }
    
    def _calculate_final_scores(self, eval_result: Dict, coverage_data: Dict) -> Dict:
        """Calculate weighted overall score."""
        weights = {
            'accuracy': 0.25,
            'clarity': 0.15,
            'completeness': 0.25,
            'relevance': 0.20,
            'coherence': 0.20
        }
        
        overall_score = sum(
            eval_result[dim]['score'] * weight 
            for dim, weight in weights.items()
        )
        
        thresholds = {
            'accuracy': 0.8,
            'clarity': 0.7,
            'completeness': 0.7,
            'relevance': 0.7,
            'coherence': 0.8
        }
        
        passes_threshold = all(
            eval_result[dim]['score'] >= threshold
            for dim, threshold in thresholds.items()
        )
        
        confidence = "high" if overall_score >= 0.85 else "medium" if overall_score >= 0.70 else "low"
        
        eval_result.update({
            'overall_score': round(overall_score, 3),
            'passes_threshold': passes_threshold,
            'overall_confidence': confidence,
            'weights': weights,
            'thresholds': thresholds,
            'coverage_data': coverage_data
        })
        
        return eval_result
    
    def _determine_recommendation(self, eval_result: Dict) -> str:
        """Determine recommendation."""
        if eval_result['accuracy']['score'] < 0.6:
            return "BLOCK"
        
        contradictions = eval_result['coherence'].get('contradictions', [])
        if contradictions:
            text = ' '.join(contradictions).lower()
            if any(kw in text for kw in ['stay', 'evacuate', 'leave', 'remain']):
                return "BLOCK"
        
        if eval_result['passes_threshold'] and eval_result['overall_score'] >= 0.75:
            return "APPROVE"
        
        if eval_result['overall_score'] >= 0.65:
            return "REVISE"
        
        return "BLOCK"


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x