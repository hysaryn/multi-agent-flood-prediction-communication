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
from app.services.cost_tracker import get_cost_tracker
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

            pipeline_start_time = input_data.get("_pipeline_start_time")

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
            
            eval_result = await self._llm_evaluate(original_plan, location, risk_level, ctx)
            eval_result = self._calculate_final_scores(eval_result)
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
                    "location": location,
                    "_pipeline_start_time": pipeline_start_time
                }
                
                # RevisionAgent returns final selected plan
                final_response = await self._runtime.send_message(
                    Message(content=json.dumps(revision_request)),
                    AgentId("Revision", "default")
                )

                # Add original evaluation to response
                final_data = json.loads(final_response.content)
                final_data["original_evaluation"] = eval_result

                from datetime import datetime, timezone
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                 
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[EvaluatorAgent] ⚠️ Could not calculate duration: {e}")
                if "final_plan" in final_data and isinstance(final_data["final_plan"], dict):
                    final_data["final_plan"]["pipeline_start_time"] = pipeline_start_time
                    final_data["final_plan"]["pipeline_end_time"] = pipeline_end_time
                    final_data["final_plan"]["total_processing_seconds"] = total_processing_seconds
                
                if total_processing_seconds:
                    print(f"[EvaluatorAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                final_data["cost_summary"] = cost_summary
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps(final_data, indent=2, ensure_ascii=False))
            
            elif recommendation == "APPROVE":
                print(f"\n[EvaluatorAgent] ✅ Plan approved without revision")
                
                # Calculate timestamps for APPROVE case
                from datetime import datetime, timezone
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[EvaluatorAgent] ⚠️ Could not calculate duration: {e}")
                
                # Add timestamps to final_plan
                final_plan_with_timestamps = action_plan_json.copy()
                if isinstance(final_plan_with_timestamps, dict):
                    final_plan_with_timestamps["pipeline_start_time"] = pipeline_start_time
                    final_plan_with_timestamps["pipeline_end_time"] = pipeline_end_time
                    final_plan_with_timestamps["total_processing_seconds"] = total_processing_seconds
                
                if total_processing_seconds:
                    print(f"[EvaluatorAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps({
                    "status": "approved",
                    "selected_version": "original",
                    "final_plan": final_plan_with_timestamps,
                    "evaluation": eval_result,
                    "cost_summary": cost_summary
                }, indent=2, ensure_ascii=False))
            
            else:  # BLOCK
                print(f"\n[EvaluatorAgent] ❌ Plan blocked (quality too low)")
                
                # Calculate timestamps for BLOCK case
                from datetime import datetime, timezone
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[EvaluatorAgent] ⚠️ Could not calculate duration: {e}")
                
                if total_processing_seconds:
                    print(f"[EvaluatorAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps({
                    "status": "blocked",
                    "evaluation": eval_result,
                    "reason": "Plan quality below minimum acceptable threshold",
                    "pipeline_start_time": pipeline_start_time,
                    "pipeline_end_time": pipeline_end_time,
                    "total_processing_seconds": total_processing_seconds,
                    "cost_summary": cost_summary
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
    
    async def _llm_evaluate(
        self, 
        action_plan: ActionPlanResponse,
        location: str,
        risk_level: str,
        ctx: MessageContext
    ) -> Dict:
        """LLM evaluation."""
        
        prompt = f"""Evaluate this flood action plan for {location}.

PLAN:
{action_plan.model_dump_json(indent=2, exclude_none=True)}

"EVALUATION CRITERIA:\n\n"
Dimension 1: ACCURACY
Definition: "Is the information factually correct and verifiable?"
Questions:
Do the sources mentioned seem authoritative?
Do the contact numbers/websites seem legitimate?
Are the evacuation routes mentioned realistic for this area?
QUICK SCORES
5 - All sources cited, verified official
4 - Most information appears official; mostly trustworthy 
3 - Some sources, some uncertainty
2 - Hard to verify; low confidence in accuracy 
1 - Not trustworthy; information seems fabricated

Dimension 2: CLARITY
Definition: Is the guidance easy to understand for someone without emergency training?
Questions:
The language is easy to understand
The steps are presented in a logical order
Technical jargon is minimized (Ask participant to underline any words/phrases they found confusing)
Example:
✅Move valuables to 2nd floor(specific action + location; no jargon) 
❌Relocate personal effects to upper-floor proximity (vague, formal, hard to parse)
QUICK SCORES
5 - 0 confusing words; reads smoothly 
4 - 1 confusing word; minor hesitation 
3 - 2~3 confusing words; re-reading needed 
2 - more than 3 confusing words; repeated clarification 
1 - Incomprehensible; unable to understand; language is too technical

Dimension 3: COMPLETENESS
Definition: Does the guidance cover all essential preparedness categories?
Show Participant: Checklist of categories:
□ Communication (how to get alerts)
□ Evacuation (where to go, routes)
□ Property Protection (protecting belongings)
□ Insurance/Financial (recovery support) *
□ Family Planning (staying together, communication)
□ Emergency Kit (supplies, essentials)
Questions:
Which of these categories are covered in the guidance?
Are there any important categories missing? (Open-ended)
QUICK SCORES
5 - Covers all categories (alerts, evacuation, property, insurance, family, supplies) 
4 - Covers most categories; missing one 
3 - Covers roughly half the important categories 
2 - Missing several important topics 
1 - Severely incomplete; critical information missing

Dimension 4: RELEVANCE
Definition: Is the guidance specific to your location?
Questions:
The guidance mentions your specific location/neighborhood
The guidance is tailored to your area's flood risks
The recommended routes make sense for where you live
QUICK SCORES
5 = 4+ location-specific details (mentions of: neighborhood names, streets, local agencies); not transferable
4 = 2-3 specific details; mostly relevant 
3 = 1 specific detail; mix of generic/specific 
2 = Minimal specificity; mostly generic 
1 = Zero specificity; completely generic

Dimension 5: COHERENCE
Definition: Is the guidance internally consistent and logical?
Questions:
The guidance flows logically (before → during → after)
There are no contradictions in the instructions
The timeline/deadlines make sense together
Test: Ask participant to identify any contradictions they notice(open-end)
QUICK SCORES
5 - Perfect flow; before→during→after makes sense 
4 - Mostly logical; one confusing element 
3 - Some confusion; 2-3 contradictions 
2 - Confusing flow; multiple contradictions 
1 - Makes no sense; severe contradictions 

JSON output (return ONLY this JSON, no markdown, no preamble):
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
                temperature=0.0,  # Set to 0.0 for maximum consistency
                max_tokens=2000
            )
            
            # Track cost and check for truncation
            if hasattr(response, 'usage') and response.usage:
                completion_tokens = response.usage.completion_tokens
                max_tokens_set = 2000
                
                # Warn if approaching max_tokens limit (possible truncation)
                if completion_tokens >= max_tokens_set * 0.9:  # Used 90% or more
                    print(f"[EvaluatorAgent] ⚠️  High token usage: {completion_tokens}/{max_tokens_set} tokens used")
                    print(f"[EvaluatorAgent]    Consider increasing max_tokens if output is truncated")
                
                get_cost_tracker().record_usage(
                    agent_name="EvaluatorAgent",
                    operation="evaluate_plan",
                    model=self._model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            # Try to parse JSON, catch truncation errors
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[EvaluatorAgent] ❌ JSON parse error (possible truncation): {e}")
                print(f"[EvaluatorAgent]    Content length: {len(content)} chars")
                print(f"[EvaluatorAgent]    Content preview: {content[:200]}...")
                # Re-raise to trigger fallback
                raise
        
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
    
    def _calculate_final_scores(self, eval_result: Dict) -> Dict:
        """Calculate weighted overall score."""
        weights = {
            'accuracy': 0.25,
            'clarity': 0.15,
            'completeness': 0.20,
            'relevance': 0.20,
            'coherence': 0.20
        }
        
        overall_score = sum(
            eval_result[dim]['score'] * weight 
            for dim, weight in weights.items()
        )
        
        thresholds = {
            'accuracy': 4,
            'clarity': 3.5,
            'completeness': 3.5,
            'relevance': 3.5,
            'coherence': 4
        }
        
        passes_threshold = all(
            eval_result[dim]['score'] >= threshold
            for dim, threshold in thresholds.items()
        )
        
        confidence = "high" if overall_score >= 4.25 else "medium" if overall_score >= 3.5 else "low"
        
        eval_result.update({
            'overall_score': round(overall_score, 3),
            'passes_threshold': passes_threshold,
            'overall_confidence': confidence,
            'weights': weights,
            'thresholds': thresholds,
        })
        
        return eval_result
    
    def _determine_recommendation(self, eval_result: Dict) -> str:
        """Determine recommendation."""
        if eval_result['accuracy']['score'] < 3.0:
            return "BLOCK"
        
        contradictions = eval_result['coherence'].get('contradictions', [])
        if contradictions:
            text = ' '.join(contradictions).lower()
            if any(kw in text for kw in ['stay', 'evacuate', 'leave', 'remain']):
                return "BLOCK"
        
        if eval_result['passes_threshold'] and eval_result['overall_score'] >= 3.75:
            return "APPROVE"
        
        if eval_result['overall_score'] >= 3.65:
            return "REVISE"
        
        return "BLOCK"


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x