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
from app.services.token_tracker import tracker
from openai import OpenAI
from typing import Dict, List
import json
import os
import re
from datetime import datetime
from pathlib import Path


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
        Evaluate action plan and route based on 2-iteration feedback loop.
        """
        start_time = datetime.now()
        try:
            input_data = json.loads(message.content)
            action_plan_json = input_data.get("action_plan")
            govdoc_data = input_data.get("govdoc_data")
            location = input_data.get("location", "Unknown")
            iteration = input_data.get("iteration", 1)
            risk_level = "Warning"
            
            print(f"[EvaluatorAgent] ⏰ Evaluation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Check if ActionPlan failed
            if action_plan_json is None or "error" in input_data:
                error_msg = input_data.get("error", "Action plan generation failed")
                print(f"[EvaluatorAgent] ❌ Received error from upstream: {error_msg}")
                return Message(content=json.dumps({
                    "status": "error",
                    "error": f"Cannot evaluate: {error_msg}",
                    "location": location,
                    "iteration": iteration
                }))
            
            plan = ActionPlanResponse(**action_plan_json)
            
            print(f"\n[EvaluatorAgent] ═══════════════════════════════════════")
            print(f"[EvaluatorAgent] Evaluating plan for {location} (Iteration {iteration})")
            print(f"[EvaluatorAgent] ═══════════════════════════════════════")
            
            # ========== Evaluate Plan ==========
            print(f"\n[EvaluatorAgent] 📊 Evaluating plan...")
            
            eval_result = await self._llm_evaluate(plan, location, risk_level, ctx)
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
            
            eval_duration = (datetime.now() - start_time).total_seconds()
            print(f"[EvaluatorAgent] ⏱️  Evaluation took {eval_duration:.2f}s")
            
            # Print token usage summary for this agent
            summary = tracker.get_summary()
            agent_summary = summary.get("by_agent", {}).get("EvaluatorAgent", {})
            if agent_summary:
                print(f"[EvaluatorAgent] 💰 Token usage: {agent_summary.get('total_tokens', 0):,} tokens (${agent_summary.get('cost_usd', 0):.6f})")
            
            # Print iteration summary with action counts
            print(f"\n[EvaluatorAgent] {'='*50}")
            print(f"[EvaluatorAgent] 📊 ITERATION {iteration} SUMMARY")
            print(f"[EvaluatorAgent] {'='*50}")
            print(f"[EvaluatorAgent] ⏱️  Total Time: {eval_duration:.2f}s")
            print(f"[EvaluatorAgent] 📋 Action Counts:")
            print(f"[EvaluatorAgent]    Before Flood: {len(plan.before_flood)} actions")
            print(f"[EvaluatorAgent]    During Flood: {len(plan.during_flood)} actions")
            print(f"[EvaluatorAgent]    After Flood: {len(plan.after_flood)} actions")
            print(f"[EvaluatorAgent]    Total: {plan.total_actions()} actions")
            print(f"[EvaluatorAgent] {'='*50}\n")
            
            # Save iteration result
            self._save_iteration_result(iteration, location, action_plan_json, eval_result, govdoc_data)
            
            # ========== Route Based on Recommendation and Iteration ==========
            
            if recommendation == "APPROVE":
                print(f"\n[EvaluatorAgent] ✅ Plan approved (Iteration {iteration})")
                
                return Message(content=json.dumps({
                    "status": "approved",
                    "final_plan": action_plan_json,
                    "evaluation": eval_result,
                    "iteration": iteration,
                    "total_iterations": iteration,
                    "action_counts": {
                        "before_flood": len(plan.before_flood),
                        "during_flood": len(plan.during_flood),
                        "after_flood": len(plan.after_flood),
                        "total": plan.total_actions()
                    }
                }, indent=2, ensure_ascii=False))
            
            elif recommendation == "REVISE":
                # Safety check: prevent infinite loops (max 2 iterations)
                if iteration >= 2:
                    # Final iteration reached, return even if not approved
                    print(f"\n[EvaluatorAgent] ⚠️  Max iterations ({iteration}) reached, returning result")
                    
                    return Message(content=json.dumps({
                        "status": "revised",
                        "final_plan": action_plan_json,
                        "evaluation": eval_result,
                        "iteration": iteration,
                        "total_iterations": iteration,
                        "note": "Plan revised but may not meet all quality thresholds",
                        "action_counts": {
                            "before_flood": len(plan.before_flood),
                            "during_flood": len(plan.during_flood),
                            "after_flood": len(plan.after_flood),
                            "total": plan.total_actions()
                        }
                    }, indent=2, ensure_ascii=False))
                
                if iteration == 1:
                    # Iteration 2: Decide between REPHRASE or SEARCH_ADDITIONAL
                    print(f"\n[EvaluatorAgent] 🔄 Iteration 2: Deciding revision strategy...")
                    
                    # Build revision notes from evaluation
                    revision_notes = []
                    for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
                        if dim in eval_result:
                            issues = eval_result[dim].get('issues', [])
                            if issues:
                                revision_notes.extend(issues)
                    
                    # Decide strategy: REPHRASE or SEARCH_ADDITIONAL
                    strategy = await self._decide_revision_strategy(
                        plan, eval_result, revision_notes, ctx
                    )
                    
                    print(f"[EvaluatorAgent] 📋 Decision: {strategy}")
                    
                    if strategy == "REPHRASE":
                        # Option A: Just rephrase using current documents
                        print(f"[EvaluatorAgent] 🔄 Strategy: REPHRASE - Sending to ActionPlan for revision...")
                        
                        revision_request = {
                            "govdoc_data": govdoc_data,
                            "revision_notes": revision_notes,
                            "previous_plan": action_plan_json,
                            "evaluation": eval_result,
                            "location": location,
                            "iteration": 2,
                            "mode": "rephrase"  # Explicit mode
                        }
                        
                        # ActionPlan returns revised plan
                        revised_response = await self._runtime.send_message(
                            Message(content=json.dumps(revision_request)),
                            AgentId("ActionPlan", "default")
                        )
                        
                        # Re-evaluate revised plan (feedback loop)
                        return await self.on_evaluation_request(revised_response, ctx)
                    
                    else:  # SEARCH_ADDITIONAL
                        # Option B: Search for additional documents and add to existing plan
                        print(f"[EvaluatorAgent] 🔄 Strategy: SEARCH_ADDITIONAL - Requesting additional documents from GovDoc...")
                        
                        # Get previous URLs for reference
                        previous_urls = govdoc_data.get("previous_urls", [])
                        
                        search_additional_request = {
                            "location": location,
                            "iteration": 2,
                            "previous_urls": previous_urls,
                            "evaluation": eval_result,
                            "revision_notes": revision_notes,
                            "previous_plan": action_plan_json,  # Keep original plan
                            "add_to_existing": True,  # Flag to add to existing plan
                            "govdoc_data": govdoc_data  # Pass existing docs context
                        }
                        
                        # GovDoc searches for additional docs, then ActionPlan adds to existing plan
                        final_response = await self._runtime.send_message(
                            Message(content=json.dumps(search_additional_request)),
                            AgentId("GovDoc", "default")
                        )
                        
                        return final_response
            
            else:  # BLOCK
                print(f"\n[EvaluatorAgent] ❌ Plan blocked (quality too low)")
                
                return Message(content=json.dumps({
                    "status": "blocked",
                    "evaluation": eval_result,
                    "iteration": iteration,
                    "total_iterations": iteration,
                    "reason": "Plan quality below minimum acceptable threshold",
                    "action_counts": {
                        "before_flood": len(plan.before_flood),
                        "during_flood": len(plan.during_flood),
                        "after_flood": len(plan.after_flood),
                        "total": plan.total_actions()
                    }
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

SUMMARY:
- Before: {len(action_plan.before_flood)} actions
- During: {len(action_plan.during_flood)} actions
- After: {len(action_plan.after_flood)} actions
- Total: {action_plan.total_actions()} actions

PLAN:
{action_plan.model_dump_json(indent=2, exclude_none=True)}

EVALUATION CRITERIA:
"Dimension 1: ACCURACY
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
            
            # Track token usage
            if hasattr(response, 'usage'):
                tracker.record_usage(
                    agent_name="EvaluatorAgent",
                    model=self._model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    operation="evaluate_plan"
                )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                # Also remove json code block marker if present
                if content.startswith("json"):
                    content = content[4:].strip()
            
            # Try to parse JSON, with better error handling
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[EvaluatorAgent] ⚠️  JSON parse error: {e}")
                print(f"[EvaluatorAgent] ⚠️  Content preview (first 500 chars): {content[:500]}")
                # Try to extract JSON from the content if it's embedded in text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except:
                        pass
                # If all else fails, return fallback scores
                print(f"[EvaluatorAgent] ❌ Failed to parse JSON, using fallback scores")
                return self._fallback_scores()
        
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
        
        confidence = "high" if overall_score >= 0.85 else "medium" if overall_score >= 0.70 else "low"
        
        eval_result.update({
            'overall_score': round(overall_score, 3),
            'passes_threshold': passes_threshold,
            'overall_confidence': confidence,
            'weights': weights,
            'thresholds': thresholds
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
    
    async def _decide_revision_strategy(
        self,
        plan: ActionPlanResponse,
        eval_result: Dict,
        revision_notes: List[str],
        ctx: MessageContext
    ) -> str:
        """
        Decide whether to REPHRASE (using current docs) or SEARCH_ADDITIONAL (find new docs).
        
        Returns:
            "REPHRASE" or "SEARCH_ADDITIONAL"
        """
        # Build summary of issues
        issues_summary = "\n".join(f"- {note}" for note in revision_notes[:15])  # Limit length
        
        prompt = f"""You are deciding how to improve a flood action plan that failed evaluation.

CURRENT PLAN SUMMARY:
- Location: {plan.location}
- Total Actions: {plan.total_actions()}
- Before: {len(plan.before_flood)}, During: {len(plan.during_flood)}, After: {len(plan.after_flood)}

EVALUATION ISSUES:
{issues_summary}

EVALUATION SCORES:
- Accuracy: {eval_result['accuracy']['score']:.2f}
- Clarity: {eval_result['clarity']['score']:.2f}
- Completeness: {eval_result['completeness']['score']:.2f}
- Relevance: {eval_result['relevance']['score']:.2f}
- Coherence: {eval_result['coherence']['score']:.2f}
- Overall: {eval_result['overall_score']:.2f}

DECISION CRITERIA:
Choose REPHRASE if:
- Issues are mainly about wording, clarity, or phrasing (e.g., "use plain language", "make more specific")
- Issues are about organization or structure (e.g., "duplicates", "phase errors", "contradictions")
- Missing information can be inferred or rephrased from existing content
- The plan has enough content but needs better presentation

Choose SEARCH_ADDITIONAL if:
- Issues indicate missing essential information (e.g., "missing categories", "incomplete coverage")
- Issues are about accuracy or relevance that require location-specific documents
- The plan lacks critical details that cannot be inferred from current documents
- Missing categories cannot be addressed with current documents

Return ONLY a JSON object:
{{
  "strategy": "REPHRASE" or "SEARCH_ADDITIONAL",
  "reasoning": "Brief explanation (1-2 sentences)"
}}
"""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You decide the best strategy to improve action plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                tracker.record_usage(
                    agent_name="EvaluatorAgent",
                    model=self._model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    operation="decide_revision_strategy"
                )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                # Also remove json code block marker if present
                if content.startswith("json"):
                    content = content[4:].strip()
            
            # Try to parse JSON, with better error handling
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[EvaluatorAgent] ⚠️  JSON parse error in strategy decision: {e}")
                print(f"[EvaluatorAgent] ⚠️  Content preview (first 500 chars): {content[:500]}")
                # Try to extract JSON from the content if it's embedded in text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except:
                        print(f"[EvaluatorAgent] ❌ Failed to parse JSON, defaulting to REPHRASE")
                        result = {"strategy": "REPHRASE", "reasoning": "JSON parse error"}
                else:
                    print(f"[EvaluatorAgent] ❌ Failed to parse JSON, defaulting to REPHRASE")
                    result = {"strategy": "REPHRASE", "reasoning": "JSON parse error"}
            
            strategy = result.get("strategy", "REPHRASE").upper()
            reasoning = result.get("reasoning", "")
            
            print(f"[EvaluatorAgent] 💭 Strategy reasoning: {reasoning}")
            
            # Validate strategy
            if strategy not in ["REPHRASE", "SEARCH_ADDITIONAL"]:
                print(f"[EvaluatorAgent] ⚠️  Invalid strategy '{strategy}', defaulting to REPHRASE")
                strategy = "REPHRASE"
            
            return strategy
        
        except Exception as e:
            print(f"[EvaluatorAgent] ❌ Error deciding strategy: {e}")
            # Default to REPHRASE if decision fails
            return "REPHRASE"
    
    def _save_iteration_result(self, iteration: int, location: str, action_plan_json: Dict, eval_result: Dict, govdoc_data: Dict):
        """Save iteration result to a JSON file for review."""
        try:
            # Create results directory if it doesn't exist (relative to backend directory)
            # This assumes the script is run from the backend directory
            results_dir = Path("iteration_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize location for filename
            location_safe = location.replace(",", "_").replace(" ", "_").replace("/", "_")[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"iteration_{iteration}_{location_safe}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Extract action counts from action plan
            plan = ActionPlanResponse(**action_plan_json)
            
            # Prepare result data
            result_data = {
                "iteration": iteration,
                "location": location,
                "timestamp": datetime.now().isoformat(),
                "action_plan": action_plan_json,
                "evaluation": eval_result,
                "action_counts": {
                    "before_flood": len(plan.before_flood),
                    "during_flood": len(plan.during_flood),
                    "after_flood": len(plan.after_flood),
                    "total": plan.total_actions()
                },
                "govdoc_summary": {
                    "num_docs": len(govdoc_data.get("docs", [])),
                    "doc_urls": [doc.get("url") for doc in govdoc_data.get("docs", [])],
                    "doc_titles": [doc.get("title") for doc in govdoc_data.get("docs", [])]
                }
            }
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"[EvaluatorAgent] 💾 Saved iteration {iteration} result to: {filepath}")
            
        except Exception as e:
            print(f"[EvaluatorAgent] ⚠️  Failed to save iteration result: {e}")


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x