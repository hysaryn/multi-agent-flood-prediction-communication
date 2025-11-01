# from dotenv import load_dotenv 
# load_dotenv(override=True)
# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.messages import TextMessage
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_core import (
#     SingleThreadedAgentRuntime,
#     AgentId,
#     MessageContext,
#     RoutedAgent,
#     message_handler,
# )
# import os
# from openai import OpenAI
# from app.models.message_model import Message
# from app.models.action_plan_models import ActionPlanResponse
# from typing import Dict, List
# import json
# import re

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
from typing import Dict, List
from openai import OpenAI
import json
import re
import os


class ActionPlanEvaluatorAgent(RoutedAgent):
    """
    Agent that evaluates action plans using a 5-dimensional rubric.
    
    Criteria:
    1. Accuracy (0.25 weight, ≥0.8 threshold)
    2. Clarity (0.15 weight, ≥0.7 threshold)
    3. Completeness (0.25 weight, ≥0.7 threshold)
    4. Relevance (0.20 weight, ≥0.7 threshold)
    5. Coherence (0.20 weight, ≥0.8 threshold)
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("ActionPlanEvaluator")
        self._runtime = runtime
        
        # LLM for evaluation
        # self._llm = AssistantAgent(
        #     "EvaluatorLLM",
        #     model_client=OpenAIChatCompletionClient(
        #         model="gpt-4o-mini",
        #     ),
        # )
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    @message_handler
    async def on_evaluation_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Evaluate an action plan using multi-dimensional rubric.
        
        Input: JSON with action_plan, location, risk_level
        Output: JSON with scores, issues, and recommendation
        """
        try:
            # Parse input
            input_data = json.loads(message.content)
            action_plan_json = input_data.get("action_plan")
            location = input_data.get("location", "Unknown")
            risk_level = input_data.get("risk_level", "Unknown")
            
            # Reconstruct ActionPlanResponse object
            action_plan = ActionPlanResponse(**action_plan_json)
            
            print(f"[EvaluatorAgent] Evaluating plan for {location} (risk: {risk_level})")
            
            # Step 1: Programmatic pre-checks
            coverage_data = self._check_category_coverage(action_plan)
            
            # Step 2: LLM-based evaluation
            eval_result = await self._llm_evaluate(
                action_plan, location, risk_level, coverage_data, ctx
            )
            
            # Step 3: Calculate overall scores
            eval_result = self._calculate_final_scores(eval_result, coverage_data)
            
            # Step 4: Determine recommendation
            eval_result["recommendation"] = self._determine_recommendation(eval_result)
            
            print(f"[EvaluatorAgent] ✅ Overall score: {eval_result['overall_score']:.2f}")
            print(f"[EvaluatorAgent] Recommendation: {eval_result['recommendation']}")
            
            return Message(content=json.dumps(eval_result, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[EvaluatorAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "recommendation": "BLOCK"
            }))
    
    def _check_category_coverage(self, action_plan: ActionPlanResponse) -> Dict:
        """
        Programmatic check for category coverage.
        
        Essential categories: evacuation, property_protection, emergency_kit,
                              communication, insurance, family_plan
        """
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
        
        # Fuzzy matching for similar terms
        matched = set()
        for essential in essential_categories:
            for found in categories_found:
                if essential in found or found in essential:
                    matched.add(essential)
                    break
        
        # Phase distribution
        total_actions = len(all_actions)
        before_pct = (len(action_plan.before_flood) / total_actions * 100) if total_actions > 0 else 0
        during_pct = (len(action_plan.during_flood) / total_actions * 100) if total_actions > 0 else 0
        after_pct = (len(action_plan.after_flood) / total_actions * 100) if total_actions > 0 else 0
        
        return {
            'coverage_ratio': len(matched) / len(essential_categories),
            'categories_matched': list(matched),
            'categories_found': list(categories_found),
            'missing_essential': list(essential_categories - matched),
            'total_actions': total_actions,
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
        """
        Single LLM call for all evaluation dimensions.
        Uses OpenAI client directly to avoid autogen compatibility issues.
        """
        prompt = f"""You are evaluating a flood action plan for quality and safety.

    LOCATION: {location}
    RISK LEVEL: {risk_level}
    SOURCE DOCUMENTS: {len(action_plan.sources)} government PDFs

    ACTION PLAN SUMMARY:
    - Before flood: {coverage_data['before_count']} actions ({coverage_data['before_pct']}%)
    - During flood: {coverage_data['during_count']} actions ({coverage_data['during_pct']}%)
    - After flood: {coverage_data['after_count']} actions ({coverage_data['after_pct']}%)

    COVERAGE METRICS (pre-computed):
    - Categories found: {coverage_data['categories_found']}
    - Categories matched: {coverage_data['categories_matched']}
    - Coverage ratio: {coverage_data['coverage_ratio']:.2f}
    - Missing essential: {coverage_data['missing_essential']}

    FULL ACTION PLAN:
    {action_plan.model_dump_json(indent=2, exclude_none=True)}

    ---

    Evaluate this plan across FIVE dimensions. For each, provide:
    1. Numerical score (0.0-1.0)
    2. Brief justification (1-2 sentences)
    3. Specific issues found (list of strings)

    DIMENSION 1: ACCURACY (Weight: 0.25, Threshold: ≥0.8)
    Evaluate factual correctness:
    - All actions appropriate for {location}
    - No invented statistics or numbers
    - Actions derivable from government documents

    Scoring:
    - 1.0: All verifiable, no factual errors
    - 0.8: Minor ambiguity but reasonable
    - 0.6: 1-2 unverifiable claims
    - 0.4: Multiple questionable claims
    - 0.0: Clear hallucinations or wrong location

    ---

    DIMENSION 2: CLARITY (Weight: 0.15, Threshold: ≥0.7)
    Evaluate readability:
    - Plain language, short sentences
    - Action specificity (concrete details)
    - Imperative voice ("Do X", not "It is recommended...")

    Scoring:
    - 1.0: All actions clear, specific, actionable
    - 0.8: Mostly clear, 1-2 vague actions
    - 0.6: Several vague actions
    - 0.4: Majority unclear
    - 0.0: Incomprehensible

    ---

    DIMENSION 3: COMPLETENESS (Weight: 0.25, Threshold: ≥0.7)
    Evaluate coverage:

    Category Coverage:
    - {coverage_data['coverage_ratio']:.2f} ratio ({len(coverage_data['categories_matched'])}/6 essential categories)
    - Missing: {coverage_data['missing_essential']}

    Phase Distribution:
    - Before: {coverage_data['before_pct']}% (expected 50-70%)
    - During: {coverage_data['during_pct']}% (expected 20-30%)
    - After: {coverage_data['after_pct']}% (expected 10-20%)

    Scoring:
    - ≥5/6 essential categories = 1.0
    - 4/6 = 0.8
    - 3/6 = 0.6 (minimum threshold)
    - <3/6 = Fail

    Penalize if any phase deviates >20 percentage points from expected range.

    ---

    DIMENSION 4: RELEVANCE (Weight: 0.20, Threshold: ≥0.7)
    Evaluate localization for {location}:

    Geographic Specificity Scale:
    - 0.0 = Wrong region (e.g., Seattle info for Vancouver)
    - 0.6 = Generic federal (acceptable but not ideal)
    - 0.8 = Province/state specific (e.g., "BC River Forecast Centre")
    - 1.0 = City/municipality specific (e.g., "Vancouver FloodWatch")

    Risk Alignment:
    - Actions should match {risk_level} risk level
    - Not too cautious or too alarmist

    Scoring:
    - 1.0: Highly localized, perfectly aligned
    - 0.8: Good localization or appropriate generic content
    - 0.6: Generic but acceptable given sources
    - 0.4: Poor localization or risk misalignment
    - 0.0: Wrong region or completely irrelevant

    ---

    DIMENSION 5: COHERENCE (Weight: 0.20, Threshold: ≥0.8)
    Evaluate structure and internal consistency:

    1. Phase Logic (40% of coherence score):
    Check if actions are in semantically correct phases:
    
    BEFORE flood should have: insurance, prepare kit, create plans, elevate valuables
    BEFORE should NOT have: evacuate, document damage, file claims
    
    DURING flood should have: evacuate, avoid water, turn off utilities
    DURING should NOT have: purchase insurance, return home, begin repairs
    
    AFTER flood should have: document damage, clean up, inspect property
    AFTER should NOT have: pack emergency kit, plan evacuation routes
    
    List any phase mismatches found.

    2. Deduplication (30% of coherence score):
    Identify semantically duplicate actions:
    
    DUPLICATES (flag these):
    - "Prepare emergency kit" + "Create emergency supplies kit"
    - "Evacuate to safe location" + "Leave home for shelter"
    
    NOT DUPLICATES (these are fine):
    - "Pack emergency kit" + "Store kit in accessible location" (different steps)
    - "Purchase insurance" + "Review insurance policy" (related but distinct)
    
    Calculate: duplicate_rate = number_of_duplicates / total_actions

    3. Internal Consistency (30% of coherence score):
    Check for contradictions:
    - "Stay in your home" vs "Evacuate immediately"
    - "Turn off utilities" vs "Keep power on for sump pump"
    
    List any contradictions found.

    Scoring:
    - 1.0: Perfect structure, no issues
    - 0.8: 1-2 minor phase mismatches or <5% duplicates
    - 0.6: Several issues but no contradictions
    - 0.4: Multiple issues or 1 contradiction
    - 0.0: Severe structural problems or dangerous contradictions

    ---

    OUTPUT FORMAT (JSON only, no markdown, no code blocks):
    {{
    "accuracy": {{
        "score": 0.85,
        "justification": "Brief explanation here",
        "issues": ["specific issue 1", "specific issue 2"]
    }},
    "clarity": {{
        "score": 0.90,
        "justification": "Brief explanation here",
        "issues": []
    }},
    "completeness": {{
        "score": 0.75,
        "justification": "Brief explanation here",
        "issues": ["missing X category"]
    }},
    "relevance": {{
        "score": 0.80,
        "justification": "Brief explanation here",
        "issues": []
    }},
    "coherence": {{
        "score": 0.85,
        "phase_errors": ["action X in wrong phase"],
        "duplicate_actions": ["action Y and Z are duplicates"],
        "contradictions": [],
        "justification": "Brief explanation here",
        "issues": []
    }}
    }}
    """
        
        try:
            # 直接调用 OpenAI API
            print(f"[EvaluatorAgent] Calling OpenAI API ({self._model})...")
            
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert evaluator of emergency preparedness plans. You provide detailed, objective assessments using structured rubrics."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            print(f"[EvaluatorAgent] Received response ({len(content)} chars)")
            
            # Clean up markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            # Parse JSON
            eval_data = json.loads(content)
            print(f"[EvaluatorAgent] ✅ Successfully parsed evaluation")
            
            return eval_data
        
        except json.JSONDecodeError as e:
            print(f"[EvaluatorAgent] ❌ JSON parse error: {e}")
            print(f"[EvaluatorAgent] LLM response preview:\n{content[:800]}")
            return self._fallback_scores()
        except Exception as e:
            print(f"[EvaluatorAgent] ❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_scores()
    
    def _fallback_scores(self) -> Dict:
        """Return conservative fallback scores if LLM evaluation fails."""
        return {
            "accuracy": {"score": 0.5, "justification": "Evaluation failed", "issues": ["LLM evaluation error"]},
            "clarity": {"score": 0.5, "justification": "Evaluation failed", "issues": ["LLM evaluation error"]},
            "completeness": {"score": 0.5, "justification": "Evaluation failed", "issues": ["LLM evaluation error"]},
            "relevance": {"score": 0.5, "justification": "Evaluation failed", "issues": ["LLM evaluation error"]},
            "coherence": {"score": 0.5, "justification": "Evaluation failed", "issues": ["LLM evaluation error"], 
                         "phase_errors": [], "duplicate_actions": [], "contradictions": []},
        }
    
    def _calculate_final_scores(self, eval_result: Dict, coverage_data: Dict) -> Dict:
        """Calculate weighted overall score and add metadata."""
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
        
        # Check if all thresholds met
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
        
        # Determine confidence
        if overall_score >= 0.85:
            confidence = "high"
        elif overall_score >= 0.70:
            confidence = "medium"
        else:
            confidence = "low"
        
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
        """
        Determine final recommendation: APPROVE, REVISE, or BLOCK.
        
        Rules:
        - BLOCK: Accuracy <0.6 or safety contradictions
        - APPROVE: All thresholds met + overall ≥0.75
        - REVISE: Fixable issues, overall ≥0.65
        - BLOCK: Too many issues
        """
        # Critical failures (hard gates)
        if eval_result['accuracy']['score'] < 0.6:
            return "BLOCK"  # Hallucinations or wrong location
        
        # Check for dangerous contradictions
        contradictions = eval_result['coherence'].get('contradictions', [])
        dangerous_keywords = ['stay', 'evacuate', 'leave', 'remain']
        if contradictions:
            contradiction_text = ' '.join(contradictions).lower()
            if any(kw in contradiction_text for kw in dangerous_keywords):
                return "BLOCK"  # Safety contradiction
        
        # All thresholds met
        if eval_result['passes_threshold'] and eval_result['overall_score'] >= 0.75:
            return "APPROVE"
        
        # Fixable issues
        if eval_result['overall_score'] >= 0.65:
            return "REVISE"
        
        # Too many issues
        return "BLOCK"


# ---------------------------------------------------------
# Testing helper
# ---------------------------------------------------------

async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


async def main():
    """Test the ActionPlanEvaluatorAgent"""
    import asyncio
    from app.models.action_plan_models import Action
    
    runtime = SingleThreadedAgentRuntime()
    
    # Register evaluator agent
    await ActionPlanEvaluatorAgent.register(
        runtime, "ActionPlanEvaluator", 
        lambda: ActionPlanEvaluatorAgent(runtime)
    )
    
    await maybe_await(runtime.start())
    
    # Create a test action plan
    test_plan = ActionPlanResponse(
        location="Vancouver, BC",
        display_name="Vancouver, British Columbia",
        before_flood=[
            Action(title="Prepare 72-hour emergency kit", 
                   description="Pack water (1 gallon per person per day), non-perishable food, flashlight, battery radio, first aid kit, medications, and important documents in waterproof container",
                   priority="high",
                   category="emergency_kit",
                   source_doc="https://example.ca/guide.pdf"),
            Action(title="Review flood insurance coverage",
                   description="Contact insurance provider to verify flood coverage limits and deductibles",
                   priority="high",
                   category="insurance",
                   source_doc="https://example.ca/guide.pdf"),
        ],
        during_flood=[
            Action(title="Evacuate if ordered",
                   description="Follow evacuation routes to designated shelter, bring emergency kit",
                   priority="high",
                   category="evacuation",
                   source_doc="https://example.ca/guide.pdf"),
        ],
        after_flood=[
            Action(title="Document property damage",
                   description="Take photos and videos of all damage for insurance claims",
                   priority="high",
                   category="insurance",
                   source_doc="https://example.ca/guide.pdf"),
        ],
        sources=["https://example.ca/guide.pdf"],
        generated_at="2025-01-01T00:00:00Z"
    )
    
    # Test evaluation
    print("=" * 60)
    print("Testing ActionPlanEvaluatorAgent")
    print("=" * 60)
    
    eval_request = {
        "action_plan": test_plan.model_dump(mode='json'),
        "location": "Vancouver, BC",
        "risk_level": "Warning"
    }
    
    response = await runtime.send_message(
        Message(content=json.dumps(eval_request)),
        AgentId("ActionPlanEvaluator", "default")
    )
    
    # Print result
    result = json.loads(response.content)
    print("\n" + "=" * 60)
    print("EVALUATION RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    await maybe_await(runtime.stop())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())