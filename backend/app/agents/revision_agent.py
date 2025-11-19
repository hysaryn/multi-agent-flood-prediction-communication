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
from app.models.action_plan_models import Action, ActionPlanResponse
from openai import OpenAI
from pathlib import Path
from typing import List, Dict
import json
import os


class RevisionAgent(RoutedAgent):
    """
    Agent that revises, re-evaluates, and selects the better version.
    
    Sequential Architecture:
    1. Receive original plan + evaluation
    2. Perform targeted revisions
    3. Re-evaluate revised plan
    4. Compare and select better version
    5. Return final selected plan (ends pipeline)
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("Revision")
        self._runtime = runtime
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    @message_handler
    async def on_revision_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Revise, compare, and return the better version.
        """
        try:
            input_data = json.loads(message.content)
            original_plan_json = input_data.get("original_plan")
            original_evaluation = input_data.get("evaluation")
            govdoc_data = input_data.get("govdoc_data")
            location = input_data.get("location")
            
            print(f"\n[RevisionAgent] ═══════════════════════════════════════")
            print(f"[RevisionAgent] Starting revision for {location}")
            print(f"[RevisionAgent] ═══════════════════════════════════════")
            
            original_plan = ActionPlanResponse(**original_plan_json)
            
            # ========== Step 1: Perform Revisions ==========
            print(f"\n[RevisionAgent] 🔧 Step 1: Performing targeted revisions...")
            
            changes_made = []
            revised_plan = original_plan.model_copy(deep=True)
            
            # Strategy 1: Add missing categories
            coverage_data = original_evaluation.get('coverage_data', {})
            missing = coverage_data.get('missing_essential', [])
            if missing:
                print(f"[RevisionAgent]   Strategy 1: Adding missing categories: {missing}")
                new_actions = await self._add_missing_categories(missing, govdoc_data, location, ctx)
                
                if new_actions:
                    for action in new_actions:
                        if action.category in ["insurance", "emergency_kit", "property_protection", "family_plan"]:
                            revised_plan.before_flood.append(action)
                        elif action.category == "evacuation":
                            revised_plan.during_flood.append(action)
                        else:
                            revised_plan.before_flood.append(action)
                    
                    changes_made.append(f"Added {len(new_actions)} actions for {', '.join(missing)}")
            
            # Strategy 2: Remove duplicates
            coherence = original_evaluation.get('coherence', {})
            if coherence.get('duplicate_actions'):
                print(f"[RevisionAgent]   Strategy 2: Removing duplicates")
                original_count = revised_plan.total_actions()
                revised_plan = self._remove_duplicates(revised_plan, coherence)
                removed = original_count - revised_plan.total_actions()
                if removed > 0:
                    changes_made.append(f"Removed {removed} duplicates")
            
            # Strategy 3: Fix phase errors
            if coherence.get('phase_errors'):
                print(f"[RevisionAgent]   Strategy 3: Fixing phase errors")
                revised_plan = self._fix_phase_errors(revised_plan, coherence)
                changes_made.append("Fixed phase errors")
            
            # Strategy 4: Enhance clarity
            clarity_score = original_evaluation.get('clarity', {}).get('score', 1.0)
            if clarity_score < 0.7:
                print(f"[RevisionAgent]   Strategy 4: Enhancing clarity")
                count = await self._enhance_clarity(revised_plan, location, ctx)
                if count > 0:
                    changes_made.append(f"Enhanced {count} actions")
            
            print(f"[RevisionAgent] ✅ Revision complete: {len(changes_made)} strategies applied")
            for i, change in enumerate(changes_made, 1):
                print(f"[RevisionAgent]      {i}. {change}")
            
            # ========== Step 2: Re-evaluate Revised Plan ==========
            print(f"\n[RevisionAgent] 📊 Step 2: Re-evaluating revised plan...")
            
            revised_plan_json = revised_plan.model_dump(mode='python')
            coverage_data_v2 = self._check_category_coverage(revised_plan)
            revised_evaluation = await self._llm_evaluate(
                revised_plan, location, "Warning", coverage_data_v2, ctx
            )
            revised_evaluation = self._calculate_final_scores(revised_evaluation, coverage_data_v2)
            recommendation_v2 = self._determine_recommendation(revised_evaluation)
            revised_evaluation["recommendation"] = recommendation_v2
            
            print(f"[RevisionAgent] ✅ Re-evaluation complete:")
            print(f"[RevisionAgent]    Recommendation: {recommendation_v2}")
            print(f"[RevisionAgent]    Overall Score: {revised_evaluation['overall_score']:.3f}")
            
            # ========== Step 3: Compare and Select ==========
            print(f"\n[RevisionAgent] ⚖️  Step 3: Comparing versions...")
            
            comparison = self._compare_versions(
                original_evaluation, revised_evaluation,
                original_plan, revised_plan
            )
            
            print(f"[RevisionAgent] 📊 Comparison Results:")
            print(f"[RevisionAgent]    Better Version: {comparison['better_version'].upper()}")
            
            # Select better version
            if comparison['better_version'] == "revised":
                final_plan = revised_plan_json
                final_evaluation = revised_evaluation
                selected_version = "revised"
            else:
                final_plan = original_plan_json
                final_evaluation = original_evaluation
                selected_version = "original"
            
            status = "approved" if final_evaluation["recommendation"] == "APPROVE" else "needs_improvement"
            
            print(f"[RevisionAgent] ✅ Selected: {selected_version.upper()}")
            
            return Message(content=json.dumps({
                "status": status,
                "selected_version": selected_version,
                "final_plan": final_plan,
                "comparison": comparison,
                "original_evaluation": original_evaluation,
                "revised_evaluation": revised_evaluation,
                "changes_made": changes_made
            }, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[RevisionAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "status": "error"
            }))
    
    # ========== Helper Methods ==========
    
    async def _add_missing_categories(
        self, missing: List[str], govdoc_data: dict, location: str, ctx: MessageContext
    ) -> List[Action]:
        """Extract actions for missing categories."""
        docs = govdoc_data.get("docs", [])
        
        if not docs:
            return []
        
        doc_texts = []
        for doc in docs:
            clean_path = doc.get("clean_path")
            if clean_path and Path(clean_path).exists():
                text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
                doc_texts.append({
                    "url": doc.get("url"),
                    "text": text[:10000],
                    "title": doc.get("title", "")
                })
        
        if not doc_texts:
            return []
        
        combined = "\n\n---\n\n".join(f"Doc: {d['title']}\n{d['text']}" for d in doc_texts)
        
        prompt = f"""Extract flood actions for ONLY these categories: {', '.join(missing)}

Location: {location}

For each action:
- title: concise
- description: specific (50+ chars)
- priority: high/medium/low
- category: MUST be one of {', '.join(missing)}

JSON array (no markdown):
[{{"title":"...","description":"...","priority":"...","category":"..."}}]

Documents:
{combined[:8000]}

JSON:"""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "Extract specific flood actions. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            data = json.loads(content)
            
            actions = []
            for item in data:
                if item["category"] in missing:
                    actions.append(Action(
                        title=item["title"],
                        description=item["description"],
                        priority=item["priority"],
                        category=item["category"],
                        source_doc=doc_texts[0]["url"] if doc_texts else None
                    ))
            
            print(f"[RevisionAgent]   ✅ Added {len(actions)} actions")
            return actions
        
        except Exception as e:
            print(f"[RevisionAgent]   ❌ Error: {e}")
            return []
    
    def _remove_duplicates(self, action_plan: ActionPlanResponse, coherence: dict) -> ActionPlanResponse:
        """Remove duplicates."""
        seen = set()
        
        for phase_name in ["before_flood", "during_flood", "after_flood"]:
            actions = getattr(action_plan, phase_name)
            filtered = []
            
            for action in actions:
                title_norm = action.title.lower().strip()
                is_dup = any(
                    title_norm == seen_title or 
                    (len(title_norm) > 10 and seen_title in title_norm) or
                    (len(seen_title) > 10 and title_norm in seen_title)
                    for seen_title in seen
                )
                
                if not is_dup:
                    filtered.append(action)
                    seen.add(title_norm)
            
            setattr(action_plan, phase_name, filtered)
        
        return action_plan  # ✅ 必须返回
    
    def _fix_phase_errors(self, action_plan: ActionPlanResponse, coherence: dict) -> ActionPlanResponse:
        """Fix phase errors."""
        before_kw = ["prepare", "plan", "kit", "insurance", "purchase"]
        during_kw = ["evacuate", "avoid", "stay away", "monitor", "turn off"]
        after_kw = ["cleanup", "repair", "document damage", "claim", "inspect"]
        
        all_actions = action_plan.before_flood + action_plan.during_flood + action_plan.after_flood
        new_before, new_during, new_after = [], [], []
        
        for action in all_actions:
            text = (action.title + " " + action.description).lower()
            
            before_score = sum(1 for kw in before_kw if kw in text)
            during_score = sum(1 for kw in during_kw if kw in text)
            after_score = sum(1 for kw in after_kw if kw in text)
            
            if during_score > before_score and during_score > after_score:
                new_during.append(action)
            elif after_score > before_score and after_score > during_score:
                new_after.append(action)
            else:
                new_before.append(action)
        
        action_plan.before_flood = new_before
        action_plan.during_flood = new_during
        action_plan.after_flood = new_after
        
        return action_plan  # ✅ 必须返回
    
    async def _enhance_clarity(self, action_plan: ActionPlanResponse, location: str, ctx: MessageContext) -> int:
        """Enhance vague actions."""
        count = 0
        
        for phase_name in ["before_flood", "during_flood", "after_flood"]:
            actions = getattr(action_plan, phase_name)
            for i, action in enumerate(actions):
                if len(action.description) < 60:
                    enhanced = await self._enhance_single_action(action, location)
                    if enhanced and enhanced.description != action.description:
                        actions[i] = enhanced
                        count += 1
        
        return count  # ✅ 返回增强的数量
    
    async def _enhance_single_action(self, action: Action, location: str) -> Action:
        """Enhance one action."""
        prompt = f"""Enhance this action with specific details:

Location: {location}
Title: {action.title}
Description: {action.description}

Add: quantities, times, specific steps, local resources.

Return ONLY enhanced description (no JSON, no title):"""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            desc = response.choices[0].message.content.strip().strip('"').strip("'")
            
            return Action(
                title=action.title,
                description=desc,
                priority=action.priority,
                category=action.category,
                source_doc=action.source_doc
            )
        except:
            return action  # ✅ 失败时返回原 action
    
    # ... (keep all other helper methods from the evaluator_agent.py that you already have)
    # _check_category_coverage, _llm_evaluate, _calculate_final_scores, 
    # _determine_recommendation, _compare_versions

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

Evaluate on 5 dimensions (0.0-1.0):
1. ACCURACY, 2. CLARITY, 3. COMPLETENESS, 4. RELEVANCE, 5. COHERENCE

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
                content = "\n".join(content.split("\n")[1:-1]).strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"[RevisionAgent] ❌ LLM error: {e}")
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
        weights = {'accuracy': 0.25, 'clarity': 0.15, 'completeness': 0.20, 'relevance': 0.20, 'coherence': 0.20}
        overall_score = sum(eval_result[dim]['score'] * weight for dim, weight in weights.items())
        thresholds = {'accuracy': 0.8, 'clarity': 0.7, 'completeness': 0.7, 'relevance': 0.7, 'coherence': 0.8}
        passes_threshold = all(eval_result[dim]['score'] >= threshold for dim, threshold in thresholds.items())
        
        eval_result.update({
            'overall_score': round(overall_score, 3),
            'passes_threshold': passes_threshold,
            'weights': weights,
            'thresholds': thresholds,
            'coverage_data': coverage_data
        })
        return eval_result
    
    def _determine_recommendation(self, eval_result: Dict) -> str:
        """Determine recommendation."""
        if eval_result['accuracy']['score'] < 0.6:
            return "BLOCK"
        if eval_result['passes_threshold'] and eval_result['overall_score'] >= 0.75:
            return "APPROVE"
        if eval_result['overall_score'] >= 0.65:
            return "REVISE"
        return "BLOCK"
    
    def _compare_versions(self, eval_v1: Dict, eval_v2: Dict, plan_v1: ActionPlanResponse, plan_v2: ActionPlanResponse) -> Dict:
        """Compare two versions."""
        score_v1 = eval_v1['overall_score']
        score_v2 = eval_v2['overall_score']
        score_delta = score_v2 - score_v1
        
        improvements = []
        regressions = []
        
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            delta = eval_v2[dim]['score'] - eval_v1[dim]['score']
            if delta > 0.05:
                improvements.append(f"{dim.capitalize()} +{delta:.2f}")
            elif delta < -0.05:
                regressions.append(f"{dim.capitalize()} {delta:.2f}")
        
        better_version = "revised" if score_delta > 0.01 else ("original" if score_delta < -0.01 else 
                         ("revised" if len(improvements) > len(regressions) else "original"))
        
        return {
            "better_version": better_version,
            "score_delta": round(score_delta, 3),
            "original_score": round(score_v1, 3),
            "revised_score": round(score_v2, 3),
            "improvements": improvements,
            "regressions": regressions
        }


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x