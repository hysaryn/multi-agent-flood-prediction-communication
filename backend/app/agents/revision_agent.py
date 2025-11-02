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
from typing import List, Dict, Tuple
import json
import os


class RevisionAgent(RoutedAgent):
    """
    Agent that revises action plans AND selects the best version.
    
    Combines revision and selection in one agent to reduce LLM calls.
    Uses heuristic comparison instead of re-evaluating with LLM.
    
    Input: {
        "original_plan": {...},
        "evaluation": {...},
        "govdoc_data": {...},
        "location": "..."
    }
    
    Output: {
        "status": "improved|original_better|blocked",
        "selected_version": "original|revised",
        "final_plan": {...},
        "comparison": {...}
    }
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("Revision")
        self._runtime = runtime
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    @message_handler
    async def on_revision_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Revise plan, compare versions, and select the best.
        """
        try:
            input_data = json.loads(message.content)
            original_plan_data = input_data.get("original_plan")
            evaluation = input_data.get("evaluation")
            govdoc_data = input_data.get("govdoc_data")
            location = input_data.get("location")
            
            print(f"\n[RevisionAgent] ╔════════════════════════════════════════")
            print(f"[RevisionAgent] ║ Starting Revision + Selection")
            print(f"[RevisionAgent] ║ Location: {location}")
            print(f"[RevisionAgent] ╚════════════════════════════════════════")
            
            original_plan = ActionPlanResponse(**original_plan_data)
            
            # ========== Step 1: Perform Revision ==========
            print(f"\n[RevisionAgent] 🔧 STEP 1: Applying revision strategies...")
            
            revised_plan, revision_metrics = await self._revise_plan(
                original_plan, evaluation, govdoc_data, location, ctx
            )
            
            print(f"[RevisionAgent] ✅ Revision complete:")
            for metric, value in revision_metrics.items():
                print(f"[RevisionAgent]    - {metric}: {value}")
            
            # ========== Step 2: Heuristic Comparison ==========
            print(f"\n[RevisionAgent] ⚖️  STEP 2: Comparing versions (heuristic)...")
            
            comparison = self._compare_versions_heuristic(
                original_plan,
                revised_plan,
                evaluation,
                revision_metrics
            )
            
            print(f"[RevisionAgent] 📊 Comparison Results:")
            print(f"[RevisionAgent]    Original Score: {comparison['original_estimated_score']:.3f}")
            print(f"[RevisionAgent]    Revised Score:  {comparison['revised_estimated_score']:.3f}")
            print(f"[RevisionAgent]    Score Delta: {comparison['score_delta']:+.3f}")
            print(f"[RevisionAgent]    Better Version: {comparison['better_version'].upper()}")
            
            # ========== Step 3: Select Final Version ==========
            if comparison['better_version'] == "revised":
                final_plan = revised_plan
                status = "improved"
                print(f"[RevisionAgent] ✅ Selected: REVISED version")
            else:
                final_plan = original_plan
                status = "original_better"
                print(f"[RevisionAgent] ⚠️  Selected: ORIGINAL version (revision didn't improve)")
            
            # Show improvements/regressions
            if comparison['improvements']:
                print(f"[RevisionAgent]    Improvements:")
                for imp in comparison['improvements']:
                    print(f"[RevisionAgent]      ✅ {imp}")
            
            if comparison['regressions']:
                print(f"[RevisionAgent]    Regressions:")
                for reg in comparison['regressions']:
                    print(f"[RevisionAgent]      ⚠️  {reg}")
            
            print(f"\n[RevisionAgent] ╔════════════════════════════════════════")
            print(f"[RevisionAgent] ║ 🎯 FINAL: {status.upper()}")
            print(f"[RevisionAgent] ║ Version: {comparison['better_version'].upper()}")
            print(f"[RevisionAgent] ╚════════════════════════════════════════")
            
            return Message(content=json.dumps({
                "status": status,
                "selected_version": comparison['better_version'],
                "final_plan": final_plan.model_dump(mode='json'),
                "comparison": comparison,
                "revision_metrics": revision_metrics
            }, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[RevisionAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "status": "error",
                "final_plan": input_data.get("original_plan")
            }))
    
    async def _revise_plan(
        self,
        original_plan: ActionPlanResponse,
        evaluation: Dict,
        govdoc_data: Dict,
        location: str,
        ctx: MessageContext
    ) -> Tuple[ActionPlanResponse, Dict]:
        """
        Apply all revision strategies and return metrics.
        
        Returns:
            (revised_plan, metrics)
        """
        # Make a copy to modify
        revised_plan = ActionPlanResponse(
            location=original_plan.location,
            display_name=original_plan.display_name,
            before_flood=original_plan.before_flood.copy(),
            during_flood=original_plan.during_flood.copy(),
            after_flood=original_plan.after_flood.copy(),
            sources=original_plan.sources.copy(),
            generated_at=original_plan.generated_at
        )
        
        metrics = {
            "actions_added": 0,
            "actions_removed": 0,
            "duplicates_removed": 0,
            "phases_fixed": 0,
            "descriptions_enhanced": 0,
            "location_details_added": 0
        }
        
        original_count = original_plan.total_actions()
        
        # Strategy 1: Add missing categories
        missing = evaluation.get("coverage_data", {}).get("missing_essential", [])
        if missing:
            print(f"[RevisionAgent]    Strategy 1: Adding missing {missing}")
            new_actions = await self._add_missing_categories(missing, govdoc_data, location, ctx)
            
            if new_actions:
                for action in new_actions:
                    if action.category in ["insurance", "emergency_kit", "property_protection", "family_plan"]:
                        revised_plan.before_flood.append(action)
                    elif action.category == "evacuation":
                        revised_plan.during_flood.append(action)
                    else:
                        revised_plan.before_flood.append(action)
                
                metrics["actions_added"] = len(new_actions)
        
        # Strategy 2: Remove duplicates
        coherence = evaluation.get("coherence", {})
        if coherence.get("duplicate_actions"):
            print(f"[RevisionAgent]    Strategy 2: Removing duplicates")
            before_removal = revised_plan.total_actions()
            revised_plan = self._remove_duplicates(revised_plan, coherence)
            after_removal = revised_plan.total_actions()
            metrics["duplicates_removed"] = before_removal - after_removal
        
        # Strategy 3: Fix phase misclassifications
        phase_errors = coherence.get("phase_errors", [])
        if phase_errors:
            print(f"[RevisionAgent]    Strategy 3: Fixing {len(phase_errors)} phase errors")
            revised_plan, fixed_count = self._fix_phase_errors(revised_plan, coherence)
            metrics["phases_fixed"] = fixed_count
        
        # Strategy 4: Enhance clarity
        clarity_score = evaluation.get("clarity", {}).get("score", 1.0)
        if clarity_score < 0.7:
            print(f"[RevisionAgent]    Strategy 4: Enhancing clarity (score: {clarity_score:.2f})")
            count = await self._enhance_clarity(revised_plan, location, ctx)
            metrics["descriptions_enhanced"] = count
        
        # Strategy 5: Add location-specific details
        relevance_score = evaluation.get("relevance", {}).get("score", 1.0)
        if relevance_score < 0.7:
            print(f"[RevisionAgent]    Strategy 5: Adding location details (score: {relevance_score:.2f})")
            count = self._add_location_details(revised_plan, location)
            metrics["location_details_added"] = count
        
        final_count = revised_plan.total_actions()
        metrics["actions_removed"] = max(0, original_count - final_count + metrics["actions_added"])
        
        return revised_plan, metrics
    
    def _compare_versions_heuristic(
        self,
        original_plan: ActionPlanResponse,
        revised_plan: ActionPlanResponse,
        original_evaluation: Dict,
        revision_metrics: Dict
    ) -> Dict:
        """
        Compare original and revised plans using heuristic rules.
        
        This avoids calling LLM again for evaluation.
        Instead, we estimate score improvements based on:
        1. What problems the original evaluation identified
        2. What changes the revision made
        3. Objective metrics (counts, distributions, lengths)
        
        Returns:
        {
            "better_version": "original" or "revised",
            "score_delta": float,
            "original_estimated_score": float,
            "revised_estimated_score": float,
            "improvements": [...],
            "regressions": [...],
            "confidence": "high|medium|low"
        }
        """
        improvements = []
        regressions = []
        
        # Base score from original evaluation
        original_score = original_evaluation.get("overall_score", 0.7)
        estimated_score_delta = 0.0
        
        print(f"[RevisionAgent]    Starting with original score: {original_score:.3f}")
        
        # ========== Dimension 1: Completeness (25% weight) ==========
        completeness_boost = self._evaluate_completeness_improvement(
            original_plan, revised_plan, original_evaluation, revision_metrics, improvements
        )
        estimated_score_delta += completeness_boost * 0.25
        
        # ========== Dimension 2: Coherence (20% weight) ==========
        coherence_boost = self._evaluate_coherence_improvement(
            original_plan, revised_plan, original_evaluation, revision_metrics, improvements, regressions
        )
        estimated_score_delta += coherence_boost * 0.20
        
        # ========== Dimension 3: Clarity (15% weight) ==========
        clarity_boost = self._evaluate_clarity_improvement(
            original_plan, revised_plan, original_evaluation, revision_metrics, improvements
        )
        estimated_score_delta += clarity_boost * 0.15
        
        # ========== Dimension 4: Relevance (20% weight) ==========
        relevance_boost = self._evaluate_relevance_improvement(
            original_plan, revised_plan, original_evaluation, revision_metrics, improvements
        )
        estimated_score_delta += relevance_boost * 0.20
        
        # ========== Dimension 5: Accuracy (25% weight) ==========
        # Accuracy rarely changes during revision (no new content generation)
        # But can regress if we removed too many actions
        accuracy_change = self._evaluate_accuracy_change(
            original_plan, revised_plan, revision_metrics, regressions
        )
        estimated_score_delta += accuracy_change * 0.25
        
        # ========== Final Decision ==========
        revised_estimated_score = min(1.0, max(0.0, original_score + estimated_score_delta))
        
        print(f"[RevisionAgent]    Estimated score delta: {estimated_score_delta:+.3f}")
        print(f"[RevisionAgent]    Revised estimated score: {revised_estimated_score:.3f}")
        
        # Decision threshold: need at least +0.03 improvement to prefer revised
        if estimated_score_delta > 0.03:
            better_version = "revised"
            confidence = "high" if estimated_score_delta > 0.10 else "medium"
        elif estimated_score_delta < -0.03:
            better_version = "original"
            confidence = "high" if estimated_score_delta < -0.10 else "medium"
        else:
            # Tie: prefer original (safer choice)
            better_version = "original"
            confidence = "low"
            improvements.append("Minimal changes, keeping original for safety")
        
        return {
            "better_version": better_version,
            "score_delta": round(estimated_score_delta, 3),
            "original_estimated_score": round(original_score, 3),
            "revised_estimated_score": round(revised_estimated_score, 3),
            "improvements": improvements,
            "regressions": regressions,
            "confidence": confidence,
            "metrics": revision_metrics
        }
    
    def _evaluate_completeness_improvement(
        self,
        original: ActionPlanResponse,
        revised: ActionPlanResponse,
        evaluation: Dict,
        metrics: Dict,
        improvements: List[str]
    ) -> float:
        """
        Evaluate improvement in completeness dimension.
        
        Factors:
        - Missing categories added
        - Total action count increase
        - Phase distribution balance
        """
        boost = 0.0
        
        # Factor 1: Missing categories added
        if metrics["actions_added"] > 0:
            missing = evaluation.get("coverage_data", {}).get("missing_essential", [])
            if missing:
                # Each missing category filled = +0.15/category
                categories_per_action = 1.0 / max(1, metrics["actions_added"])
                boost += min(0.30, categories_per_action * len(missing) * 0.15)
                improvements.append(
                    f"Added {metrics['actions_added']} actions for missing categories: {', '.join(missing)}"
                )
        
        # Factor 2: Phase distribution improvement
        original_balance = self._calculate_phase_balance_score(original)
        revised_balance = self._calculate_phase_balance_score(revised)
        
        if revised_balance > original_balance + 0.05:
            boost += 0.10
            improvements.append(
                f"Improved phase distribution (balance: {original_balance:.2f} → {revised_balance:.2f})"
            )
        
        return boost
    
    def _evaluate_coherence_improvement(
        self,
        original: ActionPlanResponse,
        revised: ActionPlanResponse,
        evaluation: Dict,
        metrics: Dict,
        improvements: List[str],
        regressions: List[str]
    ) -> float:
        """
        Evaluate improvement in coherence dimension.
        
        Factors:
        - Duplicates removed
        - Phase errors fixed
        - No new contradictions introduced
        """
        boost = 0.0
        
        # Factor 1: Duplicates removed
        if metrics["duplicates_removed"] > 0:
            # Each duplicate removed = +0.05
            boost += min(0.20, metrics["duplicates_removed"] * 0.05)
            improvements.append(f"Removed {metrics['duplicates_removed']} duplicate actions")
        
        # Factor 2: Phase errors fixed
        if metrics["phases_fixed"] > 0:
            # Each phase fix = +0.04
            boost += min(0.15, metrics["phases_fixed"] * 0.04)
            improvements.append(f"Fixed {metrics['phases_fixed']} phase misclassifications")
        
        # Factor 3: Check if we lost too many actions (potential regression)
        if metrics["actions_removed"] > metrics["duplicates_removed"] + 2:
            # Removed more than just duplicates
            excess_removed = metrics["actions_removed"] - metrics["duplicates_removed"]
            boost -= excess_removed * 0.03
            regressions.append(f"Removed {excess_removed} potentially valid actions")
        
        return boost
    
    def _evaluate_clarity_improvement(
        self,
        original: ActionPlanResponse,
        revised: ActionPlanResponse,
        evaluation: Dict,
        metrics: Dict,
        improvements: List[str]
    ) -> float:
        """
        Evaluate improvement in clarity dimension.
        
        Factors:
        - Descriptions enhanced
        - Average description length increased
        """
        boost = 0.0
        
        # Factor 1: Descriptions enhanced
        if metrics["descriptions_enhanced"] > 0:
            # Each enhancement = +0.03
            boost += min(0.15, metrics["descriptions_enhanced"] * 0.03)
            improvements.append(f"Enhanced {metrics['descriptions_enhanced']} vague descriptions")
        
        # Factor 2: Average description length
        original_avg_len = self._avg_description_length(original)
        revised_avg_len = self._avg_description_length(revised)
        
        if revised_avg_len > original_avg_len * 1.1:  # 10% increase
            boost += 0.08
            improvements.append(
                f"Increased description detail (avg: {original_avg_len:.0f} → {revised_avg_len:.0f} chars)"
            )
        
        return boost
    
    def _evaluate_relevance_improvement(
        self,
        original: ActionPlanResponse,
        revised: ActionPlanResponse,
        evaluation: Dict,
        metrics: Dict,
        improvements: List[str]
    ) -> float:
        """
        Evaluate improvement in relevance dimension.
        
        Factors:
        - Location-specific details added
        """
        boost = 0.0
        
        if metrics["location_details_added"] > 0:
            # Each location detail = +0.04
            boost += min(0.12, metrics["location_details_added"] * 0.04)
            improvements.append(
                f"Added location-specific details to {metrics['location_details_added']} actions"
            )
        
        return boost
    
    def _evaluate_accuracy_change(
        self,
        original: ActionPlanResponse,
        revised: ActionPlanResponse,
        metrics: Dict,
        regressions: List[str]
    ) -> float:
        """
        Evaluate change in accuracy dimension.
        
        Accuracy typically doesn't improve during revision (no new content).
        But can regress if we removed too much.
        """
        change = 0.0
        
        # Regression: if we removed more than 30% of actions
        original_total = original.total_actions()
        revised_total = revised.total_actions()
        
        if revised_total < original_total * 0.7:
            change -= 0.10
            regressions.append(
                f"Removed {original_total - revised_total} actions (may have lost valid content)"
            )
        
        return change
    
    def _calculate_phase_balance_score(self, plan: ActionPlanResponse) -> float:
        """
        Calculate phase distribution balance score.
        
        Ideal distribution:
        - Before: 50-60%
        - During: 20-30%
        - After: 10-20%
        
        Returns: 0.0 (worst) to 1.0 (perfect)
        """
        total = plan.total_actions()
        if total == 0:
            return 0.0
        
        before_pct = len(plan.before_flood) / total
        during_pct = len(plan.during_flood) / total
        after_pct = len(plan.after_flood) / total
        
        # Ideal targets
        ideal_before = 0.55
        ideal_during = 0.25
        ideal_after = 0.15
        
        # Calculate normalized distance from ideal
        distance = (
            abs(before_pct - ideal_before) +
            abs(during_pct - ideal_during) +
            abs(after_pct - ideal_after)
        ) / 2.0  # Normalize to [0, 1]
        
        # Convert to score
        return max(0.0, 1.0 - distance)
    
    def _avg_description_length(self, plan: ActionPlanResponse) -> float:
        """Calculate average description length across all actions."""
        all_actions = plan.before_flood + plan.during_flood + plan.after_flood
        if not all_actions:
            return 0.0
        return sum(len(a.description) for a in all_actions) / len(all_actions)
    
    # ========== Helper Methods (from original RevisionAgent) ==========
    
    async def _add_missing_categories(
        self, missing: List[str], govdoc_data: dict, location: str, ctx: MessageContext
    ) -> List[Action]:
        """Extract actions for missing categories."""
        docs = govdoc_data.get("docs", [])
        
        doc_texts = []
        for doc in docs:
            clean_path = doc.get("clean_path")
            if clean_path and Path(clean_path).exists():
                text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
                doc_texts.append({"url": doc.get("url"), "text": text[:10000], "title": doc.get("title", "")})
        
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
            
            print(f"[RevisionAgent]      ✅ Added {len(actions)} actions")
            return actions
        
        except Exception as e:
            print(f"[RevisionAgent]      ❌ Error: {e}")
            return []
    
    def _remove_duplicates(self, action_plan: ActionPlanResponse, coherence: dict) -> ActionPlanResponse:
        """Remove duplicate actions."""
        seen = set()
        
        for phase_name in ["before_flood", "during_flood", "after_flood"]:
            actions = getattr(action_plan, phase_name)
            filtered = []
            
            for action in actions:
                title_norm = action.title.lower().strip()
                
                # Check for duplicates
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
        
        return action_plan
    
    def _fix_phase_errors(self, action_plan: ActionPlanResponse, coherence: dict) -> Tuple[ActionPlanResponse, int]:
        """Fix phase misclassifications and return count of fixes."""
        before_kw = ["prepare", "plan", "kit", "insurance", "purchase", "install", "elevate", "sign up"]
        during_kw = ["evacuate", "avoid", "stay away", "monitor", "turn off", "shut off", "leave", "move to"]
        after_kw = ["cleanup", "repair", "document damage", "claim", "inspect", "restore", "file"]
        
        all_actions = action_plan.before_flood + action_plan.during_flood + action_plan.after_flood
        new_before, new_during, new_after = [], [], []
        
        fixed_count = 0
        
        for action in all_actions:
            text = (action.title + " " + action.description).lower()
            
            # Calculate keyword scores
            before_score = sum(1 for kw in before_kw if kw in text)
            during_score = sum(1 for kw in during_kw if kw in text)
            after_score = sum(1 for kw in after_kw if kw in text)
            
            # Determine correct phase
            if during_score > before_score and during_score > after_score:
                new_during.append(action)
                if action not in action_plan.during_flood:
                    fixed_count += 1
            elif after_score > before_score and after_score > during_score:
                new_after.append(action)
                if action not in action_plan.after_flood:
                    fixed_count += 1
            else:
                new_before.append(action)
                if action not in action_plan.before_flood:
                    fixed_count += 1
        
        action_plan.before_flood = new_before
        action_plan.during_flood = new_during
        action_plan.after_flood = new_after
        
        return action_plan, fixed_count
    
    async def _enhance_clarity(self, action_plan: ActionPlanResponse, location: str, ctx: MessageContext) -> int:
        """Enhance vague action descriptions."""
        count = 0
        
        for phase_name in ["before_flood", "during_flood", "after_flood"]:
            actions = getattr(action_plan, phase_name)
            for i, action in enumerate(actions):
                if len(action.description) < 60:
                    enhanced = await self._enhance_single_action(action, location)
                    if enhanced and enhanced.description != action.description:
                        actions[i] = enhanced
                        count += 1
        
        return count
    
    async def _enhance_single_action(self, action: Action, location: str) -> Action:
        """Enhance a single action's description."""
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
            return action
    
    def _add_location_details(self, action_plan: ActionPlanResponse, location: str) -> int:
        """Add location-specific information to actions."""
        count = 0
        city = location.split(",")[0].strip()
        
        for actions in [action_plan.before_flood, action_plan.during_flood, action_plan.after_flood]:
            for action in actions:
                if city.lower() not in action.description.lower():
                    # Add location context based on category
                    if action.category == "evacuation" and "route" in action.description.lower():
                        action.description += f" Check {city} official evacuation routes."
                        count += 1
                    elif action.category == "communication" and "alert" in action.description.lower():
                        action.description += f" Sign up for {city} emergency alerts."
                        count += 1
        
        return count


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x