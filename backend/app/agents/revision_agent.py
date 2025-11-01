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
    Agent that performs targeted revisions to action plans.
    
    Strategies:
    1. Add missing category actions
    2. Remove duplicates
    3. Fix phase misclassifications
    4. Enhance clarity
    5. Add location-specific details
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("Revision")
        self._runtime = runtime
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    @message_handler
    async def on_revision_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Perform targeted revision.
        
        Input: {
            "action_plan": {...},
            "evaluation": {...},
            "govdoc_data": {...},
            "location": "..."
        }
        
        Output: {
            "revised_plan": {...},
            "changes_made": [...]
        }
        """
        try:
            input_data = json.loads(message.content)
            action_plan_data = input_data.get("action_plan")
            evaluation = input_data.get("evaluation")
            govdoc_data = input_data.get("govdoc_data")
            location = input_data.get("location")
            
            print(f"\n[RevisionAgent] Starting targeted revision for {location}")
            
            action_plan = ActionPlanResponse(**action_plan_data)
            changes_made = []
            
            # Strategy 1: Add missing categories
            missing = evaluation.get("coverage_data", {}).get("missing_essential", [])
            if missing:
                print(f"[RevisionAgent] Strategy 1: Adding {missing}")
                new_actions = await self._add_missing_categories(missing, govdoc_data, location, ctx)
                
                if new_actions:
                    for action in new_actions:
                        if action.category in ["insurance", "emergency_kit", "property_protection", "family_plan"]:
                            action_plan.before_flood.append(action)
                        elif action.category == "evacuation":
                            action_plan.during_flood.append(action)
                        else:
                            action_plan.before_flood.append(action)
                    
                    changes_made.append(f"Added {len(new_actions)} actions for {', '.join(missing)}")
            
            # Strategy 2: Remove duplicates
            coherence = evaluation.get("coherence", {})
            if coherence.get("duplicate_actions"):
                print(f"[RevisionAgent] Strategy 2: Removing duplicates")
                original = action_plan.total_actions()
                action_plan = self._remove_duplicates(action_plan, coherence)
                removed = original - action_plan.total_actions()
                if removed > 0:
                    changes_made.append(f"Removed {removed} duplicates")
            
            # Strategy 3: Fix phases
            if coherence.get("phase_errors"):
                print(f"[RevisionAgent] Strategy 3: Fixing phases")
                action_plan = self._fix_phase_errors(action_plan, coherence)
                changes_made.append("Fixed phase errors")
            
            # Strategy 4: Enhance clarity
            clarity_score = evaluation.get("clarity", {}).get("score", 1.0)
            if clarity_score < 0.7:
                print(f"[RevisionAgent] Strategy 4: Enhancing clarity")
                count = await self._enhance_clarity(action_plan, location, ctx)
                if count > 0:
                    changes_made.append(f"Enhanced {count} actions")
            
            # Strategy 5: Add location details
            relevance_score = evaluation.get("relevance", {}).get("score", 1.0)
            if relevance_score < 0.7:
                print(f"[RevisionAgent] Strategy 5: Adding location details")
                count = self._add_location_details(action_plan, location)
                if count > 0:
                    changes_made.append(f"Added location details to {count} actions")
            
            print(f"[RevisionAgent] ✅ Complete: {len(changes_made)} strategies applied")
            
            return Message(content=json.dumps({
                "revised_plan": action_plan.model_dump(mode='json'),
                "changes_made": changes_made
            }, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[RevisionAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "revised_plan": input_data.get("action_plan")
            }))
    
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
        
        return action_plan
    
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
        
        return action_plan
    
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
        
        return count
    
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
            return action
    
    def _add_location_details(self, action_plan: ActionPlanResponse, location: str) -> int:
        """Add location-specific info."""
        count = 0
        city = location.split(",")[0].strip()
        
        for actions in [action_plan.before_flood, action_plan.during_flood, action_plan.after_flood]:
            for action in actions:
                if city.lower() not in action.description.lower():
                    if action.category == "evacuation" and "route" in action.description.lower():
                        action.description += f" Check {city} official evacuation routes."
                        count += 1
        
        return count


async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x