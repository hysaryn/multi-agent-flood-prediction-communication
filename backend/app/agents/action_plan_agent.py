from dotenv import load_dotenv
load_dotenv(override=True)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import (
    SingleThreadedAgentRuntime,
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
)

from app.models.message_model import Message
from app.models.action_plan_models import Action, ActionPlanResponse
from pathlib import Path
from typing import List, Dict, Tuple
import json
import asyncio
import re
from datetime import datetime, timezone


class ActionPlanAgent(RoutedAgent):
    """
    Agent that generates structured action plans from government documents.
    
    Sequential Architecture:
    - Receives: Complete GovDocAgent output
    - Processes: Extracts and categorizes actions with phase assignment
    - Outputs: Action plan + passes through govdoc_data for downstream agents
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("ActionPlan")
        self._runtime = runtime
        
        # LLM for extracting and categorizing actions
        self._llm = AssistantAgent(
            "ActionPlanLLM",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini",
            ),
        )
    
    @message_handler
    async def on_action_plan_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Generate action plan and pass through govdoc_data for sequential flow.
        
        Input (from GovDocAgent): {
            "location": {"query": "...", "display_name": "...", ...},
            "results": [...],
            "docs": [{"url": "...", "title": "...", "clean_path": "...", ...}]
        }
        
        Output (for EvaluatorAgent): {
            "action_plan": ActionPlanResponse,
            "govdoc_data": {...},
            "location": "..."
        }
        """
        input_data = None
        
        try:
            # Parse input from GovDocAgent
            input_data = json.loads(message.content)
            
            location_info = input_data.get("location", {})
            location_query = location_info.get("query") or location_info.get("display_name", "Unknown")
            display_name = location_info.get("display_name")
            docs = input_data.get("docs", [])
            
            print(f"\n[ActionPlanAgent] Generating plan for: {location_query}")
            print(f"[ActionPlanAgent] Received {len(docs)} documents")

            if not docs:
                print(f"[ActionPlanAgent] No documents provided")
                return Message(content=json.dumps({
                    "error": "No documents provided",
                    "location": location_query,
                    "action_plan": None,
                    "govdoc_data": input_data
                }))
            
            # Read all document texts
            doc_texts = []
            for doc in docs:
                clean_path = doc.get("clean_path")
                url = doc.get("url")
                
                if not clean_path or not Path(clean_path).exists():
                    print(f"[ActionPlanAgent]   ⚠️  Skipping missing file: {clean_path}")
                    continue
                
                text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
                doc_texts.append({
                    "url": url,
                    "text": text[:15000],
                    "title": doc.get("title", "Untitled")
                })
                print(f"[ActionPlanAgent]   ✅ Loaded: {doc.get('title', 'Untitled')[:50]}... ({len(text)} chars)")
            
            if not doc_texts:
                return Message(content=json.dumps({
                    "error": "No readable documents found",
                    "location": location_query,
                    "action_plan": None,
                    "govdoc_data": input_data
                }))
            
            # Extract actions from all documents
            print(f"[ActionPlanAgent] Extracting actions from {len(doc_texts)} documents...")
            all_actions_with_phases = []
            
            for doc in doc_texts:
                actions_with_phases = await self._extract_actions_from_doc(doc, location_query, ctx)
                all_actions_with_phases.extend(actions_with_phases)
                print(f"[ActionPlanAgent]   Extracted {len(actions_with_phases)} actions")
            
            print(f"[ActionPlanAgent] Total actions extracted: {len(all_actions_with_phases)}")
            
            # Deduplicate
            deduplicated = self._deduplicate_actions_with_phases(all_actions_with_phases)
            print(f"[ActionPlanAgent] Deduplication: {len(all_actions_with_phases)} → {len(deduplicated)} actions")
            
            # Categorize using LLM phases
            before, during, after = self._categorize_by_phase(deduplicated)
            
            print(f"[ActionPlanAgent] Categorized: Before={len(before)}, During={len(during)}, After={len(after)}")
            
            # Build action plan
            action_plan = ActionPlanResponse(
                location=location_query,
                display_name=display_name,
                before_flood=before,
                during_flood=during,
                after_flood=after,
                sources=[doc["url"] for doc in doc_texts],
                generated_at=datetime.now(timezone.utc).isoformat()
            )
            
            print(f"[ActionPlanAgent] ✅ Plan complete: {action_plan.total_actions()} total actions")
            
            # Output for sequential flow
            output = {
                "action_plan": action_plan.model_dump(mode='python'),
                "govdoc_data": input_data,
                "location": location_query
            }
            
            return Message(content=json.dumps(output, ensure_ascii=False))
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent] ❌ Invalid JSON input: {e}")
            return Message(content=json.dumps({
                "error": f"Invalid JSON input: {str(e)}",
                "action_plan": None,
                "govdoc_data": None
            }))
        
        except Exception as e:
            print(f"[ActionPlanAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            location = "Unknown"
            govdoc = None
            
            if input_data:
                try:
                    loc_info = input_data.get("location", {})
                    if isinstance(loc_info, dict):
                        location = loc_info.get("query") or loc_info.get("display_name", "Unknown")
                    govdoc = input_data
                except:
                    pass
            
            return Message(content=json.dumps({
                "error": str(e),
                "location": location,
                "action_plan": None,
                "govdoc_data": govdoc
            }))
    
    async def _extract_actions_from_doc(self, doc: Dict, location: str, ctx: MessageContext) -> List[Tuple[Action, str]]:
        """
        Extract actions with phase information.
        
        Returns:
            List of (Action, phase) tuples where phase is "before"|"during"|"after"
        """
        
        prompt = f"""You are analyzing a government flood preparedness document.
        
⚠️ CRITICAL RULES TO PREVENT HALLUCINATION:
1. ONLY extract information EXPLICITLY stated in the document text below
2. If a detail is not in the document, use "Not specified" instead of inventing
3. Do NOT add information from your general knowledge
4. Do NOT make assumptions about {location} unless stated in the document
5. If unsure, mark with [UNCERTAIN] prefix

VALIDATION CHECKLIST - Before including any action, verify:
☑️ Is this EXACT action mentioned in the document text?
☑️ Are the specific details (numbers, names, locations) FROM the document?
☑️ Can I quote the source sentence that supports this?

Document: {doc['title']}
Source: {doc['url']}

Extract LOCATION-SPECIFIC actionable items for flood preparation and response.

🎯 PRIORITY: Extract actions that are:
1. UNIQUE to {location}'s geography, climate, or infrastructure
2. Based on LOCAL flood history or risk factors
3. Specific to REGIONAL emergency management systems
4. Referenced in THIS document (not generic advice)

PHASE GUIDELINES (distribute naturally based on document content):
- BEFORE flood: Preparation, planning, prevention (typically 50-60%)
- DURING flood: Immediate response, evacuation, safety (typically 20-30%)
- AFTER flood: Recovery, repair, claims (typically 10-20%)

⚠️ AVOID GENERIC ACTIONS unless they have location-specific details:
- Instead of "Purchase flood insurance" → Extract specifics about local programs
- Instead of "Prepare emergency kit" → Extract region-specific supply recommendations
- Instead of "Create evacuation plan" → Extract specific routes/shelters for {location}

FOCUS ON:
✅ Local flood warning systems (names, how to register)
✅ Specific waterways, rivers, or coastal areas mentioned
✅ Regional emergency services or agencies
✅ Local building codes or elevation requirements
✅ Historical flood events in this area
✅ Community-specific resources or shelters
✅ Climate patterns unique to this region

For each action provide:
{{
  "title": "...",
  "description": "... (specific details with quantities, times, locations)",
  "priority": "high|medium|low",
  "category": "evacuation|property_protection|emergency_kit|communication|insurance|family_plan",
  "phase": "before|during|after"
}}

IMPORTANT: 
- Distribute across all three phases
- Avoid duplicates with other documents
- Be specific in descriptions

Return ONLY valid JSON array:
[
  {{
    "title": "Purchase flood insurance",
    "description": "Contact insurance provider to obtain flood coverage...",
    "priority": "high",
    "category": "insurance",
    "phase": "before"
  }}
]

Document text:
{doc['text'][:10000]}

JSON OUTPUT:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            content = response.chat_message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            actions_data = json.loads(content)
            
            # Convert to (Action, phase) tuples
            actions_with_phases = []
            for item in actions_data:
                try:
                    action = Action(
                        title=item["title"],
                        description=item["description"],
                        priority=item["priority"],
                        category=item["category"],
                        source_doc=doc["url"]
                    )
                    phase = item.get("phase", "before")
                    actions_with_phases.append((action, phase))
                except Exception as e:
                    print(f"[ActionPlanAgent]   ⚠️  Skipping invalid action: {e}")
                    continue
            
            return actions_with_phases
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent]   ❌ JSON parse error: {e}")
            print(f"[ActionPlanAgent]   LLM response preview: {content[:300]}")
            return []
        except Exception as e:
            print(f"[ActionPlanAgent]   ❌ Error: {e}")
            return []
    
    def _deduplicate_actions_with_phases(self, actions_with_phases: List[Tuple[Action, str]]) -> List[Tuple[Action, str]]:
        """
        Remove duplicate actions based on title similarity.
        
        Args:
            actions_with_phases: List of (Action, phase) tuples
        
        Returns:
            Deduplicated list of (Action, phase) tuples
        """
        seen_titles = set()
        unique = []
        
        for action, phase in actions_with_phases:
            # Normalize title
            title_norm = action.title.lower().strip()
            # Remove common prefixes for better matching
            title_norm = re.sub(r'^(prepare|create|establish|develop|implement|conduct|review|build|install|assemble)\s+(an?\s+)?', '', title_norm)
            
            # Check similarity with existing titles
            is_duplicate = False
            for seen in seen_titles:
                # Word overlap check
                words_action = set(title_norm.split())
                words_seen = set(seen.split())
                
                if len(words_action) > 0 and len(words_seen) > 0:
                    overlap = len(words_action & words_seen) / max(len(words_action), len(words_seen))
                    if overlap > 0.7:  # 70% word overlap = duplicate
                        is_duplicate = True
                        print(f"[ActionPlanAgent]   🗑️  Duplicate: '{action.title}'")
                        break
            
            if not is_duplicate:
                unique.append((action, phase))
                seen_titles.add(title_norm)
        
        return unique
    
    def _categorize_by_phase(self, actions_with_phases: List[Tuple[Action, str]]) -> Tuple[List[Action], List[Action], List[Action]]:
        """
        Categorize actions using LLM phases with keyword validation.
        """
        before = []
        during = []
        after = []
        
        # Keywords that override LLM (only check title for precision)
        during_title_keywords = ["evacuate", "leave home", "move to higher", "avoid water", "turn off", "shut off"]
        after_title_keywords = ["document damage", "file claim", "cleanup", "clean up", "repair", "restore", "inspect"]
        
        for action, llm_phase in actions_with_phases:
            title_lower = action.title.lower()
            
            # Strong override if action is clearly during/after based on title
            if any(kw in title_lower for kw in during_title_keywords):
                during.append(action)
            elif any(kw in title_lower for kw in after_title_keywords):
                after.append(action)
            # Trust LLM otherwise
            elif llm_phase == "during":
                during.append(action)
            elif llm_phase == "after":
                after.append(action)
            else:
                before.append(action)
        
        return before, during, after


# ---------------------------------------------------------
# Testing helper
# ---------------------------------------------------------

async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


async def main():
    """Test the complete Sequential Pipeline"""
    from app.agents.govdoc_agent import GovDocAgent
    from app.agents.evaluator_agent import ActionPlanEvaluatorAgent
    from app.agents.revision_agent import RevisionAgent
    
    runtime = SingleThreadedAgentRuntime()
    
    # Register all agents in sequential order
    print("Registering agents for Sequential Pipeline...")
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    await RevisionAgent.register(runtime, "Revision", lambda: RevisionAgent(runtime))
    await ActionPlanEvaluatorAgent.register(runtime, "ActionPlanEvaluator", lambda: ActionPlanEvaluatorAgent(runtime))
    
    await maybe_await(runtime.start())
    
    print("\n" + "=" * 80)
    print("SEQUENTIAL PIPELINE WITH TARGETED REVISION")
    print("=" * 80)
    print("Architecture: Pure Sequential (no orchestrator)")
    print("Flow: GovDoc → ActionPlan → Evaluator ⇄ Revision")
    print("Entry Point: GovDocAgent")
    print("=" * 80)
    
    # Test with location
    test_location = "Vancouver, BC"
    print(f"\nQuery: {test_location}\n")
    
    # Entry point: GovDocAgent
    response = await runtime.send_message(
        Message(content=test_location),
        AgentId("GovDoc", "default")
    )
    
    # Parse final result
    result = json.loads(response.content)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    
    # Display summary
    status = result.get("status", "unknown")
    print(f"\n📊 Status: {status.upper()}")
    
    # Revision info
    if "comparison" in result:
        comp = result["comparison"]
        print(f"\n⚖️  Version Comparison:")
        print(f"  Selected: {comp.get('better_version', 'N/A').upper()}")
        print(f"  Score Delta: {comp.get('score_delta', 0):+.3f}")
        if comp.get('improvements'):
            print(f"  Improvements:")
            for imp in comp['improvements']:
                print(f"    ✅ {imp}")
        if comp.get('regressions'):
            print(f"  Regressions:")
            for reg in comp['regressions']:
                print(f"    ⚠️  {reg}")
    
    # Final action plan
    if "action_plan" in result:
        ap = result["action_plan"]
        before = len(ap.get('before_flood', []))
        during = len(ap.get('during_flood', []))
        after = len(ap.get('after_flood', []))
        total = before + during + after
        
        print(f"\n📋 Final Action Plan:")
        print(f"  Location: {ap.get('location')}")
        print(f"  Total Actions: {total}")
        if total > 0:
            print(f"    Before: {before} ({before/total*100:.0f}%)")
            print(f"    During: {during} ({during/total*100:.0f}%)")
            print(f"    After: {after} ({after/total*100:.0f}%)")
        print(f"  Sources: {len(ap.get('sources', []))} documents")
    
    # Final evaluation
    if "evaluation" in result:
        ev = result["evaluation"]
        print(f"\n✅ Final Evaluation:")
        print(f"  Score: {ev.get('overall_score', 0):.3f}")
        print(f"  Recommendation: {ev.get('recommendation')}")
        print(f"  Dimension Scores:")
        
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            if dim in ev:
                score = ev[dim].get('score', 0)
                threshold = ev.get('thresholds', {}).get(dim, 0.7)
                icon = "✅" if score >= threshold else "❌"
                print(f"    {icon} {dim.capitalize()}: {score:.2f} (threshold: {threshold:.2f})")
        
        # Show issues if any
        coverage = ev.get('coverage_data', {})
        if coverage.get('missing_essential'):
            print(f"\n  ⚠️  Missing Categories: {', '.join(coverage['missing_essential'])}")
        
        coherence = ev.get('coherence', {})
        if coherence.get('duplicate_actions'):
            print(f"  ⚠️  Duplicates Found: {len(coherence['duplicate_actions'])}")
        if coherence.get('phase_errors'):
            print(f"  ⚠️  Phase Errors: {len(coherence['phase_errors'])}")
    
    # Save output
    with open("sequential_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: sequential_output.json")
    
    await maybe_await(runtime.stop())
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())