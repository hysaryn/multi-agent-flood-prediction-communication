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
from typing import List, Dict
import json
import asyncio
from datetime import datetime


class ActionPlanAgent(RoutedAgent):
    """
    Agent that generates structured action plans from government documents.
    
    Sequential Architecture:
    - Receives: Complete GovDocAgent output
    - Processes: Extracts and categorizes actions
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
        Generate action plan from government documents.
        
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
        input_data = None  # Initialize to avoid UnboundLocalError
        
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
            all_actions = []
            
            for doc in doc_texts:
                actions = await self._extract_actions_from_doc(doc, ctx)
                all_actions.extend(actions)
                print(f"[ActionPlanAgent]   Extracted {len(actions)} actions")
            
            print(f"[ActionPlanAgent] Total actions extracted: {len(all_actions)}")
            
            # Categorize into before/during/after
            before, during, after = self._categorize_by_phase(all_actions)
            
            print(f"[ActionPlanAgent] Categorized: Before={len(before)}, During={len(during)}, After={len(after)}")
            
            # Build action plan
            from datetime import datetime, timezone
            
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
            
            # Safe error response
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
    
    async def _extract_actions_from_doc(self, doc: Dict, ctx: MessageContext) -> List[Action]:
        """
        Extract actionable items from a single document using LLM.
        """
        prompt = f"""You are analyzing a government flood preparedness document.

Document: {doc['title']}
Source: {doc['url']}

Extract ALL actionable items for residents. For each action, provide:
1. Title (concise, actionable)
2. Description (detailed steps with specific quantities, times, and locations)
3. Priority (high/medium/low)
4. Category (evacuation, property_protection, emergency_kit, communication, insurance, family_plan)

Focus on:
- Concrete actions residents can take
- Specific preparedness steps
- Emergency response procedures
- Recovery and cleanup tasks

Return ONLY valid JSON array (no markdown, no explanations):
[
  {{
    "title": "...",
    "description": "...",
    "priority": "high|medium|low",
    "category": "..."
  }}
]

Document text:
{doc['text'][:10000]}

JSON OUTPUT:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Parse LLM response
            content = response.chat_message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            actions_data = json.loads(content)
            
            # Convert to Action objects
            actions = []
            for item in actions_data:
                try:
                    action = Action(
                        title=item["title"],
                        description=item["description"],
                        priority=item["priority"],
                        category=item["category"],
                        source_doc=doc["url"]
                    )
                    actions.append(action)
                except Exception as e:
                    print(f"[ActionPlanAgent]   ⚠️  Skipping invalid action: {e}")
                    continue
            
            return actions
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent]   ❌ JSON parse error: {e}")
            print(f"[ActionPlanAgent]   LLM response preview: {content[:300]}")
            return []
        except Exception as e:
            print(f"[ActionPlanAgent]   ❌ Error extracting actions: {e}")
            return []
    
    def _categorize_by_phase(self, actions: List[Action]) -> tuple[List[Action], List[Action], List[Action]]:
        """
        Categorize actions into before/during/after flood phases using keyword matching.
        """
        before = []
        during = []
        after = []
        
        # Keywords for classification
        before_keywords = ["prepare", "plan", "kit", "insurance", "document", "elevate", "install", "purchase", "review", "create", "gather"]
        during_keywords = ["evacuate", "avoid", "stay away", "listen", "monitor", "turn off", "move to", "shut off", "leave", "go to"]
        after_keywords = ["cleanup", "repair", "document damage", "claim", "return", "inspect", "disinfect", "restore", "file claim"]
        
        for action in actions:
            text = (action.title + " " + action.description).lower()
            
            # Count keyword matches for each phase
            before_score = sum(1 for kw in before_keywords if kw in text)
            during_score = sum(1 for kw in during_keywords if kw in text)
            after_score = sum(1 for kw in after_keywords if kw in text)
            
            # Assign to phase with highest score
            if during_score > before_score and during_score > after_score:
                during.append(action)
            elif after_score > before_score and after_score > during_score:
                after.append(action)
            else:
                # Default to before (preparation is most common)
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
    
    # Revision history
    if "revision_history" in result:
        print(f"\n🔄 Revision History: {len(result['revision_history'])} iterations")
        for rev in result["revision_history"]:
            print(f"  Iteration {rev['iteration']}: {rev['recommendation']} (score: {rev['overall_score']:.3f})")
    
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
        print(f"    Before: {before} ({before/total*100:.0f}%)")
        print(f"    During: {during} ({during/total*100:.0f}%)")
        print(f"    After: {after} ({after/total*100:.0f}%)")
    
    # Final evaluation
    if "evaluation" in result:
        ev = result["evaluation"]
        print(f"\n✅ Final Evaluation:")
        print(f"  Score: {ev.get('overall_score', 0):.3f}")
        print(f"  Recommendation: {ev.get('recommendation')}")
        
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            if dim in ev:
                score = ev[dim].get('score', 0)
                threshold = ev.get('thresholds', {}).get(dim, 0.7)
                icon = "✅" if score >= threshold else "❌"
                print(f"    {icon} {dim.capitalize()}: {score:.2f}")
    
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