# python -m app.agents.action_plan_agent
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
from pathlib import Path
from typing import List, Dict
import json
import asyncio
from datetime import datetime


from app.models.action_plan_models import Action, ActionPlanResponse


class ActionPlanAgent(RoutedAgent):
    """
    Summarizer Agent: Extracts structured actions from government documents.
    
    Workflow (SEQUENTIAL):
    1. Receives location info + documents from GovDocAgent
    2. Reads cleaned text from downloaded PDFs
    3. Extracts actions using LLM
    4. Categorizes into before/during/after phases
    5. Returns structured action plan
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("ActionPlan")
        self._runtime = runtime
        
        # LLM for extracting and categorizing actions
        self._llm = AssistantAgent(
            "ActionPlanLLM",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                # model="gpt-4o",  # Use this for better quality
            ),
        )
    
    @message_handler
    async def on_action_plan_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Main handler: Generate action plan from documents.
        
        Input message.content: Complete GovDocAgent output JSON:
        {
            "location": {
                "query": "Vancouver, BC",
                "display_name": "Burnaby, British Columbia, Canada",
                "latitude": 49.2827,
                "longitude": -123.1207,
                ...
            },
            "results": [...],  // Search results metadata
            "docs": [
                {
                    "url": "https://...",
                    "title": "...",
                    "clean_path": "/path/to/extracted.txt",
                    "place_key": "vancouver-bc"
                }
            ]
        }
        
        Output: JSON string with ActionPlanResponse
        """
        try:
            # ====== CHANGED: Accept full GovDocAgent output ======
            input_data = json.loads(message.content)
            
            # Extract location info from nested structure
            location_info = input_data.get("location", {})
            location_query = location_info.get("query") or location_info.get("display_name", "Unknown")
            display_name = location_info.get("display_name")
            
            # Extract documents
            docs = input_data.get("docs", [])
            
            print(f"[ActionPlanAgent] Generating plan for: {location_query}")
            print(f"[ActionPlanAgent] Received {len(docs)} documents")

            if not docs:
                print(f"[ActionPlanAgent] No documents provided for {location_query}")
                return Message(content=json.dumps({
                    "error": "No documents provided",
                    "location": location_query
                }))
            
            # Step 1: Read all document texts (same as before)
            doc_texts = []
            for doc in docs:
                clean_path = doc.get("clean_path")
                url = doc.get("url")
                
                if not clean_path or not Path(clean_path).exists():
                    print(f"[ActionPlanAgent] ⚠️  Skipping missing file: {clean_path}")
                    continue
                
                text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
                doc_texts.append({
                    "url": url,
                    "text": text[:15000],  # Limit to first 15k chars to avoid context issues
                    "title": doc.get("title", "Untitled")
                })
                print(f"[ActionPlanAgent] ✅ Loaded: {clean_path} ({len(text)} chars)")
            
            if not doc_texts:
                return Message(content=json.dumps({
                    "error": "No readable documents found",
                    "location": location_query
                }))
            
            # Step 2: Extract actions from all documents (same as before)
            print(f"[ActionPlanAgent] Extracting actions from {len(doc_texts)} documents...")
            all_actions = []
            
            for doc in doc_texts:
                actions = await self._extract_actions_from_doc(doc, ctx)
                all_actions.extend(actions)
                print(f"[ActionPlanAgent] Extracted {len(actions)} actions from {doc['url']}")
            
            print(f"[ActionPlanAgent] Total actions extracted: {len(all_actions)}")
            
            # Step 3: Categorize into before/during/after (same as before)
            before, during, after = self._categorize_by_phase(all_actions)
            
            print(f"[ActionPlanAgent] Categorized: Before={len(before)}, During={len(during)}, After={len(after)}")
            
            # Step 4: Build final response
            response = ActionPlanResponse(
                location=location_query,
                display_name=display_name,  # Use provided display_name
                before_flood=before,
                during_flood=during,
                after_flood=after,
                sources=[doc["url"] for doc in doc_texts],
                generated_at=datetime.utcnow().isoformat()
            )
            
            print(f"[ActionPlanAgent] ✅ Plan complete: {response.total_actions()} total actions")
            
            return Message(content=response.model_dump_json(indent=2))
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent] ❌ Invalid JSON input: {e}")
            return Message(content=json.dumps({
                "error": f"Invalid JSON input: {str(e)}",
                "hint": "Expected JSON with 'location', 'display_name', and 'docs' fields"
            }))
        
        except Exception as e:
            print(f"[ActionPlanAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "location": input_data.get("location", "Unknown") if 'input_data' in locals() else "Unknown"
            }))
    
    async def _extract_actions_from_doc(self, doc: Dict, ctx: MessageContext) -> List[Action]:
        """
        Extract actionable items from a single document using LLM.
        
        Args:
            doc: Dict with 'text', 'url', 'title'
            ctx: Message context
        
        Returns:
            List of Action objects
        """
        prompt = f"""You are analyzing a government flood preparedness document.

Document: {doc['title']}
Source: {doc['url']}

Extract ALL actionable items for residents. For each action, provide:
1. Title (concise, actionable)
2. Description (detailed steps)
3. Priority (high/medium/low)
4. Category (e.g., evacuation, property_protection, emergency_kit, communication, insurance, family_plan)
5. Phase (before/during/after flood)

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
    "category": "...",
    "phase": "before|during|after"
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
                    print(f"[ActionPlanAgent] ⚠️  Skipping invalid action: {e}")
                    continue
            
            return actions
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent] ❌ JSON parse error: {e}")
            print(f"[ActionPlanAgent] LLM response: {content[:500]}")
            return []
        except Exception as e:
            print(f"[ActionPlanAgent] ❌ Error extracting actions: {e}")
            return []
    
    def _categorize_by_phase(self, actions: List[Action]) -> tuple[List[Action], List[Action], List[Action]]:
        """
        Categorize actions into before/during/after flood phases.
        
        Uses simple keyword matching as a fallback if phase wasn't determined by LLM.
        """
        before = []
        during = []
        after = []
        
        # Keywords for classification
        before_keywords = ["prepare", "plan", "kit", "insurance", "document", "elevate", "install", "purchase"]
        during_keywords = ["evacuate", "avoid", "stay", "listen", "monitor", "turn off", "move to"]
        after_keywords = ["cleanup", "repair", "document damage", "claim", "return", "inspect", "disinfect"]
        
        for action in actions:
            text = (action.title + " " + action.description).lower()
            
            # Check for explicit phase keywords
            if any(kw in text for kw in before_keywords):
                before.append(action)
            elif any(kw in text for kw in during_keywords):
                during.append(action)
            elif any(kw in text for kw in after_keywords):
                after.append(action)
            else:
                # Default: most actions are preparation (before)
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
    """Test the ActionPlanAgent in SEQUENTIAL mode"""
    from app.agents.govdoc_agent import GovDocAgent
    
    runtime = SingleThreadedAgentRuntime()
    
    # Register both agents
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    
    await maybe_await(runtime.start())
    
    print("=" * 60)
    print("Testing SEQUENTIAL Architecture")
    print("=" * 60)
    
    # ====== STEP 1: Call GovDocAgent ======
    print("\n[STEP 1] Calling GovDocAgent...")
    govdoc_response = await runtime.send_message(
        Message(content="Vnancouver, BC"),
        AgentId("GovDoc", "default")
    )
    
    govdoc_data = json.loads(govdoc_response.content)
    print(f"[STEP 1] ✅ Received {len(govdoc_data.get('docs', []))} documents")
    
    # ====== STEP 2: Call ActionPlanAgent with FULL GovDoc output ======
    print("\n[STEP 2] Calling ActionPlanAgent (Summarizer)...")
    
    # Pass the COMPLETE GovDocAgent output directly
    action_plan_response = await runtime.send_message(
        Message(content=govdoc_response.content),  # Use full output as-is
        AgentId("ActionPlan", "default")
    )
    
    # Print the result
    result = json.loads(action_plan_response.content)
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    await maybe_await(runtime.stop())


if __name__ == "__main__":
    asyncio.run(main())