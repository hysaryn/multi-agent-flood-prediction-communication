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

from backend.app.models.message_model import Message
from pathlib import Path
from typing import List, Dict
import json
import asyncio
from datetime import datetime

# Import your models (adjust path as needed)
from backend.app.models.action_plan_models import Action, ActionPlanResponse


class ActionPlanAgent(RoutedAgent):
    """
    Agent that generates structured flood action plans from government documents.
    
    Workflow:
    1. Receives location query
    2. Calls GovDocAgent to get documents
    3. Reads cleaned text from downloaded PDFs
    4. Extracts actions using LLM
    5. Categorizes into before/during/after phases
    6. Returns structured action plan
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
        Main handler: Generate action plan for a location.
        
        Input message.content: location query (e.g., "Vancouver", "Ottawa")
        Output: JSON string with ActionPlanResponse
        """
        location_query = message.content.strip()
        print(f"[ActionPlanAgent] Generating plan for: {location_query}")
        
        try:
            # Step 1: Get documents from GovDocAgent
            print(f"[ActionPlanAgent] Calling GovDocAgent...")
            govdoc_response = await self._runtime.send_message(
                Message(content=location_query),
                AgentId("GovDoc", "default")
            )
            
            govdoc_data = json.loads(govdoc_response.content)
            location_info = govdoc_data.get("location", {})
            docs = govdoc_data.get("docs", [])
            
            if not docs:
                print(f"[ActionPlanAgent] No documents found for {location_query}")
                return Message(content=json.dumps({
                    "error": "No documents found",
                    "location": location_query
                }))
            
            print(f"[ActionPlanAgent] Found {len(docs)} documents")
            
            # Step 2: Read all document texts
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
            
            # Step 3: Extract actions from all documents
            print(f"[ActionPlanAgent] Extracting actions from {len(doc_texts)} documents...")
            all_actions = []
            
            for doc in doc_texts:
                actions = await self._extract_actions_from_doc(doc, ctx)
                all_actions.extend(actions)
                print(f"[ActionPlanAgent] Extracted {len(actions)} actions from {doc['url']}")
            
            print(f"[ActionPlanAgent] Total actions extracted: {len(all_actions)}")
            
            # Step 4: Categorize into before/during/after
            before, during, after = self._categorize_by_phase(all_actions)
            
            print(f"[ActionPlanAgent] Categorized: Before={len(before)}, During={len(during)}, After={len(after)}")
            
            # Step 5: Build final response
            response = ActionPlanResponse(
                location=location_query,
                display_name=location_info.get("display_name"),
                before_flood=before,
                during_flood=during,
                after_flood=after,
                sources=[doc["url"] for doc in doc_texts],
                generated_at=datetime.utcnow().isoformat()
            )
            
            print(f"[ActionPlanAgent] ✅ Plan complete: {response.total_actions()} total actions")
            
            return Message(content=response.model_dump_json(indent=2))
        
        except Exception as e:
            print(f"[ActionPlanAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "location": location_query
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
    """Test the ActionPlanAgent"""
    from app.agents.govdoc_agent import GovDocAgent
    
    runtime = SingleThreadedAgentRuntime()
    
    # Register both agents
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    
    await maybe_await(runtime.start())
    
    # Test with a location
    print("=" * 60)
    print("Testing ActionPlanAgent with Vancouver")
    print("=" * 60)
    
    response = await runtime.send_message(
        Message(content="Vancouver, BC"),
        AgentId("ActionPlan", "default")
    )
    
    # Print the result
    result = json.loads(response.content)
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    await maybe_await(runtime.stop())


if __name__ == "__main__":
    asyncio.run(main())