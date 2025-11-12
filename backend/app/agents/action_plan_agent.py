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

# Import your models (adjust path as needed)
from app.models.action_plan_models import Action, ActionPlanResponse


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
        
        Input message.content: location query (e.g., "Vancouver", "Ottawa") OR JSON with mode field
        Output: JSON string with ActionPlanResponse
        """
        content = message.content.strip()
        
        # Check if this is a mode request (SUMMARIZE/REWRITE)
        # If it is, route it to the summarize/rewrite handler instead of processing as location
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "mode" in payload:
                mode = payload.get("mode", "").upper()
                print(f"[ActionPlanAgent] ℹ️  Message has mode={mode} - routing to summarize/rewrite handler")
                # Instead of raising an exception, directly call the correct handler
                # This ensures proper routing without relying on RoutedAgent exception handling
                return await self.on_summarize_or_rewrite(message, ctx)
        except json.JSONDecodeError:
            # Not JSON, continue processing as location query
            pass
        except Exception as e:
            # If routing to summarize/rewrite fails, log and continue as location query
            print(f"[ActionPlanAgent] ⚠️  Error routing to summarize/rewrite: {e}, treating as location query")
        
        # If we get here, it's a location query (not a mode request)
        location_query = content
        print(f"[ActionPlanAgent] ✅ Processing as location query: {location_query}")
        
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
            results = govdoc_data.get("results", [])
            
            # Format GovDocAgent response for readability
            self._print_govdoc_summary(location_info, results, docs)
            
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
            
            # Step 3.5: Deduplicate actions using LLM
            all_actions = await self._deduplicate_actions_llm(all_actions, ctx)
            print(f"[ActionPlanAgent] After deduplication: {len(all_actions)} unique actions")
            
            # Step 4: Categorize into before/during/after using LLM
            before, during, after = await self._categorize_by_phase_llm(all_actions, ctx)
            
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

    @message_handler
    async def on_summarize_or_rewrite(self, message: Message, ctx: MessageContext) -> Message:
        """
        Handle summary composition or one-shot rewrite driven by Evaluator feedback.

        Expects message.content as JSON with:
        - mode: "SUMMARIZE" | "REWRITE"
        - plan: ActionPlanResponse as dict
        - constraints: optional constraints override
        - feedback: optional critique dict with required_changes/constraints/etc.
        Returns: final end-user text (string) formatted as requested.
        """
        try:
            print(f"\n[ActionPlanAgent] 📝 Received summarize/rewrite request")
            print(f"[ActionPlanAgent] Raw message content (first 200 chars): {message.content[:200]}...")
            
            payload = json.loads(message.content)
            mode = (payload.get("mode") or "SUMMARIZE").upper()
            plan_dict = payload.get("plan") or {}
            constraints = payload.get("constraints") or {}
            feedback = payload.get("feedback") or {}
            
            # Validate that this is indeed a mode request
            if mode not in ["SUMMARIZE", "REWRITE"]:
                error_msg = f"Invalid mode: {mode}. Must be SUMMARIZE or REWRITE"
                print(f"[ActionPlanAgent] ❌ {error_msg}")
                return Message(content=error_msg)
            
            print(f"[ActionPlanAgent] ✅ Processing mode={mode} request")
            
            # Extract location from plan JSON - this is what the user requested
            location_from_plan = plan_dict.get("location") or ""
            if location_from_plan:
                print(f"[ActionPlanAgent] Plan location (from JSON): {location_from_plan}")
            else:
                # Try to get location from display_name or other fields
                display_name = plan_dict.get("display_name")
                if display_name:
                    print(f"[ActionPlanAgent] ⚠️  No 'location' field, using display_name: {display_name}")
                    location_from_plan = display_name
                else:
                    print(f"[ActionPlanAgent] ⚠️  Warning: No location found in plan_dict")
                    location_from_plan = "Unknown Location"

            # Build model object for convenience where possible
            try:
                response_model = ActionPlanResponse(**plan_dict)
            except Exception as e:
                # Fall back to raw dict usage if shape differs
                print(f"[ActionPlanAgent] ⚠️  Could not build ActionPlanResponse model: {e}")
                response_model = None

            if mode == "SUMMARIZE":
                text = await self._compose_summary_text(plan_dict, constraints, None, ctx)
                print(f"[ActionPlanAgent] ✅ Generated summary text ({len(text)} chars)")
                print(f"[ActionPlanAgent] Summary preview: {text[:150]}...")
                return Message(content=text)
            elif mode == "REWRITE":
                # For REWRITE, we need the original summary to revise it
                # Get it from feedback if available, otherwise generate initial summary first
                original_summary = feedback.get("original_summary") or ""
                if not original_summary:
                    # If no original provided, generate initial summary first
                    print(f"[ActionPlanAgent] ⚠️  No original summary in feedback, generating initial summary first...")
                    original_summary = await self._compose_summary_text(plan_dict, constraints, None, ctx)
                
                text = await self._compose_summary_text(plan_dict, feedback.get("constraints") or constraints, feedback, ctx, original_summary=original_summary)
                print(f"[ActionPlanAgent] ✅ Generated revised text ({len(text)} chars)")
                print(f"[ActionPlanAgent] Revised preview: {text[:150]}...")
                return Message(content=text)
            else:
                error_msg = f"Invalid mode: {mode}"
                print(f"[ActionPlanAgent] ❌ {error_msg}")
                return Message(content=error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in summarize/rewrite request: {e}"
            print(f"[ActionPlanAgent] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return Message(content=error_msg)
        except Exception as e:
            error_msg = f"Summarize/Rewrite error: {e}"
            print(f"[ActionPlanAgent] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return Message(content=error_msg)
    
    def _print_govdoc_summary(self, location_info: Dict, results: List[Dict], docs: List[Dict]):
        """Format and print GovDocAgent response in a readable way."""
        print("\n" + "="*80)
        print("📚 GovDocAgent Response Summary")
        print("="*80)
        
        # Location info
        if isinstance(location_info, dict):
            loc_query = location_info.get("query", "N/A")
            loc_display = location_info.get("display_name", "N/A")
            print(f"\n📍 Location: {loc_query}")
            if loc_display != loc_query:
                print(f"   Full Name: {loc_display}")
        else:
            print(f"\n📍 Location: {location_info}")
        
        # Search results
        if results:
            print(f"\n🔍 Search Results ({len(results)}):")
            for i, result in enumerate(results[:5], 1):  # Show top 5
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                filetype = result.get("filetype", "")
                
                print(f"\n  {i}. 📄 {title}")
                if snippet:
                    print(f"     📝 {snippet}")
                if url:
                    display_url = url if len(url) <= 75 else url[:72] + "..."
                    print(f"     🔗 {display_url}")
                if filetype:
                    print(f"     📎 Type: {filetype.upper()}")
            if len(results) > 5:
                print(f"\n     ... and {len(results) - 5} more search results")
        
        # Downloaded documents
        if docs:
            print(f"\n📥 Downloaded Documents ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                doc_title = doc.get("title", "Untitled")
                doc_url = doc.get("url", "")
                clean_path = doc.get("clean_path", "")
                
                print(f"  {i}. 📄 {doc_title}")
                if doc_url:
                    display_url = doc_url if len(doc_url) <= 75 else doc_url[:72] + "..."
                    print(f"     🔗 {display_url}")
                if clean_path:
                    print(f"     💾 {clean_path}")
        else:
            print("\n⚠️  No documents downloaded")
        
        print("="*80 + "\n")
    
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
1. Title (concise, actionable, unique - avoid duplicating similar actions)
2. Description (detailed steps)
3. Priority (high/medium/low)
4. Category (e.g., evacuation, property_protection, emergency_kit, communication, insurance, family_plan)
5. Phase (before/during/after flood) - CRITICAL: Assign correctly based on WHEN the action should be done

PHASE CATEGORIZATION GUIDELINES:
- BEFORE FLOOD: Actions done proactively BEFORE flooding occurs
  • Preparing emergency kits
  • Creating evacuation plans
  • Purchasing insurance
  • Installing flood protection measures
  • Planning evacuation routes
  • Stockpiling supplies
  
- DURING FLOOD: Actions taken DURING an active flood event
  • Evacuating immediately
  • Moving to higher ground
  • Avoiding flooded areas
  • Following emergency instructions
  • Monitoring alerts
  
- AFTER FLOOD: Actions taken AFTER flooding has occurred and water recedes
  • Documenting damage for insurance claims
  • Cleaning up debris
  • Repairing property
  • Reviewing and updating preparedness plans based on experience
  • Filing insurance claims
  • Returning home safely after evacuation
  • Disinfecting contaminated areas

CRITICAL RULES:
1. NO DUPLICATES: If you see "create emergency kit" and "prepare emergency kit", extract only ONE (use the more complete one)
2. CORRECT PHASE: Actions like "document damage for insurance" MUST be "after", not "before"
3. CORRECT PHASE: "Review and update flood preparedness" after experiencing a flood MUST be "after", not "before"
4. Only extract distinct, unique actions - consolidate similar actions into one

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
            
            # Convert to Action objects and store phase information
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
                    # Store phase as a temporary attribute (not in Action model, but we'll use it)
                    phase = item.get("phase", "").lower().strip()
                    # Attach phase to action object as a custom attribute
                    setattr(action, "_extracted_phase", phase)
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
    
    async def _deduplicate_actions_llm(self, actions: List[Action], ctx: MessageContext) -> List[Action]:
        """
        Remove duplicate or very similar actions using LLM semantic understanding.
        LLM understands that "create emergency kit" and "prepare emergency kit" are duplicates.
        """
        if not actions:
            return []
        if len(actions) == 1:
            return actions
        
        # Prepare actions list for LLM
        actions_data = []
        for i, action in enumerate(actions):
            actions_data.append({
                "index": i,
                "title": action.title,
                "description": action.description,
                "priority": action.priority,
                "category": action.category
            })
        
        prompt = f"""You are analyzing a list of flood action items to identify and remove duplicates.

CRITICAL: Two actions are duplicates if they refer to the SAME specific task, even if worded differently.
Examples of duplicates:
- "Create emergency kit" and "Prepare emergency kit" → SAME task
- "Make an evacuation plan" and "Create evacuation plan" → SAME task
- "Buy flood insurance" and "Purchase flood insurance" → SAME task

NOT duplicates (different tasks):
- "Prepare emergency kit" and "Prepare evacuation route" → DIFFERENT tasks
- "Document property damage" and "File insurance claim" → DIFFERENT tasks (related but distinct)

ACTIONS TO REVIEW:
{json.dumps(actions_data, indent=2, ensure_ascii=False)}

Return ONLY valid JSON (no markdown, no explanations):
{{
  "unique_actions": [
    {{
      "index": <original_index>,
      "title": "<keep_or_improve_title>",
      "description": "<keep_or_merge_description>",
      "priority": "<highest_priority>",
      "category": "<category>"
    }}
  ],
  "duplicates_removed": [<list_of_removed_indices>]
}}

RULES:
1. If two actions are semantically the same task, keep only ONE (prefer the one with more complete description)
2. If merging duplicates, keep the best title and merge descriptions if helpful
3. Keep all unique/distinct actions
4. Return all unique actions in "unique_actions" array with their original indices

JSON OUTPUT:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            content = response.chat_message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result = json.loads(content)
            unique_items = result.get("unique_actions", [])
            unique_indices = {item["index"] for item in unique_items}
            
            # Create a map of index -> improved action from LLM
            improved_map = {}
            for item in unique_items:
                idx = item["index"]
                if idx < len(actions):
                    original = actions[idx]
                    # Use LLM's improved title/description if provided, otherwise keep original
                    improved_action = Action(
                        title=item.get("title", original.title),
                        description=item.get("description", original.description),
                        priority=item.get("priority", original.priority),
                        category=item.get("category", original.category),
                        source_doc=original.source_doc
                    )
                    # Preserve extracted_phase if it exists
                    if hasattr(original, "_extracted_phase"):
                        setattr(improved_action, "_extracted_phase", getattr(original, "_extracted_phase"))
                    improved_map[idx] = improved_action
            
            # Reconstruct Action objects from unique indices (use improved versions if available)
            unique_actions = []
            for i, action in enumerate(actions):
                if i in unique_indices:
                    unique_actions.append(improved_map.get(i, action))
            
            print(f"[ActionPlanAgent] LLM removed {len(actions) - len(unique_actions)} duplicate actions")
            return unique_actions
            
        except Exception as e:
            print(f"[ActionPlanAgent] ⚠️  LLM deduplication failed: {e}, using all actions")
            return actions  # Fallback: return all actions if LLM fails

    async def _categorize_by_phase_llm(self, actions: List[Action], ctx: MessageContext) -> tuple[List[Action], List[Action], List[Action]]:
        """
        Categorize actions into before/during/after flood phases using LLM understanding.
        LLM understands context: "document damage for insurance" is AFTER, not before.
        """
        if not actions:
            return [], [], []
        
        # Prepare actions for LLM
        actions_data = []
        for i, action in enumerate(actions):
            # Get LLM-extracted phase if available
            extracted_phase = getattr(action, "_extracted_phase", None)
            actions_data.append({
                "index": i,
                "title": action.title,
                "description": action.description,
                "extracted_phase": extracted_phase  # Include if LLM already extracted it
            })
        
        prompt = f"""You are categorizing flood action items into phases based on WHEN the action should be done.

PHASE DEFINITIONS:
- BEFORE FLOOD: Actions done proactively BEFORE any flooding occurs
  Examples: Preparing emergency kits, creating evacuation plans, purchasing insurance, installing flood protection, planning routes
  
- DURING FLOOD: Actions taken DURING an active flood event when flooding is happening NOW
  Examples: Evacuating immediately, moving to higher ground, avoiding flooded areas, following emergency instructions, monitoring alerts
  
- AFTER FLOOD: Actions taken AFTER flooding has occurred and water has receded
  Examples: Documenting damage for insurance, cleaning up debris, repairing property, filing insurance claims, reviewing and updating preparedness plans based on experience, returning home safely, disinfecting contaminated areas

CRITICAL RULES:
1. "Document damage for insurance" → AFTER (you can't document damage before it happens)
2. "File insurance claim" → AFTER (claims are filed after damage occurs)
3. "Review and update flood preparedness" (after experiencing a flood) → AFTER (this is post-event learning)
4. "Review and update flood preparedness" (proactively) → BEFORE (general preparedness review)
5. Use context clues: if description mentions "after flood" or "damage" or "claim", it's likely AFTER
6. If description mentions "during flood" or "immediately" or "evacuate now", it's DURING

ACTIONS TO CATEGORIZE:
{json.dumps(actions_data, indent=2, ensure_ascii=False)}

Return ONLY valid JSON (no markdown, no explanations):
{{
  "before": [<list_of_indices>],
  "during": [<list_of_indices>],
  "after": [<list_of_indices>]
}}

Each action index should appear in exactly ONE phase array.

JSON OUTPUT:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            content = response.chat_message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result = json.loads(content)
            before_indices = set(result.get("before", []))
            during_indices = set(result.get("during", []))
            after_indices = set(result.get("after", []))
            
            before = []
            during = []
            after = []
            
            for i, action in enumerate(actions):
                if i in before_indices:
                    before.append(action)
                elif i in during_indices:
                    during.append(action)
                elif i in after_indices:
                    after.append(action)
                else:
                    # If LLM didn't categorize, default to before (safest default)
                    print(f"[ActionPlanAgent] ⚠️  Action {i} ({action.title}) not categorized by LLM, defaulting to before")
                    before.append(action)
            
            return before, during, after
            
        except Exception as e:
            print(f"[ActionPlanAgent] ⚠️  LLM categorization failed: {e}, using extracted phases")
            # Fallback: use extracted phases if available
            before = []
            during = []
            after = []
            for action in actions:
                extracted_phase = getattr(action, "_extracted_phase", None)
                if extracted_phase == "before":
                    before.append(action)
                elif extracted_phase == "during":
                    during.append(action)
                elif extracted_phase == "after":
                    after.append(action)
                else:
                    before.append(action)  # Default to before
            
            return before, during, after

    async def _compose_summary_text(
        self,
        plan_dict: Dict,
        constraints: Dict | None,
        feedback: Dict | None,
        ctx: MessageContext,
        original_summary: str | None = None,
    ) -> str:
        """
        Compose an end-user summary from the plan with optional one-shot rewrite based on feedback.
        Returns plain text with format: {title}\n- bullet\n- bullet\n- bullet\nCTA: ...\nSource: ...
        """
        plan = plan_dict or {}
        max_words = ((constraints or {}).get("max_words")) or 120
        reading_level = ((constraints or {}).get("reading_level")) or "Grade 6"
        format_hint = ((constraints or {}).get("format")) or "title + 3 bullets + CTA + source link"

        required_changes = (feedback or {}).get("required_changes") or []
        suggested_phrases = (feedback or {}).get("suggested_phrases") or []
        mandatory_facts = (feedback or {}).get("mandatory_facts") or []

        # Collect minimal structured context
        location = plan.get("location") or ""
        sources = plan.get("sources") or []

        # Build a compact content sketch for LLM
        def _take_actions(key: str, n: int) -> list[Dict[str, str]]:
            arr = plan.get(key) or []
            out = []
            for it in arr[:n]:
                title = it.get("title") or ""
                desc = it.get("description") or ""
                out.append({"title": title, "description": desc})
            return out

        sketch = {
            "location": location,
            "before": _take_actions("before_flood", 5),
            "during": _take_actions("during_flood", 5),
            "after": _take_actions("after_flood", 5),
            "sources": sources,
        }

        sys = (
            "You are a public safety writer. Produce a concise, clear flood guidance message.\n"
            "Audience: general public.\n"
        )
        
        # Build the prompt based on whether this is a revision or initial summary
        if original_summary and feedback:
            # REVISION MODE: Include original summary and revision instructions
            user = (
                f"REVISE the following summary based on the feedback provided.\n\n"
                f"ORIGINAL SUMMARY:\n{original_summary}\n\n"
                f"REVISION INSTRUCTIONS:\n"
                f"- Keep the same core message and structure from the original\n"
                f"- Incorporate REQUIRED_CHANGES into the existing summary\n"
                f"- Maintain the same format: {format_hint}\n"
                f"- Max {max_words} words. Reading level: {reading_level}.\n"
                f"- Follow REQUIRED_CHANGES strictly. Keep numeric facts from sources unless flagged incorrect.\n\n"
                f"OUTPUT EXACTLY (revised version):\n{{title}}\n- bullet\n- bullet\n- bullet\nCTA: ...\nSource: ...\n\n"
                f"REQUIRED_CHANGES: {json.dumps(required_changes, ensure_ascii=False)}\n"
                f"SUGGESTED_PHRASES: {json.dumps(suggested_phrases, ensure_ascii=False)}\n"
                f"MANDATORY_FACTS: {json.dumps(mandatory_facts, ensure_ascii=False)}\n\n"
                f"SKETCH (for reference): {json.dumps(sketch, ensure_ascii=False)}\n"
            )
        else:
            # INITIAL SUMMARY MODE: Generate from scratch
            user = (
                f"FORMAT: {format_hint}. Max {max_words} words. Reading level: {reading_level}.\n"
                "If provided, follow REQUIRED CHANGES strictly. Keep numeric facts from sources unless flagged incorrect.\n"
                "OUTPUT EXACTLY:\n{title}\n- bullet\n- bullet\n- bullet\nCTA: ...\nSource: ...\n\n"
                f"SKETCH: {json.dumps(sketch, ensure_ascii=False)}\n"
                f"REQUIRED_CHANGES: {json.dumps(required_changes, ensure_ascii=False)}\n"
                f"SUGGESTED_PHRASES: {json.dumps(suggested_phrases, ensure_ascii=False)}\n"
                f"MANDATORY_FACTS: {json.dumps(mandatory_facts, ensure_ascii=False)}\n"
            )

        tm = TextMessage(content=user, source="user")
        try:
            res = await self._llm.on_messages([tm], ctx.cancellation_token)
            text = (res.chat_message.content or "").strip()
            # Strip accidental code fences
            if text.startswith("```"):
                parts = text.split("```", 2)
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("text"):
                    text = text[4:]
            # Trim to max words softly
            words = text.split()
            if len(words) > int(max_words) + 20:
                text = " ".join(words[: int(max_words)])
            return text
        except Exception as e:
            return f"{plan.get('location') or 'Flood Guidance'}\n- Prepare supplies\n- Stay informed\n- Avoid flood zones\nCTA: Follow local alerts and instructions.\nSource: {sources[0] if sources else 'official guide'} (fallback due to {e})"


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