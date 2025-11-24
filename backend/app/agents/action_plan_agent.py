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
from app.services.token_tracker import tracker
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
        Generate action plan with support for normal and revision modes.
        
        Input (from GovDocAgent or EvaluatorAgent): {
            "location": {"query": "...", "display_name": "...", ...},
            "results": [...],
            "docs": [{"url": "...", "title": "...", "clean_path": "...", ...}],
            // For revision mode (iteration 2):
            "revision_notes": [...],
            "previous_plan": {...},
            "evaluation": {...},
            "iteration": 2
        }
        
        Output (for EvaluatorAgent): {
            "action_plan": ActionPlanResponse,
            "govdoc_data": {...},
            "location": "...",
            "iteration": 1|2
        }
        """
        input_data = None
        
        try:
            # Parse input
            input_data = json.loads(message.content)
            
            # Check if this is a revision request (iteration 2) or add_to_existing
            revision_notes = input_data.get("revision_notes")
            previous_plan_data = input_data.get("previous_plan")
            iteration = input_data.get("iteration", 1)
            add_to_existing = input_data.get("add_to_existing", False)
            
            # Check for add_to_existing mode: add new actions from additional docs to existing plan
            if add_to_existing and previous_plan_data is not None:
                # ADD_TO_EXISTING MODE: Extract from new docs and merge with existing plan
                print(f"\n[ActionPlanAgent] ➕ ADD_TO_EXISTING MODE (Iteration {iteration})")
                return await self._handle_add_to_existing_request(input_data, ctx)
            # Check for revision mode: iteration 2 AND has previous_plan (revision_notes can be empty list)
            elif iteration == 2 and previous_plan_data is not None:
                # REVISION MODE: Regenerate plan considering feedback
                print(f"\n[ActionPlanAgent] 🔄 REVISION MODE (Iteration {iteration})")
                return await self._handle_revision_request(input_data, ctx)
            else:
                # NORMAL MODE: Extract from documents
                print(f"\n[ActionPlanAgent] 📄 NORMAL MODE (Iteration {iteration})")
                return await self._handle_normal_request(input_data, ctx)
        
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
    
    async def _handle_normal_request(self, input_data: Dict, ctx: MessageContext) -> Message:
        """Normal extraction mode (iteration 1)."""
        start_time = datetime.now()
        print(f"[ActionPlanAgent] ⏰ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        location_info = input_data.get("location", {})
        
        # Handle both dict and string location formats
        if isinstance(location_info, str):
            location_query = location_info
            display_name = None
        elif isinstance(location_info, dict):
            location_query = location_info.get("query") or location_info.get("display_name", "Unknown")
            display_name = location_info.get("display_name")
        else:
            location_query = "Unknown"
            display_name = None
        
        # Get docs from either direct "docs" key or from "govdoc_data"
        docs = input_data.get("docs", [])
        if not docs:
            govdoc_data = input_data.get("govdoc_data", {})
            docs = govdoc_data.get("docs", [])
        
        iteration = input_data.get("iteration", 1)
        
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
            # Use up to 30,000 characters per document (~7,500 tokens, safe for 128k limit)
            # This leaves room for prompt overhead and response tokens
            doc_texts.append({
                "url": url,
                "text": text[:15000],
                "title": doc.get("title", "Untitled")
            })
            text_length = len(text)
            used_length = min(text_length, 15000)
            print(f"[ActionPlanAgent]   ✅ Loaded: {doc.get('title', 'Untitled')[:50]}... ({used_length}/{text_length} chars used)")
        
        if not doc_texts:
            return Message(content=json.dumps({
                "error": "No readable documents found",
                "location": location_query,
                "action_plan": None,
                "govdoc_data": input_data
            }))
        
        # Extract actions from all documents in parallel
        print(f"[ActionPlanAgent] Extracting actions from {len(doc_texts)} documents in parallel...")
        extraction_start = datetime.now()
        
        # Create parallel extraction tasks
        extraction_tasks = [
            self._extract_actions_from_doc(doc, location_query, ctx)
            for doc in doc_texts
        ]
        
        # Execute all extractions in parallel
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Combine results and handle any errors
        all_actions_with_phases = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ActionPlanAgent]   ⚠️  Error extracting from doc {i+1}: {result}")
                continue
            all_actions_with_phases.extend(result)
            print(f"[ActionPlanAgent]   Extracted {len(result)} actions from doc {i+1}")
        
        extraction_duration = (datetime.now() - extraction_start).total_seconds()
        print(f"[ActionPlanAgent] Total actions extracted: {len(all_actions_with_phases)} [⏱️  {extraction_duration:.2f}s for {len(doc_texts)} docs]")
        
        # Deduplicate using LLM
        print(f"[ActionPlanAgent] Deduplicating using LLM...")
        deduplicated = await self._deduplicate_actions_with_phases(all_actions_with_phases, ctx)
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
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[ActionPlanAgent] ✅ Plan complete: {action_plan.total_actions()} total actions [⏱️  {duration:.2f}s]")
        
        # Print token usage summary for this agent
        summary = tracker.get_summary()
        agent_summary = summary.get("by_agent", {}).get("ActionPlanAgent", {})
        if agent_summary:
            print(f"[ActionPlanAgent] 💰 Token usage: {agent_summary.get('total_tokens', 0):,} tokens (${agent_summary.get('cost_usd', 0):.6f})")
        
        # Output for feedback loop
        output = {
            "action_plan": action_plan.model_dump(mode='python'),
            "govdoc_data": input_data,
            "location": location_query,
            "iteration": iteration
        }
        
        return Message(content=json.dumps(output, ensure_ascii=False))
    
    async def _handle_revision_request(self, input_data: Dict, ctx: MessageContext) -> Message:
        """
        Revision mode: Add/revise actions based on current plan + revision notes.
        
        Strategy: Keep existing actions, add new ones that address issues, and update existing ones.
        
        Input: {
            "govdoc_data": {...},  # Same documents from iteration 1
            "revision_notes": [...],  # Issues identified by evaluator
            "previous_plan": {...},  # Original plan
            "evaluation": {...},  # Full evaluation details
            "location": "...",
            "iteration": 2
        }
        """
        start_time = datetime.now()
        print(f"[ActionPlanAgent] ⏰ Revision started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        govdoc_data = input_data.get("govdoc_data", {})
        revision_notes = input_data.get("revision_notes", [])
        previous_plan = ActionPlanResponse(**input_data.get("previous_plan", {}))
        evaluation = input_data.get("evaluation", {})
        location = input_data.get("location", "Unknown")
        
        docs = govdoc_data.get("docs", [])
        
        print(f"[ActionPlanAgent] Revision mode: {len(revision_notes)} issues to address")
        print(f"[ActionPlanAgent] Previous plan: {previous_plan.total_actions()} actions")
        print(f"[ActionPlanAgent] Using {len(docs)} existing documents")
        
        # Extract revision notes from evaluation if not provided
        if not revision_notes:
            revision_notes = []
            for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
                if dim in evaluation:
                    issues = evaluation[dim].get('issues', [])
                    if issues:
                        revision_notes.extend(issues)
            
            # Add missing categories
            coverage = evaluation.get('coverage_data', {})
            missing = coverage.get('missing_essential', [])
            if missing:
                revision_notes.append(f"Missing essential categories: {', '.join(missing)}")
        
        # Re-extract with focus on issues
        location_info = govdoc_data.get("location", {})
        
        # Handle both dict and string location formats
        if isinstance(location_info, str):
            location_query = location_info
            display_name = None
        elif isinstance(location_info, dict):
            location_query = location_info.get("query") or location_info.get("display_name", location)
            display_name = location_info.get("display_name")
        else:
            location_query = location or "Unknown"
            display_name = None
        
        if not docs:
            print(f"[ActionPlanAgent] No documents available for revision")
            return Message(content=json.dumps({
                "error": "No documents available for revision",
                "location": location_query,
                "action_plan": previous_plan.model_dump(mode='python'),
                "govdoc_data": govdoc_data,
                "iteration": 2
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
            # Use up to 30,000 characters per document (~7,500 tokens, safe for 128k limit)
            # This leaves room for prompt overhead and response tokens
            doc_texts.append({
                "url": url,
                "text": text[:30000],
                "title": doc.get("title", "Untitled")
            })
        
        if not doc_texts:
            return Message(content=json.dumps({
                "error": "No readable documents found",
                "location": location_query,
                "action_plan": previous_plan.model_dump(mode='python'),
                "govdoc_data": govdoc_data,
                "iteration": 2
            }))
        
        # Decision: Extract additional actions OR just revise with LLM?
        strategy = await self._decide_revision_strategy(revision_notes, previous_plan, ctx)
        
        if strategy == "EXTRACT_ADDITIONAL":
            # Strategy A: Extract additional actions from documents
            print(f"[ActionPlanAgent] Strategy: EXTRACT_ADDITIONAL - Extracting new actions from documents...")
            
            # Start with existing actions from previous plan
            existing_actions_with_phases = []
            for action in previous_plan.before_flood:
                existing_actions_with_phases.append((action, "before"))
            for action in previous_plan.during_flood:
                existing_actions_with_phases.append((action, "during"))
            for action in previous_plan.after_flood:
                existing_actions_with_phases.append((action, "after"))
            
            print(f"[ActionPlanAgent] Keeping {len(existing_actions_with_phases)} existing actions")
            
            # Extract NEW actions that address the revision notes in parallel
            print(f"[ActionPlanAgent] Extracting additional actions in parallel...")
            extraction_start = datetime.now()
            
            # Create parallel extraction tasks
            extraction_tasks = [
                self._extract_actions_with_revision_focus(
                    doc, location_query, revision_notes, previous_plan, ctx
                )
                for doc in doc_texts
            ]
            
            # Execute all extractions in parallel
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Combine results and handle any errors
            new_actions_with_phases = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ActionPlanAgent]   ⚠️  Error extracting from doc {i+1}: {result}")
                    continue
                new_actions_with_phases.extend(result)
            
            extraction_duration = (datetime.now() - extraction_start).total_seconds()
            print(f"[ActionPlanAgent] Extracted {len(new_actions_with_phases)} new/additional actions [⏱️  {extraction_duration:.2f}s for {len(doc_texts)} docs]")
            
            # Merge: combine existing + new, then deduplicate
            all_actions_with_phases = existing_actions_with_phases + new_actions_with_phases
            
            # Deduplicate using LLM (this will remove duplicates between existing and new)
            print(f"[ActionPlanAgent] Deduplicating using LLM...")
            deduplicated = await self._deduplicate_actions_with_phases(all_actions_with_phases, ctx)
            print(f"[ActionPlanAgent] After deduplication: {len(deduplicated)} actions (was {len(all_actions_with_phases)})")
            
            # Categorize by phase
            before, during, after = self._categorize_by_phase(deduplicated)
        else:
            # Strategy B: Use LLM to revise existing plan based on revision notes
            print(f"[ActionPlanAgent] Strategy: LLM_REVISE - Revising existing plan with LLM...")
            before, during, after = await self._llm_revise_plan(
                previous_plan, revision_notes, location_query, ctx
            )
        
        # Build revised action plan
        revised_plan = ActionPlanResponse(
            location=location_query,
            display_name=display_name,
            before_flood=before,
            during_flood=during,
            after_flood=after,
            sources=[doc["url"] for doc in doc_texts],
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[ActionPlanAgent] ✅ Revised plan complete: {revised_plan.total_actions()} total actions [⏱️  {duration:.2f}s]")
        print(f"[ActionPlanAgent]   Change: {revised_plan.total_actions() - previous_plan.total_actions():+d} actions")
        
        # Print token usage summary for this agent
        summary = tracker.get_summary()
        agent_summary = summary.get("by_agent", {}).get("ActionPlanAgent", {})
        if agent_summary:
            print(f"[ActionPlanAgent] 💰 Token usage: {agent_summary.get('total_tokens', 0):,} tokens (${agent_summary.get('cost_usd', 0):.6f})")
        
        return Message(content=json.dumps({
            "action_plan": revised_plan.model_dump(mode='python'),
            "govdoc_data": govdoc_data,
            "location": location_query,
            "iteration": 2
        }, ensure_ascii=False))
    
    async def _handle_add_to_existing_request(self, input_data: Dict, ctx: MessageContext) -> Message:
        """
        Add to existing mode: Summarize action plans from additional documents with targeted focus and combine with existing plan.
        
        Strategy: 
        1. Summarize actions from NEW documents (focusing on revision notes)
        2. Combine summarized actions with existing plan
        3. Deduplicate and merge
        
        Input: {
            "docs": [...],  # All docs (existing + new, merged by GovDocAgent)
            "previous_plan": {...},  # Original plan to keep
            "revision_notes": [...],  # Focus areas for summarization
            "location": "...",
            "iteration": 2
        }
        """
        start_time = datetime.now()
        print(f"[ActionPlanAgent] ⏰ Add to existing started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        docs = input_data.get("docs", [])
        previous_plan = ActionPlanResponse(**input_data.get("previous_plan", {}))
        revision_notes = input_data.get("revision_notes", [])
        existing_govdoc_data = input_data.get("govdoc_data", {})
        
        location_info = input_data.get("location", {})
        if isinstance(location_info, str):
            location_query = location_info
            display_name = None
        elif isinstance(location_info, dict):
            location_query = location_info.get("query") or location_info.get("display_name", "Unknown")
            display_name = location_info.get("display_name")
        else:
            location_query = "Unknown"
            display_name = None
        
        print(f"[ActionPlanAgent] Add to existing mode: Keeping {previous_plan.total_actions()} existing actions")
        
        # Identify which docs are NEW (not in previous plan sources)
        previous_sources = set(previous_plan.sources)
        new_docs = [doc for doc in docs if doc.get("url") not in previous_sources]
        existing_docs = [doc for doc in docs if doc.get("url") in previous_sources]
        
        print(f"[ActionPlanAgent] Processing {len(new_docs)} new documents (out of {len(docs)} total)")
        if revision_notes:
            print(f"[ActionPlanAgent] Focus areas: {len(revision_notes)} revision notes")
        
        if not new_docs:
            print(f"[ActionPlanAgent] No new documents to process, returning existing plan")
            return Message(content=json.dumps({
                "action_plan": previous_plan.model_dump(mode='python'),
                "govdoc_data": input_data,
                "location": location_query,
                "iteration": 2
            }, ensure_ascii=False))
        
        # Read NEW document texts only
        new_doc_texts = []
        for doc in new_docs:
            clean_path = doc.get("clean_path")
            url = doc.get("url")
            
            if not clean_path or not Path(clean_path).exists():
                print(f"[ActionPlanAgent]   ⚠️  Skipping missing file: {clean_path}")
                continue
            
            text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
            new_doc_texts.append({
                "url": url,
                "text": text[:10000],  # Use same limit as normal mode
                "title": doc.get("title", "Untitled")
            })
        
        if not new_doc_texts:
            return Message(content=json.dumps({
                "error": "No readable new documents found",
                "location": location_query,
                "action_plan": previous_plan.model_dump(mode='python'),
                "govdoc_data": input_data,
                "iteration": 2
            }))
        
        # Start with ALL existing actions from previous plan
        existing_actions_with_phases = []
        for action in previous_plan.before_flood:
            existing_actions_with_phases.append((action, "before"))
        for action in previous_plan.during_flood:
            existing_actions_with_phases.append((action, "during"))
        for action in previous_plan.after_flood:
            existing_actions_with_phases.append((action, "after"))
        
        print(f"[ActionPlanAgent] Keeping {len(existing_actions_with_phases)} existing actions")
        
        # Summarize actions from NEW documents with targeted focus on revision notes
        print(f"[ActionPlanAgent] Summarizing actions from new documents with targeted focus...")
        summarization_start = datetime.now()
        
        # Summarize with focus on revision notes
        summarization_tasks = [
            self._summarize_actions_from_doc(
                doc, location_query, revision_notes, previous_plan, ctx
            )
            for doc in new_doc_texts
        ]
        
        # Execute all summarizations in parallel
        results = await asyncio.gather(*summarization_tasks, return_exceptions=True)
        
        # Combine results and handle any errors
        summarized_actions_with_phases = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ActionPlanAgent]   ⚠️  Error summarizing doc {i+1}: {result}")
                continue
            summarized_actions_with_phases.extend(result)
        
        summarization_duration = (datetime.now() - summarization_start).total_seconds()
        print(f"[ActionPlanAgent] Summarized {len(summarized_actions_with_phases)} actions from new docs [⏱️  {summarization_duration:.2f}s for {len(new_doc_texts)} docs]")
        
        # Merge: combine existing + summarized, then deduplicate
        all_actions_with_phases = existing_actions_with_phases + summarized_actions_with_phases
        
        # Deduplicate using LLM (this will remove duplicates between existing and new)
        print(f"[ActionPlanAgent] Deduplicating using LLM...")
        deduplicated = await self._deduplicate_actions_with_phases(all_actions_with_phases, ctx)
        print(f"[ActionPlanAgent] After deduplication: {len(deduplicated)} actions (was {len(all_actions_with_phases)})")
        
        # Categorize by phase
        before, during, after = self._categorize_by_phase(deduplicated)
        
        # Build merged action plan (include all sources)
        merged_plan = ActionPlanResponse(
            location=location_query,
            display_name=display_name,
            before_flood=before,
            during_flood=during,
            after_flood=after,
            sources=[doc["url"] for doc in docs],  # All docs (existing + new)
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[ActionPlanAgent] ✅ Merged plan complete: {merged_plan.total_actions()} total actions [⏱️  {duration:.2f}s]")
        print(f"[ActionPlanAgent]   Change: {merged_plan.total_actions() - previous_plan.total_actions():+d} actions")
        
        # Print token usage summary for this agent
        summary = tracker.get_summary()
        agent_summary = summary.get("by_agent", {}).get("ActionPlanAgent", {})
        if agent_summary:
            print(f"[ActionPlanAgent] 💰 Token usage: {agent_summary.get('total_tokens', 0):,} tokens (${agent_summary.get('cost_usd', 0):.6f})")
        
        return Message(content=json.dumps({
            "action_plan": merged_plan.model_dump(mode='python'),
            "govdoc_data": input_data,
            "location": location_query,
            "iteration": 2
        }, ensure_ascii=False))
    
    async def _summarize_actions_from_doc(
        self, doc: Dict, location: str, revision_notes: List[str], previous_plan: ActionPlanResponse, ctx: MessageContext
    ) -> List[Tuple[Action, str]]:
        """
        Summarize actions from a document with targeted focus on revision notes.
        This is different from extraction - it focuses on summarizing what's relevant to the gaps.
        """
        notes_text = "\n".join(f"- {note}" for note in revision_notes[:10])
        
        # Get existing action titles to avoid duplicates
        existing_titles = set()
        for action in previous_plan.before_flood + previous_plan.during_flood + previous_plan.after_flood:
            existing_titles.add(action.title.lower().strip())
        
        existing_titles_str = "\n".join(list(existing_titles)[:20])
        
        prompt = f"""You are summarizing a government flood preparedness document to fill gaps in an existing action plan.

TARGET AUDIENCE: Individual homeowners and renters in {location}

Document: {doc['title']}
Source: {doc['url']}

REVISION NOTES (focus on these gaps):
{notes_text}

EXISTING ACTIONS (do NOT duplicate - summarize NEW actions only):
{existing_titles_str}

TASK: Summarize actions from this document that address the revision notes above.
Focus on actions that fill the identified gaps.

EXTRACTION RULES:
1. Summarize actions from the document text (no hallucination)
2. Focus on actions that address the revision notes
3. EXCLUDE: government infrastructure projects, policy frameworks, municipal planning
4. Prioritize location-specific details when available

TARGET: Summarize 5-15 actions that address the gaps, distributed as:
- BEFORE: 3-8 actions
- DURING: 1-4 actions  
- AFTER: 1-3 actions

PRIORITIZE:
- Actions that directly address issues in revision notes
- Location-specific details for {location}
- Essential actions missing from existing plan

Return ONLY valid JSON array:
[
  {{
    "title": "...",
    "description": "... (specific details)",
    "priority": "high|medium|low",
    "category": "evacuation|property_protection|emergency_kit|communication|insurance|family_plan",
    "phase": "before|during|after"
  }}
]

Document text:
{doc['text'][:10000]}

JSON:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Track token usage
            usage_found = False
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="summarize_actions_from_doc"
                        )
                        usage_found = True
                        break
            
            if not usage_found:
                prompt_text = message.content if hasattr(message, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="ActionPlanAgent",
                    model="gpt-4o-mini",
                    response=response,
                    operation="summarize_actions_from_doc",
                    prompt_text=prompt_text
                )
            
            content = response.chat_message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            actions_data = json.loads(content)
            
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
            return []
        except Exception as e:
            print(f"[ActionPlanAgent]   ❌ Error: {e}")
            return []
    
    async def _extract_actions_with_revision_focus(
        self, doc: Dict, location: str, revision_notes: List[str], previous_plan: ActionPlanResponse, ctx: MessageContext
    ) -> List[Tuple[Action, str]]:
        """Extract NEW actions that address revision notes, avoiding duplicates with existing plan."""
        
        notes_text = "\n".join(f"- {note}" for note in revision_notes[:10])  # Limit to avoid token issues
        
        # Get existing action titles to avoid duplicates
        existing_titles = set()
        for action in previous_plan.before_flood + previous_plan.during_flood + previous_plan.after_flood:
            existing_titles.add(action.title.lower().strip())
        
        existing_titles_str = "\n".join(list(existing_titles)[:20])  # Show sample to LLM
        
        prompt = f"""You are analyzing a government flood preparedness document to address specific issues.

TARGET AUDIENCE: Individual homeowners and renters in {location}

Document: {doc['title']}
Source: {doc['url']}

REVISION NOTES (address these issues):
{notes_text}

EXISTING ACTIONS (do NOT duplicate these - extract NEW actions only):
{existing_titles_str}

EXTRACTION RULES:
1. Extract actions from the document text (no hallucination)
2. INCLUDE both location-specific AND essential universal actions that address revision notes
3. EXCLUDE: government infrastructure projects, policy frameworks, municipal planning
4. Focus on actions that fill gaps identified in revision notes

TARGET: Extract 5-15 NEW actions that address the revision notes, distributed as:
- BEFORE: 3-8 actions - preparation, planning, protection
- DURING: 1-4 actions - evacuation, immediate safety
- AFTER: 1-3 actions - cleanup, documentation, recovery

MUST INCLUDE if mentioned in revision notes (even if generic):
✅ Flood insurance purchase/claims (if missing)
✅ Emergency kit preparation (if missing)
✅ Family communication plan (if missing)
✅ Evacuation procedures (if missing)
✅ Utility shutoff (if missing)
✅ Damage documentation (if missing)

PRIORITIZE location-specific details:
- Named waterways, neighborhoods, or local agencies
- Specific programs or subsidies for {location}
- Regional flood history or unique risks
- Local emergency contacts or shelters

IMPORTANT: 
- Extract ONLY NEW actions not in existing list
- Address specific issues from revision notes
- Distribute across all three phases
- Be specific in descriptions

Return ONLY valid JSON array:
[
  {{
    "title": "...",
    "description": "... (specific details with quantities, times, locations)",
    "priority": "high|medium|low",
    "category": "evacuation|property_protection|emergency_kit|communication|insurance|family_plan",
    "phase": "before|during|after"
  }}
]

Document text:
{doc['text'][:10000]}

JSON:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Track token usage - check inner_messages first (autogen stores usage there)
            usage_found = False
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="extract_actions_with_revision"
                        )
                        usage_found = True
                        break
                    elif hasattr(msg, 'response') and hasattr(msg.response, 'usage'):
                        usage = msg.response.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="extract_actions_with_revision"
                        )
                        usage_found = True
                        break
            
            if not usage_found:
                # Fallback to the general method with prompt text for estimation
                prompt_text = message.content if hasattr(message, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="ActionPlanAgent",
                    model="gpt-4o-mini",
                    response=response,
                    operation="extract_actions_with_revision",
                    prompt_text=prompt_text
                )
            
            content = response.chat_message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            actions_data = json.loads(content)
            
            actions_with_phases = []
            for item in actions_data:
                try:
                    # Filter out city/government actions
                    title_lower = item.get("title", "").lower()
                    desc_lower = (item.get("description", "") or "").lower()
                    
                    # Check for city/government action indicators
                    city_indicators = [
                        "city should", "municipality should", "government should", 
                        "authorities should", "municipal", "city will", "city must",
                        "infrastructure", "public works", "zoning", "planning department",
                        "emergency services should", "fire department should"
                    ]
                    
                    if any(indicator in title_lower or indicator in desc_lower for indicator in city_indicators):
                        print(f"[ActionPlanAgent]   🚫 Filtered out city/government action: '{item.get('title', 'Unknown')}'")
                        continue
                    
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
            return []
        except Exception as e:
            print(f"[ActionPlanAgent]   ❌ Error: {e}")
            return []
    
    async def _extract_actions_from_doc(self, doc: Dict, location: str, ctx: MessageContext) -> List[Tuple[Action, str]]:
        """
        Extract actions with phase information.
        
        Returns:
            List of (Action, phase) tuples where phase is "before"|"during"|"after"
        """
        
        prompt = f"""You are analyzing a government flood preparedness document.

TARGET AUDIENCE: Individual homeowners and renters in {location}


Document: {doc['title']}
Source: {doc['url']}

TRACTION RULES:
1. Extract actions from the document text (no hallucination)
2. INCLUDE both location-specific AND essential universal actions
3. EXCLUDE: government infrastructure projects, policy frameworks, municipal planning

TARGET: 20-30 total actions distributed as:
- BEFORE: 12-18 actions (50-60%) - preparation, planning, protection
- DURING: 4-8 actions (20-30%) - evacuation, immediate safety
- AFTER: 3-6 actions (10-20%) - cleanup, documentation, recovery

MUST INCLUDE if mentioned (even if generic):
✅ Flood insurance purchase/claims
✅ Emergency kit preparation
✅ Family communication plan
✅ Evacuation procedures
✅ Utility shutoff (gas/power/water)
✅ Damage documentation

PRIORITIZE location-specific details:
- Named waterways, neighborhoods, or local agencies
- Specific programs or subsidies for {location}
- Regional flood history or unique risks
- Local emergency contacts or shelters

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

JSON:"""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Track token usage - try multiple methods to find usage data
            usage_found = False
            
            # Method 1: Check inner_messages for usage
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        if hasattr(usage, 'prompt_tokens'):
                            tracker.record_usage(
                                agent_name="ActionPlanAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                operation="extract_actions_from_doc"
                            )
                            usage_found = True
                            break
                    elif hasattr(msg, 'response') and hasattr(msg.response, 'usage'):
                        usage = msg.response.usage
                        if hasattr(usage, 'prompt_tokens'):
                            tracker.record_usage(
                                agent_name="ActionPlanAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                operation="extract_actions_from_doc"
                            )
                            usage_found = True
                            break
            
            # Method 2: Check chat_message for usage or response_metadata
            if not usage_found and hasattr(response, 'chat_message'):
                if hasattr(response.chat_message, 'usage'):
                    usage = response.chat_message.usage
                    if hasattr(usage, 'prompt_tokens'):
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="extract_actions_from_doc"
                        )
                        usage_found = True
                elif hasattr(response.chat_message, 'response_metadata'):
                    metadata = response.chat_message.response_metadata
                    if metadata and isinstance(metadata, dict) and 'token_usage' in metadata:
                        usage = metadata['token_usage']
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0),
                            operation="extract_actions_from_doc"
                        )
                        usage_found = True
            
            # Method 3: Try to access model client directly to get the last response
            if not usage_found:
                try:
                    # Access the model client from AssistantAgent
                    if hasattr(self._llm, 'model_client'):
                        model_client = self._llm.model_client
                    elif hasattr(self._llm, '_model_client'):
                        model_client = self._llm._model_client
                    else:
                        model_client = None
                    
                    if model_client:
                        # Try different ways to access the last response
                        last_resp = None
                        if hasattr(model_client, '_last_response'):
                            last_resp = model_client._last_response
                        elif hasattr(model_client, 'last_response'):
                            last_resp = model_client.last_response
                        elif hasattr(model_client, '_response_cache') and model_client._response_cache:
                            # Check if there's a response cache
                            cache = model_client._response_cache
                            if isinstance(cache, list) and cache:
                                last_resp = cache[-1]
                            elif isinstance(cache, dict):
                                # Get the most recent response
                                last_resp = list(cache.values())[-1] if cache else None
                        
                        if last_resp and hasattr(last_resp, 'usage'):
                            usage = last_resp.usage
                            if hasattr(usage, 'prompt_tokens'):
                                tracker.record_usage(
                                    agent_name="ActionPlanAgent",
                                    model="gpt-4o-mini",
                                    prompt_tokens=usage.prompt_tokens,
                                    completion_tokens=usage.completion_tokens,
                                    operation="extract_actions_from_doc"
                                )
                                usage_found = True
                except Exception as e:
                    # Silently fail - we'll fall back to estimation
                    pass
            
            # Method 4: Fallback to general method with prompt text for estimation
            if not usage_found:
                prompt_text = message.content if hasattr(message, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="ActionPlanAgent",
                    model="gpt-4o-mini",
                    response=response,
                    operation="extract_actions_from_doc",
                    prompt_text=prompt_text
                )
            
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
                    # Filter out city/government actions
                    title_lower = item.get("title", "").lower()
                    desc_lower = (item.get("description", "") or "").lower()
                    
                    # Check for city/government action indicators
                    city_indicators = [
                        "city should", "municipality should", "government should", 
                        "authorities should", "municipal", "city will", "city must",
                        "infrastructure", "public works", "zoning", "planning department",
                        "emergency services should", "fire department should"
                    ]
                    
                    if any(indicator in title_lower or indicator in desc_lower for indicator in city_indicators):
                        print(f"[ActionPlanAgent]   🚫 Filtered out city/government action: '{item.get('title', 'Unknown')}'")
                        continue
                    
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
    
    async def _deduplicate_actions_with_phases(self, actions_with_phases: List[Tuple[Action, str]], ctx: MessageContext) -> List[Tuple[Action, str]]:
        """
        Remove duplicate actions using LLM to identify semantic duplicates.
        Uses batching for large lists to improve performance.
        
        Args:
            actions_with_phases: List of (Action, phase) tuples
            ctx: MessageContext for LLM calls
        
        Returns:
            Deduplicated list of (Action, phase) tuples
        """
        if len(actions_with_phases) <= 1:
            return actions_with_phases
        
        # For large lists (>30), use batching to reduce latency
        BATCH_SIZE = 30
        if len(actions_with_phases) > BATCH_SIZE:
            print(f"[ActionPlanAgent]   Large list ({len(actions_with_phases)} actions), using batched deduplication...")
            # Split into batches and deduplicate each batch
            batches = [
                actions_with_phases[i:i + BATCH_SIZE]
                for i in range(0, len(actions_with_phases), BATCH_SIZE)
            ]
            
            # Deduplicate each batch in parallel
            batch_tasks = [
                self._deduplicate_batch(batch, batch_idx * BATCH_SIZE, ctx)
                for batch_idx, batch in enumerate(batches)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine deduplicated batches
            deduplicated_batches = []
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[ActionPlanAgent]   ⚠️  Batch deduplication error: {result}")
                    continue
                deduplicated_batches.extend(result)
            
            # Final deduplication pass on combined results to catch cross-batch duplicates
            # This is critical because duplicates can exist between batches
            print(f"[ActionPlanAgent]   Final cross-batch deduplication pass on {len(deduplicated_batches)} actions...")
            if len(deduplicated_batches) > BATCH_SIZE:
                # If combined results are still large, do a two-pass deduplication
                # First pass: deduplicate in chunks
                final_batches = [
                    deduplicated_batches[i:i + BATCH_SIZE]
                    for i in range(0, len(deduplicated_batches), BATCH_SIZE)
                ]
                final_results = []
                for final_batch in final_batches:
                    deduped = await self._deduplicate_batch(final_batch, 0, ctx)
                    final_results.extend(deduped)
                # Second pass: final deduplication on all combined results
                print(f"[ActionPlanAgent]   Second pass: deduplicating {len(final_results)} actions from chunks...")
                return await self._deduplicate_batch(final_results, 0, ctx)
            else:
                # Single final pass for smaller lists
                return await self._deduplicate_batch(deduplicated_batches, 0, ctx)
        
        # For smaller lists, process directly
        return await self._deduplicate_batch(actions_with_phases, 0, ctx)
    
    async def _deduplicate_batch(self, actions_with_phases: List[Tuple[Action, str]], start_index: int, ctx: MessageContext) -> List[Tuple[Action, str]]:
        """
        Deduplicate a batch of actions using LLM.
        
        Args:
            actions_with_phases: List of (Action, phase) tuples to deduplicate
            start_index: Starting index for this batch (for logging)
            ctx: MessageContext for LLM calls
        
        Returns:
            Deduplicated list of (Action, phase) tuples
        """
        if len(actions_with_phases) <= 1:
            return actions_with_phases
        
        # Prepare actions list for LLM
        actions_list = []
        for idx, (action, phase) in enumerate(actions_with_phases):
            actions_list.append({
                "index": idx,
                "title": action.title,
                "description": action.description or "",
                "category": action.category,
                "phase": phase
            })
        
        prompt = f"""You are analyzing a list of flood preparedness actions to identify and remove duplicates.

A duplicate is defined as:
- Two actions that convey the same or very similar information
- Actions with different wording but identical meaning
- Actions that are subsets or variations of each other

IMPORTANT: Keep the action with the most detailed description if duplicates are found. If descriptions are equally detailed, keep the first one.

Here are {len(actions_list)} actions to analyze:

{json.dumps(actions_list, indent=2, ensure_ascii=False)}

Return a JSON object with:
{{
  "unique_indices": [0, 1, 3, ...],  // List of indices to KEEP (0-based)
  "duplicates_removed": [
    {{"removed_index": 2, "kept_index": 0, "reason": "Same as action 0 but less detailed"}},
    ...
  ]
}}

Only include indices of actions that should be KEPT. Remove all duplicates.
Return ONLY valid JSON, no markdown, no explanation."""

        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Track token usage - check inner_messages first (autogen stores usage there)
            usage_found = False
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="deduplicate_actions"
                        )
                        usage_found = True
                        break
                    elif hasattr(msg, 'response') and hasattr(msg.response, 'usage'):
                        usage = msg.response.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="deduplicate_actions"
                        )
                        usage_found = True
                        break
            
            if not usage_found:
                prompt_text = message.content if hasattr(message, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="ActionPlanAgent",
                    model="gpt-4o-mini",
                    response=response,
                    operation="deduplicate_actions",
                    prompt_text=prompt_text
                )
            
            content = response.chat_message.content.strip()
            
            # Extract JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result = json.loads(content)
            unique_indices = set(result.get("unique_indices", []))
            duplicates_info = result.get("duplicates_removed", [])
            
            # Filter actions to keep only unique ones
            unique_actions = []
            for idx, (action, phase) in enumerate(actions_with_phases):
                if idx in unique_indices:
                    unique_actions.append((action, phase))
            
            # Log removed duplicates (only for small batches to avoid spam)
            if duplicates_info and len(actions_with_phases) <= 30:
                for dup in duplicates_info:
                    removed_idx = dup.get("removed_index")
                    kept_idx = dup.get("kept_index")
                    reason = dup.get("reason", "Duplicate")
                    if removed_idx < len(actions_with_phases) and kept_idx < len(actions_with_phases):
                        removed_title = actions_with_phases[removed_idx][0].title
                        kept_title = actions_with_phases[kept_idx][0].title
                        print(f"[ActionPlanAgent]   🗑️  Duplicate removed: '{removed_title}' (kept: '{kept_title}') - {reason}")
            
            removed_count = len(actions_with_phases) - len(unique_actions)
            if removed_count > 0:
                print(f"[ActionPlanAgent]   🧹 Removed {removed_count} duplicate(s) from batch")
            
            return unique_actions
        
        except json.JSONDecodeError as e:
            print(f"[ActionPlanAgent]   ⚠️  LLM deduplication JSON parse error: {e}")
            print(f"[ActionPlanAgent]   ⚠️  Falling back to keeping all actions")
            return actions_with_phases
        except Exception as e:
            print(f"[ActionPlanAgent]   ⚠️  LLM deduplication error: {e}")
            print(f"[ActionPlanAgent]   ⚠️  Falling back to keeping all actions")
            return actions_with_phases
    
    async def _decide_revision_strategy(
        self,
        revision_notes: List[str],
        previous_plan: ActionPlanResponse,
        ctx: MessageContext
    ) -> str:
        """
        Decide whether to EXTRACT_ADDITIONAL actions or use LLM_REVISE.
        
        Returns:
            "EXTRACT_ADDITIONAL" or "LLM_REVISE"
        """
        # Simple heuristic: if revision notes mention missing categories or incomplete coverage,
        # extract additional actions. Otherwise, use LLM to revise.
        notes_lower = " ".join(revision_notes).lower()
        
        # Check for indicators that we need additional extraction
        extract_indicators = [
            "missing", "incomplete", "lacks", "need more", "add", 
            "insufficient", "not enough", "missing categories"
        ]
        
        # Check for indicators that we can just revise
        revise_indicators = [
            "unclear", "vague", "improve clarity", "rephrase", 
            "better wording", "more specific", "clarify"
        ]
        
        extract_score = sum(1 for indicator in extract_indicators if indicator in notes_lower)
        revise_score = sum(1 for indicator in revise_indicators if indicator in notes_lower)
        
        # If extract indicators dominate, extract additional actions
        if extract_score > revise_score and extract_score > 0:
            return "EXTRACT_ADDITIONAL"
        else:
            # Default to LLM revise for clarity/wording issues
            return "LLM_REVISE"
    
    async def _llm_revise_plan(
        self,
        previous_plan: ActionPlanResponse,
        revision_notes: List[str],
        location: str,
        ctx: MessageContext
    ) -> Tuple[List[Action], List[Action], List[Action]]:
        """
        Use LLM to revise existing plan based on revision notes.
        Improves clarity, specificity, and addresses issues without extracting new actions.
        """
        notes_text = "\n".join(f"- {note}" for note in revision_notes[:15])
        
        # Convert plan to JSON for LLM
        plan_json = previous_plan.model_dump_json(indent=2, exclude_none=True)
        
        prompt = f"""You are revising a flood action plan for {location} based on feedback.

CURRENT PLAN:
{plan_json}

REVISION NOTES (address these issues):
{notes_text}

TASK: Revise the existing actions to:
1. Improve clarity and specificity based on revision notes
2. Fix any issues mentioned in revision notes
3. Only modify descriptions/titles to address issues
4. Maintain the same structure and phase distribution

IMPORTANT:
- DO NOT add new actions (only revise existing ones)
- Focus on improving wording, clarity, and specificity
- Address specific issues from revision notes

Return the REVISED plan in the same JSON format as the current plan, with improved actions.
Return ONLY valid JSON matching the structure above."""
        
        try:
            message = TextMessage(content=prompt, source="user")
            response = await self._llm.on_messages([message], ctx.cancellation_token)
            
            # Track token usage
            usage_found = False
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        tracker.record_usage(
                            agent_name="ActionPlanAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="llm_revise_plan"
                        )
                        usage_found = True
                        break
            
            if not usage_found:
                prompt_text = message.content if hasattr(message, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="ActionPlanAgent",
                    model="gpt-4o-mini",
                    response=response,
                    operation="llm_revise_plan",
                    prompt_text=prompt_text
                )
            
            content = response.chat_message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            revised_plan_data = json.loads(content)
            revised_plan = ActionPlanResponse(**revised_plan_data)
            
            return revised_plan.before_flood, revised_plan.during_flood, revised_plan.after_flood
        
        except Exception as e:
            print(f"[ActionPlanAgent] ❌ LLM revise error: {e}")
            # Fallback: return original plan
            return previous_plan.before_flood, previous_plan.during_flood, previous_plan.after_flood
    
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
    """Test the complete Feedback Loop Pipeline"""
    from app.agents.govdoc_agent import GovDocAgent
    from app.agents.evaluator_agent import ActionPlanEvaluatorAgent
    
    runtime = SingleThreadedAgentRuntime()
    
    # Register all agents for feedback loop
    print("Registering agents for Feedback Loop Pipeline...")
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    await ActionPlanEvaluatorAgent.register(runtime, "ActionPlanEvaluator", lambda: ActionPlanEvaluatorAgent(runtime))
    
    await maybe_await(runtime.start())
    
    print("\n" + "=" * 80)
    print("FEEDBACK LOOP PIPELINE WITH 2-ITERATION DESIGN")
    print("=" * 80)
    print("Architecture: Feedback Loop (not pure sequential)")
    print("Flow: GovDoc → ActionPlan → Evaluator")
    print("Iterations: 1) Normal, 2) REPHRASE (same docs) or SEARCH_ADDITIONAL (new docs)")
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
    
    # Print token usage summary
    from app.services.token_tracker import tracker
    tracker.print_summary()
    tracker.save_to_file("token_usage.json")
    
    # Save output
    with open("feedback_loop_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: feedback_loop_output.json")
    
    await maybe_await(runtime.stop())
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())