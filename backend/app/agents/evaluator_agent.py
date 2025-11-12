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

import json
import asyncio
from typing import Any, Dict
from pathlib import Path
from datetime import datetime


class EvaluatorAgent(RoutedAgent):
    """
    Evaluates the ActionPlanAgent's end-user summary with a compact JSON critique.
    Runs a one-shot feedback loop: if REVISE, asks ActionPlanAgent to rewrite once.

    Configurable via metadata: {"max_iterations": int}. Defaults to 1.
    """

    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("Evaluator")
        self._runtime = runtime
        self._llm = AssistantAgent(
            "EvaluatorLLM",
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        )

    @message_handler
    async def on_evaluate(self, message: Message, ctx: MessageContext) -> Message:
        """
        Orchestrates: plan -> summarize -> evaluate -> (optional) revise once -> return final text + critique.

        Input message.content: location string (e.g., "Vancouver, BC").
        Optional message.metadata: {"max_iterations": int, "debug": bool}
        """
        try:
            location = (message.content or "").strip() or "Canada"
            max_iters = 1
            debug_mode = False
            try:
                if isinstance(message.metadata, dict):
                    mi = message.metadata.get("max_iterations")
                    if isinstance(mi, int) and mi >= 0:
                        max_iters = mi
                    debug_mode = message.metadata.get("debug", False)
            except Exception:
                pass

            # 1) Ask ActionPlan for a structured plan
            print("\n" + "="*80)
            print("📋 STEP 1: Getting Action Plan from ActionPlanAgent")
            print("="*80)
            print(f"\n📤 SENDING TO ActionPlanAgent (Location Query):")
            print("-" * 80)
            print(f"Location: {location}")
            print("-" * 80)
            
            plan_msg = await self._runtime.send_message(
                Message(content=location),
                AgentId("ActionPlan", "default"),
            )
            if not plan_msg or not getattr(plan_msg, "content", None):
                return Message(content=json.dumps({
                    "error": "ActionPlan did not respond",
                    "location": location
                }))

            try:
                plan_payload = json.loads(plan_msg.content)
                print("\n📥 RECEIVED FROM ActionPlanAgent (Action Plan):")
                print("-" * 80)
                
                # Store the full action plan for comparison later
                full_action_plan = plan_payload.copy()
                
                # Show FULL action plan response in JSON format BEFORE evaluation
                print("\n📋 FULL ACTION PLAN RESPONSE (Before Evaluator) - JSON Format:")
                print("=" * 80)
                action_plan_json = {
                    "location": plan_payload.get("location", "N/A"),
                    "display_name": plan_payload.get("display_name", "N/A"),
                    "before_flood": {
                        "count": len(plan_payload.get("before_flood", [])),
                        "actions": [
                            {
                                "title": a.get("title"),
                                "description": a.get("description"),
                                "priority": a.get("priority"),
                                "category": a.get("category"),
                                "source_doc": a.get("source_doc")
                            }
                            for a in plan_payload.get("before_flood", [])
                        ]
                    },
                    "during_flood": {
                        "count": len(plan_payload.get("during_flood", [])),
                        "actions": [
                            {
                                "title": a.get("title"),
                                "description": a.get("description"),
                                "priority": a.get("priority"),
                                "category": a.get("category"),
                                "source_doc": a.get("source_doc")
                            }
                            for a in plan_payload.get("during_flood", [])
                        ]
                    },
                    "after_flood": {
                        "count": len(plan_payload.get("after_flood", [])),
                        "actions": [
                            {
                                "title": a.get("title"),
                                "description": a.get("description"),
                                "priority": a.get("priority"),
                                "category": a.get("category"),
                                "source_doc": a.get("source_doc")
                            }
                            for a in plan_payload.get("after_flood", [])
                        ]
                    },
                    "total_actions": (
                        len(plan_payload.get("before_flood", [])) +
                        len(plan_payload.get("during_flood", [])) +
                        len(plan_payload.get("after_flood", []))
                    ),
                    "sources": plan_payload.get("sources", []),
                    "generated_at": plan_payload.get("generated_at", "N/A")
                }
                print(json.dumps(action_plan_json, indent=2, ensure_ascii=False))
                print("=" * 80)
                
                # Save FULL action plan response to file (user-friendly format)
                self._log_json_to_file("action_plan_before_evaluator.json", action_plan_json)
                
                # Also show formatted summary
                self._print_formatted_json("Action Plan Summary", plan_payload, max_items=3)
                
                # Log the raw full JSON in debug mode
                if debug_mode:
                    self._log_json_to_file("evaluator_action_plan_response.json", plan_payload)
            except Exception:
                return Message(content=json.dumps({
                    "error": "Invalid ActionPlan payload",
                    "raw": plan_msg.content
                }))

            # 2) Ask ActionPlan to produce an initial concise summary (user-facing)
            print("\n" + "="*80)
            print("📝 STEP 2: Generating Initial Summary")
            print("="*80)
            summarize_req = {
                "mode": "SUMMARIZE",
                "plan": plan_payload,
                # Defaults for constraints; Evaluator can override in critique
                "constraints": {
                    "max_words": 120,
                    "reading_level": "Grade 6",
                    "format": "title + 3 bullets + CTA + source link",
                },
            }
            summarize_req_json = json.dumps(summarize_req, ensure_ascii=False, indent=2)
            
            print("\n📤 SENDING TO ActionPlanAgent (mode=SUMMARIZE):")
            print("-" * 80)
            if debug_mode:
                self._print_json_readable(summarize_req)
                print("-" * 80)
                self._log_json_to_file("evaluator_summarize_request.json", summarize_req)
            else:
                print(f"Request: mode=SUMMARIZE, plan location={plan_payload.get('location', 'N/A')}")
                print("-" * 80)
            
            init_summary_msg = await self._runtime.send_message(
                Message(content=summarize_req_json),
                AgentId("ActionPlan", "default"),
            )
            if not init_summary_msg or not getattr(init_summary_msg, "content", None):
                return Message(content=json.dumps({
                    "error": "ActionPlan summarize failed"
                }))
            
            # Check if response is an error (shouldn't happen with proper routing)
            try:
                error_check = json.loads(init_summary_msg.content)
                if isinstance(error_check, dict) and error_check.get("error"):
                    return Message(content=json.dumps({
                        "error": f"ActionPlan returned error: {error_check.get('error')}"
                    }))
            except:
                pass  # Not JSON error, continue
            current_text = init_summary_msg.content.strip()
            initial_summary = current_text  # Store for comparison
            
            # Always show the initial summary (not just in debug mode)
            print("\n📥 RECEIVED FROM ActionPlanAgent (Initial Summary):")
            print("-" * 80)
            print(current_text)
            print("-" * 80)
            if debug_mode:
                self._log_text_to_file("evaluator_summary_response.txt", current_text)

            # 3) Evaluate with compact schema
            print("\n" + "="*80)
            print("🔍 STEP 3: Evaluating Summary")
            print("="*80)
            if debug_mode:
                print("\n📄 TEXT BEFORE EVALUATION:")
                print("-" * 80)
                print(current_text)
                print("-" * 80)
                self._log_text_to_file("evaluator_before_evaluation.txt", current_text)
            
            critique = await self._critique(current_text, plan_payload, ctx)
            
            # Always show critique results (not just in debug mode)
            self._print_critique(critique)
            if debug_mode:
                self._log_json_to_file("evaluator_critique.json", critique)

            # 4) Decide: pass or one-shot revise
            iters_done = 0
            if (critique.get("decision") or "").upper() == "REVISE" and max_iters > 0:
                print("\n" + "="*80)
                print("✏️  STEP 4: Revising Summary (One-shot Revision)")
                print("="*80)
                rewrite_req = {
                    "mode": "REWRITE",
                    "plan": plan_payload,
                    "feedback": {
                        **critique,
                        "original_summary": current_text,  # Include original summary for proper revision
                    },
                }
                rewrite_req_json = json.dumps(rewrite_req, ensure_ascii=False, indent=2)
                
                if debug_mode:
                    print("\n📤 SENDING TO ActionPlanAgent (REWRITE):")
                    print("-" * 80)
                    self._print_json_readable(rewrite_req)
                    print("-" * 80)
                    self._log_json_to_file("evaluator_rewrite_request.json", rewrite_req)
                
                revised_msg = await self._runtime.send_message(
                    Message(content=rewrite_req_json),
                    AgentId("ActionPlan", "default"),
                )
                if revised_msg and getattr(revised_msg, "content", None):
                    current_text = revised_msg.content.strip()
                    iters_done = 1
                    print("\n📥 RECEIVED FROM ActionPlanAgent (Revised Summary):")
                    print("-" * 80)
                    print(current_text)
                    print("-" * 80)
                    if debug_mode:
                        self._log_text_to_file("evaluator_after_rewrite.txt", current_text)
                else:
                    print("\n⚠️  Revision failed, using initial summary")
            else:
                print("\n✅ Summary passed evaluation, no revision needed")

            # 5) Show FULL action plan response AFTER evaluation
            print("\n" + "="*80)
            print("📋 FULL ACTION PLAN RESPONSE (After Evaluator) - JSON Format:")
            print("="*80)
            
            # Format the action plan in the same user-friendly structure
            action_plan_after_json = {
                "location": full_action_plan.get("location", "N/A"),
                "display_name": full_action_plan.get("display_name", "N/A"),
                "before_flood": {
                    "count": len(full_action_plan.get("before_flood", [])),
                    "actions": [
                        {
                            "title": a.get("title"),
                            "description": a.get("description"),
                            "priority": a.get("priority"),
                            "category": a.get("category"),
                            "source_doc": a.get("source_doc")
                        }
                        for a in full_action_plan.get("before_flood", [])
                    ]
                },
                "during_flood": {
                    "count": len(full_action_plan.get("during_flood", [])),
                    "actions": [
                        {
                            "title": a.get("title"),
                            "description": a.get("description"),
                            "priority": a.get("priority"),
                            "category": a.get("category"),
                            "source_doc": a.get("source_doc")
                        }
                        for a in full_action_plan.get("during_flood", [])
                    ]
                },
                "after_flood": {
                    "count": len(full_action_plan.get("after_flood", [])),
                    "actions": [
                        {
                            "title": a.get("title"),
                            "description": a.get("description"),
                            "priority": a.get("priority"),
                            "category": a.get("category"),
                            "source_doc": a.get("source_doc")
                        }
                        for a in full_action_plan.get("after_flood", [])
                    ]
                },
                "total_actions": (
                    len(full_action_plan.get("before_flood", [])) +
                    len(full_action_plan.get("during_flood", [])) +
                    len(full_action_plan.get("after_flood", []))
                ),
                "sources": full_action_plan.get("sources", []),
                "generated_at": full_action_plan.get("generated_at", "N/A")
            }
            print(json.dumps(action_plan_after_json, indent=2, ensure_ascii=False))
            print("="*80)
            
            # Save FULL action plan response to file (user-friendly format)
            self._log_json_to_file("action_plan_after_evaluator.json", action_plan_after_json)
            
            # 6) Show comparison of summaries before and after in JSON format
            print("\n" + "="*80)
            print("📊 COMPARISON: Summary Text Before vs After Evaluation (JSON Format)")
            print("="*80)
            
            comparison_json = {
                "comparison": {
                    "initial_summary": {
                        "text": initial_summary,
                        "length": len(initial_summary),
                        "stage": "Before Evaluator"
                    },
                    "final_summary": {
                        "text": current_text,
                        "length": len(current_text),
                        "stage": "After Evaluator"
                    },
                    "was_revised": current_text != initial_summary,
                    "changes": {
                        "summary": "The summary was revised based on evaluator feedback." if current_text != initial_summary else "No changes made - summary passed evaluation as-is",
                        "length_difference": len(current_text) - len(initial_summary),
                        "initial_length": len(initial_summary),
                        "final_length": len(current_text)
                    }
                }
            }
            
            # Print JSON in readable format (line by line)
            print(json.dumps(comparison_json, indent=2, ensure_ascii=False))
            print("="*80)
            
            # Save comparison to file (user-friendly format)
            self._log_json_to_file("summary_comparison_before_after.json", comparison_json)

            result = {
                "location": location,
                "initial_summary": initial_summary,
                "final_text": current_text,
                "critique": critique,
                "iterations": {"max": max_iters, "used": iters_done},
                "was_revised": current_text != initial_summary,
            }
            return Message(content=json.dumps(result, ensure_ascii=False))
        except Exception as e:
            return Message(content=json.dumps({
                "error": f"Evaluator failed: {e}"
            }))

    def _print_formatted_json(self, title: str, data: Dict[str, Any], max_items: int = 5):
        """Helper to print JSON data in a readable format."""
        print(f"\n📋 {title}:")
        print("-" * 80)
        
        # Try to extract key information
        if isinstance(data, dict):
            # Handle GovDocAgent-style output (with "items" or "results")
            if "items" in data or "results" in data:
                self._print_govdoc_results(data, max_items)
                return
            
            # Handle location info (could be nested)
            location = data.get("location", {})
            if isinstance(location, dict):
                location_str = location.get("query") or location.get("display_name") or "N/A"
                display_name = location.get("display_name") or location_str
            else:
                location_str = location or data.get("location", "N/A")
                display_name = data.get("display_name", "N/A")
            
            print(f"📍 Location: {location_str} ({display_name})")
            
            # Show documents if available (from GovDocAgent)
            docs = data.get("docs", [])
            if docs:
                print(f"\n📄 Documents Found ({len(docs)}):")
                for i, doc in enumerate(docs[:max_items], 1):
                    doc_title = doc.get("title", "Untitled")
                    doc_url = doc.get("url", "N/A")
                    print(f"  {i}. {doc_title}")
                    print(f"     🔗 {doc_url}")
                if len(docs) > max_items:
                    print(f"     ... and {len(docs) - max_items} more documents")
            
            # Show results if available (from GovDocAgent search results)
            results = data.get("results", [])
            if results:
                print(f"\n🔍 Search Results ({len(results)}):")
                for i, result in enumerate(results[:max_items], 1):
                    result_title = result.get("title", "Untitled")
                    result_url = result.get("url", "N/A")
                    snippet = result.get("snippet", "")
                    filetype = result.get("filetype", "")
                    print(f"  {i}. {result_title}")
                    if snippet:
                        print(f"     📝 {snippet}")
                    print(f"     🔗 {result_url}")
                    if filetype:
                        print(f"     📎 Type: {filetype}")
                if len(results) > max_items:
                    print(f"     ... and {len(results) - max_items} more results")
            
            sources = data.get("sources", [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, src in enumerate(sources[:max_items], 1):
                    print(f"  {i}. {src}")
                if len(sources) > max_items:
                    print(f"     ... and {len(sources) - max_items} more sources")
            
            # Display before/during/after phases clearly
            for phase_key, phase_label in [
                ("before_flood", "Before Flood"),
                ("during_flood", "During Flood"),
                ("after_flood", "After Flood")
            ]:
                actions = data.get(phase_key, [])
                if actions:
                    print(f"\n⏰ {phase_label} ({len(actions)} actions):")
                    for i, action in enumerate(actions[:max_items], 1):
                        action_title = action.get("title", "N/A")
                        priority = action.get("priority", "N/A")
                        category = action.get("category", "N/A")
                        print(f"  {i}. [{priority.upper()}] {action_title}")
                        if category != "N/A":
                            print(f"     Category: {category}")
                    if len(actions) > max_items:
                        print(f"     ... and {len(actions) - max_items} more actions")
                else:
                    print(f"\n⏰ {phase_label}: No actions specified")
        
        print("-" * 80)

    def _print_govdoc_results(self, data: Dict[str, Any], max_items: int = 5):
        """Helper to print GovDocAgent search results in a readable format."""
        items = data.get("items") or data.get("results", [])
        
        if not items:
            print("⚠️  No documents found")
            return
        
        print(f"🔍 Found {len(items)} Document(s):")
        print()
        
        for i, item in enumerate(items[:max_items], 1):
            title = item.get("title", "Untitled Document")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            filetype = item.get("filetype", "")
            why_relevant = item.get("why_relevant", "")
            score = item.get("score", 0.0)
            
            print(f"📄 Document {i}: {title}")
            if snippet:
                print(f"   📝 {snippet}")
            if why_relevant:
                print(f"   💡 Relevance: {why_relevant}")
            if url:
                # Truncate long URLs for readability
                display_url = url if len(url) <= 70 else url[:67] + "..."
                print(f"   🔗 {display_url}")
            if filetype:
                print(f"   📎 Type: {filetype.upper()}")
            if score > 0:
                print(f"   ⭐ Score: {score:.1f}")
            print()
        
        if len(items) > max_items:
            print(f"   ... and {len(items) - max_items} more document(s)")

    def _print_critique(self, critique: Dict[str, Any]):
        """Helper to print critique in a readable format."""
        print("\n📊 Critique Results:")
        print("-" * 80)
        
        decision = critique.get("decision", "UNKNOWN")
        decision_emoji = "✅" if decision == "PASS" else "⚠️"
        print(f"{decision_emoji} Decision: {decision}")
        
        scores = critique.get("score", {})
        if scores:
            print(f"\n📈 Scores (0.0-1.0, equal weight):")
            for key, value in scores.items():
                # Handle both 0-1.0 and 0-5 scales for display
                display_value = value
                max_value = 1.0
                if isinstance(value, (int, float)) and value > 1.0:
                    # Old format (0-5), normalize for display
                    display_value = value / 5.0
                    max_value = 5.0
                
                # Create visual bar (scale to 20 chars for 1.0 max)
                bar_length = int(display_value * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {key.replace('_', ' ').title():25} {display_value:.2f}/{max_value} {bar}")
        
        issues = critique.get("issues", [])
        if issues:
            print(f"\n⚠️  Issues ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                issue_type = issue.get("type", "unknown")
                detail = issue.get("detail", "N/A")
                print(f"  {i}. [{issue_type.upper()}] {detail}")
        
        required_changes = critique.get("required_changes", [])
        if required_changes:
            print(f"\n✏️  Required Changes:")
            for i, change in enumerate(required_changes, 1):
                print(f"  {i}. {change}")
        
        print("-" * 80)

    def _print_json_readable(self, data: Any, max_depth: int = 3, indent: int = 0):
        """Print JSON in a more readable line-by-line format."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and max_depth > 0:
                    print("  " * indent + f"{key}:")
                    self._print_json_readable(value, max_depth - 1, indent + 1)
                else:
                    if isinstance(value, str) and len(value) > 100:
                        print("  " * indent + f"{key}: {value[:100]}... ({len(value)} chars)")
                    else:
                        print("  " * indent + f"{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data[:5]):  # Show first 5 items
                print("  " * indent + f"[{i}]:")
                if isinstance(item, (dict, list)) and max_depth > 0:
                    self._print_json_readable(item, max_depth - 1, indent + 1)
                else:
                    if isinstance(item, str) and len(item) > 100:
                        print("  " * (indent + 1) + f"{item[:100]}... ({len(item)} chars)")
                    else:
                        print("  " * (indent + 1) + str(item))
            if len(data) > 5:
                print("  " * indent + f"... and {len(data) - 5} more items")
        else:
            print("  " * indent + str(data))

    def _log_json_to_file(self, filename: str, data: Any):
        """Log JSON data to a file with pretty formatting."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / filename
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Logged to: {log_file}")
        except Exception as e:
            print(f"⚠️  Failed to log to file: {e}")

    def _log_text_to_file(self, filename: str, text: str):
        """Log text to a file."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / filename
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 Logged to: {log_file}")
        except Exception as e:
            print(f"⚠️  Failed to log to file: {e}")

    async def _critique(self, text: str, plan_payload: Dict[str, Any], ctx: MessageContext) -> Dict[str, Any]:
        """
        Use LLM to generate a compact, actionable critique JSON for one-shot revision.
        Evaluates against 5 equal-weighted criteria: Accuracy, Clarity, Completeness, Relevance, Coherence.
        Accuracy has a hard-gate threshold of 0.6 (below = automatic REVISE).
        """
        # Derive minimal facts for grounding
        sources = (plan_payload or {}).get("sources", []) or []
        location = (plan_payload or {}).get("location", "")
        risk_level = (plan_payload or {}).get("risk_level", "")
        
        # Extract location info
        location_info = {}
        if isinstance(location, dict):
            location_info = location
        else:
            location_info = {"query": location, "display_name": location}

        prompt = (
            "You are an editor evaluating a public safety flood action summary.\n"
            "Return ONLY compact JSON critique as specified.\n\n"
            f"LOCATION: {json.dumps(location_info, ensure_ascii=False)}\n"
            f"RISK_LEVEL: {risk_level}\n"
            f"SOURCES: {json.dumps(sources, ensure_ascii=False)}\n\n"
            "TEXT TO EVALUATE:\n" + text + "\n\n"
            "EVALUATION CRITERIA (each scored 0.0-1.0, equal weight):\n\n"
            "1. ACCURACY (1.0): All claims must be supported by evidence (location, risk level, geo/time alignment).\n"
            "   - Penalize any mismatch or invented numbers\n"
            "   - Verify location-specific facts match the provided location\n"
            "   - Verify risk level aligns with stated risk tier\n"
            "   - Verify geographic and temporal information is consistent\n"
            "   - HARD-GATE: Score < 0.6 = automatic REVISE\n\n"
            "2. CLARITY (1.0): Plain language, short sentences, imperative actions (\"Do XX\").\n"
            "   - Penalize jargon (e.g., \"20-yr return periods\" without explanation)\n"
            "   - Use direct, actionable language\n"
            "   - Keep sentences short and clear\n"
            "   - Use imperative mood for actions (\"Do X\", \"Go to Y\", \"Call Z\")\n\n"
            "3. COMPLETENESS (1.0): Coverage checklist - contains who/what/where/when/how.\n"
            "   - Action plans should cover 6-10 core categories (evacuation, kit, insurance, etc.)\n"
            "   - Ensure who/what/where/when/how are present\n"
            "   - Include time windows and contact information where applicable\n"
            "   - Verify essential action categories are covered\n\n"
            "4. RELEVANCE (1.0): Content tailored to user's location/risk level/phase/audience.\n"
            "   - No off-region steps\n"
            "   - Aligns with the stated risk tier\n"
            "   - Location-specific information matches the provided location\n"
            "   - Appropriate for the target audience\n\n"
            "5. COHERENCE (1.0): Logical order (Before → During → After), no contradictions, no duplicates, consistent terms.\n"
            "   - Verify logical flow: Before → During → After\n"
            "   - Check for contradictions between sections\n"
            "   - Identify duplicate or redundant information\n"
            "   - Ensure consistent terminology across sections\n\n"
            "Schema strictly (no prose outside JSON):\n"
            "{\n"
            "  \"decision\": \"REVISE\" | \"PASS\",\n"
            "  \"score\": {\n"
            "    \"accuracy\": 0.0-1.0,\n"
            "    \"clarity\": 0.0-1.0,\n"
            "    \"completeness\": 0.0-1.0,\n"
            "    \"relevance\": 0.0-1.0,\n"
            "    \"coherence\": 0.0-1.0\n"
            "  },\n"
            "  \"issues\": [\n"
            "    {\"type\":\"accuracy\",\"detail\":\"...\"},\n"
            "    {\"type\":\"clarity\",\"detail\":\"...\"},\n"
            "    {\"type\":\"completeness\",\"detail\":\"...\"},\n"
            "    {\"type\":\"relevance\",\"detail\":\"...\"},\n"
            "    {\"type\":\"coherence\",\"detail\":\"...\"}\n"
            "  ],\n"
            "  \"required_changes\": [\n"
            "    \"...\"\n"
            "  ],\n"
            "  \"constraints\": {\n"
            "    \"max_words\": 120,\n"
            "    \"reading_level\": \"Grade 6\",\n"
            "    \"format\": \"title + 3 bullets + CTA + source link\"\n"
            "  },\n"
            "  \"suggested_phrases\": [\n"
            "    \"Move to higher ground now if you live in the shaded area.\",\n"
            "    \"Pick up free sandbags at {location}, {hours}.\"\n"
            "  ],\n"
            "  \"mandatory_facts\": [\n"
            "    \"Source: one of the provided official links\"\n"
            "  ]\n"
            "}\n\n"
            "IMPORTANT:\n"
            "- Each score is 0.0 to 1.0 (not 0-5)\n"
            "- If accuracy < 0.6, set decision to \"REVISE\"\n"
            "- All criteria have equal weight in evaluation\n"
            "- Be specific in issues and required_changes\n\n"
            "Return ONLY JSON."
        )
        tm = TextMessage(content=prompt, source="user")
        try:
            res = await self._llm.on_messages([tm], ctx.cancellation_token)
            content = res.chat_message.content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            # Minimal validation
            if not isinstance(data, dict):
                raise ValueError("Critique not a dict")
            if "decision" not in data:
                data["decision"] = "REVISE"
            
            # Enforce accuracy hard-gate: if accuracy < 0.6, force REVISE
            scores = data.get("score", {})
            accuracy_score = scores.get("accuracy", 0.0)
            # Handle both 0-1.0 and 0-5 scales for backward compatibility
            if isinstance(accuracy_score, (int, float)):
                # If score is > 1.0, assume it's on 0-5 scale and normalize
                if accuracy_score > 1.0:
                    accuracy_score = accuracy_score / 5.0
                # Apply hard-gate
                if accuracy_score < 0.6:
                    data["decision"] = "REVISE"
                    # Add an issue if not already present
                    issues = data.get("issues", [])
                    accuracy_issue_exists = any(
                        issue.get("type") == "accuracy" and "hard-gate" in issue.get("detail", "").lower()
                        for issue in issues
                    )
                    if not accuracy_issue_exists:
                        issues.append({
                            "type": "accuracy",
                            "detail": f"Accuracy score {accuracy_score:.2f} below hard-gate threshold (0.6) - automatic REVISE required"
                        })
                        data["issues"] = issues
            
            # Normalize all scores to 0-1.0 if they're on 0-5 scale (for backward compatibility)
            normalized_scores = {}
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    if value > 1.0:
                        normalized_scores[key] = round(value / 5.0, 2)
                    else:
                        normalized_scores[key] = round(value, 2)
                else:
                    normalized_scores[key] = value
            data["score"] = normalized_scores
            
            return data
        except Exception as e:
            # Fallback minimal critique (all scores 0.0-1.0)
            return {
                "decision": "REVISE",
                "score": {
                    "accuracy": 0.5,
                    "clarity": 0.5,
                    "completeness": 0.5,
                    "relevance": 0.5,
                    "coherence": 0.5,
                },
                "issues": [{"type": "general", "detail": f"auto-critique-fallback: {e}"}],
                "required_changes": [
                    "Tighten wording, ensure Grade 6 reading level, add explicit CTA with hours.",
                ],
                "constraints": {
                    "max_words": 120,
                    "reading_level": "Grade 6",
                    "format": "title + 3 bullets + CTA + source link",
                },
                "suggested_phrases": [],
                "mandatory_facts": ["Source: use an official link from plan.sources"],
            }


# ---------------------------------------------------------
# Local test helper
# ---------------------------------------------------------
async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


async def main():
    from app.agents.govdoc_agent import GovDocAgent
    from app.agents.action_plan_agent import ActionPlanAgent

    runtime = SingleThreadedAgentRuntime()
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    await EvaluatorAgent.register(runtime, "Evaluator", lambda: EvaluatorAgent(runtime))

    await maybe_await(runtime.start())

    print("=" * 80)
    print("🧪 Testing EvaluatorAgent with Vancouver, BC")
    print("=" * 80)
    
    resp = await runtime.send_message(
        Message(content="Hope, QC", metadata={"max_iterations": 1, "debug": True}),
        AgentId("Evaluator", "default"),
    )
    
    print("\n" + "=" * 80)
    print("📦 FINAL RESULT")
    print("=" * 80)
    
    data = json.loads(resp.content)
    
    # Format final output nicely
    location = data.get("location", "N/A")
    initial_summary = data.get("initial_summary", "")
    final_text = data.get("final_text", "")
    critique = data.get("critique", {})
    iterations = data.get("iterations", {})
    was_revised = data.get("was_revised", False)
    
    print(f"\n📍 Location: {location}")
    print(f"🔄 Iterations: {iterations.get('used', 0)}/{iterations.get('max', 0)}")
    print(f"📝 Was Revised: {'Yes' if was_revised else 'No'}")
    
    # Show comparison if available in JSON format
    if initial_summary:
        print("\n" + "=" * 80)
        print("📊 COMPARISON: Initial vs Final Summary (JSON Format)")
        print("=" * 80)
        
        comparison_json = {
            "comparison": {
                "initial_summary": {
                    "text": initial_summary,
                    "length": len(initial_summary),
                    "stage": "Before Evaluator"
                },
                "final_summary": {
                    "text": final_text,
                    "length": len(final_text),
                    "stage": "After Evaluator"
                },
                "was_revised": was_revised,
                "changes": {
                    "summary": "The summary was revised based on evaluator feedback." if was_revised else "No changes made - summary passed evaluation as-is",
                    "length_difference": len(final_text) - len(initial_summary),
                    "initial_length": len(initial_summary),
                    "final_length": len(final_text)
                }
            }
        }
        
        # Print JSON in readable format (line by line)
        print(json.dumps(comparison_json, indent=2, ensure_ascii=False))
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("📄 FINAL SUMMARY TEXT")
        print("=" * 80)
        print(final_text)
        print("=" * 80)
    
    # Print critique summary
    decision = critique.get("decision", "UNKNOWN")
    decision_emoji = "✅" if decision == "PASS" else "⚠️"
    print(f"\n{decision_emoji} Final Decision: {decision}")
    
    scores = critique.get("score", {})
    if scores:
        print("\n📊 Final Scores (0.0-1.0, equal weight):")
        for key, value in scores.items():
            # Handle both 0-1.0 and 0-5 scales for display
            display_value = value
            max_value = 1.0
            if isinstance(value, (int, float)) and value > 1.0:
                display_value = value / 5.0
                max_value = 5.0
            print(f"  • {key.replace('_', ' ').title()}: {display_value:.2f}/{max_value}")
    
    print("\n" + "=" * 80)
    print("💾 Full JSON Response (for reference):")
    print("=" * 80)
    print(json.dumps(data, indent=2, ensure_ascii=False))

    await maybe_await(runtime.stop())


if __name__ == "__main__":
    asyncio.run(main())


