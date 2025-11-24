# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
# Single Agent Architecture - All functionality in one agent
# This is a single-agent version for comparison with sequential architecture
# ---------------------------------------------------------
from dotenv import load_dotenv 
load_dotenv(override=True)
import requests
from app.tools.storage import purge_keep_last_n, slugify_place

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
from agents.mcp.server import MCPServerStdio
from agents import Agent, Runner, trace
from pydantic import BaseModel

# Use unified Pydantic Message model
from app.models.message_model import Message
from app.models.action_plan_models import Action, ActionPlanResponse
from app.services.location_service import get_location_info, LocationInfo, LocationResult
from app.services.cost_tracker import get_cost_tracker

from urllib.parse import urlparse
from typing import List, Optional, Dict, Tuple
import os, re, json, asyncio
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI

from app.tools.downloader import download
from app.tools.text_extractor import extract_text
from app.models.govdoc_models import DocRef

# ---------------------------------------------------------
# Storage cleanup: keep last 2 accessed files per place_key
# ---------------------------------------------------------
purge_keep_last_n(n=0)

# ---------------------------------------------------------
# MCP INSTRUCTION TEMPLATE
# ---------------------------------------------------------
MCP_INSTRUCTIONS = """
You browse the internet to accomplish your instructions using the provided browser_* tools.
Always prefer OFFICIAL government sources for Canada:
- Federal: canada.ca OR site:gc.ca
- British Columbia: site:gov.bc.ca
- Municipalities: site:*.ca where host contains the city/region name (e.g., vancouver.ca, surrey.ca, burnaby.ca).
Do NOT return news, blogs, mirrors, or third–party sites.

Return ONLY compact JSON:
{"items":[{"title": "...","url": "...","snippet":"...","filetype":"pdf|html","why_relevant":"..."}]}
No prose, no markdown.
"""

# ---------------------------------------------------------
# Prompt builder for the browser MCP agent
# ---------------------------------------------------------
def build_task_prompt(place: str, extra_keywords: list[str]) -> str:
    kw = ", ".join(extra_keywords) if extra_keywords else "flood action plan, preparedness, checklist, mitigation, response, before, during, after"
    return f"""
Goal: Return ONLY PDF links (URLs ending with .pdf) about flood action plans for: {place}
Focus: {kw}

REQUIRED: URLs must be from government/official sources:
- Federal government domains: .gov, .gc.ca, .gov.uk, .gov.au, .gouv.fr
- State/provincial domains: .gov.bc.ca, .gov.on.ca, .ontario.ca, .gov.ab.ca, etc.
- Municipal/city domains: vancouver.ca, toronto.ca, ottawa.ca, etc.
- Regional authorities and emergency management agencies

Prefer domains containing: gov, government, city, municipality, emergency, regional

HARD LIMITS (DO NOT VIOLATE):
- TOTAL tool calls (SEARCH/OPEN/FIND/CLICK) <= 9
- As soon as you SEE a PDF link (URL endswith .pdf), ADD it to items immediately.
- You MUST include every PDF you CLICKED in the final items. Never drop a clicked PDF.
- NEVER collect the same URL twice; if you have seen a URL, SKIP it.
- STOP as soon as you have 5 PDF items.
- OUTPUT ONLY JSON:
  {{"items":[{{"title":"...","url":"...","snippet":"1 line","filetype":"pdf","why_relevant":"short"}}]}}

Procedure (follow exactly):
1) SEARCH once:
   "{place}" flood preparedness OR flood mitigation filetype:pdf
   If too few results, run ONE more SEARCH:
   (site:canada.ca OR site:gc.ca OR site:gov.bc.ca OR site:*.ca) "{place}" flood plan OR preparedness OR mitigation filetype:pdf

2) From SEARCH results, COPY PDF URLs directly (do NOT open PDFs). If still <5 items:
   - OPEN up to 2 promising HTML (non-PDF) official results.
   - On each opened page:
     a) FIND (case-insensitive): "before" OR "during" OR "after".
     b) CLICK up to 3 links total (across all pages) that look relevant; for any link whose href ENDSWITH .pdf:
         - ADD that PDF href to items immediately (title may be filename if unknown).
         - DO NOT OPEN the PDF; just record the href and continue.
     c) If a page shows multiple PDF links, you may record each without opening them.

3) For each item:
   - title: short title or filename if unknown
   - url: the PDF URL (must end with .pdf)
   - snippet: 1 short line (e.g., "official plan PDF" or mention Before/During/After if visible nearby)
   - filetype: "pdf"
   - why_relevant: short reason (e.g., "official flood action plan PDF")

4) Deduplicate strictly by URL. If you have ANY items at all, ALWAYS return them (never return empty).
"""

# ---------------------------------------------------------
# Main function for controlling browser via MCP server
# ---------------------------------------------------------
async def mcp_browser_collect(place: str, extra_keywords: list[str] | None = None, timeout_s: int = 120):
    params = {
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    }
    prompt = build_task_prompt(place, extra_keywords or [])
    try:
        async with MCPServerStdio(
            params=params,
            name="browser",
            client_session_timeout_seconds=120,
        ) as browser_mcp:
            agent = Agent(
                name="investigator",
                instructions=MCP_INSTRUCTIONS,
                model="gpt-4.1-mini",
                mcp_servers=[browser_mcp],
            )
            with trace("investigate"):
                task = "Find local-government flood action plans, summarize to JSON."
                result = await asyncio.wait_for(
                    Runner.run(agent, prompt + "\n" + task),
                    timeout=timeout_s
                )

            data = result.final_output

            # Try to parse JSON if the model returned string output
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass   

            # Extract "items" list from possible JSON structures
            def _extract_items(payload):
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        return []
                if isinstance(payload, dict):
                    for key in ("items", "results", "links"):
                        v = payload.get(key)
                        if isinstance(v, list):
                            return v
                if isinstance(payload, list):
                    return payload
                return []
            items = _extract_items(data)
            # Debug output
            print("[MCP] final_output type:", type(data).__name__)
            print("[MCP] items extracted:", len(items))
            if not isinstance(data, (dict, list)):
                print("[MCP] final_output preview:", str(data)[:800])
            else:
                try:
                    print("[MCP] final_output preview:", json.dumps(data, ensure_ascii=False)[:800])
                except Exception:
                    print("[MCP] final_output preview: <unserializable>")

            # ✅ Return extracted items in unified format
            return {"items": items} if items else {"items": []}

    except asyncio.TimeoutError:
        print(f"[MCP] browser collect timed out after {timeout_s}s")
        return {"items": []}
    except Exception as e:
        print("[MCP] browser collect failed:", e)
        return {"items": []}

# ---------------------------------------------------------
# Pydantic Models for govdoc results
# ---------------------------------------------------------
class GovDocLink(BaseModel):
    """Represents a single government document link result."""
    title: str
    url: str
    source: str = "mcp"
    snippet: Optional[str] = None
    filetype: Optional[str] = None
    score: float = 0.0

class GovDocResponse(BaseModel):
    """Full response containing location info and found links."""
    location: LocationInfo
    results: List[GovDocLink]

# Try to import tiktoken for accurate token-based truncation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[SingleAgent] ⚠️  tiktoken not available, using character-based truncation")

# ---------------------------------------------------------
# SingleAgent - All functionality in one agent
# ---------------------------------------------------------
class SingleAgent(RoutedAgent):
    """
    Single Agent that performs all tasks: document collection, action plan generation,
    evaluation, and revision in one unified flow.
    
    Single Agent Architecture:
    - Receives: Location query (e.g., "Toronto, Ontario")
    - Processes: All steps sequentially within this agent
    - Outputs: Final action plan with evaluation
    """
    
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("SingleAgent")
        self._runtime = runtime
        
        # Initialize tiktoken for accurate token-based truncation
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception as e:
                print(f"[SingleAgent] ⚠️  Failed to initialize tiktoken: {e}")
                self._encoding = None
        else:
            self._encoding = None
        
        # LLM for extracting and categorizing actions
        self._llm = AssistantAgent(
            "SingleAgentLLM",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini",
            ),
        )
        
        # OpenAI client for evaluation and revision
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-4o-mini"
    
    def _truncate_text_by_tokens(self, text: str, max_tokens: int = 15000) -> str:
        """
        Truncate text by token count (more accurate than character count).
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens (default: 10000)
        
        Returns:
            Truncated text
        """
        if self._encoding is None:
            # Fallback: approximate token truncation using character count
            # Rough estimate: 1 token ≈ 4 characters for English
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]
        
        # Accurate token-based truncation
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to max_tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self._encoding.decode(truncated_tokens)
    
    def _seems_pdf_by_head(self, url: str, timeout=12) -> bool:
        try:   
            r = requests.head(url, timeout=timeout, allow_redirects=True,
                            headers={"User-Agent": "flood-agent/1.0"})
            ct = r.headers.get("content-type", "").lower()
            if "pdf" in ct:
                return True
        except Exception:
            pass
        try:
            r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True,
                            headers={"User-Agent": "flood-agent/1.0"})
            ct = r.headers.get("content-type", "").lower()
            return "pdf" in ct
        except Exception:
            return False
    
    @message_handler
    async def on_request(self, message: Message, ctx: MessageContext) -> Message:
        """
        Main entry point - handles the complete pipeline in one agent.
        
        Single Agent Architecture:
        - Step 1: Document collection (GovDocAgent functionality)
        - Step 2: Action plan generation (ActionPlanAgent functionality)
        - Step 3: Evaluation (EvaluatorAgent functionality)
        - Step 4: Revision if needed (RevisionAgent functionality)
        - Step 5: Return final result
        """
        # Reset cost tracker at the start of each pipeline run
        from app.services.cost_tracker import reset_cost_tracker
        reset_cost_tracker()
        
        pipeline_start = datetime.now(timezone.utc)
        print(f"\n[{pipeline_start.isoformat()}] [Pipeline] Started")
        print(f"[SingleAgent] Processing location: {message.content.strip()}")
        
        try:
            # ========== STEP 1: Document Collection ==========
            print(f"\n[SingleAgent] ═══════════════════════════════════════")
            print(f"[SingleAgent] Step 1: Document Collection")
            print(f"[SingleAgent] ═══════════════════════════════════════")
            
            # 1) Get structured location information
            loc = await self._resolve_location(message.content, ctx)
            raw_place = (loc.location.query or loc.location.display_name or "Canada").strip()
            parts = [p.strip() for p in raw_place.split(",") if p.strip()]
            place = parts[0] if parts else "Canada"
            place = slugify_place(place) 
            print(f"[SingleAgent] place = {place}")

            # 2) Search and filter results
            links = await self._run_search(place, max_total=5)

            # 3) Optional: rerank and summarize with LLM
            if os.getenv("OPENAI_API_KEY") and links:
                links = await self._llm_rerank_and_summarize(links, loc.location, ctx)

            print(f"\n[SingleAgent] Processing {len(links)} links from MCP...")
            
            pdf_urls = []
            for i, l in enumerate(links[:5], 1):  # Only process top 5
                u = l.url.strip()
                
                print(f"\n[SingleAgent] --- Link {i}/5 ---")
                print(f"[SingleAgent] Title: {l.title[:60]}")
                print(f"[SingleAgent] URL: {u}")
                print(f"[SingleAgent] Filetype (from MCP): '{l.filetype}'")
                print(f"[SingleAgent] Score: {l.score}")
                
                # Check 1: MCP says it's PDF
                if (l.filetype or "").lower() == "pdf":
                    pdf_urls.append(u)
                    print(f"[SingleAgent] ✅ Added (MCP identified as PDF)")
                    continue
                
                # Check 2: URL ends with .pdf
                if u.lower().endswith(".pdf"):
                    pdf_urls.append(u)
                    print(f"[SingleAgent] ✅ Added (URL ends with .pdf)")
                    continue
                
                # Check 3: HEAD request to verify content-type
                print(f"[SingleAgent] 🔍 Verifying content-type...")
                try:
                    if self._seems_pdf_by_head(u, timeout=10):
                        pdf_urls.append(u)
                        print(f"[SingleAgent] ✅ Added (verified as PDF)")
                    else:
                        print(f"[SingleAgent] ❌ Rejected (not a PDF)")
                except Exception as e:
                    print(f"[SingleAgent] ⚠️  Verification failed: {str(e)[:60]}")
                    # Be lenient for high-scoring results
                    if l.score > 2.0:
                        pdf_urls.append(u)
                        print(f"[SingleAgent] 🤷 Added anyway (high score)")
            
            print(f"\n[SingleAgent] Selected {len(pdf_urls)} PDFs for download")
            
            # Deduplicate and limit to 5 PDFs
            seen = set()
            pdf_urls = [u for u in pdf_urls if not (u in seen or seen.add(u))][:5]
            doc_refs: list[DocRef] = []
            for u in pdf_urls:
                try:
                    meta = download(u, source="SingleAgent", place_key=place)
                    meta = extract_text(meta)
                    doc_refs.append(DocRef(url=meta.url, title=meta.title or "", 
                                        clean_path=meta.clean_path, place_key=place))
                    print(f"[SingleAgent] ✅ Downloaded: {u}")
                except requests.exceptions.SSLError as e:
                    print(f"[SingleAgent] ❌ SSL error for {u}: {e}")
                except Exception as e:
                    print(f"[SingleAgent] ❌ Download failed for {u}: {e}")
            
            # Check if we have documents
            if not doc_refs:
                print(f"[SingleAgent] ❌ No documents downloaded")
                return Message(content=json.dumps({
                    "status": "error",
                    "error": "No documents found or downloaded",
                    "location": message.content.strip()
                }, ensure_ascii=False))
            
            print(f"\n[SingleAgent] ✅ Document collection complete ({len(doc_refs)} docs)")
            
            # Build govdoc payload for downstream steps
            govdoc_payload = {
                "location": loc.location.model_dump(),
                "results": [l.model_dump() for l in links],
                "docs": [
                    {
                        "url": str(d.url),   
                        "title": d.title,
                        "clean_path": d.clean_path,
                        "place_key": d.place_key
                    }
                    for d in doc_refs
                ],
                "_pipeline_start_time": pipeline_start.isoformat()
            }
            
            location_info = loc.location
            location_query = location_info.query or location_info.display_name or "Unknown"
            display_name = location_info.display_name
            docs = govdoc_payload["docs"]
            pipeline_start_time = pipeline_start.isoformat()
            
            # ========== STEP 2: Action Plan Generation ==========
            print(f"\n[SingleAgent] ═══════════════════════════════════════")
            print(f"[SingleAgent] Step 2: Action Plan Generation")
            print(f"[SingleAgent] ═══════════════════════════════════════")
            
            print(f"\n[SingleAgent] Generating plan for: {location_query}")
            print(f"[SingleAgent] Received {len(docs)} documents")

            # Read all document texts
            doc_texts = []
            for doc in docs:
                clean_path = doc.get("clean_path")
                url = doc.get("url")
                
                if not clean_path or not Path(clean_path).exists():
                    print(f"[SingleAgent]   ⚠️  Skipping missing file: {clean_path}")
                    continue
                
                text = Path(clean_path).read_text(encoding='utf-8', errors='ignore')
                
                # Truncate by tokens (more accurate than characters)
                truncated_text = self._truncate_text_by_tokens(text, max_tokens=15000)
                
                # Count tokens if available for logging
                if self._encoding:
                    original_tokens = len(self._encoding.encode(text))
                    truncated_tokens = len(self._encoding.encode(truncated_text))
                    if original_tokens > truncated_tokens:
                        print(f"[SingleAgent]   ✅ Loaded: {doc.get('title', 'Untitled')[:50]}... ({len(text)} chars, {original_tokens} tokens → {truncated_tokens} tokens)")
                    else:
                        print(f"[SingleAgent]   ✅ Loaded: {doc.get('title', 'Untitled')[:50]}... ({len(text)} chars, {original_tokens} tokens)")
                else:
                    print(f"[SingleAgent]   ✅ Loaded: {doc.get('title', 'Untitled')[:50]}... ({len(text)} chars)")
                
                doc_texts.append({
                    "url": url,
                    "text": truncated_text,
                    "title": doc.get("title", "Untitled")
                })
            
            if not doc_texts:
                return Message(content=json.dumps({
                    "error": "No readable documents found",
                    "location": location_query,
                    "action_plan": None,
                    "govdoc_data": govdoc_payload
                }))
            
            # Extract actions from all documents
            print(f"[SingleAgent] Extracting actions from {len(doc_texts)} documents...")
            all_actions_with_phases = []
            
            for doc in doc_texts:
                actions_with_phases = await self._extract_actions_from_doc(doc, location_query, ctx)
                all_actions_with_phases.extend(actions_with_phases)
                print(f"[SingleAgent]   Extracted {len(actions_with_phases)} actions")
            
            print(f"[SingleAgent] Total actions extracted: {len(all_actions_with_phases)}")
            
            # Deduplicate
            deduplicated = self._deduplicate_actions_with_phases(all_actions_with_phases)
            print(f"[SingleAgent] Deduplication: {len(all_actions_with_phases)} → {len(deduplicated)} actions")
            
            # Categorize using LLM phases
            before, during, after = self._categorize_by_phase(deduplicated)
            
            print(f"[SingleAgent] Categorized: Before={len(before)}, During={len(during)}, After={len(after)}")
            
            # Build action plan
            action_plan = ActionPlanResponse(
                location=location_query,
                display_name=display_name,
                before_flood=before,
                during_flood=during,
                after_flood=after,
                sources=[doc["url"] for doc in doc_texts],
                generated_at=datetime.now(timezone.utc).isoformat(),
                pipeline_start_time=pipeline_start_time
            )

            print(f"[SingleAgent] ✅ Plan complete: {action_plan.total_actions()} total actions")
            
            action_plan_json = action_plan.model_dump(mode='python')
            
            # ========== STEP 3: Evaluation ==========
            print(f"\n[SingleAgent] ═══════════════════════════════════════")
            print(f"[SingleAgent] Step 3: Evaluation")
            print(f"[SingleAgent] ═══════════════════════════════════════")
            print(f"[SingleAgent] Evaluating plan for {location_query}")
            
            risk_level = "Warning"
            
            # Evaluate Original Plan
            print(f"\n[SingleAgent] 📊 Evaluating original plan...")
            
            coverage_data = self._check_category_coverage(action_plan)
            eval_result = await self._llm_evaluate(action_plan, location_query, risk_level, coverage_data, ctx)
            eval_result = self._calculate_final_scores(eval_result, coverage_data)
            recommendation = self._determine_recommendation(eval_result)
            eval_result["recommendation"] = recommendation
            
            print(f"[SingleAgent] ✅ Evaluation complete:")
            print(f"[SingleAgent]    Recommendation: {recommendation}")
            print(f"[SingleAgent]    Overall Score: {eval_result['overall_score']:.2f}/5.0")
            print(f"[SingleAgent]    Accuracy: {eval_result['accuracy']['score']}/5")
            print(f"[SingleAgent]    Clarity: {eval_result['clarity']['score']}/5")
            print(f"[SingleAgent]    Completeness: {eval_result['completeness']['score']}/5")
            print(f"[SingleAgent]    Relevance: {eval_result['relevance']['score']}/5")
            print(f"[SingleAgent]    Coherence: {eval_result['coherence']['score']}/5")
            
            original_evaluation = eval_result
            original_plan_json = action_plan_json
            
            # ========== STEP 4: Revision (if needed) ==========
            if recommendation == "REVISE":
                print(f"\n[SingleAgent] ═══════════════════════════════════════")
                print(f"[SingleAgent] Step 4: Revision")
                print(f"[SingleAgent] ═══════════════════════════════════════")
                print(f"[SingleAgent] Starting revision for {location_query}")
                
                original_plan = ActionPlanResponse(**original_plan_json)
                
                # Perform Revisions
                print(f"\n[SingleAgent] 🔧 Performing targeted revisions...")
                
                changes_made = []
                revised_plan = original_plan.model_copy(deep=True)
                
                # Strategy 1: Add missing categories
                coverage_data = original_evaluation.get('coverage_data', {})
                missing = coverage_data.get('missing_essential', [])
                if missing:
                    print(f"[SingleAgent]   Strategy 1: Adding missing categories: {missing}")
                    new_actions = await self._add_missing_categories(missing, govdoc_payload, location_query, ctx)
                    
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
                    print(f"[SingleAgent]   Strategy 2: Removing duplicates")
                    original_count = revised_plan.total_actions()
                    revised_plan = self._remove_duplicates(revised_plan, coherence)
                    removed = original_count - revised_plan.total_actions()
                    if removed > 0:
                        changes_made.append(f"Removed {removed} duplicates")
                
                # Strategy 3: Fix phase errors
                if coherence.get('phase_errors'):
                    print(f"[SingleAgent]   Strategy 3: Fixing phase errors")
                    revised_plan = self._fix_phase_errors(revised_plan, coherence)
                    changes_made.append("Fixed phase errors")
                
                # Strategy 4: Enhance clarity
                clarity_score = original_evaluation.get('clarity', {}).get('score', 5)
                if clarity_score < 3.5:  # 3.5/5 = 0.7 in old scale
                    print(f"[SingleAgent]   Strategy 4: Enhancing clarity")
                    count = await self._enhance_clarity(revised_plan, location_query, ctx)
                    if count > 0:
                        changes_made.append(f"Enhanced {count} actions")
                
                print(f"[SingleAgent] ✅ Revision complete: {len(changes_made)} strategies applied")
                for i, change in enumerate(changes_made, 1):
                    print(f"[SingleAgent]      {i}. {change}")
                
                # Re-evaluate Revised Plan
                print(f"\n[SingleAgent] 📊 Re-evaluating revised plan...")
                
                revised_plan_json = revised_plan.model_dump(mode='python')
                coverage_data_v2 = self._check_category_coverage(revised_plan)
                revised_evaluation = await self._llm_evaluate(
                    revised_plan, location_query, "Warning", coverage_data_v2, ctx
                )
                revised_evaluation = self._calculate_final_scores(revised_evaluation, coverage_data_v2)
                recommendation_v2 = self._determine_recommendation(revised_evaluation)
                revised_evaluation["recommendation"] = recommendation_v2
                
                print(f"[SingleAgent] ✅ Re-evaluation complete:")
                print(f"[SingleAgent]    Recommendation: {recommendation_v2}")
                print(f"[SingleAgent]    Overall Score: {revised_evaluation['overall_score']:.2f}/5.0")
                
                # Compare and Select
                print(f"\n[SingleAgent] ⚖️  Comparing versions...")
                
                comparison = self._compare_versions(
                    original_evaluation, revised_evaluation,
                    original_plan, revised_plan
                )
                
                print(f"[SingleAgent] 📊 Comparison Results:")
                print(f"[SingleAgent]    Better Version: {comparison['better_version'].upper()}")
                
                # Select better version
                if comparison['better_version'] == "revised":
                    final_plan = revised_plan_json
                    final_evaluation = revised_evaluation
                    selected_version = "revised"
                else:
                    final_plan = original_plan_json
                    final_evaluation = original_evaluation
                    selected_version = "original"

                if isinstance(final_plan, dict):
                    final_plan["pipeline_start_time"] = pipeline_start_time
                
                status = "approved" if final_evaluation["recommendation"] == "APPROVE" else "needs_improvement"
                
                print(f"[SingleAgent] ✅ Selected: {selected_version.upper()}")
                
                # Calculate timestamps
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[SingleAgent] ⚠️ Could not calculate duration: {e}")
                
                if isinstance(final_plan, dict):
                    final_plan["pipeline_end_time"] = pipeline_end_time
                    final_plan["total_processing_seconds"] = total_processing_seconds
                
                if total_processing_seconds:
                    print(f"[SingleAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps({
                    "status": status,
                    "selected_version": selected_version,
                    "final_plan": final_plan,
                    "comparison": comparison,
                    "original_evaluation": original_evaluation,
                    "revised_evaluation": revised_evaluation,
                    "changes_made": changes_made,
                    "cost_summary": cost_summary
                }, indent=2, ensure_ascii=False))
            
            elif recommendation == "APPROVE":
                print(f"\n[SingleAgent] ✅ Plan approved without revision")
                
                # Calculate timestamps for APPROVE case
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[SingleAgent] ⚠️ Could not calculate duration: {e}")
                
                # Add timestamps to final_plan
                final_plan_with_timestamps = action_plan_json.copy()
                if isinstance(final_plan_with_timestamps, dict):
                    final_plan_with_timestamps["pipeline_start_time"] = pipeline_start_time
                    final_plan_with_timestamps["pipeline_end_time"] = pipeline_end_time
                    final_plan_with_timestamps["total_processing_seconds"] = total_processing_seconds
                
                if total_processing_seconds:
                    print(f"[SingleAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps({
                    "status": "approved",
                    "selected_version": "original",
                    "final_plan": final_plan_with_timestamps,
                    "evaluation": eval_result,
                    "cost_summary": cost_summary
                }, indent=2, ensure_ascii=False))
            
            else:  # BLOCK
                print(f"\n[SingleAgent] ❌ Plan blocked (quality too low)")
                
                # Calculate timestamps for BLOCK case
                pipeline_end = datetime.now(timezone.utc)
                pipeline_end_time = pipeline_end.isoformat()
                
                total_processing_seconds = None
                if pipeline_start_time:
                    try:
                        start_dt = datetime.fromisoformat(pipeline_start_time.replace('Z', '+00:00'))
                        total_processing_seconds = (pipeline_end - start_dt).total_seconds()
                    except Exception as e:
                        print(f"[SingleAgent] ⚠️ Could not calculate duration: {e}")
                
                if total_processing_seconds:
                    print(f"[SingleAgent] ⏱️  Total Processing Time: {total_processing_seconds:.2f}s")
                
                # Add cost summary to response
                cost_summary = get_cost_tracker().get_summary()
                get_cost_tracker().print_summary()
                
                print(f"[{pipeline_end_time}] [Pipeline] Completed\n")
                
                return Message(content=json.dumps({
                    "status": "blocked",
                    "evaluation": eval_result,
                    "reason": "Plan quality below minimum acceptable threshold",
                    "pipeline_start_time": pipeline_start_time,
                    "pipeline_end_time": pipeline_end_time,
                    "total_processing_seconds": total_processing_seconds,
                    "cost_summary": cost_summary
                }, indent=2, ensure_ascii=False))
        
        except Exception as e:
            print(f"[SingleAgent] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return Message(content=json.dumps({
                "error": str(e),
                "status": "error"
            }))
    
    # ========== Document Collection methods ==========
    
    async def _resolve_location(self, raw: str, ctx: MessageContext) -> LocationResult:
        """Resolve a location name or coordinates to structured location info."""
        return await get_location_info(raw)
    
    async def _run_search(self, place: str, max_total: int = 12) -> List[GovDocLink]:
        """Perform MCP-based web search for government flood PDFs."""
        browser_links: List[GovDocLink] = []
        try:
            extra_kw = ["flood action plan", "preparedness", "checklist", "mitigation", "response", "before", "during", "after"]
            data = await mcp_browser_collect(place, extra_kw, timeout_s=120)
            items = (data or {}).get("items", [])
            
            print(f"[_run_search] MCP returned {len(items)} items")
            
            seen = set()
            for i, it in enumerate(items, 1):
                u = (it.get("url") or "").strip()
                
                print(f"[_run_search] Item {i}: {it.get('title', 'No title')[:50]}")
                print(f"[_run_search]   URL: {u}")
                
                if not u:
                    print(f"[_run_search]   ❌ Skipped: empty URL")
                    continue
                if u in seen:
                    print(f"[_run_search]   ❌ Skipped: duplicate")
                    continue
                is_pdf = u.lower().endswith(".pdf")
                browser_links.append(GovDocLink(
                    title=(it.get("title") or "")[:300],
                    url=u,
                    snippet=(it.get("snippet") or it.get("why_relevant") or "")[:500],
                    filetype="pdf" if is_pdf else None,
                    score=2.5 if is_pdf else 0.5,
                ))
                seen.add(u)
                print(f"[_run_search]   ✅ Added (PDF: {is_pdf})")
                
                if len(browser_links) >= max_total:
                    break
            
            print(f"[_run_search] Final: {len(browser_links)} links")
        
        except Exception as e:
            print("[MCP] fallback due to:", e)
        # Deterministic sorting: PDF first, then by score, then by URL (for consistency)
        browser_links.sort(key=lambda x: (
            -(1 if x.filetype == "pdf" else 0),  # PDF files first
            -x.score,                              # Higher score first
            x.url.lower()                          # Then by URL (deterministic)
        ))
        return browser_links

    async def _llm_rerank_and_summarize(self, links: List[GovDocLink], loc: LocationInfo, ctx: MessageContext):
        """Use LLM to rank and summarize found government PDFs."""
        items = [{"title": l.title, "url": l.url, "snippet": l.snippet or "", "score": l.score} for l in links]
        prompt = (
            "You are ranking official government flood preparation documents for the given place.\n"
            f"PLACE: {loc.display_name or loc.query}\n\n"
            "Return JSON with a list 'items', each item: "
            "{'url':..., 'boost':0..3, 'summary':'1-2 sentences concise summary'}.\n"
            "Higher 'boost' for PDFs and clearly official plans/guides.\n"
            "Only return JSON."
        )
        tm = TextMessage(
            content=prompt + "\n\nCANDIDATES:\n" + json.dumps(items, ensure_ascii=False),
            source="user"
        )
        try:
            res = await self._llm.on_messages([tm], ctx.cancellation_token)
            
            # Try to track cost
            prompt_text = prompt + "\n\nCANDIDATES:\n" + json.dumps(items, ensure_ascii=False)
            prompt_tokens_est = len(prompt_text.split()) * 1.3
            content = res.chat_message.content.strip()
            completion_tokens_est = len(content.split()) * 1.3
            
            # Try to get actual usage if available
            usage = None
            if hasattr(res, 'usage'):
                usage = res.usage
            elif hasattr(res, 'chat_message') and hasattr(res.chat_message, 'usage'):
                usage = res.chat_message.usage
            
            if usage:
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="rerank_documents",
                    model="gpt-4o-mini",
                    usage={
                        "prompt_tokens": getattr(usage, 'prompt_tokens', int(prompt_tokens_est)),
                        "completion_tokens": getattr(usage, 'completion_tokens', int(completion_tokens_est)),
                        "total_tokens": getattr(usage, 'total_tokens', int(prompt_tokens_est + completion_tokens_est))
                    }
                )
            else:
                # Fallback: estimate
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="rerank_documents",
                    model="gpt-4o-mini",
                    prompt_tokens=int(prompt_tokens_est),
                    completion_tokens=int(completion_tokens_est),
                    total_tokens=int(prompt_tokens_est + completion_tokens_est)
                )
            
            data = json.loads(content)
            boosts = {d["url"]: (float(d.get("boost", 0)), d.get("summary", "")) for d in data.get("items", [])}
            for l in links:
                if l.url in boosts:
                    b, s = boosts[l.url]
                    l.score += b
                    if s:
                        l.snippet = (s.strip() + " ") + (l.snippet or "")
            # Deterministic sorting after LLM reranking
            links.sort(key=lambda x: (
                -(1 if x.filetype == "pdf" else 0),  # PDF files first
                -x.score,                              # Higher score first
                x.url.lower()                          # Then by URL (deterministic)
            ))
        except Exception:
            pass
        return links
    
    # ========== Action Plan Generation methods ==========
    
    async def _extract_actions_from_doc(self, doc: Dict, location: str, ctx: MessageContext) -> List[Tuple[Action, str]]:
        """
        Extract ONLY **resident-facing flood preparedness and response actions**.
        
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
            
            # Try to track cost - autogen may not expose usage directly
            # Estimate based on prompt and response length if usage not available
            prompt_tokens_est = len(prompt.split()) * 1.3  # Rough estimate: 1.3 tokens per word
            content = response.chat_message.content.strip()
            completion_tokens_est = len(content.split()) * 1.3
            
            # Try to get actual usage if available
            usage = None
            if hasattr(response, 'usage'):
                usage = response.usage
            elif hasattr(response, 'chat_message') and hasattr(response.chat_message, 'usage'):
                usage = response.chat_message.usage
            
            if usage:
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="extract_actions",
                    model="gpt-4o-mini",
                    usage={
                        "prompt_tokens": getattr(usage, 'prompt_tokens', int(prompt_tokens_est)),
                        "completion_tokens": getattr(usage, 'completion_tokens', int(completion_tokens_est)),
                        "total_tokens": getattr(usage, 'total_tokens', int(prompt_tokens_est + completion_tokens_est))
                    }
                )
            else:
                # Fallback: estimate based on content length
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="extract_actions",
                    model="gpt-4o-mini",
                    prompt_tokens=int(prompt_tokens_est),
                    completion_tokens=int(completion_tokens_est),
                    total_tokens=int(prompt_tokens_est + completion_tokens_est)
                )
            
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
                    print(f"[SingleAgent]   ⚠️  Skipping invalid action: {e}")
                    continue
            
            return actions_with_phases
        
        except json.JSONDecodeError as e:
            print(f"[SingleAgent]   ❌ JSON parse error: {e}")
            print(f"[SingleAgent]   LLM response preview: {content[:300]}")
            return []
        except Exception as e:
            print(f"[SingleAgent]   ❌ Error: {e}")
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
                        print(f"[SingleAgent]   🗑️  Duplicate: '{action.title}'")
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
    
    # ========== Evaluation methods ==========
    
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
- Coverage: {coverage_data['coverage_ratio']:.0%}, Missing: {coverage_data['missing_essential']}

PLAN:
{action_plan.model_dump_json(indent=2, exclude_none=True)}

EVALUATION CRITERIA (each scored 1-5, integer scale):

1. ACCURACY (1-5): "Is the information factually correct and verifiable?"
    5: All sources cited, verified official
    4: Most information appears official; mostly trustworthy
    3: Some sources, some uncertainty
    2: Hard to verify; low confidence in accuracy
    1: Not trustworthy; information seems fabricated
---
2. CLARITY (1-5): "Is the guidance easy to understand for someone without emergency training?"
    5: 0 confusing words; reads smoothly
    4: 1 confusing word; minor hesitation
    3: 2~3 confusing words; re-reading needed
    2: more than 3 confusing words; repeated clarification
    1: Incomprehensible; unable to understand; language is too technical
---
3. COMPLETENESS (1-5): Does the guidance cover all essential preparedness categories?
    Check if the plan includes actions for these 6 essential categories:
    - Communication: alerts, information, monitoring, emergency contacts
    - Evacuation: evacuation procedures, routes, shelters, evacuation planning
    - Property protection: protecting home, property, belongings, property safety
    - Insurance: flood insurance purchase, claims, coverage, insurance planning
    - Family planning: family communication plan, family coordination, family safety
    - Emergency kit: emergency supplies, preparedness kit, essentials, emergency preparation
    Score based on how many of these 6 categories are covered:
    5: All 6 categories covered (communication, evacuation, property protection, insurance, family planning, emergency kit)
    4: 5 categories covered (missing one)
    3: 4 categories covered (missing two)
    2: 3 or fewer categories covered (missing three or more)
    1: Severely incomplete; critical information missing
---
4. RELEVANCE (1-5): Is the guidance specific to your location?
    5: 4+ location-specific details (mentions of: neighborhood names, streets, local agencies); not transferable
    4: 2-3 specific details; mostly relevant
    3: 1 specific detail; mix of generic/specific
    2: Minimal specificity; mostly generic
    1: Zero specificity; completely generic
---
5. COHERENCE (1-5): Is the guidance internally consistent and logical?
    5: Perfect flow; before→during→after makes sense
    4: Mostly logical; one confusing element
    3: Some confusion; 2-3 contradictions
    2: Confusing flow; multiple contradictions
    1: Makes no sense; severe contradictions
---
JSON output:
{{
  "accuracy": {{"score": 1-5, "justification": "...", "issues": []}},
  "clarity": {{"score": 1-5, "justification": "...", "issues": []}},
  "completeness": {{"score": 1-5, "justification": "...", "issues": []}},
  "relevance": {{"score": 1-5, "justification": "...", "issues": []}},
  "coherence": {{"score": 1-5, "phase_errors": [], "duplicate_actions": [], "contradictions": [], "justification": "...", "issues": []}}
}}
"""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You evaluate emergency plans objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Set to 0.0 for maximum consistency
                max_tokens=2000
            )
            
            # Track cost and check for truncation
            if hasattr(response, 'usage') and response.usage:
                completion_tokens = response.usage.completion_tokens
                max_tokens_set = 2000
                
                # Warn if approaching max_tokens limit (possible truncation)
                if completion_tokens >= max_tokens_set * 0.9:  # Used 90% or more
                    print(f"[SingleAgent] ⚠️  High token usage: {completion_tokens}/{max_tokens_set} tokens used")
                    print(f"[SingleAgent]    Consider increasing max_tokens if output is truncated")
                
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="evaluate_plan",
                    model=self._model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            # Try to parse JSON, catch truncation errors
            try:
                eval_data = json.loads(content)
                
                # Debug: Print what LLM actually returned
                print(f"[SingleAgent] 🔍 LLM returned keys: {list(eval_data.keys())}")
                if eval_data:
                    first_key = list(eval_data.keys())[0]
                    print(f"[SingleAgent] 🔍 Sample entry ({first_key}): {eval_data.get(first_key, {})}")
                
                # CRITICAL: Ensure ALL required dimensions exist, even if LLM didn't return them
                required_dims = ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']
                
                for dim in required_dims:
                    # If dimension doesn't exist in LLM response, create it with default values
                    if dim not in eval_data:
                        print(f"[SingleAgent] ⚠️  {dim}: Missing from LLM response, creating default")
                        eval_data[dim] = {
                            "score": 3,
                            "justification": "Dimension not evaluated by LLM",
                            "issues": []
                        }
                        if dim == 'coherence':
                            eval_data[dim]['phase_errors'] = []
                            eval_data[dim]['duplicate_actions'] = []
                            eval_data[dim]['contradictions'] = []
                    # Ensure score field exists and is valid (1-5)
                    elif 'score' not in eval_data[dim]:
                        print(f"[SingleAgent] ⚠️  {dim}: Missing score field, using default")
                        eval_data[dim]['score'] = 3
                    # Validate score is in 1-5 range
                    else:
                        score = eval_data[dim]['score']
                        if not isinstance(score, (int, float)) or score < 1 or score > 5:
                            print(f"[SingleAgent] ⚠️  {dim}: Invalid score {score}, clamping to 1-5")
                            eval_data[dim]['score'] = max(1, min(5, int(round(score))))
                
                return eval_data
            except json.JSONDecodeError as e:
                print(f"[SingleAgent] ❌ JSON parse error (possible truncation): {e}")
                print(f"[SingleAgent]    Content length: {len(content)} chars")
                print(f"[SingleAgent]    Content preview: {content[:200]}...")
                # Re-raise to trigger fallback
                raise
        
        except Exception as e:
            print(f"[SingleAgent] ❌ LLM error: {e}")
            return self._fallback_scores()
    
    def _fallback_scores(self) -> Dict:
        """Fallback scores if LLM fails."""
        return {
            "accuracy": {"score": 3, "justification": "Evaluation failed", "issues": []},
            "clarity": {"score": 3, "justification": "Evaluation failed", "issues": []},
            "completeness": {"score": 3, "justification": "Evaluation failed", "issues": []},
            "relevance": {"score": 3, "justification": "Evaluation failed", "issues": []},
            "coherence": {"score": 3, "justification": "Evaluation failed", "issues": [], 
                         "phase_errors": [], "duplicate_actions": [], "contradictions": []},
        }
    
    def _calculate_final_scores(self, eval_result: Dict, coverage_data: Dict) -> Dict:
        """Calculate weighted overall score using 1-5 scale."""
        weights = {
            'accuracy': 0.25,
            'clarity': 0.15,
            'completeness': 0.20,
            'relevance': 0.20,
            'coherence': 0.20
        }
        
        # Calculate weighted average (1-5 scale)
        overall_score = sum(
            eval_result[dim]['score'] * weight 
            for dim, weight in weights.items()
        )
        
        # Thresholds in 1-5 scale (equivalent to 0.8, 0.7, 0.7, 0.7, 0.8 in 0.0-1.0)
        thresholds = {
            'accuracy': 4,      # 4/5 = 0.8 in old scale
            'clarity': 3.5,     # 3.5/5 = 0.7 in old scale
            'completeness': 3.5, # 3.5/5 = 0.7 in old scale
            'relevance': 3.5,   # 3.5/5 = 0.7 in old scale
            'coherence': 4      # 4/5 = 0.8 in old scale
        }
        
        passes_threshold = all(
            eval_result[dim]['score'] >= threshold
            for dim, threshold in thresholds.items()
        )
        
        # Confidence levels in 1-5 scale
        # 0.85 in old scale = 4.25/5, 0.70 = 3.5/5
        confidence = "high" if overall_score >= 4.25 else "medium" if overall_score >= 3.5 else "low"
        
        eval_result.update({
            'overall_score': round(overall_score, 2),  # 1-5 scale, 2 decimal places
            'passes_threshold': passes_threshold,
            'overall_confidence': confidence,
            'weights': weights,
            'thresholds': thresholds,
            'coverage_data': coverage_data
        })
        
        return eval_result
    
    def _determine_recommendation(self, eval_result: Dict) -> str:
        """Determine recommendation using 1-5 scale."""
        # 0.6 in old scale = 3.0/5, 0.75 = 3.75/5, 0.65 = 3.25/5
        if eval_result['accuracy']['score'] < 3:
            return "BLOCK"
        
        contradictions = eval_result['coherence'].get('contradictions', [])
        if contradictions:
            text = ' '.join(contradictions).lower()
            if any(kw in text for kw in ['stay', 'evacuate', 'leave', 'remain']):
                return "BLOCK"
        
        if eval_result['passes_threshold'] and eval_result['overall_score'] >= 3.75:
            return "APPROVE"
        
        if eval_result['overall_score'] >= 3.25:
            return "REVISE"
        
        return "BLOCK"
    
    # ========== Revision methods ==========
    
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
                temperature=0.0,  # Set to 0.0 for maximum consistency
                max_tokens=1500
            )
            
            # Track cost
            if hasattr(response, 'usage') and response.usage:
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="add_missing_categories",
                    model=self._model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
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
            
            print(f"[SingleAgent]   ✅ Added {len(actions)} actions")
            return actions
        
        except Exception as e:
            print(f"[SingleAgent]   ❌ Error: {e}")
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
                temperature=0.0,  # Set to 0.0 for maximum consistency
                max_tokens=200
            )
            
            # Track cost
            if hasattr(response, 'usage') and response.usage:
                get_cost_tracker().record_usage(
                    agent_name="SingleAgent",
                    operation="enhance_clarity",
                    model=self._model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
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
    
    def _compare_versions(self, eval_v1: Dict, eval_v2: Dict, plan_v1: ActionPlanResponse, plan_v2: ActionPlanResponse) -> Dict:
        """Compare two versions."""
        score_v1 = eval_v1['overall_score']
        score_v2 = eval_v2['overall_score']
        score_delta = score_v2 - score_v1
        
        improvements = []
        regressions = []
        
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            delta = eval_v2[dim]['score'] - eval_v1[dim]['score']
            if delta > 0.25:  # 0.25 in 1-5 scale (equivalent to 0.05 in 0.0-1.0)
                improvements.append(f"{dim.capitalize()} +{delta:.1f}")
            elif delta < -0.25:
                regressions.append(f"{dim.capitalize()} {delta:.1f}")
        
        better_version = "revised" if score_delta > 0.1 else ("original" if score_delta < -0.1 else 
                         ("revised" if len(improvements) > len(regressions) else "original"))
        
        return {
            "better_version": better_version,
            "score_delta": round(score_delta, 3),
            "original_score": round(score_v1, 3),
            "revised_score": round(score_v2, 3),
            "improvements": improvements,
            "regressions": regressions
        }


# ---------------------------------------------------------
# Helper for safe await calls
# ---------------------------------------------------------
async def maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


# ---------------------------------------------------------
# Entry point for testing the single agent
# ---------------------------------------------------------
async def main():
    """Test the Single Agent Architecture"""
    runtime = SingleThreadedAgentRuntime()
    
    # Register single agent
    print("Registering Single Agent...")
    await SingleAgent.register(runtime, "SingleAgent", lambda: SingleAgent(runtime))
    
    await maybe_await(runtime.start())
    
    print("\n" + "=" * 80)
    print("SINGLE AGENT ARCHITECTURE")
    print("=" * 80)
    print("Architecture: Single Agent (all functionality in one agent)")
    print("Flow: SingleAgent handles all steps internally")
    print("Entry Point: SingleAgent")
    print("=" * 80)
    
    # Test with location
    test_location = "Toronto, Ontario"
    print(f"\nQuery: {test_location}\n")
    
    # Entry point: SingleAgent
    response = await runtime.send_message(
        Message(content=test_location),
        AgentId("SingleAgent", "default")
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
    if "final_plan" in result:
        ap = result["final_plan"]
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
    eval_key = "evaluation" if "evaluation" in result else "revised_evaluation" if "revised_evaluation" in result else None
    if eval_key and eval_key in result:
        ev = result[eval_key]
        print(f"\n✅ Final Evaluation:")
        print(f"  Score: {ev.get('overall_score', 0):.2f}/5.0")
        print(f"  Recommendation: {ev.get('recommendation')}")
        print(f"  Dimension Scores:")
        
        for dim in ['accuracy', 'clarity', 'completeness', 'relevance', 'coherence']:
            if dim in ev:
                score = ev[dim].get('score', 0)
                threshold = ev.get('thresholds', {}).get(dim, 3.5)
                icon = "✅" if score >= threshold else "❌"
                print(f"    {icon} {dim.capitalize()}: {score}/5 (threshold: {threshold:.1f}/5)")
        
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
    with open("single_agent_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: single_agent_output.json")
    
    await maybe_await(runtime.stop())
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

