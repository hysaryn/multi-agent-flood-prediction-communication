# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
# cd "/Users/carrietong/Desktop/CS7980 Capstone/multi-agent-flood-prediction-communication/backend"
# source .venv/bin/activate
# python -m app.agents.govdoc_agent
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
from app.services.location_service import get_location_info, LocationInfo, LocationResult

from urllib.parse import urlparse
from typing import List, Optional
import os, re, json, asyncio
from datetime import datetime

from ..tools.downloader import download
from ..tools.text_extractor import extract_text
from ..models.govdoc_models import DocRef
from ..services.token_tracker import tracker

# ---------------------------------------------------------
# Storage cleanup: keep last 5 accessed files per place_key
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
def build_task_prompt(
    place: str, 
    extra_keywords: list[str], 
    revision_notes: list[str] | None = None,
    mode: str = "normal"  # "normal" or "additional"
) -> str:
    """
    Build task prompt for MCP browser agent.
    
    Args:
        place: Location to search for
        extra_keywords: Keywords to use in search
        revision_notes: Optional revision notes to guide additional document search
        mode: "normal" for first iteration, "additional" for adding documents
    """
    kw = ", ".join(extra_keywords) if extra_keywords else "flood action plan, preparedness, checklist, mitigation, response, before, during, after"
    
    if mode == "additional" and revision_notes:
        # Build search query based on revision notes - context-aware, not broader
        notes_text = "\n".join(f"- {note}" for note in revision_notes[:10])
        
        # Extract key terms from revision notes for targeted search
        search_terms = []
        for note in revision_notes:
            note_lower = note.lower()
            # Look for specific missing categories or issues
            if "evacuation" in note_lower and "evacuation" not in search_terms:
                search_terms.append("evacuation")
            if "insurance" in note_lower and "insurance" not in search_terms:
                search_terms.append("insurance")
            if ("emergency kit" in note_lower or "kit" in note_lower) and "emergency kit" not in search_terms:
                search_terms.append("emergency kit")
            if "communication" in note_lower and "communication" not in search_terms:
                search_terms.append("communication")
            if "property protection" in note_lower or ("property" in note_lower and "protection" in note_lower):
                if "property protection" not in search_terms:
                    search_terms.append("property protection")
            if "family plan" in note_lower or ("family" in note_lower and "plan" in note_lower):
                if "family plan" not in search_terms:
                    search_terms.append("family plan")
            # Look for other specific issues mentioned
            if "clarity" in note_lower or "unclear" in note_lower:
                # Don't add generic terms, but note that clarity issues might need more specific docs
                pass
            if "missing" in note_lower and "information" in note_lower:
                # Extract what's missing if mentioned
                if "detail" in note_lower or "specific" in note_lower:
                    # Look for more detailed documents
                    pass
        
        # Build targeted search query - use specific terms, not generic broader terms
        # ALWAYS include "flood" in the search query
        if search_terms:
            # Use the specific terms found, combine with "flood" and location
            search_query = f"flood ({' OR '.join(search_terms[:4])})"  # Use up to 4 specific terms, always with "flood"
        else:
            # If no specific terms found, use minimal focused search with "flood"
            search_query = "flood preparedness"
        
        return f"""
Goal: Find ADDITIONAL PDF documents to address specific gaps in the existing flood action plan for {place}.

CONTEXT - Issues to Address:
{notes_text}

FOCUS: Find documents that specifically address these gaps. Look for:
- Documents covering missing categories or topics mentioned above
- More detailed information on areas where the current plan is incomplete
- Location-specific resources that weren't found in the initial search

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
1) SEARCH once with targeted terms based on the gaps above (MUST include "flood"):
   "{place}" {search_query} filetype:pdf
   
   If too few results, run ONE more SEARCH with site-specific targeting:
   (site:canada.ca OR site:gc.ca OR site:gov.bc.ca OR site:*.ca) "{place}" {search_query} filetype:pdf

2) From SEARCH results, COPY PDF URLs directly (do NOT open PDFs). If still <5 items:
   - OPEN up to 2 promising HTML (non-PDF) official results.
   - On each opened page:
     a) FIND (case-insensitive): terms related to the gaps above (e.g., "evacuation", "insurance", "kit", etc.)
     b) CLICK up to 3 links total (across all pages) that look relevant to addressing the gaps; for any link whose href ENDSWITH .pdf:
         - ADD that PDF href to items immediately (title may be filename if unknown).
         - DO NOT OPEN the PDF; just record the href and continue.
     c) If a page shows multiple PDF links, you may record each without opening them.

3) For each item:
   - title: short title or filename if unknown
   - url: the PDF URL (must end with .pdf)
   - snippet: 1 short line describing how it addresses the gaps
   - filetype: "pdf"
   - why_relevant: short reason explaining how it addresses the specific gaps (e.g., "covers missing evacuation procedures")

4) Deduplicate strictly by URL. If you have ANY items at all, ALWAYS return them (never return empty).
"""
    else:
        # Normal search (first iteration)
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
async def mcp_browser_collect(
    place: str, 
    extra_keywords: list[str] | None = None, 
    timeout_s: int = 120,
    revision_notes: list[str] | None = None,
    mode: str = "normal"
):
    params = {
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    }
    prompt = build_task_prompt(place, extra_keywords or [], revision_notes=revision_notes, mode=mode)
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

# ---------------------------------------------------------
# GovDocAgent - orchestrates gov doc retrieval and ranking
# ---------------------------------------------------------
class GovDocAgent(RoutedAgent):
    def __init__(self, runtime: SingleThreadedAgentRuntime):
        super().__init__("GovDoc")
        self._runtime = runtime
        self._llm = AssistantAgent(
            "GovDocLLM",
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        )

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
    async def on_govdoc_request(self, message: Message, ctx: MessageContext) -> Message:
        """Handle requests to search for government flood preparedness PDFs."""
        start_time = datetime.now()
        print(f"[GovDocAgent] ⏰ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse input
        input_data = {}
        try:
            if message.content.startswith("{"):
                input_data = json.loads(message.content)
        except:
            pass
        
        # Determine mode: add_to_existing (additional documents) or first call (normal)
        add_to_existing = input_data.get("add_to_existing", False)
        previous_urls = set(input_data.get("previous_urls", []))
        previous_plan = input_data.get("previous_plan")
        revision_notes = input_data.get("revision_notes", [])
        existing_govdoc_data = input_data.get("govdoc_data", {})
        
        if add_to_existing:
            print(f"[GovDocAgent] ➕ ADDITIONAL MODE: Searching for additional documents with context")
            govdoc_call_number = 2  # Additional search uses context from revision notes
        else:
            print(f"[GovDocAgent] 📄 NORMAL MODE: First iteration search")
            govdoc_call_number = 1  # First call to GovDoc
        
        # 1) Get structured location information
        location_input = input_data.get("location", message.content)
        loc = await self._resolve_location(location_input, ctx)
        raw_place = (loc.location.query or loc.location.display_name or "Canada").strip()
        parts = [p.strip() for p in raw_place.split(",") if p.strip()]
        place = parts[0] if parts else "Canada"
        place = slugify_place(place) 
        print(f"[GovDocAgent] place = {place}")

        # 2) Search with different keywords based on call number
        links = await self._run_search(
            place, 
            max_total=5,
            govdoc_call_number=govdoc_call_number,
            previous_urls=previous_urls,
            revision_notes=revision_notes if add_to_existing else None,
            mode="additional" if add_to_existing else "normal"
        )
        
        # 2b) Fallback: If additional search found no documents, try normal search
        if add_to_existing and len(links) == 0:
            print(f"[GovDocAgent] ⚠️  Additional search found 0 documents, falling back to normal search...")
            links = await self._run_search(
                place,
                max_total=5,
                govdoc_call_number=1,  # Use normal search
                previous_urls=previous_urls,
                revision_notes=None,  # No revision notes for normal search
                mode="normal"
            )
            print(f"[GovDocAgent] Fallback normal search returned {len(links)} links")

        # 3) Optional: rerank and summarize with LLM
        if os.getenv("OPENAI_API_KEY") and links:
            links = await self._llm_rerank_and_summarize(links, loc.location, ctx)

        print(f"\n[GovDoc] Processing {len(links)} links from MCP...")
        
        pdf_urls = []
        for i, l in enumerate(links[:5], 1):  # Only process top 5
            u = l.url.strip()
            
            print(f"\n[GovDoc] --- Link {i}/5 ---")
            print(f"[GovDoc] Title: {l.title[:60]}")
            print(f"[GovDoc] URL: {u}")
            print(f"[GovDoc] Filetype (from MCP): '{l.filetype}'")
            print(f"[GovDoc] Score: {l.score}")
            
            # Check 1: MCP says it's PDF
            if (l.filetype or "").lower() == "pdf":
                pdf_urls.append(u)
                print(f"[GovDoc] ✅ Added (MCP identified as PDF)")
                continue
            
            # Check 2: URL ends with .pdf
            if u.lower().endswith(".pdf"):
                pdf_urls.append(u)
                print(f"[GovDoc] ✅ Added (URL ends with .pdf)")
                continue
            
            # Check 3: HEAD request to verify content-type
            print(f"[GovDoc] 🔍 Verifying content-type...")
            try:
                if self._seems_pdf_by_head(u, timeout=10):
                    pdf_urls.append(u)
                    print(f"[GovDoc] ✅ Added (verified as PDF)")
                else:
                    print(f"[GovDoc] ❌ Rejected (not a PDF)")
            except Exception as e:
                print(f"[GovDoc] ⚠️  Verification failed: {str(e)[:60]}")
                # Be lenient for high-scoring results
                if l.score > 2.0:
                    pdf_urls.append(u)
                    print(f"[GovDoc] 🤷 Added anyway (high score)")
        
        print(f"\n[GovDoc] Selected {len(pdf_urls)} PDFs for download")
        
        # Deduplicate and limit to 5 PDFs
        seen = set()
        pdf_urls = [u for u in pdf_urls if not (u in seen or seen.add(u))][:5]
        doc_refs: list[DocRef] = []
        for u in pdf_urls:
            try:
                meta = download(u, source="GovDocAgent", place_key=place)
                meta = extract_text(meta)
                doc_refs.append(DocRef(url=meta.url, title=meta.title or "", 
                                    clean_path=meta.clean_path, place_key=place))
                print(f"[GovDoc] ✅ Downloaded: {u}")
            except requests.exceptions.SSLError as e:
                print(f"[GovDoc] ❌ SSL error for {u}: {e}")
                # Could add to a "failed_urls" list for future reference
            except Exception as e:
                print(f"[GovDoc] ❌ Download failed for {u}: {e}")
        # 4) Build payload with documents
        # Track URLs for reference (to avoid duplicates in additional searches)
        found_urls = [str(d.url) for d in doc_refs]
        all_urls = list(previous_urls | set(found_urls))
        
        # Track iteration number for proper flow control
        if add_to_existing:
            current_iteration = 2  # Iteration 2: adding to existing
        else:
            current_iteration = 1  # Iteration 1: normal search
        
        # If add_to_existing, merge with existing docs
        if add_to_existing and existing_govdoc_data:
            existing_docs = existing_govdoc_data.get("docs", [])
            # Merge new docs with existing (avoid duplicates by URL)
            existing_urls = {doc.get("url") for doc in existing_docs}
            new_docs = [
                {
                    "url": str(d.url),   
                    "title": d.title,
                    "clean_path": d.clean_path,
                    "place_key": d.place_key
                }
                for d in doc_refs
                if str(d.url) not in existing_urls
            ]
            all_docs = existing_docs + new_docs
            print(f"[GovDocAgent] 📚 Merged docs: {len(existing_docs)} existing + {len(new_docs)} new = {len(all_docs)} total")
        else:
            all_docs = [
                {
                    "url": str(d.url),   
                    "title": d.title,
                    "clean_path": d.clean_path,
                    "place_key": d.place_key
                }
                for d in doc_refs
            ]
        
        govdoc_payload = {
            "location": loc.location.model_dump(),
            "results": [l.model_dump() for l in links],
            "docs": all_docs,
            "previous_urls": all_urls,  # Track for reference (but don't skip based on this)
            "iteration": current_iteration,  # Pass iteration to downstream agents
        }
        
        # If add_to_existing, pass previous plan and revision notes to ActionPlan
        if add_to_existing:
            govdoc_payload["previous_plan"] = previous_plan
            govdoc_payload["revision_notes"] = revision_notes
            govdoc_payload["add_to_existing"] = True
            print(f"[GovDocAgent] ➕ Passing previous plan and revision notes to ActionPlanAgent")
            
        # 5) Check if we have documents
        if not doc_refs:
            print(f"[GovDocAgent] ❌ No documents downloaded")
            return Message(content=json.dumps({
                "status": "error",
                "error": "No documents found or downloaded",
                "location": message.content.strip()
            }, ensure_ascii=False))
            
        # 6) Sequential flow: Pass govdoc_payload directly to ActionPlanAgent
        doc_collection_time = datetime.now()
        doc_duration = (doc_collection_time - start_time).total_seconds()
        print(f"\n[GovDocAgent] ✅ Document collection complete ({len(doc_refs)} docs) [⏱️  {doc_duration:.2f}s]")
        print(f"[GovDocAgent] → Calling ActionPlanAgent...")
            
            # Send govdoc_payload as-is (no wrapping!)
        action_plan_response = await self._runtime.send_message(
            Message(content=json.dumps(govdoc_payload, ensure_ascii=False)),
            AgentId("ActionPlan", "default")
        )
            
        # 7) ActionPlanAgent returns: {action_plan, govdoc_data, location}
        # Pass it directly to Evaluator (sequential!)
        print(f"[GovDocAgent] → Calling EvaluatorAgent...")
        
        final_response = await self._runtime.send_message(
            action_plan_response,  # Pass through
            AgentId("ActionPlanEvaluator", "default")
        )
        
        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"[GovDocAgent] ⏰ Total duration: {total_duration:.2f}s")
        
        # Print token usage summary for this agent
        summary = tracker.get_summary()
        agent_summary = summary.get("by_agent", {}).get("GovDocAgent", {})
        if agent_summary:
            print(f"[GovDocAgent] 💰 Token usage: {agent_summary.get('total_tokens', 0):,} tokens (${agent_summary.get('cost_usd', 0):.6f})")
        
        return final_response

    async def _resolve_location(self, raw: str, ctx: MessageContext) -> LocationResult:
        """Resolve a location name or coordinates to structured location info."""
        return await get_location_info(raw)
    
    async def _run_search(
        self, 
        place: str, 
        max_total: int = 12,
        govdoc_call_number: int = 1,  # 1 = normal, 2 = additional
        previous_urls: set[str] | None = None,
        revision_notes: list[str] | None = None,
        mode: str = "normal"  # "normal" or "additional"
    ) -> List[GovDocLink]:
        """
        Perform MCP-based web search with different keywords.
        
        Args:
            place: Location to search for
            max_total: Maximum number of links to return
            govdoc_call_number: 1 for normal search, 2 for additional search
            previous_urls: URLs from previous search to avoid duplicates
            revision_notes: Optional revision notes to guide additional document search
            mode: "normal" for first iteration, "additional" for adding documents
        """
        # Determine keywords based on mode
        if mode == "additional":
            # Additional search: Use context from revision notes
            # Extract targeted keywords from revision notes
            extra_kw = []
            if revision_notes:
                # Extract specific terms from revision notes
                for note in revision_notes:
                    note_lower = note.lower()
                    # Look for specific categories/topics
                    if "evacuation" in note_lower and "evacuation" not in extra_kw:
                        extra_kw.append("evacuation")
                    if "insurance" in note_lower and "insurance" not in extra_kw:
                        extra_kw.append("insurance")
                    if ("emergency kit" in note_lower or "kit" in note_lower) and "emergency kit" not in extra_kw:
                        extra_kw.append("emergency kit")
                    if "communication" in note_lower and "communication" not in extra_kw:
                        extra_kw.append("communication")
                    if "property protection" in note_lower or "property" in note_lower:
                        if "property protection" not in extra_kw:
                            extra_kw.append("property protection")
                    if "family plan" in note_lower or "family" in note_lower:
                        if "family plan" not in extra_kw:
                            extra_kw.append("family plan")
            
            # If no specific terms found, use minimal focused terms related to the gaps
            if not extra_kw:
                extra_kw = ["flood preparedness"]  # Keep it minimal and focused
            
            print(f"[_run_search] Using CONTEXT-AWARE search with revision notes")
            print(f"[_run_search] Extracted search terms: {extra_kw}")
        else:  # mode == "normal"
            # Normal search: Focused keywords
            extra_kw = [
                "flood action plan", "preparedness", "checklist", 
                "mitigation", "response", "before", "during", "after"
            ]
            print(f"[_run_search] Using FOCUSED keywords (normal mode)")
        
        browser_links: List[GovDocLink] = []
        try:
            data = await mcp_browser_collect(
                place, 
                extra_kw, 
                timeout_s=120,
                revision_notes=revision_notes,
                mode=mode
            )
            items = (data or {}).get("items", [])
            
            print(f"[_run_search] MCP returned {len(items)} items")
            
            # Only deduplicate within current search, not against previous searches
            # This allows us to get more documents even if some URLs were seen before
            seen_in_current_search = set()
            for i, it in enumerate(items, 1):
                u = (it.get("url") or "").strip()
                
                print(f"[_run_search] Item {i}: {it.get('title', 'No title')[:50]}")
                print(f"[_run_search]   URL: {u}")
                
                if not u:
                    print(f"[_run_search]   ❌ Skipped: empty URL")
                    continue
                
                # Only skip if we've seen this URL in the current search
                if u in seen_in_current_search:
                    print(f"[_run_search]   ⚠️  Skipped: duplicate within current search")
                    continue
                
                # Note if URL was seen in previous searches, but still include it
                if previous_urls and u in previous_urls:
                    print(f"[_run_search]   ℹ️  URL seen in previous search, but including anyway")
                    
                is_pdf = u.lower().endswith(".pdf")
                browser_links.append(GovDocLink(
                    title=(it.get("title") or "")[:300],
                    url=u,
                    snippet=(it.get("snippet") or it.get("why_relevant") or "")[:500],
                    filetype="pdf" if is_pdf else None,
                    score=2.5 if is_pdf else 0.5,
                ))
                seen_in_current_search.add(u)
                print(f"[_run_search]   ✅ Added (PDF: {is_pdf})")
                
                if len(browser_links) >= max_total:
                    break
            
            print(f"[_run_search] Final: {len(browser_links)} links")
        
        except Exception as e:
            print(f"[_run_search] Error: {e}")
        browser_links.sort(key=lambda x: (-(1 if x.filetype == "pdf" else 0), -x.score, x.title.lower()))
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
            # Track token usage - try multiple methods to find usage data
            usage_found = False
            
            # Method 1: Check inner_messages for usage
            if hasattr(res, 'inner_messages') and res.inner_messages:
                for i, msg in enumerate(res.inner_messages):
                    # Check direct usage attribute
                    if hasattr(msg, 'usage'):
                        usage = msg.usage
                        if hasattr(usage, 'prompt_tokens'):
                            tracker.record_usage(
                                agent_name="GovDocAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                operation="rerank_and_summarize"
                            )
                            usage_found = True
                            break
                    # Check nested response
                    if hasattr(msg, 'response') and hasattr(msg.response, 'usage'):
                        usage = msg.response.usage
                        if hasattr(usage, 'prompt_tokens'):
                            tracker.record_usage(
                                agent_name="GovDocAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                operation="rerank_and_summarize"
                            )
                            usage_found = True
                            break
                    # Check if message itself is a dict with usage
                    if isinstance(msg, dict) and 'usage' in msg:
                        usage = msg['usage']
                        tracker.record_usage(
                            agent_name="GovDocAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.get('prompt_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'completion_tokens', 0),
                            operation="rerank_and_summarize"
                        )
                        usage_found = True
                        break
            
            # Method 2: Check chat_message for usage or response_metadata
            if not usage_found and hasattr(res, 'chat_message'):
                if hasattr(res.chat_message, 'usage'):
                    usage = res.chat_message.usage
                    if hasattr(usage, 'prompt_tokens'):
                        tracker.record_usage(
                            agent_name="GovDocAgent",
                            model="gpt-4o-mini",
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            operation="rerank_and_summarize"
                        )
                        usage_found = True
                elif hasattr(res.chat_message, 'response_metadata'):
                    metadata = res.chat_message.response_metadata
                    if metadata:
                        if isinstance(metadata, dict) and 'token_usage' in metadata:
                            usage = metadata['token_usage']
                            tracker.record_usage(
                                agent_name="GovDocAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=usage.get('prompt_tokens', 0),
                                completion_tokens=usage.get('completion_tokens', 0),
                                operation="rerank_and_summarize"
                            )
                            usage_found = True
                        elif hasattr(metadata, 'token_usage'):
                            usage = metadata.token_usage
                            tracker.record_usage(
                                agent_name="GovDocAgent",
                                model="gpt-4o-mini",
                                prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                                completion_tokens=getattr(usage, 'completion_tokens', 0),
                                operation="rerank_and_summarize"
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
                                    agent_name="GovDocAgent",
                                    model="gpt-4o-mini",
                                    prompt_tokens=usage.prompt_tokens,
                                    completion_tokens=usage.completion_tokens,
                                    operation="rerank_and_summarize"
                                )
                                usage_found = True
                except Exception as e:
                    # Silently fail - we'll fall back to estimation
                    pass
            
            # Method 4: Fallback to general method with prompt text for estimation
            if not usage_found:
                prompt_text = tm.content if hasattr(tm, 'content') else ""
                tracker.record_from_openai_response(
                    agent_name="GovDocAgent",
                    model="gpt-4o-mini",
                    response=res,
                    operation="rerank_and_summarize",
                    prompt_text=prompt_text
                )
            data = json.loads(res.chat_message.content)
            boosts = {d["url"]: (float(d.get("boost", 0)), d.get("summary", "")) for d in data.get("items", [])}
            for l in links:
                if l.url in boosts:
                    b, s = boosts[l.url]
                    l.score += b
                    if s:
                        l.snippet = (s.strip() + " ") + (l.snippet or "")
            links.sort(key=lambda x: (-(1 if x.filetype == "pdf" else 0), -x.score, x.title.lower()))
        except Exception:
            pass
        return links

def fetch_docs(urls: list[str], place: str, source: str = "Prepared") -> list[DocRef]:
    docs: list[DocRef] = []
    for u in urls:
        meta = download(u, source=source,place_key=place)
        meta = extract_text(meta)
        docs.append(DocRef(url=meta.url, title=meta.title, clean_path=meta.clean_path, place_key=place))
    return docs
# ---------------------------------------------------------
# Helper for safe await calls
# ---------------------------------------------------------
import inspect
async def maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x
# ---------------------------------------------------------
# Entry point for testing the agent directly
# ---------------------------------------------------------
async def main(): 
    from app.agents.action_plan_agent import ActionPlanAgent
    from app.agents.evaluator_agent import ActionPlanEvaluatorAgent
    
    runtime = SingleThreadedAgentRuntime()

    print("Registering Feedback Loop Pipeline agents...")
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await ActionPlanAgent.register(runtime, "ActionPlan", lambda: ActionPlanAgent(runtime))
    await ActionPlanEvaluatorAgent.register(runtime, "ActionPlanEvaluator", lambda: ActionPlanEvaluatorAgent(runtime))
    
    await maybe_await(runtime.start())

    print("\n" + "=" * 80)
    print("FEEDBACK LOOP PIPELINE TEST")
    print("=" * 80)
    print("Entry: GovDocAgent\n")
    
    pipeline_start = datetime.now()
    print(f"⏰ Pipeline started at {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    resp = await runtime.send_message(
        Message(content="Brisbane, Queensland"),
        AgentId("GovDoc", "default")
    )
    
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
    
    result = json.loads(resp.content)
    
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"Status: {result.get('status')}")
    print(f"Iterations: {result.get('total_iterations', 1)}")
    print(f"⏱️  Total pipeline duration: {pipeline_duration:.2f}s ({pipeline_duration/60:.1f} minutes)")
    
    if "action_plan" in result:
        ap = result["action_plan"]
        print(f"Actions: {len(ap.get('before_flood', []))} before, {len(ap.get('during_flood', []))} during, {len(ap.get('after_flood', []))} after")
    
    # Print token usage summary
    from ..services.token_tracker import tracker
    tracker.print_summary()
    tracker.save_to_file("token_usage.json")
    
    with open("feedback_loop_output.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to: feedback_loop_output.json")

    await maybe_await(runtime.stop())

if __name__ == "__main__":
    asyncio.run(main())
