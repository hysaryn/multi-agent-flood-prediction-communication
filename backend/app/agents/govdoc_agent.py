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
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, trace
from pydantic import BaseModel

# Use unified Pydantic Message model
from app.models.message_model import Message
from app.services.location_service import get_location_info, LocationInfo, LocationResult

from urllib.parse import urlparse
from typing import List, Optional
import os, re, json, asyncio

from ..tools.downloader import download
from ..tools.text_extractor import extract_text
from ..models.govdoc_models import DocRef

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
                    pass  # If parsing fails, keep as raw text

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
        content = message.content.strip()
        
        # CRITICAL: Reject JSON messages - GovDocAgent only accepts location strings
        if content.startswith('{') or content.startswith('['):
            error_msg = f"CRITICAL ERROR: GovDocAgent received JSON instead of location! First 200 chars: {content[:200]}"
            print(f"[GovDocAgent] ❌ {error_msg}")
            # Try to parse to see what it is
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "mode" in payload:
                    mode = payload.get('mode', 'unknown')
                    error_msg += f"\n[GovDocAgent] This appears to be a {mode} request for ActionPlanAgent!"
                    print(f"[GovDocAgent] {error_msg}")
            except:
                pass
            return Message(content=json.dumps({
                "error": "GovDocAgent only accepts location strings, not JSON requests",
                "received": content[:100]
            }))
        
        # 1) Get structured location information
        loc = await self._resolve_location(content, ctx)
        raw_place = (loc.location.query or loc.location.display_name or "Canada").strip()
        parts = [p.strip() for p in raw_place.split(",") if p.strip()]
        place = parts[0] if parts else "Canada"
        place = slugify_place(place) 
        print(f"[GovDocAgent] ✅ place = {place}")

        # 2) Search and filter results
        links = await self._run_search(place, max_total=5)

        # 3) Optional: rerank and summarize with LLM
        if os.getenv("OPENAI_API_KEY") and links:
            links = await self._llm_rerank_and_summarize(links, loc.location, ctx)

        pdf_urls = []
        for l in links[:5]:  # Limit to top 5 to avoid too many requests
            u = l.url.strip()
            
            # Method 1: Check if MCP already identified it as PDF
            if (l.filetype or "").lower() == "pdf":
                pdf_urls.append(u)
                print(f"[GovDoc] ✅ PDF (by filetype): {u}")
                continue
            
            # Method 2: URL ends with .pdf
            if u.lower().endswith(".pdf"):
                pdf_urls.append(u)
                print(f"[GovDoc] ✅ PDF (by extension): {u}")
                continue
            
            # Method 3: Do HEAD/GET to check content-type
            print(f"[GovDoc] 🔍 Checking content-type: {u}")
            try:
                if self._seems_pdf_by_head(u, timeout=10):
                    pdf_urls.append(u)
                    print(f"[GovDoc] ✅ PDF (by content-type): {u}")
                else:
                    print(f"[GovDoc] ❌ Not a PDF: {u}")
            except Exception as e:
                print(f"[GovDoc] ⚠️  Could not check {u}: {e}")
                # If we can't verify, but it's from results, try downloading anyway
                if l.score > 2.0:  # High-scoring results
                    pdf_urls.append(u)
                    print(f"[GovDoc] 🤷 Adding anyway (high score)")
        # Deduplicate and limit to 5 PDFs
        seen = set()
        pdf_urls = [u for u in pdf_urls if not (u in seen or seen.add(u))][:5]
        doc_refs: list[DocRef] = []
        for u in pdf_urls:
            # try:
            #     meta = download(u, source="GovDocAgent", place_key=place)
            #     meta = extract_text(meta)
            #     doc_refs.append(DocRef(url=meta.url, title=meta.title or "", clean_path=meta.clean_path, place_key=place))
            # except Exception as e:
            #     print(f"[GovDocAgent] download/extract failed for {u}: {e}")
            try:
                meta = download(u, source="GovDocAgent", place_key=place)
                meta = extract_text(meta)
                doc_refs.append(DocRef(url=meta.url, title=meta.title or "", 
                                    clean_path=meta.clean_path, place_key=place))
                print(f"[GovDoc] ✅ Downloaded: {u}")
            except requests.exceptions.SSLError as e:
                print(f"[GovDoc] ❌ SSL error for {u}: {e}")
                # Could add to a "failed_urls" list for retry later
            except Exception as e:
                print(f"[GovDoc] ❌ Download failed for {u}: {e}")
        # 4) Build final payload
        payload = {
            "location": loc.location.model_dump(mode='json'),
            "results": [l.model_dump(mode='json') for l in links],
            "docs": [d.model_dump(mode='json') for d in doc_refs],
        }
        return Message(content=json.dumps(payload, ensure_ascii=False))

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
            seen = set()
            for it in items:
                u = (it.get("url") or "").strip()
                if not u or u in seen or not self._is_gov_or_official(u):
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
                if len(browser_links) >= max_total:
                    break
        except Exception as e:
            print("[MCP] fallback due to:", e)
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

    def _is_gov_or_official(self, url: str) -> bool:
        """Check if a URL belongs to an official government domain."""
        h = (urlparse(url).hostname or "").lower()
        if any(h.endswith(s) for s in (".gov", ".gc.ca", ".gov.uk", ".gov.au", ".gouv.fr")):
            return True
        if h.endswith(".ca") and re.search(r"(city|municipal|regional|vancouver|toronto|ottawa)", h):
            return True
        if re.search(r"(cityof)[a-z\-]+", h):
            return True
        return False


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
    runtime = SingleThreadedAgentRuntime()

    # Register the GovDoc agent into the runtime
    await GovDocAgent.register(runtime, "GovDoc", lambda: GovDocAgent(runtime))
    await maybe_await(runtime.start())

    # Test: ask the GovDoc agent to search for flood-related docs
    resp = await runtime.send_message(Message(content="Vancouver"), AgentId("GovDoc", "default"))
    print(resp.content)

    await maybe_await(runtime.stop())

if __name__ == "__main__":
    asyncio.run(main())

# ---------------------------------------------------------
# (Optional) Manual MCP test helper
# ---------------------------------------------------------
# import asyncio, json
# from agents.mcp.server import MCPServerStdio   
# from agents import Agent, Runner   
# from dotenv import load_dotenv  
# load_dotenv(override=True)
#
# MCP_PARAMS = {
#     "command": "npx",
#     "args": ["-y", "@playwright/mcp@latest"],   
# }
#
# async def main():
#     async with MCPServerStdio(params=MCP_PARAMS, name="browser") as browser_mcp:
#         tools = await browser_mcp.list_tools()
#         print("✅ list_tools:", [t.name for t in tools])
#
#         agent = Agent(
#             name="smoke",
#             instructions="You can browse with the 'browser' MCP server.",
#             model="gpt-4.1-mini",
#             mcp_servers=[browser_mcp],
#         )
#         res = await Runner.run(agent, "List available tools and return them as JSON.")
#         print("✅ Runner result:", res.final_output)
#
# if __name__ == "__main__":
#     asyncio.run(main())
