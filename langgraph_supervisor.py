# langgraph_supervisor.py
import os, json, subprocess, pathlib, time
from typing import Literal, TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()  # pulls OPENAI_API_KEY, etc. from .env

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

DATA = pathlib.Path("data")
NEWS = DATA / "news.json"
SENT = DATA / "sentiment.json"
ACTIONS_PATH = DATA / "actions.json"

from datetime import datetime, timezone

import json
from typing import Tuple

ALLOWED_ACTIONS = {"fetch_news", "analyze_sentiment", "trade", "finish"}

# 1) Tool registry
TOOL_REGISTRY = {
    "fetch_news": {
        "schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "lookback_hours": {"type": "integer", "minimum": 1, "maximum": 72},
                "sources": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        }
    },
    "analyze_sentiment": {
        "schema": {
            "type": "object",
            "properties": {
                "recompute_only_new": {"type": "boolean"},
                "models": {"type": "array", "items": {"type":"string"}}
            },
            "additionalProperties": False
        }
    },
    "trade": {
        "schema": {
            "type": "object",
            "properties": {
                "paper_trade": {"type":"boolean"},
                "confidence_min": {"type":"number", "minimum":0, "maximum":1},
                "max_positions": {"type":"integer", "minimum":1, "maximum":50}
            },
            "additionalProperties": False
        }
    }
}

# 2) Validate args
def _validate_args(action: str, args: dict) -> dict:
    schema = TOOL_REGISTRY.get(action, {}).get("schema")
    if not schema:
        return {}
    try:
        import jsonschema
        jsonschema.validate(args, schema)
        return args
    except Exception:
        return {}

# 3) Return args from the LLM
def _llm_choose_action(payload: dict) -> Tuple[str, dict, str]:
    resp = llm.invoke([
        {"role":"system","content": SUPERVISOR_SYS},
        {"role":"user","content": json.dumps(payload)}
    ])
    raw = getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
    data = json.loads(raw)  # may raise
    action = data.get("action","")
    reason = data.get("reason","") or "no reason given"
    args = data.get("args", {}) or {}
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    args = _validate_args(action, args)
    return action, args, reason.strip()

def parse_trader_output(out: str):
    """
    Be tolerant of extra lines like 'Container ... Starting' around the real JSON.
    Try whole string, then { ... } slice, then per-line attempts.
    Return dict or None.
    """
    try:
        data = json.loads(out)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    # Try first '{' .. last '}' slice
    first = out.find("{")
    last = out.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = out[first:last+1]
        try:
            data = json.loads(snippet)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

    # Try line-by-line JSON object
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return data if isinstance(data, dict) else None
            except Exception:
                continue

    return None

def _llm_choose_action(payload: dict) -> Tuple[str, str]:
    """
    Ask the LLM to pick the next action. Returns (action, reason).
    If the LLM returns invalid JSON or an unknown action, we raise ValueError.
    """
    resp = llm.invoke([
        {"role": "system", "content": SUPERVISOR_SYS},
        {"role": "user", "content": json.dumps(payload)}
    ])
    # langchain_openai returns an object with .content (string)
    raw = getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
    data = json.loads(raw)  # may raise

    action = data.get("action", "")
    args = data.get("args", {}) or {}
    reason = data.get("reason", "")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    if not isinstance(reason, str) or not reason.strip():
        reason = "no reason given"
    args = _validate_args(action, args)
    return action, args, reason.strip()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}

# ------------ Utilities ------------
def run_service(name: str, args: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, str]] = None, timeout_s: int = 90) -> str:
    """
    Run a docker compose service once and return its stdout.
    Args are passed as CLI flags: --key value (lists join with comma; booleans add flag when True).
    Extra environment variables can be provided via 'env' (e.g., credentials, toggles).
    """
    cmd: List[str] = ["docker", "compose", "run"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--rm", name]
    if args:
        for k, v in args.items():
            flag = f"--{str(k).replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif isinstance(v, (int, float, str)):
                cmd += [flag, str(v)]
            elif isinstance(v, list):
                cmd += [flag, ",".join(map(str, v))]
            # ignore nested/dict types silently
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout_s)
    return out.strip()

def obs_summary() -> str:
    parts = []
    if NEWS.exists():
        try: 
            with open(NEWS, "r", encoding="utf-8") as f: parts.append(f"news.json({len(json.load(f))} items)")
        except: parts.append("news.json(unreadable)")
    else:
        parts.append("news.json(missing)")
    if SENT.exists():
        try: parts.append(f"sentiment.json({len(json.load(open(SENT)))} items)")
        except: parts.append("sentiment.json(unreadable)")
    else:
        parts.append("sentiment.json(missing)")
    return " | ".join(parts)

# ------------ Agent State ------------
class AgentState(TypedDict):
    logs: List[Dict[str, Any]]
    final: Optional[Dict[str, str]]
    ran_fetch: bool
    ran_analyze: bool
    ran_trade: bool
    plan: Optional[List[Dict[str, Any]]]  
    pending_call: Optional[Dict[str, Any]] 

def _count_json(path: pathlib.Path) -> int:
    try:
        return len(json.load(open(path, "r", encoding="utf-8")))
    except Exception:
        return 0

def _mtime(path: pathlib.Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0

# ------------ LLM (Supervisor brain) ------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- REPLACE your SUPERVISOR_SYS with this ---
SUPERVISOR_SYS = """You are the supervisor for a trading workflow.
You control a graph with these nodes (choose ONE per turn):
- fetch_news      : run a news gatherer (updates data/news.json)
- analyze_sentiment: run a sentiment analyzer (updates data/sentiment.json)
- trade           : run the trader (prints JSON)
- finish          : stop the run

Goal:
- Ensure we have enough recent news, analyze it, then trade when ready.
- Prefer efficient paths; avoid loops.
- If everything useful is done, finish.

Input you'll receive:
- A JSON object with:
  - obs: a one-line summary of file status (counts / missing)
  - files: detailed counts & mtimes for news.json and sentiment.json
  - flags: {ran_fetch, ran_analyze, ran_trade}
  - final_ready: whether a final trade decision already exists (bool)
  - last_log: last routing or node log if present
  - tool_schemas: JSON Schemas for valid args per tool (for reference)

Output:
Return STRICT JSON:
{
  "action": "<fetch_news|analyze_sentiment|trade|finish>",
  "args": { },  // optional; must match the tool's schema
  "reason": "<brief, 1-2 sentences>"
}

Rules:
- Only one action.
- Keep the reason short and actionable.
- You may re-run tools if you judge outputs stale/insufficient.
"""

# --- Replace supervisor_route with this enhanced version ---
def supervisor_route(state: AgentState) -> Literal["planner","fetch_news","analyze_sentiment","trade","__end__"]:
    # If a plan exists, consume the next step first (no LLM hop needed)
    plan = state.get("plan") or []
    if plan:
        step = plan.pop(0)
        state["plan"] = plan
        action = step.get("action")
        args = _validate_args(action, step.get("args") or {})
        reason = step.get("reason") or "from plan"
        # Log and stage the call
        state["pending_call"] = {"action": action, "args": args, "reason": reason}
        state["logs"].append({"node": "supervisor", "thought": f"plan: {reason}", "action": action, "args": args})
        if action == "finish":
            return "__end__"
        return action  # follow the planned action

    # Bootstrap: first hop to planner if nothing has run and no plan exists
    if not state.get("logs") and not plan:
        state["pending_call"] = {"action": "planner", "args": {}, "reason": "bootstrap plan"}
        state["logs"].append({"node":"supervisor","thought":"bootstrap plan","action":"planner","args":{}})
        return "planner"

    # No plan -> gather file stats & ask router
    news_n = _count_json(NEWS)
    sent_n = _count_json(SENT)
    news_m = _mtime(NEWS)
    sent_m = _mtime(SENT)

    payload = {
        "obs": obs_summary(),
        "files": {
            "news": {"count": news_n, "mtime": news_m, "exists": NEWS.exists()},
            "sentiment": {"count": sent_n, "mtime": sent_m, "exists": SENT.exists()},
        },
        "flags": {
            "ran_fetch": bool(state.get("ran_fetch")),
            "ran_analyze": bool(state.get("ran_analyze")),
            "ran_trade": bool(state.get("ran_trade")),
        },
        "final_ready": bool(state.get("final")),
        "last_log": (state["logs"][-1] if state.get("logs") else None),
        "tool_schemas": {k: v.get("schema") for k, v in TOOL_REGISTRY.items()},
    }

    try:
        recent_actions = [log.get("action") for log in state.get("logs", []) if "action" in log]
        payload["recent_actions"] = recent_actions[-3:]  # last three decisions

        action, args, reason = _llm_choose_action(payload)
        state["pending_call"] = {"action": action, "args": args, "reason": reason}
    except Exception as e:
        # Minimal, pragmatic fallback: a very small heuristic,
        # just to keep the app from crashing if the LLM reply is malformed.
        if news_n < 1:
            action, args, reason = "fetch_news", {}, "Fallback: no news; fetching."
        elif sent_n < 1:
            action, args, reason = "analyze_sentiment", {}, "Fallback: no sentiment; analyzing."
        elif not state.get("ran_trade", False) and not state.get("final"):
            action, args, reason = "trade", {}, "Fallback: attempt trade."
        else:
            action, args, reason = "finish", {}, "Fallback: finish."

    # Log what the LLM decided
    state["logs"].append({"node":"supervisor","thought":reason,"action":action,"args":state.get("pending_call",{}).get("args",{})})

    # Map to graph edges
    if action == "finish":
        return "__end__"
    return action  # "fetch_news" | "analyze_sentiment" | "trade"

# ------------ Planner Node ------------
PLANNER_SYS = """You are a short-horizon planner for a trading workflow.
Propose 1–3 next steps to efficiently achieve: fetch sufficient recent news -> analyze -> trade (if ready).
Only use these actions: fetch_news, analyze_sentiment, trade, finish.
For each step, include concise 'reason' and optional 'args' following the tool schemas below.
Return STRICT JSON: { "plan": [ { "action": "...", "args": { ... }, "reason": "..." }, ... ] }
Rules: Be efficient, avoid loops, skip steps already fresh/satisfied if not needed.
"""

def node_planner(state: AgentState) -> AgentState:
    # Build a compact context for the planner
    news_n = _count_json(NEWS)
    sent_n = _count_json(SENT)
    payload = {
        "obs": obs_summary(),
        "files": {
            "news": {"count": news_n, "mtime": _mtime(NEWS), "exists": NEWS.exists()},
            "sentiment": {"count": sent_n, "mtime": _mtime(SENT), "exists": SENT.exists()},
        },
        "flags": {
            "ran_fetch": bool(state.get("ran_fetch")),
            "ran_analyze": bool(state.get("ran_analyze")),
            "ran_trade": bool(state.get("ran_trade")),
        },
        "final_ready": bool(state.get("final")),
        "recent_actions": [log.get("action") for log in state.get("logs", []) if "action" in log][-3:],
        "tool_schemas": {k: v.get("schema") for k, v in TOOL_REGISTRY.items()},
    }
    try:
        resp = llm.invoke([
            {"role": "system", "content": PLANNER_SYS},
            {"role": "user", "content": json.dumps(payload)}
        ])
        raw = getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
        data = json.loads(raw)  # may raise
        plan_items = data.get("plan") or []
    except Exception:
        plan_items = []

    # Validate args for each plan item
    validated_plan: List[Dict[str, Any]] = []
    for item in plan_items:
        action = item.get("action")
        if action not in ALLOWED_ACTIONS:
            continue
        args = _validate_args(action, item.get("args") or {})
        reason = item.get("reason") or "planned step"
        validated_plan.append({"action": action, "args": args, "reason": reason})

    # Fallback minimal plan if LLM failed
    if not validated_plan:
        if news_n < 1:
            validated_plan = [{"action": "fetch_news", "args": {"lookback_hours": 6}, "reason": "no news; fetch recent"}]
        elif sent_n < 1:
            validated_plan = [{"action": "analyze_sentiment", "args": {"recompute_only_new": True}, "reason": "no sentiment; analyze"}]
        else:
            validated_plan = [{"action": "trade", "args": {"paper_trade": True}, "reason": "attempt trade"}]

    state["plan"] = validated_plan
    state["logs"].append({"node": "planner", "observation": f"planned {len(validated_plan)} step(s)", "plan": validated_plan})
    return state

# ------------ Tool Nodes ------------
def node_fetch(state: AgentState) -> AgentState:
    try:
        args = ((state.get("pending_call") or {}).get("args")) or {}
        out = run_service("news_fetcher", args=args)
    except subprocess.CalledProcessError as e:
        out = f"ERROR: {e.output.strip()}"
    print("\nAction: fetch_news")
    print(f"Observation: {(out[:400] + ' ...[truncated]') if len(out)>400 else out}")
    state["ran_fetch"] = True
    state["logs"].append({"node":"fetch_news","observation":"done", "snapshot":obs_summary()})
    return state

def node_analyze(state: AgentState) -> AgentState:
    try:
        args = ((state.get("pending_call") or {}).get("args")) or {}
        out = run_service("sentiment", args=args)
    except subprocess.CalledProcessError as e:
        out = f"ERROR: {e.output.strip()}"
    print("\nAction: analyze_sentiment")
    print(f"Observation: {(out[:400] + ' ...[truncated]') if len(out)>400 else out}")
    state["ran_analyze"] = True
    state["logs"].append({"node":"analyze_sentiment","observation":"done", "snapshot":obs_summary()})
    return state

def node_trade(state: AgentState) -> AgentState:
    try:
        args = ((state.get("pending_call") or {}).get("args")) or {}
        out = run_service("trader", args=args)
    except subprocess.CalledProcessError as e:
        out = f"ERROR: {e.output.strip()}"
    print("\nAction: trade")
    print(f"Observation: {(out[:400] + ' ...[truncated]') if len(out)>400 else out}")
    try:
        parsed = parse_trader_output(out)
        if isinstance(parsed, dict):
            state["final"] = parsed
    except Exception:
        pass
    state["ran_trade"] = True
    state["logs"].append({"node":"trade","observation":"done", "snapshot":obs_summary()})
    return state

def write_json_atomic(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path) 

# ------------ Build the graph ------------
# --- tiny cosmetic tweak so your console shows when supervisor regains control ---
def supervisor_node(state: AgentState) -> AgentState:
    last = state["logs"][-1] if state["logs"] else None
    if last and last.get("node") != "supervisor":
        print(f"[{_now_iso()}] SUPERVISOR routing… | {obs_summary()}")
        state["logs"].append({"node": "supervisor", "note": "routing...", "ts": _now_iso()})
    return state

graph = StateGraph(AgentState)

# Add nodes (including the new 'planner' and 'supervisor')
graph.add_node("supervisor", supervisor_node)
graph.add_node("planner", node_planner)
graph.add_node("fetch_news", node_fetch)
graph.add_node("analyze_sentiment", node_analyze)
graph.add_node("trade", node_trade)

# Entry point is the supervisor node
graph.set_entry_point("supervisor")

# From supervisor, choose next node via the LLM router
graph.add_conditional_edges(
    "supervisor",
    supervisor_route,
    {
        "planner": "planner",
        "fetch_news": "fetch_news",
        "analyze_sentiment": "analyze_sentiment",
        "trade": "trade",
        "__end__": END,
    },
)

# After any tool node, return to supervisor for the next decision
for n in ["planner", "fetch_news", "analyze_sentiment", "trade"]:
    graph.add_edge(n, "supervisor")

app = graph.compile()

# ------------ Runner ------------
if __name__ == "__main__":
    print("=== LangGraph Agent Run ===")
    init_state = {
        "logs": [],
        "final": None,
        "ran_fetch": False,
        "ran_analyze": False,
        "ran_trade": False,
        "plan": [],
        "pending_call": None,
    }
    cfg = {"recursion_limit": 12, "configurable": {"thread_id": "demo-run-1"}}

    last_values = None
    for values in app.stream(init_state, stream_mode="values", config=cfg):
        last_values = values  # capture the latest state snapshot

    print("\n=== Final Answer ===")
    final = (last_values or {}).get("final")

    if final is None:
        final = {"AAPL": "HOLD", "TSLA": "HOLD", "BK": "HOLD"}
        
    write_json_atomic(ACTIONS_PATH, final)
    print(json.dumps(final, indent=2))
