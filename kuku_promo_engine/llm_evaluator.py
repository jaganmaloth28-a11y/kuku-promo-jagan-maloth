"""Editorial judgment only. Parsing, cut selection, and timestamp validation stay in code."""
import hashlib
import json
import os
import re
import requests
from typing import Any, Dict
from .config import Config
from .window_generator import CandidateWindow

_CACHE = {}

def _cached_call(kind: str, content: str, system: str, cache_dir: str = "") -> Dict[str, Any]:
    key = hashlib.sha256((Config.MODEL_BASE_URL + Config.MODEL_NAME + kind + system + content).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    cache_path = os.path.join(cache_dir or os.getenv("KUKU_OUTPUT_DIR", "outputs"), "model_cache.json")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            disk = json.load(f)
        if key in disk:
            _CACHE[key] = disk[key]
            return disk[key]
    except (OSError, ValueError):
        disk = {}
    api_key = Config.MODEL_API_KEY or Config.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY (or OPENAI_API_KEY) to call the selected model.")
    url = Config.MODEL_BASE_URL.rstrip("/") + "/chat/completions"
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json={
        "model": Config.MODEL_NAME,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, timeout=60)
    response.raise_for_status()
    result = json.loads(response.json()["choices"][0]["message"]["content"])
    _CACHE[key] = result
    disk[key] = result
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(disk, f)
    except OSError:
        pass
    return result

class LLMEvaluator:
    @classmethod
    def evaluate(cls, candidate: CandidateWindow, retry: bool = False, cache_dir: str = "") -> Dict[str, Any]:
        provider = Config.LLM_PROVIDER.lower()
        if provider not in {"openai", "compatible"}:
            return HeuristicEvaluator.evaluate(candidate)
        try:
            hook = _first_dialogue(candidate)
            prompt = ("Score only editorial judgment; do not propose or alter cuts. Return JSON: "
                      '{"hook_score":1,"cliffhanger_score":1,"standalone_score":1,"spoiler_level":"none|mild|major",'
                      '"hook_quote":"exact supplied opening dialogue","why":"brief"}. Scores 0-10. '
                      "Use only this clip text. Exact hook quote must be copied from the supplied opening line.\n\n"
                      f"CODE-SELECTED OPENING LINE: {hook}\nCLIP:\n{candidate.formatted_transcript}")
            result = _cached_call("score_retry" if retry else "score", prompt, "You are a short-drama promo editor. Judge hook, cliffhanger, standalone clarity and spoiler risk only. Copy the supplied opening dialogue exactly for hook_quote.", cache_dir)
            result.setdefault("hook_quote", hook)
            result["hook_score"] = _score(result.get("hook_score"))
            result["cliffhanger_score"] = _score(result.get("cliffhanger_score", result.get("cliff_score")))
            result["standalone_score"] = _score(result.get("standalone_score", 5))
            result["hook_description"] = result.get("hook_description") or hook
            result["cliffhanger_description"] = result.get("cliffhanger_description") or candidate.ending_text
            why = result.get("why_it_works") or result.get("why") or ""
            result["why_it_works"] = " ".join(str(why).split())[:240] or _code_reason(candidate)
            result.setdefault("stands_alone_reason", "Model estimate; editor review recommended.")
            result["stands_alone"] = result["standalone_score"] >= 6
            result["overall_score"] = round((result["hook_score"] + result["cliffhanger_score"] + result["standalone_score"]) / 30 * 100)
            result["evaluator"] = f"OpenAI-compatible ({Config.MODEL_NAME})"
            return result
        except Exception as exc:
            result = HeuristicEvaluator.evaluate(candidate)
            result["evaluator"] = f"heuristic_fallback: {str(exc)[:100]}"
            return result

    @classmethod
    def cold_viewer_check(cls, clip_text: str, cache_dir: str = "") -> Dict[str, Any]:
        prompt = ("Read only this clip. Return JSON with fields: who (brief), stakes (brief), "
                  "what_happens_next (brief), needs_earlier_context (boolean), confidence (0-10). "
                  "Do not use episode context.\n\nCLIP:\n" + clip_text)
        return _cached_call("cold", prompt, "You are a first-time viewer checking whether a short promo is understandable without prior episodes.", cache_dir)

def _score(value):
    try:
        return max(0, min(10, int(float(value))))
    except (TypeError, ValueError):
        return 5

def _first_dialogue(candidate):
    for seg in candidate.segments:
        for b in seg["blocks"]:
            if b["type"] == "DIALOGUE":
                return f'[{b["start"]}] {b["speaker"]}: "{b["text"]}"'
    return ""

def _code_reason(candidate) -> str:
    """Summarize measured hook and ending signals without claiming audience evidence."""
    from .window_generator import HOOK_KEYWORDS, CLIFFHANGER_KEYWORDS
    first = candidate.segments[0]
    opening = [b for b in first["blocks"] if b["start_sec"] <= first["start_sec"] + 3.0]
    hook_words = sorted({k for b in opening for k in HOOK_KEYWORDS if k in b["text"].lower()})[:2]
    last = candidate.segments[-1]
    tail = [b for b in last["blocks"] if b["end_sec"] >= last["end_sec"] - 5.0]
    final_line = next((b for b in reversed(last["blocks"]) if b["type"] in ("DIALOGUE", "ACTION")), None)
    ends_question = bool(final_line and final_line["type"] == "DIALOGUE"
                         and re.search(r"\?(?:[!\"'”’]*)$", final_line["text"].strip()))
    question_near_end = any(b["type"] == "DIALOGUE" and "?" in b["text"] for b in tail)
    cliff_words = sorted({k for b in tail for k in CLIFFHANGER_KEYWORDS if k in b["text"].lower()})[:2]
    if getattr(candidate, "relaxed_hook", False):
        opening_reason = "Relaxed fallback: opens on a question, exclamation, or high-emotion dialogue"
    else:
        opening_reason = ("Opens on hook words " + "/".join(f"'{w}'" for w in hook_words)
                          if hook_words else "Opens on dialogue within 3s")
    if ends_question:
        ending_reason = "ends on a question"
    elif question_near_end:
        ending_reason = "a question appears within the final five seconds, but the final line is declarative"
    elif cliff_words:
        ending_reason = "ends near tension words " + "/".join(f"'{w}'" for w in cliff_words)
    else:
        ending_reason = "ending signal is weak"
    kind = "joined moments" if candidate.candidate_type == "stitched" else "one continuous cut"
    return f"{opening_reason}; {ending_reason} ({kind}). Keyword-based, not viewer-tested."

class HeuristicEvaluator:
    @classmethod
    def evaluate(cls, candidate: CandidateWindow) -> Dict[str, Any]:
        hook = _first_dialogue(candidate)
        hook_score = round(candidate.code_hook_score)
        cliff = round(candidate.code_cliffhanger_score)
        transcript = candidate.formatted_transcript.lower()
        suspicious = any(x in transcript for x in ("ledger", "inheritance", "secret", "real father", "duke"))
        standalone = 4 if suspicious else 7
        return {
            "hook_score": hook_score, "cliffhanger_score": cliff,
            "standalone_score": standalone, "stands_alone": standalone >= 6,
            "hook_quote": hook, "hook_description": hook,
            "cliffhanger_description": candidate.ending_text,
            "stands_alone_reason": "Code-only estimate; human cold-viewer review recommended.",
            "why_it_works": _code_reason(candidate),
            "spoiler_level": "mild" if suspicious else "none",
            "overall_score": round((hook_score + cliff + standalone) / 30 * 100),
            "evaluator": "heuristic_engine"
        }
