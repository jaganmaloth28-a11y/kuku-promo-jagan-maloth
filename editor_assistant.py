"""
editor_assistant.py - a grounded "Ask the editor" assistant for the promo results.

Rules it enforces:
  * It only sees the tool's results (ids, episodes, timestamps, scores, decisions, scrubbed explanations).
  * Common questions (rank comparisons, episode filters, decisions, top N) are answered by code, with no model.
  * Open questions go to the swappable OpenAI-compatible model, with a strict system prompt.
  * Any answer that cites a promo ID that is not in the results is replaced with "Not in the results."
No Streamlit imports, so it can be tested.
"""
from __future__ import annotations

import re

import editor_tools as et

NOT_IN_RESULTS = "Not in the results."

SYSTEM_PROMPT = (
    "You are the Editor's Room assistant for a short-drama promo tool. "
    "Answer ONLY from the RESULTS block below. Quote promo IDs and timestamps exactly as written there. "
    "If the answer is not in RESULTS, reply exactly: Not in the results. "
    "Never invent promo IDs, scores, timestamps, dialogue or plot. Never quote script lines. "
    "If RESULTS has no explanation text, say you can only answer from scores and timings. "
    "Keep answers under 120 words."
)

_RANGE_RE = re.compile(r"episodes?\s*(\d+)\s*(?:-|–|—|to|through)\s*(\d+)", re.IGNORECASE)
_ONE_EP_RE = re.compile(r"episode\s*(\d+)", re.IGNORECASE)
_TOP_RE = re.compile(r"\btop\s*(\d+)\b", re.IGNORECASE)
_TOP_QUALIFIER_RE = re.compile(r"\b(?:best|highest)\b", re.IGNORECASE)
_PROMO_NOUN_RE = re.compile(r"\b(?:promos?|clips?|cuts?|picks?)\b", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(?:why|above|over|higher|better|compare|versus|vs)\b", re.IGNORECASE)
_ID_RE = re.compile(r"\b[A-Za-z]+-?\d+\b")
_TIMESTAMP_RE = re.compile(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?!\d)")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_KEYWORDS = ("reveal", "twist", "betray", "secret", "confront")


# ------------------------------------------------------------------ helpers
def _episodes(promo: dict) -> list:
    out = []
    for seg in promo.get("segments", []):
        try:
            out.append(int(seg.get("episode")))
        except (TypeError, ValueError):
            pass
    return out


def _timing(promo: dict) -> str:
    parts = []
    for seg in promo.get("segments", []):
        parts.append(f"Ep {seg.get('episode')} {seg.get('start')}-{seg.get('end')}")
    return "; ".join(parts) or "no timing"


def _ev(promo: dict) -> dict:
    return promo.get("evaluation", {}) or {}


def _status(promo: dict, decisions: dict) -> str:
    return decisions.get(promo["id"], {}).get("status", "PENDING")


def _line(promo: dict, decisions: dict, scrub=None, with_why: bool = False) -> str:
    ev = _ev(promo)
    bits = [
        f"{promo['id']}", f"rank {promo.get('rank', '?')}", _timing(promo),
        f"{promo.get('total_duration', '?')}s", f"fit {ev.get('overall_score', '?')}/100",
        f"hook {ev.get('hook_score', '?')}/10", f"cliffhanger {ev.get('cliffhanger_score', '?')}/10",
        f"stands_alone {ev.get('stands_alone', '?')}", f"decision {_status(promo, decisions)}",
    ]
    if promo.get("rank_reason"):
        bits.append(f"rank reason {promo['rank_reason']}")
    text = " | ".join(str(b) for b in bits)
    if with_why:
        why = str(ev.get("why_it_works", "") or "").strip()
        if scrub:
            why = scrub(why)
        if why:
            text += f" | why: {why[:240]}"
    return text


def build_context(promos: list, decisions: dict, adjustments: dict, scrub=None, max_promos: int = 60) -> str:
    """Text-only RESULTS block for the model. `scrub` hides quoted script lines in redacted mode."""
    ranked = et.rank_sorted(promos)
    chosen = ranked[:max_promos]
    seen = {p["id"] for p in chosen}
    chosen += [p for p in ranked[max_promos:] if _status(p, decisions) in {"APPROVED", "FLAGGED"} and p["id"] not in seen]
    lines = [_line(p, decisions, scrub, with_why=True) for p in chosen]
    for pid, adj in adjustments.items():
        if pid in {p["id"] for p in chosen}:
            lines.append(f"{pid} editor-adjusted cut: {et.sec_to_tc(adj['start_sec'], True)}-{et.sec_to_tc(adj['end_sec'], True)}")
    header = f"{len(promos)} promos in total; {len(chosen)} listed below."
    return "RESULTS\n" + header + "\n" + "\n".join(lines)


def known_ids(promos: list) -> set:
    return {p["id"] for p in promos}


def _find_ids(question: str, ids: set) -> list:
    lookup = {i.lower(): i for i in ids}
    found = []
    for token in _ID_RE.findall(question):
        real = lookup.get(token.lower())
        if real and real not in found:
            found.append(real)
    return found


def ungrounded_ids(answer: str, ids: set) -> list:
    """IDs in the answer that look like promo IDs (same prefix as real ones) but are not in the results."""
    prefixes = {m.group(0).lower() for i in ids if (m := re.match(r"[A-Za-z]+-?", i))}
    lowered = {i.lower() for i in ids}
    bad = []
    for token in _ID_RE.findall(answer):
        m = re.match(r"[A-Za-z]+-?", token)
        if m and m.group(0).lower() in prefixes and token.lower() not in lowered:
            bad.append(token)
    return bad


# ------------------------------------------------------------------ answers computed by code
def answer_deterministic(question: str, promos: list, decisions: dict, scrub=None):
    """Return an answer string for questions the data can settle directly, else None."""
    if not promos:
        return "No results are loaded yet. Analyse a script first."
    ranked = et.rank_sorted(promos)
    by_id = {p["id"]: p for p in promos}
    q = question.lower()
    mentioned = _find_ids(question, set(by_id))

    # why is A ranked above B
    if len(mentioned) >= 2 and _COMPARE_RE.search(q):
        a, b = by_id[mentioned[0]], by_id[mentioned[1]]
        return _compare(a, b, scrub)

    # a single promo
    if len(mentioned) == 1 and not _RANGE_RE.search(question):
        return _line(by_id[mentioned[0]], decisions, scrub, with_why=True)

    # episode filters (+ optional keyword in the explanation)
    lo = hi = None
    m = _RANGE_RE.search(question)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
    elif _ONE_EP_RE.search(question):
        lo = hi = int(_ONE_EP_RE.search(question).group(1))
    keyword = next((k for k in _KEYWORDS if k in q), None)
    if lo is not None or keyword:
        hits = []
        for p in ranked:
            if lo is not None and not any(lo <= e <= hi for e in _episodes(p)):
                continue
            if keyword:
                why = str(_ev(p).get("why_it_works", "") or "")
                if keyword not in why.lower():
                    continue
            hits.append(p)
        scope = f"episodes {lo}-{hi}" if lo is not None else "all episodes"
        if keyword and not any(_ev(p).get("why_it_works") for p in ranked):
            return (f"Not in the results. These results have no explanation text, so I can't search for "
                    f"'{keyword}'. I can filter by episode, score, timing and decision.")
        if not hits:
            return f"Not in the results: no promos for {scope}" + (f" mention '{keyword}'." if keyword else ".")
        shown = hits[:10]
        body = "\n".join(f"- {_line(p, decisions, scrub)}" for p in shown)
        more = f"\n(+{len(hits) - len(shown)} more)" if len(hits) > len(shown) else ""
        qual = f" mentioning '{keyword}' in the explanation" if keyword else ""
        return f"{len(hits)} promo(s) in {scope}{qual}:\n{body}{more}"

    # editor decisions
    for word, status in (("approved", "APPROVED"), ("flagged", "FLAGGED"), ("skipped", "REJECTED"), ("rejected", "REJECTED"), ("pending", "PENDING")):
        if word in q:
            hits = [p for p in ranked if _status(p, decisions) == status]
            if not hits:
                return f"No promos are marked {status.lower()} yet."
            return f"{len(hits)} {status.lower()}:\n" + "\n".join(f"- {_line(p, decisions, scrub)}" for p in hits[:15])

    # top N
    t = _TOP_RE.search(question)
    if t or (_TOP_QUALIFIER_RE.search(q) and _PROMO_NOUN_RE.search(q)):
        n = int(t.group(1)) if t else 5
        return f"Top {min(n, len(ranked))} by rank:\n" + "\n".join(f"- {_line(p, decisions, scrub)}" for p in ranked[:n])
    return None


def _compare(a: dict, b: dict, scrub=None) -> str:
    first, second = (a, b) if (a.get("rank") or 10**9) <= (b.get("rank") or 10**9) else (b, a)
    ea, eb = _ev(first), _ev(second)
    rows = []
    for label, key in (("fit score", "overall_score"), ("hook", "hook_score"), ("cliffhanger", "cliffhanger_score")):
        va, vb = ea.get(key), eb.get(key)
        if va is not None and vb is not None:
            rows.append(f"{label}: {first['id']} {va} vs {second['id']} {vb}")
    rows.append(f"stands alone: {first['id']} {ea.get('stands_alone')} vs {second['id']} {eb.get('stands_alone')}")
    rows.append(f"duration: {first['id']} {first.get('total_duration')}s vs {second['id']} {second.get('total_duration')}s")
    if first.get("rank_reason"):
        rows.append(f"ranking reason: {first['id']} {first['rank_reason']}")
    out = (f"{first['id']} (rank {first.get('rank')}, {_timing(first)}) is ranked above "
           f"{second['id']} (rank {second.get('rank')}, {_timing(second)}).\n" + "\n".join(f"- {r}" for r in rows))
    why = str(ea.get("why_it_works", "") or "")
    if why:
        out += f"\nTool's explanation for {first['id']}: {(scrub(why) if scrub else why)[:240]}"
    else:
        out += "\nThese results carry scores and timings only, so that is all I can compare."
    return out


# ------------------------------------------------------------------ model path
def call_model(question: str, context: str, history: list, settings: dict, post=None, timeout: int = 40) -> str:
    """OpenAI-compatible chat call. `post` is injectable for tests. Raises RuntimeError on failure."""
    if post is None:
        import requests
        post = requests.post
    base = (settings.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    try:
        resp = post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"},
            json={"model": settings["model_name"], "messages": messages, "temperature": 0, "max_tokens": 300},
            timeout=timeout,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()
    except Exception as exc:  # network, auth, bad payload
        raise RuntimeError(f"The model call failed: {exc}") from exc


def _ungrounded_values(answer: str, context: str) -> tuple[list[str], list[str]]:
    """Return numeric values and timestamps in the answer absent from RESULTS."""
    context_ids_removed = _ID_RE.sub(" ", context)
    answer_ids_removed = _ID_RE.sub(" ", answer)
    allowed_timestamps = set(_TIMESTAMP_RE.findall(context_ids_removed))
    answer_timestamps = set(_TIMESTAMP_RE.findall(answer_ids_removed))
    bad_timestamps = sorted(answer_timestamps - allowed_timestamps)

    context_without_times = _TIMESTAMP_RE.sub(" ", context_ids_removed)
    answer_without_times = _TIMESTAMP_RE.sub(" ", answer_ids_removed)
    allowed_numbers = set(_NUMBER_RE.findall(context_without_times))
    answer_numbers = set(_NUMBER_RE.findall(answer_without_times))
    bad_numbers = sorted(answer_numbers - allowed_numbers)
    return bad_numbers, bad_timestamps


def guard_answer(answer: str, promos: list, context: str | None = None) -> str:
    """Replace answers containing promo IDs, numbers, or timestamps absent from RESULTS."""
    bad = ungrounded_ids(answer, known_ids(promos))
    if bad:
        return f"{NOT_IN_RESULTS} (The model mentioned {', '.join(sorted(set(bad)))}, which is not in these results, so I dropped its answer.)"
    results_context = context if context is not None else build_context(promos, {}, {})
    bad_numbers, bad_timestamps = _ungrounded_values(answer, results_context)
    unsupported = [*(f"number {n}" for n in bad_numbers), *(f"timestamp {t}" for t in bad_timestamps)]
    if unsupported:
        return f"{NOT_IN_RESULTS} (The model mentioned {', '.join(unsupported)}, which is not in the results, so I dropped its answer.)"
    return answer or NOT_IN_RESULTS


def ask(question: str, promos: list, decisions: dict, adjustments: dict, settings: dict | None = None,
        history: list | None = None, scrub=None, post=None) -> dict:
    """Answer one question. Returns {"answer", "source": "results"|"model"|"fallback"}."""
    direct = answer_deterministic(question, promos, decisions, scrub)
    if direct is not None:
        return {"answer": direct, "source": "results"}
    if settings and settings.get("provider") == "compatible" and settings.get("api_key"):
        context = build_context(promos, decisions, adjustments, scrub)
        raw = call_model(question, context, history or [], settings, post=post)
        return {"answer": guard_answer(raw, promos, context), "source": "model"}
    return {
        "answer": (f"{NOT_IN_RESULTS} Without a model I can answer: why one promo is ranked above another, "
                   "promos in an episode range, top N, approved/flagged/skipped lists, and a single promo's details. "
                   "Switch the Judgment model to an OpenAI-compatible API for open questions."),
        "source": "fallback",
    }
