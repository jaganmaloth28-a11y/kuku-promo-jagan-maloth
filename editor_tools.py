"""
editor_tools.py - logic for the Editor's Room upgrades. No Streamlit imports, so it can be unit-tested.

  1. Quality checking : IoU, "was the human pick in the tool's top 15?", hold-out labels,
                        side-by-side timeline, run-to-run disagreement.
  2. Editor adjustments: line-boundary options and live validation (20-60s, hook in 3s, ending).
  3. Cut-list export  : one row per segment with in/out timecodes for approved promos.

Promo shape assumed (same as app.py): {"id", "rank", "candidate_type", "total_duration",
"segments": [{"episode", "start", "end", "start_sec"?, "end_sec"?, "blocks": [...]}],
"evaluation": {"overall_score", ...}}. Timestamps look like "MM:SS.ss" or "HH:MM:SS.ss".
"""
from __future__ import annotations

import csv
import html
import io
import json
import re

import pandas as pd

_TS_FULL = re.compile(r"\d+(?::\d+)+(?:\.\d+)?")

CUT_LIST_COLUMNS = [
    "Promo_ID", "Rank", "Segment", "Episode", "In", "Out",
    "Duration_sec", "Cut_Source", "Decision",
]


# ---------------------------------------------------------------- time helpers
def ts_to_sec(ts):
    """'00:35.00' or '01:02:03.5' -> seconds. Numbers pass through. Anything else -> None."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    text = str(ts).strip()
    if not _TS_FULL.fullmatch(text):
        return None
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def sec_to_tc(sec, short: bool = False) -> str:
    """Seconds -> 'HH:MM:SS.mmm' (editor tools) or 'MM:SS.ss' (matches the app's display)."""
    ms = int(round(max(0.0, float(sec)) * 1000))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if short:
        return f"{hours * 60 + minutes:02d}:{seconds:02d}.{millis // 10:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_range(text):
    """First two timestamps found in a string such as '00:10.00 - 00:45.00' -> (start, end) or None."""
    found = [ts_to_sec(t) for t in _TS_FULL.findall(str(text or ""))]
    if len(found) < 2 or found[1] <= found[0]:
        return None
    return found[0], found[1]


def segment_range(segment: dict):
    start = segment.get("start_sec")
    end = segment.get("end_sec")
    start = float(start) if isinstance(start, (int, float)) else ts_to_sec(segment.get("start"))
    end = float(end) if isinstance(end, (int, float)) else ts_to_sec(segment.get("end"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def _same_episode(a, b) -> bool:
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def rank_sorted(promos: list) -> list:
    """Promos ordered by their rank field (missing rank keeps original order, after ranked ones)."""
    indexed = list(enumerate(promos))
    indexed.sort(key=lambda ip: (ip[1].get("rank") if isinstance(ip[1].get("rank"), (int, float)) else 10**9, ip[0]))
    return [p for _, p in indexed]


# ---------------------------------------------------------------- 1. quality checking
def _best_iou_in_episode(gt_range, episode, promo):
    best, best_range = 0.0, None
    for seg in promo.get("segments", []):
        if not _same_episode(seg.get("episode"), episode):
            continue
        rng = segment_range(seg)
        if rng is None:
            continue
        value = iou(gt_range, rng)
        if best_range is None or value > best:
            best, best_range = value, rng
    return best, best_range


def evaluate_labels(labels: list, ranked_promos: list, k: int = 15, thr: float = 0.35, scope: str = "show") -> list:
    """Compare human cuts with the tool's ranked promos.

    labels: [{"id", "episode", "start_sec", "end_sec", "title"?}]
    scope : "show"    -> is the human pick inside the tool's top-k across the whole show (what an editor sees)?
            "episode" -> inside the top-k promos of that episode?
    Returns one row per label with the top-pick IoU, best-of-top-k IoU and the hit flag.
    """
    rows = []
    numbered = list(enumerate(ranked_promos, start=1))
    for lab in labels:
        gt = (float(lab["start_sec"]), float(lab["end_sec"]))
        ep = lab["episode"]
        ep_pool = [(r, p) for r, p in numbered if any(_same_episode(s.get("episode"), ep) for s in p.get("segments", []))]
        pool = numbered[:k] if scope == "show" else ep_pool[:k]

        best_iou, best_id, best_rank, best_range = 0.0, None, None, None
        for rank, promo in pool:
            value, rng = _best_iou_in_episode(gt, ep, promo)
            if rng is not None and (best_range is None or value > best_iou):
                best_iou, best_id, best_rank, best_range = value, promo["id"], rank, rng

        top_pick = ep_pool[0][1] if ep_pool else None
        top_iou, top_range = (0.0, None)
        if top_pick is not None:
            top_iou, top_range = _best_iou_in_episode(gt, ep, top_pick)
        rows.append({
            "id": lab["id"], "episode": ep, "title": lab.get("title", ""),
            "human_range": gt,
            "hit_top_k": best_iou >= thr,
            "best_iou": round(best_iou, 3), "best_id": best_id, "best_rank": best_rank, "best_range": best_range,
            "top_pick_id": top_pick["id"] if top_pick else None,
            "top_pick_rank": ep_pool[0][0] if ep_pool else None,
            "top_pick_range": top_range,
            "top_pick_iou": round(top_iou, 3),
            "top_pick": top_pick,
            "candidates_in_episode": len(ep_pool),
        })
    return rows


def hit_summary(rows: list, thr: float = 0.35) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "hits": 0, "hit_pct": None, "mean_top_pick_iou": None, "top_pick_match_pct": None}
    hits = sum(1 for r in rows if r["hit_top_k"])
    mean_iou = sum(r["top_pick_iou"] for r in rows) / n
    top_match = sum(1 for r in rows if r["top_pick_iou"] >= thr)
    return {
        "n": n, "hits": hits, "hit_pct": round(100 * hits / n, 1),
        "mean_top_pick_iou": round(mean_iou, 3), "top_pick_match_pct": round(100 * top_match / n, 1),
    }


def parse_label_upload(raw: bytes, filename: str) -> list:
    """Hold-out labels from CSV (columns id, episode, start, end[, title]) or JSON (list of same keys)."""
    text = raw.decode("utf-8", errors="ignore")
    if filename.lower().endswith(".json"):
        data = json.loads(text)
        records = data.get("labels", []) if isinstance(data, dict) else data
    else:
        records = list(csv.DictReader(io.StringIO(text)))
    labels = []
    for n, rec in enumerate(records, start=1):
        rec = {str(k).strip().lower(): v for k, v in rec.items()}
        try:
            start = _to_sec(rec["start"])
            end = _to_sec(rec["end"])
            episode = rec["episode"]
        except KeyError as exc:
            raise ValueError(f"Missing column {exc}. Expected: id, episode, start, end.") from exc
        if start is None or end is None or end <= start:
            raise ValueError(f"Row {n}: start/end must be timestamps (MM:SS.ss) or seconds, with end after start.")
        labels.append({
            "id": str(rec.get("id") or f"H{n:02d}"), "episode": episode,
            "start_sec": start, "end_sec": end, "title": str(rec.get("title") or ""),
        })
    return labels


def _to_sec(value):
    if isinstance(value, (int, float)):
        return float(value)
    as_ts = ts_to_sec(value)
    if as_ts is not None:
        return as_ts
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def timeline_html(human, tool) -> str:
    """Tiny two-bar timeline: the editor's cut (green) over the tool's cut (orange)."""
    items = [r for r in (human, tool) if r]
    if not items:
        return ""
    lo = min(r[0] for r in items) - 5
    hi = max(r[1] for r in items) + 5
    span = max(hi - lo, 1e-6)

    def bar(rng, color, label):
        if not rng:
            return f"<div style='font:12px sans-serif;color:#888;margin:4px 0'>{html.escape(label)}: none</div>"
        left = 100 * (rng[0] - lo) / span
        width = max(1.0, 100 * (rng[1] - rng[0]) / span)
        return (
            f"<div style='font:12px sans-serif;margin:6px 0 2px'>{html.escape(label)} "
            f"{sec_to_tc(rng[0], True)} - {sec_to_tc(rng[1], True)}</div>"
            f"<div style='position:relative;height:12px;background:#e8e8e8;border-radius:6px'>"
            f"<div style='position:absolute;left:{left:.2f}%;width:{width:.2f}%;height:12px;"
            f"background:{color};border-radius:6px'></div></div>"
        )

    return bar(human, "#2e8b57", "Editor's cut") + bar(tool, "#e08a1e", "Tool's cut")


def _promo_key(promo: dict):
    key = []
    for seg in promo.get("segments", []):
        rng = segment_range(seg)
        key.append((str(seg.get("episode")), round(rng[0], 1) if rng else None, round(rng[1], 1) if rng else None))
    return tuple(key)


def _score(promo: dict):
    value = promo.get("evaluation", {}).get("overall_score")
    return float(value) if isinstance(value, (int, float)) else None


def compare_runs(run_a: list, run_b: list, k: int = 15, gap: float = 10.0) -> dict:
    """Flag promos in run A's top-k whose result changes in run B (missing, dropped from top-k, or score moved)."""
    ra, rb = rank_sorted(run_a), rank_sorted(run_b)
    top_b = {_promo_key(p) for p in rb[:k]}
    by_key_b = {_promo_key(p): p for p in rb}
    flags = {}
    for promo in ra[:k]:
        key = _promo_key(promo)
        other = by_key_b.get(key)
        reasons = []
        if other is None:
            reasons.append("not produced in the second run")
        else:
            if key not in top_b:
                reasons.append(f"dropped out of the top {k} in the second run")
            sa, sb = _score(promo), _score(other)
            if sa is not None and sb is not None and abs(sa - sb) >= gap:
                reasons.append(f"fit score {sa:g} vs {sb:g}")
        if reasons:
            flags[promo["id"]] = "; ".join(reasons)
    top_a = {_promo_key(p) for p in ra[:k]}
    return {
        "flags": flags,
        "compared": min(k, len(ra)),
        "top_overlap": len(top_a & top_b),
        "new_in_second_run": len(top_b - top_a),
    }


# ---------------------------------------------------------------- 2. editor adjustments
def line_boundaries(blocks: list, segment: dict):
    """Timestamps (seconds) where a cut may start or end: line starts, and line ends.

    The cut's current start/end are always included so the original cut can be reproduced.
    Returns (starts, ends); both empty when the result has no line-level timing.
    """
    seg_rng = segment_range(segment)
    timed = sorted(
        (b for b in blocks if isinstance(b.get("start_sec"), (int, float))),
        key=lambda b: b["start_sec"],
    )
    if not timed or seg_rng is None:
        return [], []
    starts = {round(float(b["start_sec"]), 3) for b in timed} | {round(seg_rng[0], 3)}
    ends = {round(seg_rng[1], 3)}
    for i, block in enumerate(timed):
        end = block.get("end_sec")
        if not isinstance(end, (int, float)):
            end = timed[i + 1]["start_sec"] if i + 1 < len(timed) else seg_rng[1]
        ends.add(round(float(end), 3))
    return sorted(starts), sorted(ends)


def nearest_index(options: list, value: float) -> int:
    return min(range(len(options)), key=lambda i: abs(options[i] - value)) if options else 0


def validate_cut(blocks, start, end, ending_suggestion=None, min_d=20.0, max_d=60.0, hook_limit=3.0, eps=0.05) -> list:
    """Live checks on an adjusted cut. Each result: {"name", "status": pass|warn|fail, "message"}."""
    checks = []
    duration = end - start
    if duration <= 0:
        return [{"name": "Duration", "status": "fail", "message": "The end must come after the start."}]
    if min_d <= duration <= max_d:
        checks.append({"name": "Duration", "status": "pass", "message": f"{duration:.1f}s is inside {min_d:g}-{max_d:g}s."})
    else:
        checks.append({"name": "Duration", "status": "fail", "message": f"{duration:.1f}s is outside {min_d:g}-{max_d:g}s."})

    inside = sorted(
        (b for b in blocks if isinstance(b.get("start_sec"), (int, float)) and start - eps <= b["start_sec"] < end),
        key=lambda b: b["start_sec"],
    )
    if not inside:
        checks.append({"name": "Hook", "status": "fail", "message": "No dialogue or action beat falls inside this cut."})
    else:
        lead = inside[0]["start_sec"] - start
        if lead <= hook_limit:
            checks.append({"name": "Hook", "status": "pass", "message": f"First beat lands {max(lead, 0):.1f}s in (limit {hook_limit:g}s)."})
        else:
            checks.append({"name": "Hook", "status": "fail", "message": f"First beat lands {lead:.1f}s in; the brief asks for {hook_limit:g}s or less."})

    sug_end = ts_to_sec(ending_suggestion.get("end")) if ending_suggestion else None
    kind = (ending_suggestion or {}).get("kind")
    if sug_end is None:
        checks.append({"name": "Ending", "status": "pass",
                       "message": "The rule check found no later line that resolves the tension (not a guarantee; read the last beat)."})
    elif kind == "resolved ending" and end > sug_end + eps:
        checks.append({"name": "Ending", "status": "fail",
                       "message": f"This cut runs past the line that appears to resolve the tension. Suggested end: {sec_to_tc(sug_end, True)}."})
    elif kind != "resolved ending" and end > sug_end + eps:
        checks.append({"name": "Ending", "status": "warn",
                       "message": f"The strongest tension beat may come earlier, at {sec_to_tc(sug_end, True)}."})
    else:
        checks.append({"name": "Ending", "status": "pass", "message": "Ends before the point the rule check flags as resolving."})
    return checks


# ---------------------------------------------------------------- 3. cut-list export
def build_cut_list(promos: list, decisions: dict, adjustments: dict, statuses=("APPROVED",)) -> pd.DataFrame:
    """One row per segment, in/out as HH:MM:SS.mmm, using the editor's adjusted cut where one was saved."""
    rows = []
    for promo in rank_sorted(promos):
        status = decisions.get(promo["id"], {}).get("status", "PENDING")
        if status not in statuses:
            continue
        segments = promo.get("segments", [])
        adj = adjustments.get(promo["id"]) if len(segments) == 1 else None
        for number, seg in enumerate(segments, start=1):
            rng, source = segment_range(seg), "Tool"
            if adj:
                rng, source = (float(adj["start_sec"]), float(adj["end_sec"])), "Editor-adjusted"
            if rng is None:
                continue
            rows.append({
                "Promo_ID": promo["id"], "Rank": promo.get("rank"), "Segment": number,
                "Episode": seg.get("episode"), "In": sec_to_tc(rng[0]), "Out": sec_to_tc(rng[1]),
                "Duration_sec": round(rng[1] - rng[0], 2), "Cut_Source": source, "Decision": status,
            })
    return pd.DataFrame(rows, columns=CUT_LIST_COLUMNS)
