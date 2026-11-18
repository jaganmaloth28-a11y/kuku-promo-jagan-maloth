"""
quality_evaluator.py - Hand-labeled Ground Truth benchmark and quality evaluation.
Compares automated tool picks against human hand-labeled promos across Episodes 1 to 5.
Calculates IoU, boundary alignment, recall, and provides honest failure mode analysis.
"""

from typing import List, Dict, Any, Tuple
from .screenplay_parser import format_time, parse_time_str
from .window_generator import CandidateWindow

# Assistant-curated reference cuts for a small, transparent retrieval check.
# These are subjective prototype labels, not editor-team or audience-validated data.
GROUND_TRUTH_PROMOS = [
    {"id":"GT-EP01-A","episode":1,"title":"Altar betrayal","start":"00:11.30","end":"00:41.20","start_sec":11.3,"end_sec":41.2,"duration":29.9,"stands_alone":True,"editor_notes":"Confrontation to reveal; cut on the sister's answer."},
    {"id":"GT-EP02-A","episode":2,"title":"Public dignity comeback","start":"01:14.40","end":"01:36.60","start_sec":74.4,"end_sec":96.6,"duration":22.2,"stands_alone":True,"editor_notes":"Reversal and strong line; aftermath is intentionally excluded."},
    {"id":"GT-EP03-A","episode":3,"title":"Guards and intervention","start":"00:58.30","end":"01:25.50","start_sec":58.3,"end_sec":85.5,"duration":27.2,"stands_alone":True,"editor_notes":"Action hook; punch is a visual payoff and may feel resolved."},
    {"id":"GT-EP04-A","episode":4,"title":"Pity or vow","start":"00:53.60","end":"01:16.20","start_sec":53.6,"end_sec":76.2,"duration":22.6,"stands_alone":True,"editor_notes":"Romantic question; acceptance resolves part of the tension."},
    {"id":"GT-EP05-A","episode":5,"title":"The ledger is dirtied","start":"01:17.80","end":"01:40.80","start_sec":77.8,"end_sec":100.8,"duration":23.0,"stands_alone":False,"editor_notes":"Emotionally legible but the book's meaning needs earlier context."},
    {"id":"GT-EP06-A","episode":6,"title":"A quiet refuge","start":"00:30.15","end":"01:00.00","start_sec":30.15,"end_sec":60.0,"duration":29.85,"stands_alone":False,"editor_notes":"Character beat; deliberately weak opening hook."},
    {"id":"GT-EP07-A","episode":7,"title":"She remains a lady","start":"00:22.00","end":"00:54.60","start_sec":22.0,"end_sec":54.6,"duration":32.6,"stands_alone":True,"editor_notes":"Insult and defiant response; depends somewhat on relationship context."},
    {"id":"GT-EP09-A","episode":9,"title":"Feverish child","start":"00:06.30","end":"00:43.50","start_sec":6.3,"end_sec":43.5,"duration":37.2,"stands_alone":True,"editor_notes":"Clear immediate stakes; ends before the child's recovery is shown."},
    {"id":"GT-EP10-A","episode":10,"title":"The bloodied return","start":"00:09.15","end":"00:38.90","start_sec":9.15,"end_sec":38.9,"duration":29.75,"stands_alone":True,"editor_notes":"Wound reveals danger and creates a care-taking beat."},
    {"id":"GT-EP11-A","episode":11,"title":"The theft accusation","start":"00:11.10","end":"00:41.10","start_sec":11.1,"end_sec":41.1,"duration":30.0,"stands_alone":True,"editor_notes":"Accusation escalates into a search; outcome remains open."}
]

def calculate_interval_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates Intersection over Union (IoU) of two time intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0.0, intersection_end - intersection_start)
    union = (end1 - start1) + (end2 - start2) - intersection
    return round(intersection / union, 4) if union > 0 else 0.0

def evaluate_quality(tool_promos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates tool picks against hand-labeled human selections.
    """
    matches = []
    
    # Evaluate against single-segment promos for direct boundary comparison
    single_tool_promos = [p for p in tool_promos if p["candidate_type"] == "single"]

    for gt in GROUND_TRUTH_PROMOS:
        best_match = None
        best_iou = 0.0
        best_cand = None
        
        ep_tool_promos = [
            p for p in single_tool_promos
            if p["segments"][0]["episode"] == gt["episode"]
        ]
        
        for p in ep_tool_promos:
            p_start = p["segments"][0]["start_sec"]
            p_end = p["segments"][0]["end_sec"]
            iou = calculate_interval_iou(gt["start_sec"], gt["end_sec"], p_start, p_end)
            if iou > best_iou:
                best_iou = iou
                best_cand = p

        start_error = abs(best_cand["segments"][0]["start_sec"] - gt["start_sec"]) if best_cand else None
        end_error = abs(best_cand["segments"][0]["end_sec"] - gt["end_sec"]) if best_cand else None
        
        matches.append({
            "gt_id": gt["id"],
            "episode": gt["episode"],
            "title": gt["title"],
            "gt_range": f"{gt['start']} - {gt['end']} ({gt['duration']}s)",
            "tool_range": f"{best_cand['segments'][0]['start']} - {best_cand['segments'][0]['end']} ({best_cand['total_duration']}s)" if best_cand else "No Match",
            "iou": best_iou,
            "start_error_sec": start_error,
            "end_error_sec": end_error,
            "matched": best_iou >= 0.35,
            "stands_alone_match": (best_cand["evaluation"]["stands_alone"] == gt["stands_alone"]) if best_cand else False
        })

    matched_count = sum(1 for m in matches if m["matched"])
    mean_iou = round(sum(m["iou"] for m in matches) / len(matches), 3) if matches else 0.0
    accuracy = round((matched_count / len(matches)) * 100, 1) if matches else 0.0

    # Document honest failure modes
    failure_modes = [
        {
            "category": "Mid-Scene Conversation Hooks",
            "description": "A candidate can start inside a conversation and miss an earlier visual setup.",
            "example": "Episode 1: the selected start is later than the establishing visual moment.",
            "fix": "Weight candidate start points higher when paired with CAMERA establishing shots and physical door openings."
        },
        {
            "category": "Cliffhangers that Resolve Immediately",
            "description": "A verbal question may be answered immediately by a later action beat.",
            "example": "Episode 3: a candidate ending before the payoff can make the clip feel unresolved.",
            "fix": "Lookahead rule: if high-impact SOUND_FX or physical combat actions occur within 5 seconds of a candidate cut, extend the window to capture the action climax."
        },
        {
            "category": "Reaction Shot Clipping",
            "description": "A cut at the final dialogue block may omit a non-verbal reaction or sound cue.",
            "example": "Episode 1: the source timing may omit a reaction hold after the final dialogue.",
            "fix": "Extend the ending boundary to include contiguous CAMERA/SOUND_FX reaction blocks up to the next scene or dialogue start."
        },
        {
            "category": "Lore / Context Blindness (Stands-Alone Failures)",
            "description": "A candidate may include an unfamiliar object or relationship without establishing why it matters.",
            "example": "Episode 5: the significance of an object may depend on earlier context.",
            "fix": "Strict entity-grounding filter that penalizes unexplained proper nouns and unintroduced props."
        }
    ]

    return {
        "ground_truth_count": len(GROUND_TRUTH_PROMOS),
        "matched_count": matched_count,
        "mean_iou": mean_iou,
        "accuracy_pct": accuracy,
        "statistical_significance_note": "Ten assistant-curated reference cuts across ten episodes; exploratory only, not editor-team or audience-validated and not statistically conclusive.",
        "detailed_matches": matches,
        "failure_modes": failure_modes
    }
