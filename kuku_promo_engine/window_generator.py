"""
window_generator.py - Enhanced candidate window generator.
Enforces 20-60s durations, block boundaries, and balances dialogue density with high-impact action beats.
"""

from typing import List, Dict, Any, Tuple
from .screenplay_parser import EpisodeScript, ScriptBlock, format_time

# Dramatic keywords for hook and cliffhanger detection in code
HOOK_KEYWORDS = {
    "slap", "shame", "betray", "leave", "altar", "marry", "wedding", "husband", "wife",
    "liar", "bastard", "dare", "guards", "remove", "kill", "die", "disgrace", "worth",
    "turn around", "not anymore", "changed my mind", "stop", "never", "hands off",
    "dignity", "cannot marry", "better choice", "plainness", "tolerate", "blood", "fever", "burning", "steal", "stole", "theft", "accuse", "accusation"
}

CLIFFHANGER_KEYWORDS = {
    "why", "how", "what", "who", "never", "promise", "regret", "curse", "swear",
    "ashamed", "daughter", "disown", "marry", "ring", "choice", "dignity",
    "cliffhanger", "cut to black", "silence", "drone", "ominous", "swell", "alone",
    "touch her again", "take back", "jewel", "value"
}

HIGH_EMOTION_TAGS = {
    "cold", "commanding", "shouting", "screeching", "cruel", "callous", "defiant",
    "furious", "desperate", "shaken", "mocking", "imperious", "harsh", "broken",
    "heartbroken", "gloating", "venomous", "resolute", "proud"
}

ACTION_IMPACT_KEYWORDS = {
    "punch", "strike", "knock", "drag", "grab", "kneel", "knee", "ring", "slap", "doors", "cane"
}

class CandidateWindow:
    def __init__(self, segments: List[Dict[str, Any]], candidate_type: str = "single"):
        self.candidate_type = candidate_type
        self.segments = segments
        self.total_duration = round(sum(s["duration"] for s in segments), 2)
        
        self.formatted_transcript = self._build_transcript()
        self.hook_text = self._extract_hook_text()
        self.ending_text = self._extract_ending_text()
        
        self.code_hook_score = 0.0
        self.code_cliffhanger_score = 0.0
        self.code_tension_score = 0.0
        self.code_prelim_rank = 0.0
        self._calculate_code_scores()

    def _build_transcript(self) -> str:
        lines = []
        for i, seg in enumerate(self.segments):
            if len(self.segments) > 1:
                lines.append(f"--- SEGMENT {i+1} [Ep {seg['episode']} | {seg['start']} - {seg['end']} ({seg['duration']}s)] ---")
            else:
                lines.append(f"--- [Ep {seg['episode']} | {seg['start']} - {seg['end']} ({seg['duration']}s)] ---")
            for b in seg["blocks"]:
                if b["type"] == "DIALOGUE":
                    speaker = b["speaker"]
                    emotion = f" ({b['emotion']})" if b.get("emotion") else ""
                    lines.append(f"[{b['start']}] {speaker}{emotion}: \"{b['text']}\"")
                elif b["type"] == "ACTION":
                    lines.append(f"[{b['start']}] [ACTION] {b['text']}")
                elif b["type"] == "CAMERA":
                    lines.append(f"[{b['start']}] [CAMERA] {b['text']}")
                elif b["type"] == "SOUND_FX":
                    lines.append(f"[{b['start']}] [SOUND] {b['text']}")
        return "\n".join(lines)

    def _extract_hook_text(self) -> str:
        first_seg = self.segments[0]
        hook_blocks = []
        seg_start = first_seg["start_sec"]
        for b in first_seg["blocks"]:
            if b["start_sec"] <= seg_start + 3.0:
                if b["type"] == "DIALOGUE":
                    hook_blocks.append(f"{b['speaker']}: \"{b['text']}\"")
                else:
                    hook_blocks.append(f"[{b['type']}] {b['text']}")
        return " | ".join(hook_blocks[:3]) if hook_blocks else "Visual opening"

    def _extract_ending_text(self) -> str:
        last_seg = self.segments[-1]
        end_blocks = []
        seg_end = last_seg["end_sec"]
        for b in reversed(last_seg["blocks"]):
            if b["end_sec"] >= seg_end - 5.0:
                if b["type"] == "DIALOGUE":
                    end_blocks.insert(0, f"{b['speaker']}: \"{b['text']}\"")
                else:
                    end_blocks.insert(0, f"[{b['type']}] {b['text']}")
        return " | ".join(end_blocks[-3:]) if end_blocks else "Final beat"

    def _calculate_code_scores(self):
        first_seg = self.segments[0]
        seg_start = first_seg["start_sec"]
        hook_pts = 4.5
        
        early_blocks = [b for b in first_seg["blocks"] if b["start_sec"] <= seg_start + 3.0]
        for b in early_blocks:
            t = b["text"].lower()
            if b["type"] == "DIALOGUE":
                hook_pts += 2.0
                if any(k in t for k in HOOK_KEYWORDS):
                    hook_pts += 2.5
                if b.get("emotion") and any(e in b["emotion"].lower() for e in HIGH_EMOTION_TAGS):
                    hook_pts += 1.5
            elif b["type"] in ("ACTION", "SOUND_FX"):
                if any(k in t for k in ACTION_IMPACT_KEYWORDS):
                    hook_pts += 2.5
        self.code_hook_score = min(10.0, round(hook_pts, 2))

        last_seg = self.segments[-1]
        seg_end = last_seg["end_sec"]
        cliff_pts = 4.5
        late_blocks = [b for b in last_seg["blocks"] if b["end_sec"] >= seg_end - 5.0]
        for b in late_blocks:
            t = b["text"].lower()
            if b["type"] == "DIALOGUE":
                if "?" in t:
                    cliff_pts += 2.0
                if any(k in t for k in CLIFFHANGER_KEYWORDS):
                    cliff_pts += 2.2
                if b.get("emotion") and any(e in b["emotion"].lower() for e in HIGH_EMOTION_TAGS):
                    cliff_pts += 1.2
            elif b["type"] in ("SOUND_FX", "CAMERA", "ACTION"):
                if any(k in t for k in ["cliffhanger", "cut to black", "swell", "drone", "ring", "defiance", "kneel"]):
                    cliff_pts += 2.8
        self.code_cliffhanger_score = min(10.0, round(cliff_pts, 2))

        all_blocks = [b for seg in self.segments for b in seg["blocks"]]
        dialogue_blocks = [b for b in all_blocks if b["type"] == "DIALOGUE"]
        action_blocks = [b for b in all_blocks if b["type"] in ("ACTION", "SOUND_FX")]
        
        d_ratio = len(dialogue_blocks) / max(1, len(all_blocks))
        a_ratio = len(action_blocks) / max(1, len(all_blocks))
        tension_pts = 4.0 + (d_ratio * 3.5) + (a_ratio * 2.5)
        if len(self.segments) > 1:
            tension_pts += 1.0
        self.code_tension_score = min(10.0, round(tension_pts, 2))

        self.code_prelim_rank = round((self.code_hook_score * 0.4) + (self.code_cliffhanger_score * 0.4) + (self.code_tension_score * 0.2), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": self.candidate_type,
            "total_duration": self.total_duration,
            "segments": [
                {
                    "episode": s["episode"],
                    "start": s["start"],
                    "end": s["end"],
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "duration": s["duration"],
                    "blocks": s["blocks"]
                } for s in self.segments
            ],
            "code_hook_score": self.code_hook_score,
            "code_cliffhanger_score": self.code_cliffhanger_score,
            "code_tension_score": self.code_tension_score,
            "code_prelim_rank": self.code_prelim_rank,
            "hook_text": self.hook_text,
            "ending_text": self.ending_text,
            "formatted_transcript": self.formatted_transcript
        }

def generate_single_episode_windows(
    episode: EpisodeScript,
    min_duration: float = 20.0,
    max_duration: float = 60.0,
    max_candidates: int = 12,
    relaxed: bool = False,
) -> List[CandidateWindow]:
    """
    Extracts all strictly valid single-segment candidate windows within [min_duration, max_duration].
    Start and end points are strictly constrained to script block boundaries.
    """
    blocks = episode.blocks
    n = len(blocks)
    candidates = []

    # Valid start indices
    # Preserve exact line boundaries and anchor every cut on spoken dialogue.
    valid_start_indices = [i for i, b in enumerate(blocks) if b.block_type == "DIALOGUE"]

    for i in valid_start_indices:
        start_sec = blocks[i].start_sec
        for j in range(i, n):
            end_sec = blocks[j].end_sec
            end_idx = j
            if (j + 1 < n and blocks[j + 1].block_type == "CAMERA"
                    and blocks[j + 1].end_sec - start_sec <= max_duration):
                end_idx = j + 1
                end_sec = blocks[end_idx].end_sec
            dur = end_sec - start_sec
            if dur > max_duration:
                break
            first = blocks[i]
            has_early_hook = (first.start_sec - start_sec <= 3.0 and
                              any(k in first.text.lower() for k in HOOK_KEYWORDS))
            if relaxed and not has_early_hook:
                emotion = (first.emotion or "").lower()
                has_early_hook = ("?" in first.text or "!" in first.text
                                  or any(tag in emotion for tag in HIGH_EMOTION_TAGS))
            if dur >= min_duration and has_early_hook:
                seg = {
                    "episode": episode.episode_num,
                    "start": format_time(start_sec),
                    "end": format_time(end_sec),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration": round(dur, 2),
                    "blocks": [b.to_dict() for b in blocks[i:end_idx+1]]
                }
                cand = CandidateWindow(segments=[seg], candidate_type="single")
                cand.relaxed_hook = relaxed and not any(k in first.text.lower() for k in HOOK_KEYWORDS)
                candidates.append(cand)

    candidates.sort(key=lambda c: c.code_prelim_rank, reverse=True)
    
    # Non-maximum suppression with 6s tolerance
    selected: List[CandidateWindow] = []
    for cand in candidates:
        cand_start = cand.segments[0]["start_sec"]
        cand_end = cand.segments[0]["end_sec"]
        is_duplicate = False
        for sel in selected:
            sel_start = sel.segments[0]["start_sec"]
            sel_end = sel.segments[0]["end_sec"]
            if abs(cand_start - sel_start) < 6.0 and abs(cand_end - sel_end) < 6.0:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(cand)
            if len(selected) >= max_candidates:
                break

    return selected

def generate_stitched_windows(
    episodes: List[EpisodeScript],
    min_total_dur: float = 20.0,
    max_total_dur: float = 60.0,
    max_stitched: int = 15
) -> List[CandidateWindow]:
    """
    Builds multi-segment stitched promos joining moments across scenes or episodes.
    Enforces total duration between 20.0 and 60.0 seconds.
    """
    stitched_candidates = []
    setups = []
    for ep in episodes[:10]:
        for i, b in enumerate(ep.blocks):
            if b.block_type == "DIALOGUE" and any(k in b.text.lower() for k in HOOK_KEYWORDS):
                start_sec = b.start_sec
                for j in range(i, min(i + 15, len(ep.blocks))):
                    end_sec = ep.blocks[j].end_sec
                    dur = end_sec - start_sec
                    if 8.0 <= dur <= 25.0:
                        setups.append({
                            "episode": ep.episode_num,
                            "start": format_time(start_sec),
                            "end": format_time(end_sec),
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration": round(dur, 2),
                            "blocks": [blk.to_dict() for blk in ep.blocks[i:j+1]]
                        })

    payoffs = []
    for ep in episodes[:10]:
        for i, b in enumerate(ep.blocks):
            if any(k in b.text.lower() for k in CLIFFHANGER_KEYWORDS) or (b.block_type in ("ACTION", "SOUND_FX") and any(k in b.text.lower() for k in ACTION_IMPACT_KEYWORDS)):
                start_sec = b.start_sec
                for j in range(i, min(i + 20, len(ep.blocks))):
                    end_sec = ep.blocks[j].end_sec
                    dur = end_sec - start_sec
                    if 12.0 <= dur <= 35.0:
                        payoffs.append({
                            "episode": ep.episode_num,
                            "start": format_time(start_sec),
                            "end": format_time(end_sec),
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration": round(dur, 2),
                            "blocks": [blk.to_dict() for blk in ep.blocks[i:j+1]]
                        })

    # Bound pair-combination work for long shows. These are machine-ranked
    # segment snippets; only the strongest 80 per side enter pair generation.
    def segment_strength(segment, keywords):
        text = " ".join(b["text"].lower() for b in segment["blocks"])
        return sum(1 for word in keywords if word in text)
    setups.sort(key=lambda seg: segment_strength(seg, HOOK_KEYWORDS), reverse=True)
    payoffs.sort(key=lambda seg: segment_strength(seg, CLIFFHANGER_KEYWORDS | ACTION_IMPACT_KEYWORDS), reverse=True)
    setups = setups[:80]
    payoffs = payoffs[:80]

    for s in setups:
        for p in payoffs:
            if s["episode"] == p["episode"] and abs(s["start_sec"] - p["start_sec"]) < 20.0:
                continue
            # Cross-episode stitches require explicit continuity notes on the
            # payoff episode; same-episode joins remain available without them.
            if s["episode"] != p["episode"]:
                payoff_ep = next((e for e in episodes if e.episode_num == p["episode"]), None)
                if not payoff_ep or not payoff_ep.continuity.strip() or s["episode"] >= p["episode"]:
                    continue
            total_dur = s["duration"] + p["duration"]
            if min_total_dur <= total_dur <= max_total_dur:
                cand = CandidateWindow(segments=[s, p], candidate_type="stitched")
                stitched_candidates.append(cand)

    # Extend cross-episode stitching through the full show. A later episode must
    # carry CONTINUITY NOTES; code selects a short earlier setup plus current beat.
    ordered = sorted(episodes, key=lambda e: e.episode_num)
    for previous, current in zip(ordered, ordered[1:]):
        if not current.continuity.strip():
            continue
        local_setups, local_payoffs = [], []
        for i, block in enumerate(previous.blocks):
            if block.block_type != "DIALOGUE" or not any(k in block.text.lower() for k in HOOK_KEYWORDS):
                continue
            for j in range(i, min(i + 15, len(previous.blocks))):
                duration = previous.blocks[j].end_sec - block.start_sec
                if 8.0 <= duration <= 25.0:
                    local_setups.append({"episode": previous.episode_num,"start":format_time(block.start_sec),"end":format_time(previous.blocks[j].end_sec),"start_sec":block.start_sec,"end_sec":previous.blocks[j].end_sec,"duration":round(duration,2),"blocks":[x.to_dict() for x in previous.blocks[i:j+1]]})
                    break
        for i, block in enumerate(current.blocks):
            if not (any(k in block.text.lower() for k in CLIFFHANGER_KEYWORDS) or (block.block_type in ("ACTION","SOUND_FX") and any(k in block.text.lower() for k in ACTION_IMPACT_KEYWORDS))):
                continue
            for j in range(i, min(i + 20, len(current.blocks))):
                duration = current.blocks[j].end_sec - block.start_sec
                if 12.0 <= duration <= 35.0:
                    local_payoffs.append({"episode":current.episode_num,"start":format_time(block.start_sec),"end":format_time(current.blocks[j].end_sec),"start_sec":block.start_sec,"end_sec":current.blocks[j].end_sec,"duration":round(duration,2),"blocks":[x.to_dict() for x in current.blocks[i:j+1]]})
                    break
        for setup in local_setups[:8]:
            for payoff in local_payoffs[:8]:
                duration = setup["duration"] + payoff["duration"]
                if min_total_dur <= duration <= max_total_dur:
                    stitched_candidates.append(CandidateWindow([setup, payoff], candidate_type="stitched"))
        if len(stitched_candidates) >= max_stitched * 8:
            break

    stitched_candidates.sort(key=lambda c: c.code_prelim_rank, reverse=True)
    
    unique_stitched = []
    seen_pairs = set()
    for c in stitched_candidates:
        pair_key = (c.segments[0]["episode"], c.segments[0]["start"], c.segments[1]["episode"], c.segments[1]["start"])
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_stitched.append(c)
            if len(unique_stitched) >= max_stitched:
                break

    return unique_stitched
