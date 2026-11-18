"""
pipeline.py - Orchestrates end-to-end promo candidate extraction, evaluation, ranking, and export.
Runs on single episodes or an entire 50-episode show folder.
Outputs structured JSON and CSV files.
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
from .screenplay_parser import load_all_episodes, EpisodeScript, format_time
from .window_generator import generate_single_episode_windows, generate_stitched_windows, CandidateWindow
from .llm_evaluator import LLMEvaluator
from .quality_evaluator import evaluate_quality
from .config import Config

_HOOK_EVENTS = ("slap", "betray", "leave", "marry", "guards", "remove", "kill", "die", "shame", "liar", "stop", "hands off", "turn around", "changed my mind", "punch", "grab", "drag", "scream", "screams", "blood", "fever", "burning", "steal", "stole", "theft", "accuse", "accusation")

def _validate_candidate(cand, episodes):
    """Hard mechanical checks: duration, exact source ranges, order, opening hook."""
    if not Config.MIN_PROMO_DURATION <= cand.total_duration <= min(60.0, Config.MAX_PROMO_DURATION):
        return False
    known = {e.episode_num: e for e in episodes}
    previous = None
    for seg in cand.segments:
        ep = known.get(seg["episode"])
        if not ep or seg["start_sec"] < 0 or seg["end_sec"] > ep.total_duration or seg["end_sec"] <= seg["start_sec"]:
            return False
        key = (seg["episode"], seg["start_sec"])
        if previous is not None and key < previous:
            return False
        previous = (seg["episode"], seg["end_sec"])
        source_starts = {round(b.start_sec, 2) for b in ep.blocks}
        source_ends = {round(b.end_sec, 2) for b in ep.blocks}
        if round(seg["start_sec"], 2) not in source_starts or round(seg["end_sec"], 2) not in source_ends:
            return False
    first = cand.segments[0]
    blocks = first["blocks"]
    if not blocks or blocks[0]["type"] != "DIALOGUE":
        return False
    opening = [b for b in blocks if b["start_sec"] <= first["start_sec"] + 3.0]
    if getattr(cand, "relaxed_hook", False):
        return any(b["type"] == "DIALOGUE" for b in opening)
    return any(b["type"] == "DIALOGUE" and any(term in b["text"].lower() for term in _HOOK_EVENTS) for b in opening)

def _opening_dialogue(cand):
    for seg in cand.segments:
        for block in seg["blocks"]:
            if block["type"] == "DIALOGUE":
                return f'[{block["start"]}] {block["speaker"]}: "{block["text"]}"'
    return ""

def _timestamped_line_exists(quote, cand):
    # The quoted hook is required to be the code-selected opening dialogue,
    # including its exact source timestamp, not merely any line in the clip.
    return bool(quote and quote == _opening_dialogue(cand))

def _diverse_rank(promos):
    remaining = list(promos)
    selected = []
    used_episodes = {}
    used_terms = set()
    while remaining:
        def utility(p):
            ev = p["evaluation"]
            eps = {s["episode"] for s in p["segments"]}
            penalty = sum(7 for ep in eps if used_episodes.get(ep, 0))
            spoiler = str(ev.get("spoiler_level", "none")).lower()
            penalty += 28 if spoiler == "major" else 10 if spoiler == "mild" else 0
            words = {w.lower() for w in p.get("formatted_transcript", "").split() if len(w) > 5}
            penalty += 6 if len(words & used_terms) >= 8 else 0
            return float(ev.get("overall_score", 0)) - penalty
        chosen = max(remaining, key=utility)
        adjusted_score = round(utility(chosen), 2)
        fit_score = float(chosen["evaluation"].get("overall_score", 0))
        penalty = round(fit_score - adjusted_score, 2)
        chosen["diversity_adjusted_score"] = adjusted_score
        chosen["rank_reason"] = (
            f"Highest remaining adjusted score ({adjusted_score:g}): fit {fit_score:g}"
            f" minus {penalty:g} diversity/spoiler penalty."
            if penalty else f"Highest remaining fit score ({fit_score:g}); no diversity/spoiler penalty."
        )
        selected.append(chosen)
        remaining.remove(chosen)
        for seg in chosen["segments"]:
            used_episodes[seg["episode"]] = used_episodes.get(seg["episode"], 0) + 1
        used_terms.update(w.lower() for w in chosen.get("formatted_transcript", "").split() if len(w) > 5)
    for rank, promo in enumerate(selected, 1):
        promo["rank"] = rank
    return selected

class PromoPipeline:
    def __init__(self, input_dir: str, output_dir: str = "outputs"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.episodes: List[EpisodeScript] = []
        self.all_evaluated_promos: List[Dict[str, Any]] = []
        self.top_promos: List[Dict[str, Any]] = []
        self.quality_report: Optional[Dict[str, Any]] = None

    def run(self, max_episodes: Optional[int] = None, include_stitched: bool = True) -> Dict[str, Any]:
        print(f"[Pipeline] Loading screenplays from {self.input_dir}...")
        self.episodes = load_all_episodes(self.input_dir)
        if max_episodes:
            self.episodes = self.episodes[:max_episodes]
        print(f"[Pipeline] Loaded {len(self.episodes)} episodes.")
        if not self.episodes:
            return {
                "total_episodes": 0,
                "total_evaluated_promos": 0,
                "top_promos_count": 0,
                "quality_report": None,
                "message": "No supported, time-coded scripts were found. See parser messages for skipped files.",
            }

        all_candidates: List[CandidateWindow] = []

        # 1. Generate single-segment candidates per episode
        print("[Pipeline] Generating single-segment candidate windows (20-60s)...")
        coverage_debug_episodes = {27, 36, 47}
        for ep in self.episodes:
            # Generator ranks and deduplicates before returning; retain enough
            # valid alternatives without constructing thousands of transcripts.
            cands = generate_single_episode_windows(ep, min_duration=20.0, max_duration=60.0, max_candidates=10)
            strict_generated = len(cands)
            cands = [c for c in cands if _validate_candidate(c, self.episodes)]
            strict_valid = len(cands)
            relaxed_generated = 0
            if not cands:
                relaxed = generate_single_episode_windows(
                    ep, min_duration=20.0, max_duration=60.0, max_candidates=3, relaxed=True
                )
                relaxed_generated = len(relaxed)
                cands = [c for c in relaxed if _validate_candidate(c, self.episodes)]
            if ep.episode_num in coverage_debug_episodes:
                print(f"[Coverage] Episode {ep.episode_num}: strict generated={strict_generated}, strict valid={strict_valid}, relaxed generated={relaxed_generated}, valid before prefilter={len(cands)}")
            all_candidates.extend(cands)
        print(f"[Pipeline] Generated {len(all_candidates)} single-segment candidates.")

        # 2. Generate stitched cross-episode candidates
        if include_stitched and len(self.episodes) >= 2:
            print("[Pipeline] Generating stitched cross-episode candidate windows (20-60s)...")
            stitched = generate_stitched_windows(self.episodes, min_total_dur=20.0, max_total_dur=60.0, max_stitched=20)
            all_candidates.extend(stitched)
            print(f"[Pipeline] Generated {len(stitched)} stitched candidates.")

        if not all_candidates:
            return {
                "total_episodes": len(self.episodes),
                "total_evaluated_promos": 0,
                "top_promos_count": 0,
                "quality_report": None,
                "message": "Scripts loaded, but no 20–60 second windows could be formed from their timestamps.",
            }

        # Mechanical prefilter: at most a few candidates per episode reach the
        # model; all cut boundaries and timing remain code-selected.
        per_episode = {}
        stitched_candidates = []
        for cand in all_candidates:
            if cand.candidate_type == "stitched":
                stitched_candidates.append(cand)
            else:
                per_episode.setdefault(cand.segments[0]["episode"], []).append(cand)
        shortlist = []
        for episode_candidates in per_episode.values():
            shortlist.extend(sorted(episode_candidates, key=lambda c: c.code_prelim_rank, reverse=True)[:Config.PREFILTER_PER_EPISODE])
        shortlist.extend(sorted(stitched_candidates, key=lambda c: c.code_prelim_rank, reverse=True)[:20])
        all_candidates = [c for c in shortlist if _validate_candidate(c, self.episodes)]
        for episode_num in sorted(coverage_debug_episodes):
            generated = sum(1 for c in all_candidates if c.candidate_type != "stitched" and c.segments[0]["episode"] == episode_num)
            before = sum(1 for c in shortlist if c.candidate_type != "stitched" and c.segments[0]["episode"] == episode_num)
            if any(ep.episode_num == episode_num for ep in self.episodes):
                print(f"[Coverage] Episode {episode_num}: shortlisted={before}, valid after prefilter={generated}")
        print(f"[Pipeline] Code prefilter retained {len(all_candidates)} valid candidates.")

        if not all_candidates:
            self.all_evaluated_promos = []
            self.top_promos = []
            self.quality_report = evaluate_quality([])
            self.export_results()
            return {"total_episodes": len(self.episodes), "total_evaluated_promos": 0,
                    "top_promos_count": 0, "quality_report": self.quality_report,
                    "message": "Scripts parsed, but none passed the 20–60s dialogue-hook cut rules. Add more timed dialogue/action or adjust source timestamps."}

        # 3. Evaluate candidates with swappable model (judgment only)
        print(f"[Pipeline] Evaluating {len(all_candidates)} candidates using {LLMEvaluator.__name__}...")
        self.all_evaluated_promos = []
        for idx, cand in enumerate(all_candidates):
            eval_res = LLMEvaluator.evaluate(cand, cache_dir=self.output_dir)
            hook_quote = _opening_dialogue(cand)
            quoted = str(eval_res.get("hook_quote", ""))
            quote_ok = bool(quoted and quoted in cand.formatted_transcript and _timestamped_line_exists(quoted, cand))
            if not quote_ok:
                # Retry a malformed quote once with a fresh cache key; if it still
                # fails, discard model judgment and retain the deterministic score.
                retry_res = LLMEvaluator.evaluate(cand, retry=True, cache_dir=self.output_dir)
                retry_quote = str(retry_res.get("hook_quote", ""))
                if retry_quote and _timestamped_line_exists(retry_quote, cand):
                    eval_res = retry_res
                else:
                    from .llm_evaluator import HeuristicEvaluator
                    eval_res = HeuristicEvaluator.evaluate(cand)
                    eval_res["validation_retry"] = "model_output_rejected_after_retry"
                eval_res["hook_quote"] = hook_quote
            eval_res["hook_quote_verified"] = True
            promo_dict = cand.to_dict()
            promo_dict["id"] = f"PROMO-{idx+1:04d}"
            promo_dict["relaxed_hook"] = bool(getattr(cand, "relaxed_hook", False))
            promo_dict["evaluation"] = eval_res
            self.all_evaluated_promos.append(promo_dict)

        # Second model call sees only clip text, and only for the best 20.
        for promo in sorted(self.all_evaluated_promos, key=lambda p: p["evaluation"].get("overall_score", 0), reverse=True)[:20]:
            try:
                check = LLMEvaluator.cold_viewer_check(promo["formatted_transcript"], cache_dir=self.output_dir)
                promo["evaluation"]["cold_viewer"] = check
                if check.get("needs_earlier_context") is True:
                    promo["evaluation"]["stands_alone"] = False
                    promo["evaluation"]["standalone_score"] = max(0, int(promo["evaluation"].get("standalone_score", 5)) - 3)
                    promo["evaluation"]["overall_score"] = max(0, int(promo["evaluation"].get("overall_score", 50)) - 10)
            except Exception as exc:
                promo["evaluation"]["cold_viewer"] = {"status": "unavailable", "reason": str(exc)[:100]}

        # 4. Diversity-aware greedy rank penalizes repeated episode/topic and reveals.
        self.all_evaluated_promos = _diverse_rank(self.all_evaluated_promos)

        for rank, p in enumerate(self.all_evaluated_promos, start=1):
            p["rank"] = rank

        self.top_promos = self.all_evaluated_promos[:25]
        print(f"[Pipeline] Top promo selected: {self.top_promos[0]['id']} with score {self.top_promos[0]['evaluation']['overall_score']}")

        # 5. Exploratory comparison against assistant-curated reference cuts.
        print("[Pipeline] Comparing against assistant-curated reference cuts (exploratory)...")
        self.quality_report = evaluate_quality(self.all_evaluated_promos)

        # 6. Export outputs to JSON and CSV
        self.export_results()

        return {
            "total_episodes": len(self.episodes),
            "total_evaluated_promos": len(self.all_evaluated_promos),
            "top_promos_count": len(self.top_promos),
            "quality_report": self.quality_report
        }

    def export_results(self):
        json_path = os.path.join(self.output_dir, "promos_all.json")
        csv_path = os.path.join(self.output_dir, "promos_all.csv")
        top_csv_path = os.path.join(self.output_dir, "promos_top_ranked.csv")
        benchmark_path = os.path.join(self.output_dir, "quality_benchmark.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_promos": len(self.all_evaluated_promos),
                    "episodes_processed": len(self.episodes),
                },
                "promos": self.all_evaluated_promos,
                "quality_benchmark": self.quality_report
            }, f, indent=2)
        print(f"[Export] Saved JSON to {json_path}")

        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"[Export] Saved Quality Benchmark to {benchmark_path}")

        fieldnames = [
            "Rank", "Rank_Reason", "ID", "Candidate_Type", "Hook_Method", "Total_Duration_Sec",
            "Episodes", "Segment_Timestamps", "Hook_Score", "Hook_Description",
            "Cliffhanger_Score", "Cliffhanger_Description", "Stands_Alone",
            "Stands_Alone_Reason", "Overall_Score", "Why_It_Works", "Evaluator"
        ]

        def promo_to_row(p):
            seg_str = " + ".join(f"[Ep {s['episode']}: {s['start']}-{s['end']}]" for s in p["segments"])
            ep_str = ", ".join(str(s["episode"]) for s in p["segments"])
            return {
                "Rank": p.get("rank", ""),
                "Rank_Reason": p.get("rank_reason", ""),
                "ID": p["id"],
                "Candidate_Type": p["candidate_type"],
                "Hook_Method": "relaxed fallback" if p.get("relaxed_hook") else "keyword rule",
                "Total_Duration_Sec": p["total_duration"],
                "Episodes": ep_str,
                "Segment_Timestamps": seg_str,
                "Hook_Score": p["evaluation"]["hook_score"],
                "Hook_Description": p["evaluation"]["hook_description"],
                "Cliffhanger_Score": p["evaluation"]["cliffhanger_score"],
                "Cliffhanger_Description": p["evaluation"]["cliffhanger_description"],
                "Stands_Alone": "YES" if p["evaluation"]["stands_alone"] else "NO",
                "Stands_Alone_Reason": p["evaluation"]["stands_alone_reason"],
                "Overall_Score": p["evaluation"]["overall_score"],
                "Why_It_Works": p["evaluation"]["why_it_works"],
                "Evaluator": p["evaluation"].get("evaluator", "")
            }

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in self.all_evaluated_promos:
                writer.writerow(promo_to_row(p))
        print(f"[Export] Saved CSV to {csv_path}")

        with open(top_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in self.top_promos:
                writer.writerow(promo_to_row(p))
        print(f"[Export] Saved Top Promos CSV to {top_csv_path}")

if __name__ == "__main__":
    input_pack = r"kuku_scripts\I Married a Broken Soldier. He was a Duke"
    pipeline = PromoPipeline(input_pack, output_dir="outputs")
    res = pipeline.run()
    print("Pipeline run complete.")
