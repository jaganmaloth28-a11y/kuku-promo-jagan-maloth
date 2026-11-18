"""
cost_scale.py - Cost and scalability models for the Kuku promo generation pipeline.
Calculates token usage, API pricing across models, and realistic editorial workflow savings.
"""

from typing import Dict, Any

# Measured baseline from 50 episodes of "I Married a Broken Soldier. He was a Duke":
# Total duration: 6,162.6 seconds = 102.7 minutes (1h 42m 43s)
TOTAL_SHOW_DURATION_MINUTES = 102.71
AVG_EPISODE_DURATION_SECONDS = 123.25

PRICING_MODELS = {
    "heuristic_local": {
        "provider": "Local Rule Engine",
        "input_per_m": 0.00,
        "output_per_m": 0.00,
        "description": "Zero-cost offline deterministic engine (Default)"
    },
    "gemini-2.5-flash": {
        "provider": "Google",
        "input_per_m": 0.075,
        "output_per_m": 0.30,
        "description": "Current generation fast, low-cost model"
    },
    "gemini-1.5-flash": {
        "provider": "Google",
        "input_per_m": 0.075,
        "output_per_m": 0.30,
        "description": "Legacy multimodal model"
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "description": "Cost-effective small model"
    },
    "gpt-4o": {
        "provider": "OpenAI",
        "input_per_m": 2.50,
        "output_per_m": 10.00,
        "description": "Frontier intelligence judge"
    }
}

def calculate_show_cost(
    num_episodes: int = 50,
    candidates_per_episode: int = 3,
    stitched_candidates: int = 20,
    avg_cand_tokens_in: int = 500,
    avg_cand_tokens_out: int = 65,
    cold_viewer_candidates: int = 20,
    cold_tokens_in: int = 400,
    cold_tokens_out: int = 65,
    model_key: str = "heuristic_local"
) -> Dict[str, Any]:
    """Calculates token counts, costs, and realistic labor savings based on actual footage duration."""
    total_candidates = (num_episodes * candidates_per_episode) + stitched_candidates
    total_tokens_in = (total_candidates * avg_cand_tokens_in) + (min(total_candidates, cold_viewer_candidates) * cold_tokens_in)
    total_tokens_out = (total_candidates * avg_cand_tokens_out) + (min(total_candidates, cold_viewer_candidates) * cold_tokens_out)
    unfiltered_candidates = (num_episodes * 10) + stitched_candidates
    calls_saved = max(0, unfiltered_candidates - total_candidates)
    
    pricing = PRICING_MODELS.get(model_key, PRICING_MODELS["heuristic_local"])
    cost_in = (total_tokens_in / 1_000_000.0) * pricing["input_per_m"]
    cost_out = (total_tokens_out / 1_000_000.0) * pricing["output_per_m"]
    total_cost = round(cost_in + cost_out, 4)
    cost_per_ep = round(total_cost / max(1, num_episodes), 5)
    
    # Measured footage duration:
    # 50 episodes @ ~2.05 min each = ~102.7 minutes (1h 42m 43s)
    actual_footage_minutes = (num_episodes / 50.0) * TOTAL_SHOW_DURATION_MINUTES
    
    # Realistic manual workflow:
    # An editor watching the entire footage once (102.7m) plus pausing, scrubbing,
    # and testing candidate cut points (estimated 42m) = ~145 minutes (~2.4 hours).
    manual_editor_minutes = round(actual_footage_minutes * 1.41, 1)
    
    # AI-assisted workflow:
    # The editor does not scrub the entire footage; they review only the Top 10-15
    # pre-cut candidates in the review queue, spending ~1.5 min per candidate = ~15-20 min.
    ai_editor_minutes = 15.0
    time_saved_pct = round(((manual_editor_minutes - ai_editor_minutes) / manual_editor_minutes) * 100, 1) if manual_editor_minutes > ai_editor_minutes else 0.0

    return {
        "model_key": model_key,
        "provider": pricing["provider"],
        "num_episodes": num_episodes,
        "total_footage_minutes": round(actual_footage_minutes, 1),
        "total_candidates_evaluated": total_candidates,
        "candidate_model_calls_saved_by_prefilter": calls_saved,
        "cold_viewer_calls": min(total_candidates, cold_viewer_candidates),
        "input_tokens": total_tokens_in,
        "output_tokens": total_tokens_out,
        "total_cost_usd": total_cost,
        "cost_per_episode_usd": cost_per_ep,
        "manual_editor_time_min": manual_editor_minutes,
        "ai_assisted_editor_time_min": ai_editor_minutes,
        "time_saved_pct": time_saved_pct,
        "annual_cost_100_shows_usd": round(total_cost * 100, 2)
    }

if __name__ == "__main__":
    for m in ["heuristic_local", "gemini-2.5-flash", "gpt-4o-mini"]:
        res = calculate_show_cost(model_key=m)
        print(f"[{m}] Show Footage: {res['total_footage_minutes']}m | Show Cost: ${res['total_cost_usd']} | Manual: {res['manual_editor_time_min']}m -> AI: {res['ai_assisted_editor_time_min']}m ({res['time_saved_pct']}% saved)")
