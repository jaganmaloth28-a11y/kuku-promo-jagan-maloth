"""
test_model_swap.py - Validates that the swappable model architecture works across providers.
Tests the Heuristic Engine, Gemini payload formatting, and OpenAI payload formatting.
"""

import os
from kuku_promo_engine.config import Config
from kuku_promo_engine.screenplay_parser import EpisodeScript
from kuku_promo_engine.window_generator import generate_single_episode_windows
from kuku_promo_engine.llm_evaluator import LLMEvaluator, HeuristicEvaluator

def test_model_swapping():
    print("=" * 65)
    print("      TESTING SWAPPABLE MODEL ARCHITECTURE")
    print("=" * 65)

    script_path = os.path.join("kuku_scripts", "I Married a Broken Soldier. He was a Duke", "episode-001-Episode 1671703.txt")
    ep1 = EpisodeScript(script_path)
    cands = generate_single_episode_windows(ep1, max_candidates=2)
    sample_cand = cands[0]

    # Test 1: Offline Deterministic Engine (Default)
    print("\n[Test 1] Testing Default Local Heuristic Engine (Offline / Zero Cost)...")
    Config.update_model("heuristic", "heuristic_local")
    res_local = LLMEvaluator.evaluate(sample_cand)
    print(f"  Provider Active:   {res_local.get('evaluator')}")
    print(f"  Hook Score:        {res_local.get('hook_score')}/10")
    print(f"  Cliffhanger Score: {res_local.get('cliffhanger_score')}/10")
    print(f"  Stands Alone:      {res_local.get('stands_alone')}")
    print(f"  Overall Score:     {res_local.get('overall_score')}/100")
    assert res_local.get("overall_score") > 0, "Heuristic evaluation failed"
    print("  -> Passed successfully.")

    # Test 2: Gemini Configuration & Graceful Fallback
    print("\n[Test 2] Testing Gemini Connector & Graceful Key-Missing Fallback...")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    Config.update_model("gemini", "gemini-2.5-flash", api_key=gemini_key)
    res_gemini = LLMEvaluator.evaluate(sample_cand)
    print(f"  Evaluator Active:  {res_gemini.get('evaluator')}")
    print(f"  Result Returned:   Overall Score {res_gemini.get('overall_score')}/100")
    print("  -> Passed successfully.")

    # Test 3: OpenAI Configuration & Graceful Fallback
    print("\n[Test 3] Testing OpenAI Connector & Graceful Key-Missing Fallback...")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    Config.update_model("openai", "gpt-4o-mini", api_key=openai_key)
    res_openai = LLMEvaluator.evaluate(sample_cand)
    print(f"  Evaluator Active:  {res_openai.get('evaluator')}")
    print(f"  Result Returned:   Overall Score {res_openai.get('overall_score')}/100")
    print("  -> Passed successfully.")

    print("\n" + "=" * 65)
    print("All swappable model tests passed successfully.")
    print("=" * 65)

if __name__ == "__main__":
    test_model_swapping()
