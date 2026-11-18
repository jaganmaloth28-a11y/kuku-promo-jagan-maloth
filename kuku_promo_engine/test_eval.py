"""Quick test script to verify end-to-end evaluation"""
from kuku_promo_engine.screenplay_parser import EpisodeScript
from kuku_promo_engine.window_generator import generate_single_episode_windows
from kuku_promo_engine.llm_evaluator import LLMEvaluator

ep1 = EpisodeScript(r"kuku_scripts/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt")
cands = generate_single_episode_windows(ep1)
eval_res = LLMEvaluator.evaluate(cands[0])

print("Scored Candidate 0:")
print(f"  Duration: {cands[0].total_duration} seconds")
print(f"  Segments: {cands[0].segments[0]['start']} - {cands[0].segments[0]['end']}")
print(f"  Hook Score: {eval_res['hook_score']}")
print(f"  Cliffhanger Score: {eval_res['cliffhanger_score']}")
print(f"  Stands Alone: {eval_res['stands_alone']}")
print(f"  Reason: {eval_res['stands_alone_reason']}")
print(f"  Overall Score: {eval_res['overall_score']}")
print(f"  Why it works: {eval_res['why_it_works']}")
print(f"  Evaluator: {eval_res['evaluator']}")
