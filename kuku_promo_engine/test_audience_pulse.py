"""
test_audience_pulse.py - Verifies the precision promo editor prompt and evaluation output.
"""

import os
from kuku_promo_engine.screenplay_parser import EpisodeScript
from kuku_promo_engine.window_generator import generate_single_episode_windows
from kuku_promo_engine.audience_pulse import AudiencePulseEvaluator, DEFAULT_VIEWER_COMMENTS, DEFAULT_PATTERNS_THAT_WORK

def test_pulse():
    print("=" * 65)
    print("      TESTING AUDIENCE PULSE & PRECISION CUT EVALUATOR")
    print("=" * 65)

    script_path = os.path.join("kuku_scripts", "I Married a Broken Soldier. He was a Duke", "episode-001-Episode 1671703.txt")
    ep1 = EpisodeScript(script_path)
    cands = generate_single_episode_windows(ep1, max_candidates=3)
    for i, c in enumerate(cands):
        c.id = f"EP01-CAND-{i+1:02d}"

    evals = AudiencePulseEvaluator.evaluate_candidates(
        episode_num=1,
        series_context=ep1.logline,
        candidates=cands
    )

    print(f"Evaluated {len(evals)} candidates for Episode 1:")
    for ev in evals:
        print(f"\nCandidate: {ev['id']}")
        print(f"  Hook Score:       {ev['hook_score']}/10")
        print(f"  Cliff Score:      {ev['cliff_score']}/10")
        print(f"  Standalone Score: {ev['standalone_score']}/10")
        print(f"  Weak Spot:        {ev['weak_spot']}")
        print(f"  Better Cut:       {ev['better_cut']}")
        print(f"  Spoiler Level:    {ev['spoiler_level']}")
        print(f"  Pulse Match:      {ev['pulse_match']}")
        print(f"  Why (<= 15 words): \"{ev['why']}\" (Words: {len(ev['why'].split())})")

    print("\n" + "=" * 65)
    print("Audience Pulse Evaluator test completed successfully.")
    print("=" * 65)

if __name__ == "__main__":
    test_pulse()
