from kuku_promo_engine.screenplay_parser import EpisodeScript
from kuku_promo_engine.window_generator import generate_single_episode_windows

for ep_num in [1, 2, 3]:
    ep_path = f"kuku_scripts/I Married a Broken Soldier. He was a Duke/episode-{ep_num:03d}"
    import glob
    f = glob.glob(f"{ep_path}*.txt")[0]
    ep = EpisodeScript(f)
    # Generate without cap
    cands = generate_single_episode_windows(ep, max_candidates=30)
    print(f"\n=== Episode {ep_num} Candidates (Top 8 of {len(cands)}) ===")
    for i, c in enumerate(cands[:8]):
        s = c.segments[0]
        print(f"  #{i+1}: {s['start']} - {s['end']} ({s['duration']}s) | Rank: {c.code_prelim_rank} | Hook: {c.hook_text[:40]} | End: {c.ending_text[:40]}")
