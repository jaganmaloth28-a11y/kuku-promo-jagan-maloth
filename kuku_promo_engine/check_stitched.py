import json

with open("outputs/promos_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)

stitched = [p for p in data["promos"] if p["candidate_type"] == "stitched"]
print(f"Total stitched promos in dataset: {len(stitched)}")
print("First 3 stitched promos:")
for p in stitched[:3]:
    ev = p["evaluation"]
    segs = " + ".join(f"Ep {s['episode']}: {s['start']}-{s['end']} ({s['duration']}s)" for s in p["segments"])
    print(f"Rank {p.get('rank')}: {p['id']} | Score: {ev['overall_score']} | Dur: {p['total_duration']}s")
    print(f"  Segments: {segs}")
    print(f"  Hook: {ev['hook_description'][:70]}")
    print(f"  Cliffhanger: {ev['cliffhanger_description'][:70]}")
    print(f"  Why it works: {ev['why_it_works'][:70]}")
