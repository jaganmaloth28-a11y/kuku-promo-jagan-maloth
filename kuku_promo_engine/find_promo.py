import json

with open("outputs/promos_all.json", "r") as f:
    data = json.load(f)

for p in data["promos"]:
    for s in p["segments"]:
        if "00:11" in s["start"]:
            print(f"Promo {p['id']}: Ep {s['episode']}, start {s['start']} ({s['start_sec']}), end {s['end']} ({s['end_sec']}), dur {s['duration']}")
