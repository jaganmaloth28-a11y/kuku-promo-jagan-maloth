import json

with open("outputs/promos_all.json", "r") as f:
    data = json.load(f)

print("First 5 promos:")
for p in data["promos"][:5]:
    seg = p["segments"][0]
    print(f"ID: {p['id']}, Ep: {seg['episode']}, Start: {seg['start']}, End: {seg['end']}, Dur: {p['total_duration']}, Score: {p['evaluation']['overall_score']}")
