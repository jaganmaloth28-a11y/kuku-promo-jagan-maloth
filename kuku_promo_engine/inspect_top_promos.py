import json
import csv

with open("outputs/promos_top_ranked.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in top ranked CSV: {len(rows)}")
print("\nFirst 5 Rows in CSV:")
for r in rows[:5]:
    print(f"Rank {r['Rank']}: {r['ID']} | Type: {r['Candidate_Type']} | Dur: {r['Total_Duration_Sec']}s | Score: {r['Overall_Score']}")
    print(f"  Episodes: {r['Episodes']} | Segments: {r['Segment_Timestamps']}")
    print(f"  Hook: {r['Hook_Description']}")
    print(f"  Cliffhanger: {r['Cliffhanger_Description']}")
    print(f"  Why it works: {r['Why_It_Works']}")
    print("-" * 60)
