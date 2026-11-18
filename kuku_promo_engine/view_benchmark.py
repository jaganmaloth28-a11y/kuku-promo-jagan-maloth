import json

with open("outputs/quality_benchmark.json", "r") as f:
    bench = json.load(f)

print(f"Ground Truth count: {bench['ground_truth_count']}")
print(f"Matched count: {bench['matched_count']}")
print(f"Mean IoU: {bench['mean_iou']}")
print(f"Accuracy: {bench['accuracy_pct']}%\n")
print("Detailed matches:")
for m in bench["detailed_matches"]:
    print(f"  [{m['gt_id']}] Ep {m['episode']}: GT {m['gt_range']} vs Tool {m['tool_range']} | IoU: {m['iou']} | Matched: {m['matched']}")
