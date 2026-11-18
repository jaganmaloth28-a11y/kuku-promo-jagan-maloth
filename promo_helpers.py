import re


def redact_transcript(transcript: str) -> str:
    """Hide dialogue while preserving timestamps, speakers, and non-dialogue cues."""
    redacted_lines = []
    for line in str(transcript or "").splitlines():
        if any(tag in line for tag in ("[ACTION]", "[CAMERA]", "[SOUND]")) or ': "' not in line:
            redacted_lines.append(line)
            continue
        prefix, dialogue = line.split(': "', 1)
        if dialogue.endswith('"'):
            redacted_lines.append(f'{prefix}: "[Dialogue redacted for confidentiality]"')
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def scrub_quotes_text(text: str) -> str:
    """Remove quoted script fragments from explanations in confidential mode."""
    value = str(text or "")
    return re.sub(r'“[^”]*”|"[^"\n]*"|‘[^’]*’|\'[^\'\n]*\'', "[quoted text hidden]", value)


def suggest_ending_cut(promo: dict, confidential_mode: bool = True) -> dict | None:
    """Suggest a valid existing line boundary before a resolved or weaker ending."""
    segments = promo.get("segments", [])
    if promo.get("candidate_type") != "single" or len(segments) != 1:
        return None
    segment = segments[0]
    blocks = sorted(
        [b for b in segment.get("blocks", []) if b.get("type") in {"DIALOGUE", "ACTION"}],
        key=lambda b: b.get("start_sec", 0),
    )
    if len(blocks) < 2:
        return None
    start_sec = float(segment.get("start_sec", 0))
    segment_end = float(segment.get("end_sec", 0))
    resolution = re.compile(
        r"\b(because|of course|long enough|i knew|he knew|she knew|the truth is|"
        r"that is why|that's why|i understand|understood|yes,? i|no,? i|"
        r"never going to|was never going to|it was|so you were)\b", re.IGNORECASE
    )
    tension = re.compile(r"\b(why|how|what|who|never|regret|promise|swear|don't|do not|touch her|marry)\b", re.IGNORECASE)
    for index, block in enumerate(blocks[:-1]):
        text = str(block.get("text", ""))
        if "?" not in text and not tension.search(text):
            continue
        for answer in blocks[index + 1:]:
            if not resolution.search(str(answer.get("text", ""))):
                continue
            cut_at = answer.get("start")
            cut_seconds = float(answer.get("start_sec", 0))
            duration = round(cut_seconds - start_sec, 2)
            if cut_at and start_sec < cut_seconds < segment_end and 20 <= duration <= 60:
                evidence = f"{answer.get('speaker')}: {answer.get('text', '')}" if answer.get("speaker") else f"Action: {answer.get('text', '')}"
                return {
                    "start": segment["start"], "end": cut_at, "duration": duration,
                    "kind": "resolved ending",
                    "reason": "A later line appears to answer the suspense question.",
                    "evidence": evidence if not confidential_mode else "[source line hidden]",
                }
            break
    if promo.get("evaluation", {}).get("cliffhanger_score", 10) <= 6:
        for index in range(len(blocks) - 2, 0, -1):
            text = str(blocks[index].get("text", ""))
            if "?" not in text and not re.search(r"\b(why|how|what|who|never|regret|promise|swear)\b", text, re.IGNORECASE):
                continue
            next_block = blocks[index + 1]
            cut_at = next_block.get("start")
            cut_seconds = float(next_block.get("start_sec", 0))
            duration = round(cut_seconds - start_sec, 2)
            if cut_at and start_sec < cut_seconds < segment_end and 20 <= duration <= 60:
                return {
                    "start": segment["start"], "end": cut_at, "duration": duration,
                    "kind": None,
                    "reason": "The strongest tension beat may arrive before the current ending.",
                }
    return None
