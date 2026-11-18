"""
screenplay_parser.py - Robust deterministic parser for Kuku microdrama screenplays.
Parses episodes, scenes, dialogue, actions, camera directions, and exact float timestamps.
"""

import os
import re
import html
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional

SUPPORTED_SCRIPT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".srt", ".vtt", ".docx", ".pdf",
    ".rtf", ".html", ".htm", ".csv"
}


class _HTMLTextExtractor(HTMLParser):
    """Small standard-library HTML to text converter."""
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        value = data.strip()
        if value:
            self.parts.append(value)


def _read_script_text(file_path: str) -> str:
    """Extract text from supported document formats without changing timestamps."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        with zipfile.ZipFile(file_path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", ns):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
            if text.strip():
                paragraphs.append(text)
        return "\n".join(paragraphs)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support needs pypdf. Install it with: pip install pypdf") from exc
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as source:
        content = source.read()
    if ext in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(content)
        return "\n".join(parser.parts)
    if ext == ".rtf":
        content = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("cp1252"), content)
        content = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), content)
        content = re.sub(r"\\par[d]?\b", "\n", content)
        content = re.sub(r"\\'[0-9a-fA-F]{2}", "", content)
        content = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", content)
        return content.replace("{", "").replace("}", "")
    return content


def _subtitle_time(value: str) -> float:
    value = value.strip().replace(",", ".")
    if value.count(":") == 1:
        return parse_time_str(value)
    return parse_time_str(value)

def parse_time_str(ts: str) -> float:
    """Converts MM:SS.SS or M:SS.SS to total float seconds."""
    ts = ts.strip("() ")
    parts = ts.split(":")
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return round(minutes * 60.0 + seconds, 2)
    elif len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return round(hours * 3600.0 + minutes * 60.0 + seconds, 2)
    return round(float(ts), 2)

def format_time(seconds: float) -> str:
    """Formats float seconds into MM:SS.SS format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"

class ScriptBlock:
    def __init__(self, tag: str, start_sec: float, end_sec: float, text: str, scene_idx: int = 1):
        self.raw_tag = tag.strip()
        self.start_sec = round(start_sec, 2)
        self.end_sec = round(end_sec, 2)
        self.duration = round(max(0.0, self.end_sec - self.start_sec), 2)
        # A malformed block match can absorb the following scene heading.
        text = re.split(r"\n\s*SCENE\s+\d+\s*\n\s*Scene heading:", text, maxsplit=1, flags=re.IGNORECASE)[0]
        self.scene_idx = scene_idx
        
        # Categorize
        upper_tag = self.raw_tag.upper()
        if "CAMERA" in upper_tag:
            self.block_type = "CAMERA"
            self.speaker = None
            self.emotion = None
        elif "ACTION" in upper_tag:
            self.block_type = "ACTION"
            self.speaker = None
            self.emotion = None
        elif "SOUND" in upper_tag or "MUSIC" in upper_tag or "SFX" in upper_tag:
            self.block_type = "SOUND_FX"
            self.speaker = None
            self.emotion = None
        else:
            self.block_type = "DIALOGUE"
            # Extract speaker & emotion: e.g. "LADY JULIANA (Cold, Resolute):"
            speaker_match = re.match(r"^([^(:*]+)(?:\(([^)]+)\))?:?", self.raw_tag)
            if speaker_match:
                self.speaker = speaker_match.group(1).strip()
                self.emotion = speaker_match.group(2).strip() if speaker_match.group(2) else None
            else:
                self.speaker = self.raw_tag.rstrip(":")
                self.emotion = None

        if self.block_type == "DIALOGUE":
            text = text.strip().strip('"“”')
        self.text = text.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.block_type,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "start": format_time(self.start_sec),
            "end": format_time(self.end_sec),
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration": self.duration,
            "text": self.text
        }

    def __repr__(self):
        return f"[{format_time(self.start_sec)}-{format_time(self.end_sec)}] {self.raw_tag}: {self.text[:40]}"

class EpisodeScript:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.episode_num = self._extract_ep_number(self.filename)
        self.logline = ""
        self.continuity = ""
        self.scenes: List[Dict[str, Any]] = []
        self.blocks: List[ScriptBlock] = []
        self._parse()

    def _extract_ep_number(self, filename: str) -> int:
        match = re.search(r'episode[_-]?0*(\d+)', filename, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _parse(self):
        content = _read_script_text(self.file_path)
        extension = os.path.splitext(self.file_path)[1].lower()

        if extension in {".srt", ".vtt"}:
            self._parse_subtitles(content)
            self.blocks.sort(key=lambda b: (b.start_sec, b.end_sec))
            if not self.blocks:
                raise ValueError("No timed subtitle cues found. Expected SRT/VTT timestamps.")
            return

        # Parse Header
        num_m = re.search(r'EPISODE NUMBER:\s*(.*)', content)
        if num_m:
            ep_str = num_m.group(1).strip()
            digit_m = re.search(r'\d+', ep_str)
            if digit_m:
                self.episode_num = int(digit_m.group(0))

        log_m = re.search(r'EPISODE LOGLINE:\s*(.*)', content)
        if log_m:
            self.logline = log_m.group(1).strip()

        cont_m = re.search(r'CONTINUITY NOTES:\s*(.*)', content)
        if cont_m:
            self.continuity = cont_m.group(1).strip()

        # Parse Scenes
        scene_headings = list(re.finditer(r'SCENE\s+(\d+)[\r\n]+Scene heading:\s*(.*?)[\r\n]+Scene start:\s*([0-9:.]+)[\r\n]+Scene end:\s*([0-9:.]+)', content, re.IGNORECASE))
        for sh in scene_headings:
            self.scenes.append({
                "scene_num": int(sh.group(1)),
                "heading": sh.group(2).strip(),
                "start_sec": parse_time_str(sh.group(3)),
                "end_sec": parse_time_str(sh.group(4)),
                "start": sh.group(3).strip(),
                "end": sh.group(4).strip()
            })

        # Parse timestamped blocks
        block_pattern = re.compile(
            r'^\s*\*{0,2}([^*\r\n]+?)\*{0,2}\s*\r?\n[ \t]*Start:\s*\(([0-9:.]+)\)\s*\|\s*End:\s*\(([0-9:.]+)\)\s*\r?\n([\s\S]*?)(?=^\s*\*{0,2}[^*\r\n]+\*{0,2}\s*\r?\n[ \t]*Start:|--- SCRIPT END ---|\Z)',
            re.MULTILINE
        )

        for match in block_pattern.finditer(content):
            tag = match.group(1)
            start_str = match.group(2)
            end_str = match.group(3)
            body = match.group(4).strip()

            st_sec = parse_time_str(start_str)
            en_sec = parse_time_str(end_str)

            # Determine scene index based on start time
            sc_idx = 1
            for sc in self.scenes:
                if sc["start_sec"] <= st_sec <= sc["end_sec"]:
                    sc_idx = sc["scene_num"]
                    break

            block = ScriptBlock(tag=tag, start_sec=st_sec, end_sec=en_sec, text=body, scene_idx=sc_idx)
            self.blocks.append(block)

        # Sort blocks by start_sec
        self.blocks.sort(key=lambda b: (b.start_sec, b.end_sec))

        # Also accept simple timestamped text rows such as:
        # [00:12.00 - 00:14.50] JULIANA: Don't touch me.
        if not self.blocks:
            row_pattern = re.compile(
                r"^\s*\[?(\d{1,2}:\d{2}(?:\.\d{1,3})?)\s*(?:--?>|[-–])\s*"
                r"(\d{1,2}:\d{2}(?:\.\d{1,3})?)\]?\s*(?:\*\*)?\s*"
                r"([^:\n]{1,80}):\s*(.*?)\s*$"
            )
            for line in content.splitlines():
                match = row_pattern.match(line)
                if not match:
                    continue
                start_sec = parse_time_str(match.group(1))
                end_sec = parse_time_str(match.group(2))
                if end_sec <= start_sec:
                    continue
                self.blocks.append(ScriptBlock(
                    tag=match.group(3).strip(" *"), start_sec=start_sec,
                    end_sec=end_sec, text=match.group(4).strip(),
                ))
            self.blocks.sort(key=lambda b: (b.start_sec, b.end_sec))
        if not self.blocks:
            raise ValueError(
                "No timestamped screenplay blocks found. Add time-coded dialogue/action "
                "or provide an SRT/VTT subtitle file; untimed prose cannot support exact promo cuts."
            )

    def _parse_subtitles(self, content: str):
        """Convert timed SRT/VTT cues to screenplay blocks, preserving cue timestamps."""
        cue_pattern = re.compile(
            r"(?m)^[ \t]*((?:\d{2}:)?\d{2}:\d{2}[,.]\d{3})[ \t]*-->[ \t]*"
            r"((?:\d{2}:)?\d{2}:\d{2}[,.]\d{3})(?:[ \t]+[^\r\n]*)?[ \t]*$"
        )
        matches = list(cue_pattern.finditer(content))
        for index, match in enumerate(matches):
            cue_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            cue_text = content[match.end():cue_end]
            voice = re.search(r"<v(?:\.[^ >]+)?\s+([^>]+)>", cue_text, re.IGNORECASE)
            cue_text = re.sub(r"<[^>]+>", "", cue_text)
            cue_text = html.unescape(cue_text)
            lines = [line.strip() for line in cue_text.splitlines() if line.strip()]
            lines = [line for line in lines if not re.fullmatch(r"\d+", line)]
            if not lines:
                continue
            body = " ".join(lines)
            start_sec = _subtitle_time(match.group(1))
            end_sec = _subtitle_time(match.group(2))
            if end_sec <= start_sec:
                continue
            if voice:
                tag = voice.group(1).strip()
            elif re.match(r"^\[[^]]+\]$", body):
                tag = "ACTION"
            elif re.match(r"^[A-Z][A-Z0-9 _'-]{1,40}:\s*", body):
                tag, body = body.split(":", 1)
                body = body.strip()
            else:
                tag = "DIALOGUE"
            self.blocks.append(ScriptBlock(tag, start_sec, end_sec, body))

    @property
    def total_duration(self) -> float:
        if self.blocks:
            return max(b.end_sec for b in self.blocks)
        if self.scenes:
            return max(s["end_sec"] for s in self.scenes)
        return 0.0

def load_all_episodes(directory: str) -> List[EpisodeScript]:
    """Load supported screenplay documents from a file or recursively from a folder."""
    if os.path.isfile(directory):
        files = [directory] if os.path.splitext(directory)[1].lower() in SUPPORTED_SCRIPT_EXTENSIONS else []
    elif os.path.isdir(directory):
        files = []
        for root, _, names in os.walk(directory):
            for name in names:
                if name.startswith("~$") or name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in SUPPORTED_SCRIPT_EXTENSIONS:
                    files.append(os.path.join(root, name))
    else:
        raise FileNotFoundError(f"Input path does not exist: {directory}")
    files.sort(key=lambda value: value.lower())
    episodes = []
    for full_path in files:
        try:
            ep = EpisodeScript(full_path)
            episodes.append(ep)
        except Exception as e:
            print(f"Skipping {os.path.basename(full_path)}: {e}")
    # Scripts without episode headers are still usable: preserve file order and
    # assign unused sequence numbers instead of collapsing every file into Ep 1.
    used = {ep.episode_num for ep in episodes if ep.episode_num > 0}
    next_number = 1
    for ep in episodes:
        if ep.episode_num <= 0:
            while next_number in used:
                next_number += 1
            ep.episode_num = next_number
            used.add(next_number)
    episodes.sort(key=lambda ep: ep.episode_num)
    return episodes

if __name__ == "__main__":
    import sys
    test_dir = r"kuku_scripts\I Married a Broken Soldier. He was a Duke"
    eps = load_all_episodes(test_dir)
    print(f"Successfully loaded {len(eps)} episodes.")
    print(f"Episode 1: {len(eps[0].blocks)} blocks, Total Duration: {format_time(eps[0].total_duration)}")
    print("First 3 blocks:", eps[0].blocks[:3])
