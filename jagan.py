"""
app.py - Interactive Streamlit UI for Kuku Promo Studio.
Run with: streamlit run app.py

Optional server-side settings (Streamlit secrets or environment variables):
  MODEL_NAME, MODEL_BASE_URL, MODEL_API_KEY  Demo model. The key stays on the server and is never sent to the browser.
  DEMO_KEY_RUN_LIMIT                         Runs per visitor session allowed on the demo key (default 5).
  KUKU_ALLOW_UNREDACTED=1                    Local use only: shows a switch to display script text.
                                             Leave it unset on any public host.
"""

import glob
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import streamlit as st

from kuku_promo_engine.pipeline import PromoPipeline
from kuku_promo_engine.config import Config
from kuku_promo_engine.cost_scale import PRICING_MODELS, calculate_show_cost, TOTAL_SHOW_DURATION_MINUTES
from kuku_promo_engine.quality_evaluator import GROUND_TRUTH_PROMOS
from kuku_promo_engine.audience_pulse import (
    AudiencePulseEvaluator,
    DEFAULT_VIEWER_COMMENTS,
    DEFAULT_PATTERNS_THAT_WORK,
)
from kuku_promo_engine.window_generator import CandidateWindow
from promo_helpers import (
    redact_transcript as _redact_transcript,
    scrub_quotes_text,
    suggest_ending_cut as _suggest_ending_cut,
)

import editor_tools as et
import editor_assistant as ea

st.set_page_config(
    page_title="Kuku Promo Studio | Editor's Room",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium blue hero styling
st.html("""
<style>
    .hero { background: radial-gradient(ellipse at 85% 12%, #315d9b 0%, transparent 42%), linear-gradient(115deg, #081a38, #123c78); color: #f3f7ff; border-radius: 22px; padding: 28px 32px 24px; margin-bottom: 18px; box-shadow: 0 12px 32px rgba(16,42,33,.12); }
    .hero-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 34px; }
    .hero-brand { color: #f3f7ff; font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .16em; text-transform: uppercase; }
    .hero-pill { border: 1px solid rgba(180,210,255,.42); border-radius: 30px; color: #dce8ff; padding: 7px 12px; font: 600 11px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .09em; text-transform: uppercase; }
    .hero-title { font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-size: clamp(2.4rem, 5vw, 4rem); line-height: .99; letter-spacing: -.055em; font-weight: 780; margin: 0 0 18px; color: #fbfcf6; }
    .hero-title span { color: #9bc7ff; }
    .hero-copy { max-width: 720px; color: #d1def4; font-size: 1rem; line-height: 1.65; margin: 0 0 22px; }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 12px 26px; border-top: 1px solid rgba(193,215,255,.25); padding-top: 14px; color: #c8d8f3; font: 600 12px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; text-transform: uppercase; }
    @media (max-width: 700px) { .hero { padding: 22px 19px; } .hero-top { align-items: flex-start; flex-direction: column; margin-bottom: 24px; } .hero-pill { font-size: 9px; } }
</style>
""")

# ----------------- SERVER-SIDE SETTINGS (never sent to the browser) -----------------
def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value) if value else os.getenv(name, default)


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SERVER_KEY = _secret("MODEL_API_KEY", getattr(Config, "MODEL_API_KEY", "") or "")
SERVER_MODEL = _secret("MODEL_NAME", getattr(Config, "MODEL_NAME", "") or "gpt-4o-mini")
SERVER_BASE = _secret("MODEL_BASE_URL", getattr(Config, "MODEL_BASE_URL", "") or "")
DEMO_KEY_RUN_LIMIT = _as_int(_secret("DEMO_KEY_RUN_LIMIT", "5"), 5)
ALLOW_UNREDACTED = os.getenv("KUKU_ALLOW_UNREDACTED", "0") == "1"
DEFAULT_MIN_DURATION = 20.0
DEFAULT_MAX_DURATION = 60.0
# Pre-computed results for the sample show, with all script text removed (see promo_helpers.py).
SAMPLE_RESULTS = os.path.join("sample_results", "promos_all.json")

HEURISTIC_SETTINGS = {
    "provider": "heuristic",
    "model_name": "heuristic_local",
    "api_key": "",
    "base_url": SERVER_BASE,
    "uses_demo_key": False,
    "error": None,
}


@contextmanager
def model_config(settings: dict, durations=None):
    """Apply model and duration settings to the shared Config for ONE run, then reset it.

    Config is process-wide state, so a visitor's key or settings must not outlive their run.
    (Fully fixing this means the engine taking a settings object instead of reading Config.)
    """
    try:
        Config.MODEL_BASE_URL = settings["base_url"]
        Config.update_model(settings["provider"], settings["model_name"], settings["api_key"])
        if durations:
            Config.MIN_PROMO_DURATION = float(durations[0])
            Config.MAX_PROMO_DURATION = float(durations[1])
        yield
    finally:
        Config.MODEL_BASE_URL = SERVER_BASE
        Config.update_model("heuristic", "heuristic_local", "")
        Config.MIN_PROMO_DURATION = DEFAULT_MIN_DURATION
        Config.MAX_PROMO_DURATION = DEFAULT_MAX_DURATION


# ----------------- SESSION STORAGE -----------------
SESSIONS_ROOT = os.path.join("outputs", "sessions")


def cleanup_old_sessions(max_age_hours: int = 24):
    """Delete old per-session output folders (they hold full script text)."""
    if not os.path.isdir(SESSIONS_ROOT):
        return
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(SESSIONS_ROOT):
        path = os.path.join(SESSIONS_ROOT, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


if "_promo_session_id" not in st.session_state:
    st.session_state["_promo_session_id"] = uuid.uuid4().hex
    cleanup_old_sessions()
st.session_state.setdefault("run_token", 0)
st.session_state.setdefault("demo_key_runs", 0)

SESSION_ID = st.session_state["_promo_session_id"]
SESSION_OUTPUT_DIR = os.path.join(SESSIONS_ROOT, SESSION_ID)
DECISIONS_FILE = os.path.join(SESSION_OUTPUT_DIR, "editorial_decisions.json")
ADJUSTMENTS_FILE = os.path.join(SESSION_OUTPUT_DIR, "editor_adjustments.json")


def load_editorial_decisions() -> dict:
    if os.path.exists(DECISIONS_FILE):
        try:
            with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_editorial_decision(promo_id: str, status: str, notes: str = ""):
    decisions = load_editorial_decisions()
    decisions[promo_id] = {
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "notes": notes
    }
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)


def load_editor_adjustments() -> dict:
    if os.path.exists(ADJUSTMENTS_FILE):
        try:
            with open(ADJUSTMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_editor_adjustment(promo_id: str, start_sec=None, end_sec=None):
    """Store the editor's nudged cut. Passing None resets the promo to the tool's original cut."""
    adjustments = load_editor_adjustments()
    if start_sec is None:
        adjustments.pop(promo_id, None)
    else:
        adjustments[promo_id] = {"start_sec": start_sec, "end_sec": end_sec, "updated_at": datetime.now().isoformat()}
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    with open(ADJUSTMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(adjustments, f, indent=2)


# ----------------- INPUT HELPERS -----------------
_EP_RE = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.IGNORECASE)
_LOGLINE_RE = re.compile(r"EPISODE LOGLINE:\s*(.+)", re.IGNORECASE)


def read_folder_texts(path: str, limit: int = 200) -> list:
    texts = []
    try:
        if path and os.path.isfile(path):
            files = [path]
        elif path and os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.txt")))[:limit]
        else:
            return []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read())
    except OSError:
        pass
    return texts


def extract_loglines(texts: list) -> dict:
    """Pull 'EPISODE LOGLINE' per episode when the script format has it; otherwise returns {}."""
    loglines = {}
    for text in texts:
        ep, logline = _EP_RE.search(text), _LOGLINE_RE.search(text)
        if ep and logline:
            loglines[int(ep.group(1))] = logline.group(1).strip()
    return loglines


# Sidebar Controls
st.sidebar.title("🎬 Promo Studio")
st.sidebar.caption("Turn episode scripts into short promo ideas.")

# Cross-platform relative path
_local_sample = os.path.join("kuku_scripts", "I Married a Broken Soldier. He was a Duke")
_demo_dir = "demo_scripts"  # put a small NON-confidential demo script here for public hosting
default_input = _local_sample if os.path.isdir(_local_sample) else (_demo_dir if os.path.isdir(_demo_dir) else "")
st.sidebar.markdown("### 1. Add your episodes")
uploaded_scripts = st.sidebar.file_uploader(
    "Choose script files",
    type=["txt", "md", "markdown", "srt", "vtt", "docx", "pdf", "rtf", "html", "htm", "csv"],
    accept_multiple_files=True,
    key="screenplay_uploads",
)
with st.sidebar.expander("Choose a folder instead"):
    input_dir = st.text_input("Folder or file path", value=default_input, key="screenplay_path")
    st.caption(
        "TXT, Markdown, SRT/VTT, DOCX, PDF, RTF, HTML, and CSV. "
        "Promo cuts need timestamps; untimed prose is skipped."
    )

with st.sidebar.expander('Advanced settings', expanded=False):
    st.subheader("🔒 Privacy")
    if ALLOW_UNREDACTED:
        confidential_mode = not st.checkbox(
            "Show script text (local use only)",
            value=False,
            help="Only available because KUKU_ALLOW_UNREDACTED=1 is set on this machine.",
        )
    else:
        confidential_mode = True
        st.caption("Script text is hidden in this view to protect Kuku's screenplays.")

    st.subheader("Scoring model")
    model_choice = st.selectbox("Judgment model", ["Local rules (free)", "OpenAI-compatible API"], index=0)
    typed_key = ""
    model_name = "heuristic_local"
    base_url = SERVER_BASE
    if model_choice == "OpenAI-compatible API":
        model_name = st.text_input("Model name", value=SERVER_MODEL)
        base_url = st.text_input("API base URL", value=SERVER_BASE)
        typed_key = st.text_input(
            "Your API key",
            value="",
            type="password",
            placeholder="Leave blank to use the demo key" if SERVER_KEY else "Paste a key",
        )
        if SERVER_KEY and not typed_key:
            left = max(0, DEMO_KEY_RUN_LIMIT - st.session_state["demo_key_runs"])
            st.caption(f"Demo key in use: {left} run(s) left in this session.")
    else:
        st.caption("Candidate cuts, timestamps and validation always run locally.")

    st.subheader("Promo length")
    dur_range = st.slider("Promo duration (seconds)", min_value=15, max_value=75, value=(20, 60))
    st.caption("This filters existing results. Click Analyze scripts to rebuild candidates.")


def build_model_settings() -> dict:
    """Resolve the model/key for one run. Falls back to local rules with a message when no key is usable."""
    if model_choice != "OpenAI-compatible API":
        return dict(HEURISTIC_SETTINGS)
    key = typed_key.strip()
    uses_demo_key = False
    error = None
    if not key and SERVER_KEY:
        if st.session_state["demo_key_runs"] < DEMO_KEY_RUN_LIMIT:
            key, uses_demo_key = SERVER_KEY, True
        else:
            error = "Demo key limit reached for this session. Paste your own key or use Local rules."
    if not key:
        return {**HEURISTIC_SETTINGS, "error": error or "Add an API key, or switch to Local rules."}
    return {
        "provider": "compatible",
        "model_name": model_name,
        "api_key": key,
        "base_url": base_url,
        "uses_demo_key": uses_demo_key,
        "error": None,
    }


# Data Loading. The run token (not a global cache clear) invalidates results after each Analyze.
@st.cache_data(show_spinner=False)
def load_pipeline_data(script_folder: str, output_dir: str, run_token: int):
    """This session's results if any; else the bundled sample results; else run on the folder."""
    json_path = os.path.join(output_dir, "promos_all.json")
    folder_exists = bool(script_folder) and os.path.exists(script_folder)

    def _read(path, source):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source"] = source
        return data

    if os.path.exists(json_path):
        return _read(json_path, "session")
    # Locally with the real scripts and unredacted mode on, prefer a live run over the text-free sample.
    if os.path.exists(SAMPLE_RESULTS) and not (ALLOW_UNREDACTED and folder_exists):
        return _read(SAMPLE_RESULTS, "sample")
    if folder_exists:
        with model_config(HEURISTIC_SETTINGS):
            PromoPipeline(input_dir=script_folder, output_dir=output_dir).run()
        if os.path.exists(json_path):
            return _read(json_path, "session")
    return {"summary": {"episodes_processed": 0}, "promos": [], "quality_benchmark": {}, "_source": "none"}


if st.sidebar.button("✨ Analyze scripts", type="primary"):
    settings = build_model_settings()
    if settings["error"]:
        st.sidebar.warning(f"{settings['error']} Using local rules for this run.")
    with st.spinner("Finding promo moments in your scripts..."):
        try:
            stale_json = os.path.join(SESSION_OUTPUT_DIR, "promos_all.json")
            if os.path.exists(stale_json):
                os.remove(stale_json)  # a failed run must never show the previous run's results
            if uploaded_scripts:
                texts = [u.getvalue().decode("utf-8", errors="ignore") for u in uploaded_scripts]
                with tempfile.TemporaryDirectory(prefix="kuku_screenplays_") as upload_dir:
                    for uploaded in uploaded_scripts:
                        safe_name = os.path.basename(uploaded.name)
                        with open(os.path.join(upload_dir, safe_name), "wb") as destination:
                            destination.write(uploaded.getbuffer())
                    with model_config(settings, dur_range):
                        run_result = PromoPipeline(input_dir=upload_dir, output_dir=SESSION_OUTPUT_DIR).run()
            else:
                texts = read_folder_texts(input_dir)
                with model_config(settings, dur_range):
                    run_result = PromoPipeline(input_dir=input_dir, output_dir=SESSION_OUTPUT_DIR).run()
            if settings["uses_demo_key"]:
                st.session_state["demo_key_runs"] += 1
            st.session_state["loglines"] = extract_loglines(texts)
            st.session_state["run_token"] += 1
            if run_result.get("message"):
                st.session_state["script_input_message"] = run_result["message"]
                st.sidebar.warning(run_result["message"])
            else:
                st.session_state.pop("script_input_message", None)
                st.sidebar.success(
                    f"Ready: {run_result['total_episodes']} episodes, "
                    f"{run_result['total_evaluated_promos']} promo ideas."
                )
        except Exception as e:
            st.session_state["script_input_message"] = str(e)
            st.sidebar.error(f"Could not process the selected input: {e}")

if (os.path.exists(SAMPLE_RESULTS)
        and os.path.exists(os.path.join(SESSION_OUTPUT_DIR, "promos_all.json"))
        and st.sidebar.button("↩ Return to demo results")):
    session_json = os.path.join(SESSION_OUTPUT_DIR, "promos_all.json")
    if os.path.exists(session_json):
        os.remove(session_json)
    st.session_state.pop("script_input_message", None)
    st.session_state["run_token"] += 1

if "loglines" not in st.session_state:
    st.session_state["loglines"] = extract_loglines(read_folder_texts(input_dir))

try:
    pipeline_data = load_pipeline_data(input_dir, SESSION_OUTPUT_DIR, st.session_state["run_token"])
    all_promos = pipeline_data.get("promos", [])
    benchmark_data = pipeline_data.get("quality_benchmark", {})
except Exception as e:
    st.error(f"Error loading pipeline data from {input_dir}: {e}")
    pipeline_data = {}
    all_promos = []
    benchmark_data = {}

if st.session_state.get("script_input_message"):
    all_promos = []
    benchmark_data = {}

is_sample = pipeline_data.get("_source") == "sample" and not st.session_state.get("script_input_message")

# Load editorial decisions
editorial_decisions = load_editorial_decisions()
editor_adjustments = load_editor_adjustments()
_stability = st.session_state.get("stability") or {}
stability_flags = _stability.get("flags", {}) if _stability.get("token") == st.session_state["run_token"] else {}

# Editorial room header
total_eps = pipeline_data.get('summary', {}).get('episodes_processed', 0)
approved_count = sum(1 for d in editorial_decisions.values() if d.get("status") == "APPROVED")
decision_count = sum(1 for d in editorial_decisions.values() if d.get("status") in {"APPROVED", "FLAGGED", "REJECTED"})
episode_meta = (f"{total_eps} episodes · prepared shortlist" if is_sample else f"{total_eps} episodes loaded") if total_eps else "Ready for your first script"

st.html(f"""
<section class="hero">
  <div class="hero-top">
    <div class="hero-brand">Kuku Promo Studio / Editor's Room</div>
    <div class="hero-pill">Audience evidence &nbsp;→&nbsp; Editorial decision</div>
  </div>
  <h1 class="hero-title">Give every promo cut<br><span>a closer read.</span></h1>
  <p class="hero-copy">A focused workspace for short-drama promos. Compare the opening, ending tension, viewer fit, and reveal risk—then decide what to keep.</p>
  <div class="hero-meta"><span>{episode_meta}</span><span>20–60 second promo windows</span><span>Editor-led review</span></div>
</section>
""")

def render_summary_cards():
    summary_cards = st.columns(4)
    summary_values = [
        ("Source coverage", str(total_eps), "episodes parsed"),
        ("Promo review queue", str(len(all_promos)), "suggested cuts"),
        ("Approved picks", str(approved_count), "saved for use"),
        ("Reviewed", str(decision_count), "editor decisions saved"),
    ]
    for column, (label, value, note) in zip(summary_cards, summary_values):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.subheader(value)
                st.caption(note)


if is_sample:
    with st.expander("Run summary", expanded=False):
        render_summary_cards()
        st.caption("These are script-review signals. Viewer retention and conversion need separate audience testing.")
        st.caption("Standalone clarity is a model estimate; the additional cold-viewer check runs on the top 20 candidates.")
else:
    render_summary_cards()
    st.caption("These are script-review signals. Viewer retention and conversion need separate audience testing.")
    st.caption("Standalone clarity is a model estimate; the additional cold-viewer check runs on the top 20 candidates.")
if is_sample and all_promos:
    st.success(
        "The demo is already loaded: 50 episodes and 150 promo suggestions. Script dialogue and full transcripts are hidden. Upload your own scripts in the left panel to analyze another show.",
        title="Ready to review the solution",
        icon=":material/check_circle:",
    )
    featured = all_promos[0]
    featured_segment = featured["segments"][0]
    with st.container(border=True):
        featured_text, featured_fit = st.columns([5, 1], vertical_alignment="center")
        featured_text.markdown("#### Start with the top suggestion")
        featured_text.caption(
            f"Rank #{featured.get('rank', 1)} · {featured['id']} · Episode {featured_segment['episode']} · "
            f"{featured['total_duration']} seconds"
        )
        featured_text.markdown(
            f"**Suggested cut:** `{featured_segment['start']} → {featured_segment['end']}`"
        )
        featured_text.caption("Open the Promo picks tab below to approve, flag, or adjust this suggestion.")
        featured_fit.metric("Fit", f"{featured['evaluation'].get('overall_score', 0)}/100")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 Promo picks",
    "02 Episodes",
    "03 Audience review",
    "04 Method & downloads",
    "05 Ask the editor"
])

# ----------------- REDACTION HELPERS -----------------
def scrub_quotes(text) -> str:
    """Hide quoted script lines inside free text (model explanations, failure examples)."""
    return scrub_quotes_text(text) if confidential_mode else str(text or "")


def redact_transcript(transcript: str) -> str:
    return _redact_transcript(transcript)


def _ordered_blocks(promo: dict) -> list:
    blocks = []
    for segment in promo.get("segments", []):
        segment_blocks = [
            block for block in segment.get("blocks", [])
            if block.get("type") in {"DIALOGUE", "ACTION"}
        ]
        blocks.extend(sorted(segment_blocks, key=lambda block: block.get("start_sec", 0)))
    return blocks


def get_candidate_edge_text(promo: dict, first: bool) -> str:
    """The first or last dialogue/action beat of a candidate (text hidden when redacted)."""
    blocks = _ordered_blocks(promo)
    if not blocks:
        return "No dialogue or action text is available."
    block = blocks[0] if first else blocks[-1]
    text = str(block.get("text", "")).strip()
    timestamp = f"[{block.get('start', '')}] "
    speaker = block.get("speaker")
    if confidential_mode:
        public_edge = promo.get("public_hook" if first else "public_cliffhanger")
        if public_edge:
            edges = public_edge if isinstance(public_edge, list) else [public_edge]
            return "\n".join(
                f"[{edge.get('timestamp', '')}] {edge.get('speaker') or ('Action' if edge.get('type') == 'ACTION' else 'Character')}: “{edge.get('text', '')}”"
                for edge in edges
            )
        if block.get("type") == "ACTION":
            return f"{timestamp}Action: [text hidden]"
        if block.get("redacted"):
            label = "Action" if block.get("type") == "ACTION" else (speaker or "Character")
            return f"{timestamp}{label}: [source text hidden]"
        words = len(text.split())
        return f"{timestamp}{speaker}: [line hidden, {words} words]" if speaker else f"{timestamp}[line hidden, {words} words]"
    if block.get("type") == "ACTION":
        return f"{timestamp}Action: {text}"
    return f"{timestamp}{speaker}: “{text}”" if speaker else f"{timestamp}{text}"


def first_beat_lead(promo: dict):
    """Seconds between the clip start and its first dialogue/action beat (brief: hook within 3 seconds)."""
    segments = promo.get("segments", [])
    if not segments:
        return None
    first_segment_blocks = sorted(
        [b for b in segments[0].get("blocks", []) if b.get("type") in {"DIALOGUE", "ACTION"}],
        key=lambda block: block.get("start_sec", 0),
    )
    if not first_segment_blocks:
        return None
    try:
        return float(first_segment_blocks[0].get("start_sec", 0)) - float(segments[0].get("start_sec", 0))
    except (TypeError, ValueError):
        return None


def get_ending_suggestion(promo: dict):
    """Use the suggestion pre-computed for the sample results; otherwise compute it from the script text."""
    if "precomputed_ending" in promo:
        return promo["precomputed_ending"]
    return _suggest_ending_cut(promo, confidential_mode)


def promos_export_df(promos: list, decisions: dict) -> pd.DataFrame:
    """Text-free table of promo picks: timings, scores and the editor's decision."""
    rows = []
    for p in promos:
        ev = p.get("evaluation", {})
        decision = decisions.get(p["id"], {})
        rows.append({
            "ID": p["id"],
            "Rank": p.get("rank"),
            "Rank_Reason": p.get("rank_reason", ""),
            "Type": p.get("candidate_type"),
            "Hook_Method": "Relaxed fallback" if p.get("relaxed_hook") else "Keyword rule",
            "Segments": "; ".join(f"Ep {s['episode']} {s['start']}-{s['end']}" for s in p.get("segments", [])),
            "Duration_sec": p.get("total_duration"),
            "Fit_score": ev.get("overall_score"),
            "Hook_score": ev.get("hook_score"),
            "Cliffhanger_score": ev.get("cliffhanger_score"),
            "Stands_alone": ev.get("stands_alone"),
            "Editor_Status": decision.get("status", "PENDING"),
            "Editor_Notes": decision.get("notes", ""),
        })
    return pd.DataFrame(rows)


def render_cut_adjuster(promo: dict, ending_suggestion):
    """Let the editor move a cut's start/end to a line boundary, with live checks."""
    segments = promo.get("segments", [])
    if promo.get("candidate_type") != "single" or len(segments) != 1:
        st.caption("Adjusting is available for single-episode cuts. Joined moments keep the tool's cut.")
        return
    blocks = _ordered_blocks(promo)
    starts, ends = et.line_boundaries(blocks, segments[0])
    if not starts or not ends:
        st.caption("This result has no line-level timing, so the cut cannot be adjusted.")
        return

    original = et.segment_range(segments[0])
    saved = editor_adjustments.get(promo["id"])
    base_start, base_end = (saved["start_sec"], saved["end_sec"]) if saved else original
    pid = promo["id"]
    c1, c2 = st.columns(2)
    start_i = c1.selectbox(
        "Start at line boundary", range(len(starts)), index=et.nearest_index(starts, base_start),
        format_func=lambda i: et.sec_to_tc(starts[i], True), key=f"adj_start_{pid}",
    )
    end_i = c2.selectbox(
        "End at line boundary", range(len(ends)), index=et.nearest_index(ends, base_end),
        format_func=lambda i: et.sec_to_tc(ends[i], True), key=f"adj_end_{pid}",
    )
    new_start, new_end = starts[start_i], ends[end_i]

    for check in et.validate_cut(blocks, new_start, new_end, ending_suggestion,
                                 min_d=float(dur_range[0]) if dur_range[0] >= 20 else 20.0, max_d=60.0):
        message = f"**{check['name']}:** {check['message']}"
        if check["status"] == "pass":
            st.success(message, icon=":material/check_circle:")
        elif check["status"] == "warn":
            st.warning(message, icon=":material/warning:")
        else:
            st.error(message, icon=":material/error:")

    b1, b2 = st.columns(2)
    if b1.button("Save adjusted cut", key=f"adj_save_{pid}"):
        save_editor_adjustment(pid, new_start, new_end)
        st.toast(f"{pid}: adjusted cut saved. The cut-list export will use it.")
        st.rerun()
    if saved and b2.button("Reset to tool's cut", key=f"adj_reset_{pid}"):
        save_editor_adjustment(pid)
        st.rerun()
    if saved:
        st.caption(f"Saved cut: {et.sec_to_tc(saved['start_sec'], True)} to {et.sec_to_tc(saved['end_sec'], True)}.")


def run_pipeline_once(settings: dict) -> list:
    """Re-run the pipeline on the current input and return its promos (used for the confidence check)."""
    with tempfile.TemporaryDirectory(prefix="kuku_stability_out_") as out_dir:
        if uploaded_scripts:
            with tempfile.TemporaryDirectory(prefix="kuku_stability_in_") as in_dir:
                for uploaded in uploaded_scripts:
                    with open(os.path.join(in_dir, os.path.basename(uploaded.name)), "wb") as dest:
                        dest.write(uploaded.getbuffer())
                with model_config(settings, dur_range):
                    PromoPipeline(input_dir=in_dir, output_dir=out_dir).run()
        else:
            with model_config(settings, dur_range):
                PromoPipeline(input_dir=input_dir, output_dir=out_dir).run()
        result_path = os.path.join(out_dir, "promos_all.json")
        if not os.path.exists(result_path):
            return []
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f).get("promos", [])


def render_disagreement(rows: list, key_prefix: str):
    """Editor's cut and the tool's cut side by side, for labels the tool missed or only partly matched."""
    if not rows:
        return
    ordered = sorted(rows, key=lambda r: (r["hit_top_k"], r["top_pick_iou"]))
    labels = {
        r["id"]: f"{r['id']} · Ep {r['episode']} · "
                 f"{'in top 15' if r['hit_top_k'] else 'NOT in top 15'} · top-pick IoU {r['top_pick_iou']}"
        for r in ordered
    }
    choice = st.selectbox("Pick a labelled moment", list(labels), format_func=lambda i: labels[i],
                          key=f"{key_prefix}_disagree")
    row = next(r for r in ordered if r["id"] == choice)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Editor's cut**")
            st.write(f"Episode {row['episode']} · `{et.sec_to_tc(row['human_range'][0], True)} → {et.sec_to_tc(row['human_range'][1], True)}`")
            if row.get("title"):
                st.caption(row["title"])
    with right:
        with st.container(border=True):
            st.markdown("**Tool's top pick for this episode**")
            if row["top_pick"] is None:
                st.write("The tool produced no candidate for this episode.")
            else:
                pk = row["top_pick"]
                st.write(f"`{pk['id']}` · rank {row['top_pick_rank']} · Fit {pk['evaluation'].get('overall_score')}/100")
                if row["top_pick_range"]:
                    st.write(f"`{et.sec_to_tc(row['top_pick_range'][0], True)} → {et.sec_to_tc(row['top_pick_range'][1], True)}`")
                st.caption(f"Overlap with the editor's cut (IoU): {row['top_pick_iou']}")
    st.html(et.timeline_html(row["human_range"], row["top_pick_range"]))
    if row["best_id"] and row["best_id"] != row["top_pick_id"]:
        st.caption(f"Closest candidate inside the top 15: `{row['best_id']}` (rank {row['best_rank']}, IoU {row['best_iou']}).")
    elif not row["hit_top_k"]:
        st.caption("No candidate in the top 15 overlaps this cut enough to count as the same moment.")
    if row["top_pick"] is not None and not confidential_mode:
        with st.expander("Script excerpt of the tool's pick"):
            st.text(row["top_pick"].get("formatted_transcript", ""))


# ----------------- TAB 1: PROMO PICKS -----------------
with tab1:
    st.subheader("Pick a promo to review")
    st.caption("Start with the top suggestions, then approve, flag, or reject each cut.")

    # Posting order, derived from the real top promos, with a one-line reason each
    if len(all_promos) >= 3:
        order_lines = []
        for label, p in zip(("First", "Next", "Also consider"), all_promos[:3]):
            why = scrub_quotes(p["evaluation"].get("why_it_works", "")).strip()
            why = (why[:140] + "…") if len(why) > 140 else why
            order_lines.append(
                f"- **{label}:** `{p['id']}` · Episode {p['segments'][0]['episode']} · "
                f"{p['total_duration']} sec. {why}"
            )
        stitched_examples = [p for p in all_promos if p["candidate_type"] == "stitched"]
        if stitched_examples:
            p_stitch = stitched_examples[0]
            seg_parts = " + ".join(f"Ep {s['episode']}: {s['start']}-{s['end']}" for s in p_stitch['segments'])
            order_lines.append(
                f"- **One clip from several episodes:** `{p_stitch['id']}` ({p_stitch['total_duration']} sec), using {seg_parts}."
            )
        with st.expander("Suggested order for posting", expanded=False):
            st.markdown("\n".join(order_lines))

    # Filters
    with st.expander("Filter promo ideas", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        type_filter = fc1.selectbox("Clip type", ["All types", "One episode", "Joined moments"])
        alone_filter = fc2.selectbox("First-time viewer", ["All", "Easy to follow", "Needs context"])
        status_filter = fc3.selectbox("Your decision", ["All", "Approved", "Flagged", "Pending"])
        min_score = fc4.slider("Minimum fit score", 50, 100, 50)

    search_query = st.text_input(
        "Find a promo",
        placeholder="Search promo ID, episode, or reason it works…",
        key="promo_search",
        help="Searches promo IDs, episode numbers, and the reason each promo works.",
    ).strip().casefold()
    sort_order = st.segmented_control(
        "Sort promo ideas",
        ["Recommended", "Highest fit", "Shortest first"],
        default="Recommended",
        key="promo_sort_order",
    )

    # Live filter application
    filtered_top = []
    for p in all_promos:
        if not (dur_range[0] <= p["total_duration"] <= dur_range[1]):
            continue
        if p["evaluation"]["overall_score"] < min_score:
            continue
        if type_filter == "One episode" and p["candidate_type"] != "single":
            continue
        if type_filter == "Joined moments" and p["candidate_type"] != "stitched":
            continue
        if alone_filter == "Easy to follow" and not p["evaluation"]["stands_alone"]:
            continue
        if alone_filter == "Needs context" and p["evaluation"]["stands_alone"]:
            continue

        p_status = editorial_decisions.get(p["id"], {}).get("status", "PENDING")
        if status_filter == "Approved" and p_status != "APPROVED":
            continue
        if status_filter == "Flagged" and p_status != "FLAGGED":
            continue
        if status_filter == "Pending" and p_status != "PENDING":
            continue

        searchable = " ".join([
            str(p.get("id", "")),
            " ".join(str(seg.get("episode", "")) for seg in p.get("segments", [])),
            str(p["evaluation"].get("why_it_works", "")),
        ]).casefold()
        if search_query and search_query not in searchable:
            continue
        filtered_top.append(p)

    if sort_order == "Highest fit":
        filtered_top.sort(key=lambda p: p["evaluation"].get("overall_score", 0), reverse=True)
    elif sort_order == "Shortest first":
        filtered_top.sort(key=lambda p: p.get("total_duration", 0))

    st.write(f"Showing **{len(filtered_top[:15])}** promo ideas")
    if not all_promos:
        st.info("Add a timestamped script in the left panel, then click **Analyze scripts** to see promo ideas here.")
    elif not filtered_top:
        st.info("No promo ideas match these filters. Try lowering the fit score or clearing a filter.")

    for idx, promo in enumerate(filtered_top[:15]):
        ev = promo["evaluation"]
        is_stitched = promo["candidate_type"] == "stitched"
        stands_alone = ev["stands_alone"]
        p_decision = editorial_decisions.get(promo["id"], {})
        curr_status = p_decision.get("status", "PENDING")

        status_labels = {
            "APPROVED": ("Approved", "green"),
            "FLAGGED": ("Flagged for a closer look", "orange"),
            "REJECTED": ("Skipped", "gray"),
            "PENDING": ("Awaiting your decision", "blue"),
        }
        status_label, status_color = status_labels.get(curr_status, status_labels["PENDING"])
        with st.container(border=True):
            head, fit = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"### Episode {promo['segments'][0]['episode']}")
            head.caption(f"Suggestion {promo.get('rank', idx + 1)} · {promo['id']} · {promo['total_duration']} sec")
            head.caption(promo.get("rank_reason", "Ranking reason unavailable."))
            fit.badge(f"Fit {ev['overall_score']}/100", color="green")

            with st.container(horizontal=True, gap="small"):
                st.badge(status_label, color=status_color)
                if is_stitched:
                    st.badge("Joined moments", color="blue")
                if promo.get("relaxed_hook"):
                    st.badge("Relaxed hook fallback", color="orange")
                st.badge("Easy to follow" if stands_alone else "Needs context", color="green" if stands_alone else "orange")
                if promo["id"] in stability_flags:
                    st.badge("Runs disagree: review", color="red")
                if promo["id"] in editor_adjustments:
                    st.badge("Cut adjusted", color="violet")

            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown("**Clip timing**")
                for segment in promo["segments"]:
                    st.markdown(
                        f"Episode {segment['episode']} · `{segment['start']} → {segment['end']}` · {segment['duration']} sec"
                    )

                st.markdown("**Hook** · first beat")
                st.write(get_candidate_edge_text(promo, first=True))
                lead = first_beat_lead(promo)
                if lead is not None and lead > 3.0:
                    st.warning(
                        f"The first beat arrives {lead:.1f}s into the clip; the brief asks for a hook in the first 3 seconds.",
                        icon=":material/timer:",
                    )

                st.markdown("**Cliffhanger** · last beat")
                st.write(get_candidate_edge_text(promo, first=False))

                score_bits = []
                if ev.get("hook_score") is not None:
                    score_bits.append(f"Hook {ev['hook_score']}/10")
                if ev.get("cliffhanger_score") is not None:
                    score_bits.append(f"Cliffhanger {ev['cliffhanger_score']}/10")
                if score_bits:
                    st.caption(" · ".join(score_bits))

                st.markdown("**Why it works**")
                st.write(scrub_quotes(ev.get("why_it_works", "")))

                ending_suggestion = get_ending_suggestion(promo)
                if ending_suggestion:
                    suggestion_intro = (
                        "A later line appears to resolve the suspense."
                        if ending_suggestion.get("kind") == "resolved ending"
                        else "The strongest tension may arrive before the current ending."
                    )
                    st.info(
                        f"{suggestion_intro} Try ending at "
                        f"**{ending_suggestion['end']}** ({ending_suggestion['duration']} sec total). "
                        f"Later source beat: {ending_suggestion.get('evidence', 'the current excerpt continues after the tension beat')}. "
                        f"Rule-based suggestion; the dialogue stays unchanged.",
                        title="Possible ending improvement",
                        icon=":material/lightbulb:",
                    )

            with col_b:
                st.markdown("**For a new viewer**")
                if stands_alone:
                    st.success("This moment should make sense without earlier episodes.")
                else:
                    st.info(f"This moment may need context: {scrub_quotes(ev.get('stands_alone_reason', ''))}")

                st.markdown("**Your decision**")
                act_c1, act_c2, act_c3 = st.columns(3)
                if act_c1.button("Use it", icon=":material/check:", key=f"app_{promo['id']}"):
                    save_editorial_decision(promo["id"], "APPROVED")
                    st.toast(f"{promo['id']} added to approved picks.")
                    st.rerun()

                if act_c2.button("Review", icon=":material/flag:", key=f"flg_{promo['id']}"):
                    save_editorial_decision(promo["id"], "FLAGGED")
                    st.toast(f"{promo['id']} flagged for another look.")
                    st.rerun()

                if act_c3.button("Skip", icon=":material/skip_next:", key=f"rej_{promo['id']}"):
                    save_editorial_decision(promo["id"], "REJECTED")
                    st.toast(f"{promo['id']} skipped.")
                    st.rerun()

            if promo["id"] in stability_flags:
                st.warning(f"Two runs of the model disagreed on this promo: {stability_flags[promo['id']]}.",
                           icon=":material/rule:")
            with st.expander("Adjust this cut"):
                render_cut_adjuster(promo, ending_suggestion)

            raw_transcript = promo.get("formatted_transcript", "Transcript unavailable.")
            displayed_transcript = redact_transcript(raw_transcript) if confidential_mode else raw_transcript
            with st.expander("View script text" + (" (hidden in public demo)" if confidential_mode else "")):
                st.text(displayed_transcript)

# ----------------- TAB 2: BROWSE EPISODES -----------------
with tab2:
    st.subheader("Browse moments by episode")
    st.caption("Choose an episode to review suggested clip timings. Script dialogue is hidden in the public demo.")
    ep_nums = sorted(list(set(s["episode"] for p in all_promos for s in p["segments"])))
    if ep_nums:
        selected_ep = st.selectbox("Episode", ep_nums, index=0)
        ep_promos = [p for p in all_promos if any(s["episode"] == selected_ep for s in p["segments"])]
        st.write(f"Found **{len(ep_promos)}** candidate promo windows in Episode {selected_ep}:")

        for p in ep_promos:
            with st.expander(f"{p['total_duration']} sec · Fit {p['evaluation']['overall_score']}/100"):
                st.write(f"**Clip timing:** {[s['start'] + ' – ' + s['end'] for s in p['segments']]}")
                st.write(f"**Hook (first beat):** {get_candidate_edge_text(p, first=True)}")
                st.write(f"**Cliffhanger (last beat):** {get_candidate_edge_text(p, first=False)}")
                st.write(f"**For a new viewer:** {'Easy to follow' if p['evaluation']['stands_alone'] else 'May need earlier context'}")
                raw_t = p.get("formatted_transcript", "")
                st.text(redact_transcript(raw_t) if confidential_mode else raw_t)
    else:
        st.info("No episodes loaded.")

# ----------------- TAB 4: METHOD, QUALITY, COST, DOWNLOADS -----------------
with tab4:
    st.subheader("Reports and downloads")
    st.caption("Optional details for checking quality, estimating costs, and downloading files.")

    mean_iou_val = benchmark_data.get("mean_iou")
    acc_val = benchmark_data.get("accuracy_pct")
    gt_count = benchmark_data.get("ground_truth_count", len(GROUND_TRUTH_PROMOS))

    benchmark_note = benchmark_data.get(
        "statistical_significance_note",
        "Reference cuts were used while building the tool; this is an exploratory sanity check, not independent validation.",
    )
    st.markdown(f"**Reference comparison:** {benchmark_note} Below is the comparison between the reference cuts and the tool's picks.")
    st.warning(
        f"These {gt_count} reference cuts were used while building the tool. Treat the numbers as an exploratory sanity check, not an independent accuracy benchmark."
    )

    bm_c1, bm_c2, bm_c3 = st.columns(3)
    bm_c1.metric("Reference Promos", f"{gt_count} Moments")
    bm_c2.metric("Mean IoU (Temporal Overlap)", f"{mean_iou_val}" if mean_iou_val is not None else "N/A")
    bm_c3.metric("Match Rate (IoU >= 0.35)", f"{acc_val}%" if acc_val is not None else "N/A")

    ranked_promos = et.rank_sorted(all_promos)
    gt_items = []
    for m in benchmark_data.get("detailed_matches", []):
        rng = et.parse_range(m.get("gt_range", ""))
        if rng:
            gt_items.append({"id": m["gt_id"], "episode": m["episode"], "start_sec": rng[0],
                             "end_sec": rng[1], "title": m.get("title", "")})
    scope_label = st.radio(
        "Top-15 check counts a hit when the human pick overlaps (IoU ≥ 0.35) a candidate in:",
        ["The whole show's top 15 (what the editor reviews)", "The top 15 within that episode"],
        horizontal=True, key="topk_scope",
    )
    topk_scope = "show" if scope_label.startswith("The whole") else "episode"
    gt_rows = et.evaluate_labels(gt_items, ranked_promos, k=15, thr=0.35, scope=topk_scope) if (gt_items and ranked_promos) else []
    gt_sum = et.hit_summary(gt_rows)
    tk_c1, tk_c2 = st.columns(2)
    tk_c1.metric("Human pick in tool's top 15", f"{gt_sum['hits']} of {gt_sum['n']} ({gt_sum['hit_pct']}%)" if gt_sum["n"] else "N/A")
    tk_c2.metric("Tool's #1 pick per episode: mean IoU", gt_sum["mean_top_pick_iou"] if gt_sum["n"] else "N/A")
    st.caption("These labels were used while building the tool, so treat them as a sanity check. The hold-out below is the fairer test.")

    st.markdown("### 📊 Side-by-Side Comparison")
    detailed_matches = benchmark_data.get("detailed_matches", [])
    if detailed_matches:
        match_rows = []
        for m in detailed_matches:
            match_rows.append({
                "Ground Truth ID": m["gt_id"],
                "Ep": m["episode"],
                "Title": m["title"],
                "Human Editor Range": m["gt_range"],
                "AI Tool Pick Range": m["tool_range"],
                "IoU (Overlap)": m["iou"],
                "Match?": "✅ Match" if m["matched"] else "❌ Divergence",
                "Stands Alone Match": "✅" if m["stands_alone_match"] else "❌"
            })
        st.dataframe(pd.DataFrame(match_rows), width="stretch")
    else:
        st.info("No benchmark matches found. Run pipeline to generate quality benchmark.")

    st.markdown("### Potential failure modes")
    failure_modes = benchmark_data.get("failure_modes", [])
    if failure_modes:
        st.markdown("Review notes from development:")
        for fm in failure_modes:
            with st.expander(f"🚨 {fm['category']}"):
                st.markdown(f"**Description:** {scrub_quotes(fm['description'])}")
                st.markdown(f"**Example:** *{scrub_quotes(fm['example'])}*")
                st.markdown(f"**Possible fix:** `{fm['fix']}`")
    else:
        st.info("No independent failure-mode review is included with these prepared results.")

    # ---- Hold-out labels ----
    st.markdown("### Hold-out check (labels from later episodes)")
    st.caption(
        "Upload hand-labels for episodes the tool was never tuned on (CSV or JSON with id, episode, start, end). "
        "They are scored on their own and never mixed into the numbers above."
    )
    holdout_file = st.file_uploader("Hold-out labels", type=["csv", "json"], key="holdout_labels")
    holdout_rows = []
    if holdout_file is not None:
        try:
            holdout_labels = et.parse_label_upload(holdout_file.getvalue(), holdout_file.name)
        except (ValueError, json.JSONDecodeError) as exc:
            holdout_labels = []
            st.error(f"Could not read the labels: {exc}")
        if holdout_labels and not ranked_promos:
            st.info("Run the tool on these episodes first, then upload the labels.")
        elif holdout_labels:
            holdout_rows = et.evaluate_labels(holdout_labels, ranked_promos, k=15, thr=0.35, scope=topk_scope)
            ho = et.hit_summary(holdout_rows)
            h1, h2, h3 = st.columns(3)
            h1.metric("Hold-out labels", ho["n"])
            h2.metric("Human pick in tool's top 15", f"{ho['hit_pct']}%")
            h3.metric("Mean IoU (tool's #1 pick)", ho["mean_top_pick_iou"])
            if ho["n"] < 10:
                st.warning(f"{ho['n']} hold-out labels is still a small sample; report it as indicative, not conclusive.")

    st.markdown("### Where we disagree")
    st.caption("The editor's cut next to the tool's top pick, worst misses first.")
    disagree_rows = holdout_rows if holdout_rows else gt_rows
    if disagree_rows:
        st.caption("Showing hold-out labels." if holdout_rows else "Showing the original labels (upload hold-out labels above to see those instead).")
        render_disagreement(disagree_rows, "holdout" if holdout_rows else "gt")
    else:
        st.info("Needs labelled moments and an analysed show.")

    st.markdown("### Run-to-run confidence")
    st.caption(
        "Re-runs the scoring once and compares it with the results above. Use the same model settings as the original run. "
        "Local rules are deterministic, so they should agree exactly; a model may not."
    )
    can_recheck = bool(all_promos) and not is_sample and bool(uploaded_scripts or (input_dir and os.path.exists(input_dir)))
    if st.button("Re-run and compare", key="stability_btn", disabled=not can_recheck):
        recheck_settings = build_model_settings()
        if recheck_settings["error"]:
            st.warning(f"{recheck_settings['error']} Using local rules for this check.")
        with st.spinner("Re-running the scoring..."):
            try:
                second = run_pipeline_once(recheck_settings)
                if recheck_settings["uses_demo_key"]:
                    st.session_state["demo_key_runs"] += 1
                comparison = et.compare_runs(all_promos, second)
                st.session_state["stability"] = {"token": st.session_state["run_token"], **comparison}
                st.rerun()
            except Exception as exc:
                st.error(f"Could not complete the second run: {exc}")
    if not can_recheck:
        st.caption("Available after you analyse your own scripts (not on the redacted sample).")
    if _stability.get("token") == st.session_state["run_token"]:
        st.metric("Top-15 picks that agree across the two runs", f"{_stability['top_overlap']} of {_stability['compared']}")
        if stability_flags:
            st.warning(f"{len(stability_flags)} promo(s) are flagged for human review: {', '.join(stability_flags)}.")
        else:
            st.success("No promo in the top 15 changed between the two runs.")

    # ---- Cost and scale ----
    st.subheader("Cost and scale (estimates)")
    avg_ep_min = TOTAL_SHOW_DURATION_MINUTES / 50.0
    st.markdown(
        f"Reference show: 50 episodes, **{TOTAL_SHOW_DURATION_MINUTES:.1f} minutes** "
        f"(~{avg_ep_min:.2f} min per episode, measured from the sample pack). "
        "The editor-time figures below are **assumptions you can change**, not measurements."
    )

    default_eps = min(max(_as_int(total_eps, 50) or 50, 20), 100)
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        ep_count = st.slider("Episodes per show", 20, 100, default_eps)
        cands_per_ep = st.slider("Candidates evaluated per episode", 5, 20, 10)
        manual_multiplier = st.slider("Assumed editor logging time (× footage length)", 0.5, 3.0, 1.4, 0.1)
    with col_sc2:
        model_sim = st.selectbox("Model pricing simulation", list(PRICING_MODELS.keys()), index=0)
        shows_per_year = st.slider("Shows released per year", 10, 200, 50)
        review_minutes = st.slider("Assumed shortlist review time (minutes)", 5, 60, 15)

    sim_res = calculate_show_cost(
        num_episodes=ep_count,
        candidates_per_episode=cands_per_ep,
        stitched_candidates=20,
        model_key=model_sim
    )
    footage_minutes = avg_ep_min * ep_count
    manual_minutes = footage_minutes * manual_multiplier
    time_saved_pct = max(0.0, (1 - review_minutes / manual_minutes) * 100) if manual_minutes else 0.0

    sc_m1, sc_m2, sc_m3, sc_m4 = st.columns(4)
    sc_m1.metric("Show total cost", f"${sim_res['total_cost_usd']:.4f}")
    sc_m2.metric("Cost per episode", f"${sim_res['cost_per_episode_usd']:.5f}")
    sc_m3.metric("Annual catalog cost", f"${sim_res['total_cost_usd'] * shows_per_year:.2f}")
    sc_m4.metric("Editor time saved (assumed)", f"{time_saved_pct:.1f}%")
    st.caption(
        f"Assumption: logging {footage_minutes:.0f} min of footage by hand takes ~{manual_minutes:.0f} min; "
        f"reviewing the shortlist takes ~{review_minutes} min."
    )

    st.markdown("### How a promo gets ready")
    workflow = [
        ("1", "Add scripts", "Choose an episode file or folder."),
        ("2", "Review moments", "Compare hooks, tension, and context."),
        ("3", "Choose a cut", "Use it, flag it, or skip it."),
        ("4", "Download", "Export your approved promo list."),
    ]
    workflow_columns = st.columns(4)
    for column, (number, title, detail) in zip(workflow_columns, workflow):
        with column:
            with st.container(border=True):
                st.badge(number, color="green")
                st.markdown(f"**{title}**")
                st.caption(detail)

    st.markdown(f"""
    - **Step 1 (deterministic filtering):** code builds candidate windows on shot/speech boundaries; this run kept **{len(all_promos)}** candidates.
    - **Step 2 (model or rule scoring):** hook, cliffhanger and stands-alone criteria, without moving timestamps.
    - **Step 3 (human review):** the editor reviews the top 10-15 instead of scrubbing the full footage.
    """)

    # ---- Downloads ----
    st.subheader("Data export & AI attribution")
    st.markdown("### 📥 Download outputs")
    d1, d2, d3, d4 = st.columns(4)
    export_df = promos_export_df(all_promos, editorial_decisions) if all_promos else pd.DataFrame()
    if not export_df.empty:
        approved_df = export_df[export_df["Editor_Status"] == "APPROVED"]
        d1.download_button("📥 Approved promos (CSV)", approved_df.to_csv(index=False),
                           "kuku_approved_promos.csv", "text/csv")
        d2.download_button("📥 Top 15 (CSV)", export_df.head(15).to_csv(index=False),
                           "kuku_top_promos.csv", "text/csv")
        d3.download_button("📥 All candidates (CSV)", export_df.to_csv(index=False),
                           "kuku_all_promos.csv", "text/csv")
    json_all_path = os.path.join(SESSION_OUTPUT_DIR, "promos_all.json")
    if not confidential_mode and os.path.exists(json_all_path):
        with open(json_all_path, "rb") as f:
            d4.download_button("📥 Full dataset with text (JSON)", f.read(), "kuku_promos_full.json", "application/json")
    else:
        d4.caption("Full-text JSON is disabled in the redacted view. The CSVs contain timings and scores only.")

    cut_df = et.build_cut_list(all_promos, editorial_decisions, editor_adjustments) if all_promos else pd.DataFrame()
    if not cut_df.empty:
        st.download_button("📥 Cut list for editors (CSV)", cut_df.to_csv(index=False),
                           "kuku_cut_list.csv", "text/csv", key="cut_list_download")
        st.caption(
            "One row per segment, with In and Out as HH:MM:SS.mmm, using your adjusted cut where you saved one. "
            "Times come from the script timestamps, so confirm they match the video's timecode start."
        )
    else:
        st.caption("Approve at least one promo to enable the editor cut list.")

    st.markdown("---")
    st.markdown("### Tools used and how the suggestions are produced")
    # NOTE: confirm every claim below is true of the engine and visible in your raw outputs before submitting.
    st.markdown("""
    **Tools used in development:**
    1. **Rule and heuristic engine:** produces the baseline results (free, offline, reproducible).
    2. **Antigravity AI assistant and Python AST tools:** built the regex parsers, boundary validation and Streamlit state handling.
    3. **Swappable model connector:** any OpenAI-compatible API (model name, base URL and key are settings).

    **Code** parses supported files, selects timestamped line boundaries, enforces duration and opening-hook rules, validates quoted lines, and ranks a small shortlist. **The selected model** judges hook, cliffhanger, standalone clarity, and spoiler risk; it never creates or moves timestamps. The top 20 also receive a clip-only cold-viewer check. Review the top 10-15 suggestions before use.

    **Known failure:** In an earlier experiment, an LLM proposed an Episode 1 cut ending at `00:35.00`, after Juliana says “Gerald?” but before “Why are you holding her hand?” The current deterministic line-boundary selector prevents that type of mid-line cut. It can still miss action-only or music-only hooks, and the continuity/spoiler judgment needs an editor.
    """)

# ----------------- TAB 3: AUDIENCE REVIEW -----------------
with tab3:
    st.subheader("Shape a promo around viewer reactions")
    st.caption(
        "Choose an episode and 3–4 promo ideas. Add the episode context and review how each cut fits your audience."
    )

    eligible_promos = [
        p for p in all_promos
        if p.get("candidate_type") == "single"
        and len(p.get("segments", [])) == 1
        and 20 <= float(p.get("total_duration", 0)) <= 60
    ]
    pulse_episodes = sorted({p["segments"][0]["episode"] for p in eligible_promos})
    if is_sample:
        st.info(
            "This tab needs the script text, which the bundled sample leaves out. "
            "Upload the scripts (or choose a folder) and click **Analyze scripts**, then come back."
        )
    elif not pulse_episodes:
        st.info("No single-episode candidates between 20 and 60 seconds are available.")
    else:
        pulse_ep = st.selectbox("1. Choose an episode", pulse_episodes, key="pulse_episode")
        episode_candidates = [p for p in eligible_promos if p["segments"][0]["episode"] == pulse_ep]
        episode_candidates.sort(key=lambda p: p.get("rank", 9999))
        cand_labels = {
            p["id"]: f"{p['id']} · {p['segments'][0]['start']}–{p['segments'][0]['end']} · {p['total_duration']:.1f}s"
            for p in episode_candidates
        }
        default_ids = [p["id"] for p in episode_candidates[:min(3, len(episode_candidates))]]
        selected_ids = st.multiselect(
            "2. Choose 3 or 4 promo ideas",
            options=list(cand_labels),
            default=default_ids,
            format_func=lambda candidate_id: cand_labels[candidate_id],
            max_selections=4,
            key=f"pulse_candidates_{pulse_ep}",
        )

        default_context = st.session_state.get("loglines", {}).get(_as_int(pulse_ep), "")
        pulse_context = st.text_input(
            "3. What is this episode about?",
            value=default_context,
            placeholder="One sentence about the story. Add any later reveal we should protect.",
            key=f"pulse_context_{pulse_ep}",
            help="Filled from the script's EPISODE LOGLINE when it has one. Edit freely.",
        )
        with st.expander("4. Viewer feedback and what usually works (sample text: replace with real comments)", expanded=False):
            comments_default = "\n".join(DEFAULT_VIEWER_COMMENTS)
            patterns_default = "\n".join(DEFAULT_PATTERNS_THAT_WORK)
            comments_text = st.text_area(
                "Viewer comments (one per line)", value=comments_default, height=190, key="pulse_comments"
            )
            patterns_text = st.text_area(
                "Patterns that work (one per line)", value=patterns_default, height=120, key="pulse_patterns"
            )

        st.markdown("**The selected moments**")
        promo_by_id = {p["id"]: p for p in episode_candidates}
        for candidate_id in selected_ids:
            with st.expander(cand_labels[candidate_id]):
                candidate_text = "\n".join(
                    line for line in promo_by_id[candidate_id].get("formatted_transcript", "").splitlines()
                    if "[CAMERA]" not in line and "[SOUND]" not in line
                )
                st.text(redact_transcript(candidate_text) if confidential_mode else candidate_text)
                ending_suggestion = get_ending_suggestion(promo_by_id[candidate_id])
                if ending_suggestion:
                    suggestion_label = (
                        "Possible resolved ending"
                        if ending_suggestion.get("kind") == "resolved ending"
                        else "Possible mid-scene ending"
                    )
                    before_text = "before the answer" if ending_suggestion.get("kind") == "resolved ending" else "at the earlier tension beat"
                    source_evidence = ending_suggestion.get("evidence")
                    consider = (
                        f"Consider {ending_suggestion['start']}–{ending_suggestion['end']} "
                        f"({ending_suggestion['duration']} sec), {before_text}."
                    )
                    st.info(
                        f"{consider} Source line: {source_evidence}" if source_evidence else consider,
                        title=suggestion_label,
                        icon=":material/lightbulb:",
                    )

        if st.button("✨ Review these promo ideas", type="primary", key="evaluate_pulse"):
            if len(selected_ids) not in (3, 4):
                st.error("Select exactly 3 or 4 candidates to run the review.")
            elif not pulse_context.strip():
                st.error("Add a one-line summary of the episode before reviewing.")
            else:
                pulse_candidates = []
                for candidate_id in selected_ids:
                    promo = promo_by_id[candidate_id]
                    candidate_segments = []
                    for segment in promo["segments"]:
                        segment_for_review = dict(segment)
                        segment_for_review["blocks"] = [
                            block for block in segment.get("blocks", [])
                            if block.get("type") in {"DIALOGUE", "ACTION"}
                        ]
                        candidate_segments.append(segment_for_review)
                    candidate = CandidateWindow(
                        segments=candidate_segments,
                        candidate_type=promo.get("candidate_type", "single"),
                    )
                    candidate.id = candidate_id
                    pulse_candidates.append(candidate)

                viewer_comments = [line.strip() for line in comments_text.splitlines() if line.strip()]
                patterns = [line.strip() for line in patterns_text.splitlines() if line.strip()]
                pulse_settings = build_model_settings()
                if pulse_settings["error"]:
                    st.warning(f"{pulse_settings['error']} Using local rules for this review.")
                try:
                    with st.spinner("Reviewing the selected windows..."):
                        with model_config(pulse_settings):
                            pulse_results = AudiencePulseEvaluator.evaluate_candidates(
                                episode_num=int(pulse_ep),
                                series_context=pulse_context.strip(),
                                candidates=pulse_candidates,
                                viewer_comments=viewer_comments,
                                patterns=patterns,
                                provider=pulse_settings["provider"],
                                api_key=pulse_settings["api_key"],
                                model_name=pulse_settings["model_name"],
                            )
                    if pulse_settings["uses_demo_key"]:
                        st.session_state["demo_key_runs"] += 1

                    # Guard model output: IDs and cut timestamps must belong to the supplied windows.
                    allowed_times = {
                        c.id: (
                            {line[1:line.find("]")] for line in c.formatted_transcript.splitlines()
                             if line.startswith("[") and "]" in line}
                            | {promo_by_id[c.id]["segments"][0]["start"], promo_by_id[c.id]["segments"][0]["end"]}
                        )
                        for c in pulse_candidates
                    }
                    by_result_id = {str(r.get("id")): r for r in pulse_results if isinstance(r, dict)}
                    clean_results = []
                    allowed_spoilers = {"none", "mild", "major"}
                    allowed_weak_kinds = {"dead air", "exposition", "resolved ending", "needs setup"}
                    for candidate_id in selected_ids:
                        result = by_result_id.get(candidate_id, {})
                        clean = {"id": candidate_id}
                        for key in ("hook_score", "cliff_score", "standalone_score"):
                            value = result.get(key, "unclear")
                            clean[key] = max(0, min(10, int(value))) if isinstance(value, (int, float)) else "unclear"
                        weak = result.get("weak_spot")
                        if (isinstance(weak, dict) and weak.get("at") in allowed_times[candidate_id]
                                and weak.get("kind") in allowed_weak_kinds):
                            clean["weak_spot"] = {"at": weak["at"], "kind": weak["kind"], "source": pulse_settings["provider"]}
                        else:
                            clean["weak_spot"] = None
                        better = result.get("better_cut")
                        if isinstance(better, dict) and better.get("start") in allowed_times[candidate_id] and better.get("end") in allowed_times[candidate_id]:
                            try:
                                start_seconds = float(better["start"].split(":")[0]) * 60 + float(better["start"].split(":")[1])
                                end_seconds = float(better["end"].split(":")[0]) * 60 + float(better["end"].split(":")[1])
                                valid_duration = 20 <= end_seconds - start_seconds <= 60
                            except (TypeError, ValueError, IndexError):
                                valid_duration = False
                            clean["better_cut"] = (
                                {"start": better["start"], "end": better["end"], "source": pulse_settings["provider"]}
                                if valid_duration else None
                            )
                        else:
                            clean["better_cut"] = None
                        # Fall back to the rule-based suggestion, and say so in the output.
                        ending_suggestion = get_ending_suggestion(promo_by_id[candidate_id])
                        if clean["better_cut"] is None and ending_suggestion:
                            clean["better_cut"] = {
                                "start": ending_suggestion["start"],
                                "end": ending_suggestion["end"],
                                "source": "rule-based",
                            }
                        if clean["weak_spot"] is None and ending_suggestion and ending_suggestion.get("kind") in allowed_weak_kinds:
                            clean["weak_spot"] = {
                                "at": ending_suggestion["end"],
                                "kind": ending_suggestion["kind"],
                                "source": "rule-based",
                            }
                        spoiler = result.get("spoiler_level")
                        clean["spoiler_level"] = spoiler if spoiler in allowed_spoilers else "none"
                        clean["pulse_match"] = str(result.get("pulse_match", "none"))
                        clean["why"] = scrub_quotes(" ".join(str(result.get("why", "unclear")).split()[:15]))
                        clean_results.append(clean)
                    st.session_state["pulse_results"] = clean_results
                    st.session_state["pulse_result_episode"] = pulse_ep
                except Exception as exc:
                    st.error(f"Audience Pulse review failed: {exc}")

        if st.session_state.get("pulse_results") and st.session_state.get("pulse_result_episode") == pulse_ep:
            result_json = json.dumps(st.session_state["pulse_results"], ensure_ascii=False, indent=2)
            st.markdown("**JSON review output**")
            st.code(result_json, language="json")
            st.download_button(
                "📥 Download review JSON",
                result_json,
                file_name=f"episode_{pulse_ep}_audience_pulse_review.json",
                mime="application/json",
                key=f"download_pulse_{pulse_ep}",
            )


# ----------------- TAB 5: ASK THE EDITOR (grounded assistant) -----------------
with tab5:
    st.subheader("Ask the editor")
    st.caption(
        "Ask about the results on screen, such as why one promo ranks above another or which promos sit in episodes 10-20. "
        "Answers use only this run's scores, timings and your decisions. Anything else gets \"Not in the results.\""
    )
    if is_sample:
        st.info("This is the redacted sample, so answers come from scores and timings only.", icon=":material/info:")
    use_model = st.toggle(
        "Use the selected model for open questions (optional)", value=False, key="assistant_use_model",
        help="Off: common questions are answered by code with no model call. On: other questions go to the Judgment model chosen in Advanced settings.",
    )
    chat_key = ("assistant_chat", st.session_state["run_token"])
    if st.session_state.get("assistant_chat_key") != chat_key:
        st.session_state["assistant_chat_key"] = chat_key
        st.session_state["assistant_chat"] = []
    history = st.session_state["assistant_chat"]

    if not all_promos:
        st.info("Analyse a script first, then ask questions about the results.")
    else:
        examples = ["Why is %s ranked above %s?" % (all_promos[0]["id"], all_promos[min(1, len(all_promos) - 1)]["id"]),
                    "Show promos in episodes 10-20", "Which promos are approved?", "Top 5 promos"]
        st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("source"):
                    st.caption({"results": "Answered from the results by code",
                                "model": "Answered by the model, checked against the results",
                                "fallback": "No model used"}.get(turn["source"], ""))
        question = st.chat_input("Ask about the promo results")
        if question:
            history.append({"role": "user", "content": question})
            model_settings = None
            if use_model:
                model_settings = build_model_settings()
                if model_settings["error"]:
                    st.warning(f"{model_settings['error']} Answering without a model.")
                    model_settings = None
            try:
                with st.spinner("Checking the results..."):
                    reply = ea.ask(
                        question, all_promos, editorial_decisions, editor_adjustments,
                        settings=model_settings, history=[{"role": t["role"], "content": t["content"]} for t in history[:-1]],
                        scrub=scrub_quotes,
                    )
                if reply["source"] == "model" and model_settings and model_settings["uses_demo_key"]:
                    st.session_state["demo_key_runs"] += 1
            except RuntimeError as exc:
                reply = {"answer": f"{ea.NOT_IN_RESULTS} {exc}", "source": "fallback"}
            history.append({"role": "assistant", "content": reply["answer"], "source": reply["source"]})
            st.rerun()
        if history and st.button("Clear chat", key="assistant_clear"):
            st.session_state["assistant_chat"] = []
            st.rerun()

